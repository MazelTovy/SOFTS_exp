from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def apply_mask(self, batch_x, mask_length):
        """Apply masking to the beginning of sequence up to mask_length"""
        masked_batch = batch_x.clone()
        masked_batch[:, :mask_length, :] = 0.
        return masked_batch
        
    def compute_mask_penalty_loss(self, outputs, batch_y, criterion, batch_x, f_dim):
        """Compute penalty loss when mask performs better than original sequence"""
        original_loss = criterion(outputs, batch_y)
        mask_losses = []
        mask_lengths = []
        
        # Calculate max mask length (up to 2/3 of sequence)
        seq_len = batch_x.shape[1]
        max_mask_len = int(seq_len * 2 / 3)
        
        # Create list of mask lengths with increasing step sizes
        step = 3
        for i in range(0, max_mask_len, step):
            mask_lengths.append(i)
            # Increase step size as we get deeper into the sequence
            if i > 30:
                step = 6
            if i > 60:
                step = 9
                
        # Compute loss for each mask length
        for mask_len in mask_lengths:
            if mask_len == 0:  # Skip when mask length is 0 (same as original)
                continue
                
            # Apply mask
            masked_batch_x = self.apply_mask(batch_x, mask_len)
            
            # Forward pass with masked input
            if self.args.output_attention:
                masked_outputs = self.model(masked_batch_x, None, None, None)[0]
            else:
                masked_outputs = self.model(masked_batch_x, None, None, None)
                
            masked_outputs = masked_outputs[:, -self.args.pred_len:, f_dim:]
            masked_loss = criterion(masked_outputs, batch_y)
            mask_losses.append(masked_loss.item())
        
        # If any mask performs better than original, apply penalty
        mask_losses = np.array(mask_losses)
        if len(mask_losses) > 0 and np.min(mask_losses) < original_loss.item():
            # Calculate penalty based on improvement
            min_mask_loss = np.min(mask_losses)
            best_mask_idx = np.argmin(mask_losses)
            best_mask_len = mask_lengths[best_mask_idx+1] if best_mask_idx+1 < len(mask_lengths) else mask_lengths[best_mask_idx]
            improvement = original_loss.item() - min_mask_loss
            # penalty_factor = 0.5  # Adjustable factor
            penalty_factor = 5
            penalty_loss = original_loss + improvement * penalty_factor
            
            # Print detailed comparison info (every 100 iters to avoid flooding logs)
            if hasattr(self, 'log_counter'):
                self.log_counter += 1
            else:
                self.log_counter = 0
                
            if self.log_counter % 100 == 0:
                print("\n------ Mask Loss Comparison ------")
                print(f"Original Loss: {original_loss.item():.7f}")
                print(f"Best Mask Loss: {min_mask_loss:.7f} (mask length: {best_mask_len})")
                print(f"Improvement: {improvement:.7f} ({improvement/original_loss.item()*100:.2f}%)")
                print(f"Applied Penalty Factor: {penalty_factor}")
                print(f"Final Loss with Penalty: {penalty_loss.item():.7f}")
                print("---------------------------------\n")
                
            return penalty_loss
            
        return original_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))
        total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Flag to enable/disable mask penalty
        use_mask_penalty = getattr(self.args, 'use_mask_penalty', True)
        
        # Track original vs penalty losses
        epoch_orig_losses = []
        epoch_penalty_losses = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            orig_losses = []
            penalty_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        # Calculate original loss for comparison
                        original_loss = criterion(outputs, batch_y)
                        orig_losses.append(original_loss.item())
                        
                        if use_mask_penalty:
                            loss = self.compute_mask_penalty_loss(outputs, batch_y, criterion, batch_x, f_dim)
                            penalty_losses.append(loss.item())
                        else:
                            loss = original_loss

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    # Calculate original loss for comparison
                    original_loss = criterion(outputs, batch_y)
                    orig_losses.append(original_loss.item())
                    
                    if use_mask_penalty:
                        loss = self.compute_mask_penalty_loss(outputs, batch_y, criterion, batch_x, f_dim)
                        penalty_losses.append(loss.item())
                    else:
                        loss = original_loss

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    
                    # Display additional loss info
                    if use_mask_penalty:
                        orig_avg = np.mean(orig_losses[-100:])
                        penalty_avg = np.mean(penalty_losses[-100:])
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | orig_loss: {3:.7f} | penalty: {4:.7f}".format(
                            i + 1, epoch + 1, loss_float, orig_avg, penalty_avg))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            # Average losses for this epoch
            train_loss = np.average(train_loss) if train_loss else 0
            
            # Calculate and store epoch avg losses
            if use_mask_penalty and orig_losses and penalty_losses:
                epoch_orig_avg = np.average(orig_losses)
                epoch_penalty_avg = np.average(penalty_losses)
                epoch_orig_losses.append(epoch_orig_avg)
                epoch_penalty_losses.append(epoch_penalty_avg)
                
                print("Epoch Losses - Original: {:.7f}, With Penalty: {:.7f}, Difference: {:.7f} ({:.2f}%)".format(
                    epoch_orig_avg, epoch_penalty_avg, 
                    epoch_penalty_avg - epoch_orig_avg,
                    (epoch_penalty_avg - epoch_orig_avg) / epoch_orig_avg * 100 if epoch_orig_avg != 0 else 0
                ))
            
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # Print final summary of loss comparison if using mask penalty
        if use_mask_penalty and epoch_orig_losses and epoch_penalty_losses:
            print("\n===== Training Loss Summary =====")
            print("Dataset: {}, seq_len: {}, pred_len: {}".format(
                self.args.data, self.args.seq_len, self.args.pred_len))
            print("Avg Original Loss: {:.7f}".format(np.mean(epoch_orig_losses)))
            print("Avg Penalty Loss: {:.7f}".format(np.mean(epoch_penalty_losses)))
            print("Avg Difference: {:.7f} ({:.2f}%)".format(
                np.mean(epoch_penalty_losses) - np.mean(epoch_orig_losses),
                (np.mean(epoch_penalty_losses) - np.mean(epoch_orig_losses)) / np.mean(epoch_orig_losses) * 100 
                if np.mean(epoch_orig_losses) != 0 else 0
            ))
            print("================================\n")
            
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
            
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mse = AverageMeter()
        mae = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

        mse = mse.avg
        mae = mae.avg
        print('mse:{}, mae:{}'.format(mse, mae))

        return
