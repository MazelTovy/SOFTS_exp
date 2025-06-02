from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import random

warnings.filterwarnings('ignore')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    AMRC: Adaptive Mask with Representation Consistency
    This is the full method with both mask penalty and embedding consistency penalty
    """
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.mask_penalty_weight = getattr(args, 'mask_penalty_weight', 0.1)  # Weight for mask penalty
        self.emb_penalty_weight = getattr(args, 'emb_penalty_weight', 1)  # Weight for embedding penalty

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
        
    def compute_mask_penalty_loss(self, outputs, batch_y, batch_x, batch_x_mark, batch_y_mark, f_dim):
        """Compute penalty loss when mask performs better than original sequence"""
        criterion = nn.MSELoss(reduction='none')
        original_loss = criterion(outputs, batch_y).mean(dim=[1, 2])

        population = list(range(1, 48))
        local_rng = random.Random()
        local_rng.seed()
        mask_lengths = local_rng.sample(population, 12)

        mask_losses = []
        
        # Calculate max mask length (up to 2/3 of sequence)
        seq_len = batch_x.shape[1]
        max_mask_len = int(seq_len * 2 / 3)
        
        # Compute loss for each mask length
        for mask_len in mask_lengths:
            if mask_len > max_mask_len:
                continue
                
            # Apply mask
            masked_batch_x = self.apply_mask(batch_x, mask_len)
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            # Forward pass with masked input
            if self.args.output_attention:
                masked_outputs = self.model(masked_batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                masked_outputs = self.model(masked_batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
            masked_outputs = masked_outputs[:, -self.args.pred_len:, f_dim:]
            masked_loss = criterion(masked_outputs, batch_y).mean(dim=[1, 2])
            mask_losses.append(masked_loss)
        
        if not mask_losses:
            return torch.tensor(0.).to(self.device)
            
        # If any mask performs better than original, apply penalty
        mask_losses = torch.stack(mask_losses).permute(1, 0)
        improvement = original_loss.unsqueeze(1) - mask_losses
        improvement = torch.relu(improvement)
        d = improvement.max(-1).values
        indices = improvement.max(-1).indices
        
        mask_batch = []
        for i in range(batch_x.size(0)):
            if d[i] > 0:
                best_mask_idx = indices[i].item()
                mask_len = mask_lengths[best_mask_idx] if best_mask_idx < len(mask_lengths) else 1
                mask_x = self.apply_mask(batch_x[i:i+1], mask_len)
                mask_batch.append(mask_x)
            else:
                mask_batch.append(batch_x[i:i+1])
        
        if mask_batch:
            mask_batch = torch.cat(mask_batch, dim=0)
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            if self.args.output_attention:
                mask_output = self.model(mask_batch, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                mask_output = self.model(mask_batch, batch_x_mark, dec_inp, batch_y_mark)
            
            mask_output = mask_output[:, -self.args.pred_len:, f_dim:]
            penalty = nn.MSELoss()(outputs, mask_output)
        else:
            penalty = torch.tensor(0.).to(self.device)

        penalty_loss = penalty * self.mask_penalty_weight
        return penalty_loss

    def compute_embedding_penalty(self, outputs, batch_y, batch_x, batch_x_mark, f_dim):
        """Compute embedding-based consistency penalty loss"""
        criterion = nn.MSELoss()
        
        # Get input embeddings directly from encoder
        if hasattr(self.model, 'module'):
            # Multi-GPU case
            model = self.model.module
        else:
            # Single GPU case
            model = self.model
            
        # Check model type and get embeddings accordingly
        model_name = self.args.model
        
        if model_name.startswith('i'):  # iTransformer, iInformer, etc.
            # For inverted models, embeddings are already processed differently
            # B L N -> B N E
            enc_out = model.enc_embedding(batch_x, batch_x_mark)
            if hasattr(model, 'encoder'):
                enc_out, _ = model.encoder(enc_out, attn_mask=None)
            input_emb = enc_out
        else:
            # For standard models
            # B L N -> B L E
            enc_out = model.enc_embedding(batch_x, batch_x_mark)
            if hasattr(model, 'encoder'):
                enc_out, _ = model.encoder(enc_out, attn_mask=None)
            input_emb = enc_out
        
        # First pass MSE loss
        mse_loss = criterion(outputs, batch_y)
        
        # Calculate embedding difference for corresponding input time steps
        # For inverted models, we need to handle the different dimension ordering
        if model_name.startswith('i'):
            # For inverted models: B N E
            input_emb_last = input_emb
        else:
            # For standard models: B L E -> focus on last seq_len steps
            input_emb_last = input_emb[:, -self.args.seq_len:, :]
        
        # Calculate sample-wise embedding similarity within each batch
        batch_size = input_emb_last.size(0)
        
        # Flatten the embeddings for comparison
        input_emb_flat = input_emb_last.reshape(batch_size, -1)
        
        emb_sim_matrix = torch.zeros(batch_size, batch_size, device=self.device)
        
        for j in range(batch_size):
            # Calculate MSE distance between this sample's embedding and all others
            sample_emb = input_emb_flat[j:j+1]  # [1, flatten_dim]
            emb_diff = F.mse_loss(
                sample_emb.expand(batch_size, -1),
                input_emb_flat,
                reduction='none'
            ).mean(dim=1)  # [batch_size]
            emb_sim_matrix[j] = emb_diff
        
        # Normalize similarity scores
        emb_sim_matrix = emb_sim_matrix / (emb_sim_matrix.mean() + 1e-8)
        
        # Calculate output difference matrix
        output_diff_matrix = torch.zeros(batch_size, batch_size, device=self.device)
        
        for j in range(batch_size):
            # Calculate MSE between this sample's target and all others' targets
            sample_target = batch_y[j:j+1]  # [1, pred_len, dim]
            target_diff = F.mse_loss(
                sample_target.expand(batch_size, -1, -1),
                batch_y,
                reduction='none'
            ).mean(dim=(1, 2))  # [batch_size]
            output_diff_matrix[j] = target_diff
        
        # Normalize output differences
        output_diff_matrix = output_diff_matrix / (output_diff_matrix.mean() + 1e-8)
        
        # Identify pairs where:
        # 1. Embeddings are similar (low emb_sim)
        # 2. But outputs differ greatly (high output_diff)
        # Such pairs indicate potentially problematic data
        penalty_matrix1 = torch.relu(output_diff_matrix - emb_sim_matrix)
        
        # Also identify pairs where:
        # 1. Embeddings are different (high emb_sim)
        # 2. But outputs are similar (low output_diff)
        penalty_matrix2 = torch.relu(emb_sim_matrix - output_diff_matrix)
        
        # Combine both penalties
        penalty = penalty_matrix1.mean() + penalty_matrix2.mean()
        
        return mse_loss, penalty

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

        # Flags to enable/disable different penalties
        use_mask_penalty = getattr(self.args, 'use_mask_penalty', True)
        use_emb_penalty = getattr(self.args, 'use_emb_penalty', True)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            mask_penalty_values = []
            emb_penalty_values = []
            mse_losses = []

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
                        
                        # Compute MSE loss
                        mse_loss = criterion(outputs, batch_y)
                        total_loss = mse_loss
                        
                        # Add mask penalty if enabled
                        if use_mask_penalty:
                            mask_penalty = self.compute_mask_penalty_loss(outputs, batch_y, batch_x, batch_x_mark, batch_y_mark, f_dim)
                            total_loss = total_loss + mask_penalty
                            mask_penalty_values.append(mask_penalty.item())
                        
                        # Add embedding penalty if enabled
                        if use_emb_penalty:
                            _, emb_penalty = self.compute_embedding_penalty(outputs, batch_y, batch_x, batch_x_mark, f_dim)
                            total_loss = total_loss + self.emb_penalty_weight * emb_penalty
                            emb_penalty_values.append(emb_penalty.item())
                        
                        mse_losses.append(mse_loss.item())
                        loss = total_loss

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    # Compute MSE loss
                    mse_loss = criterion(outputs, batch_y)
                    total_loss = mse_loss
                    
                    # Add mask penalty if enabled
                    if use_mask_penalty:
                        mask_penalty = self.compute_mask_penalty_loss(outputs, batch_y, batch_x, batch_x_mark, batch_y_mark, f_dim)
                        total_loss = total_loss + mask_penalty
                        mask_penalty_values.append(mask_penalty.item())
                    
                    # Add embedding penalty if enabled
                    if use_emb_penalty:
                        _, emb_penalty = self.compute_embedding_penalty(outputs, batch_y, batch_x, batch_x_mark, f_dim)
                        total_loss = total_loss + self.emb_penalty_weight * emb_penalty
                        emb_penalty_values.append(emb_penalty.item())
                    
                    mse_losses.append(mse_loss.item())
                    loss = total_loss

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    
                    # Display detailed loss info
                    mse_avg = np.mean(mse_losses[-100:])
                    msg = f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss_float:.7f} | mse: {mse_avg:.7f}"
                    
                    if use_mask_penalty and mask_penalty_values:
                        mask_avg = np.mean(mask_penalty_values[-100:])
                        msg += f" | mask_penalty: {mask_avg:.7f}"
                    
                    if use_emb_penalty and emb_penalty_values:
                        emb_avg = np.mean(emb_penalty_values[-100:])
                        msg += f" | emb_penalty: {emb_avg:.7f}"
                    
                    print(msg)
                    
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
            
            train_loss = np.average(train_loss) if train_loss else 0
            mask_penalty_avg = np.average(mask_penalty_values) if mask_penalty_values else 0
            emb_penalty_avg = np.average(emb_penalty_values) if emb_penalty_values else 0
            
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Mask Penalty: {3:.7f} Emb Penalty: {4:.7f} Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, mask_penalty_avg, emb_penalty_avg, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
            
        # Check if save_model attribute exists, if not, set default to True
        if not hasattr(self.args, 'save_model'):
            self.args.save_model = True
            
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
            
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
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
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return 