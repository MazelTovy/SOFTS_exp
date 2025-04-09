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


class CombinedLoss(nn.Module):
    def __init__(self, lambda_penalty=1, sim_threshold=0.7, dist_threshold=0.5, eps=1e-8, skipping_steps=5, activate_after=1000, logging_freq=50):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_penalty = lambda_penalty
        self.sim_threshold = sim_threshold
        self.dist_threshold = dist_threshold
        self.eps = eps  # For numerical stability in cosine similarity
        
        # Parameters for controlling when to apply penalty
        self.step_counter = 0
        self.skipping_steps = skipping_steps  # Apply penalty every n steps
        self.activate_after = activate_after  # Only start applying penalty after n steps
        self.logging_freq = logging_freq      # How often to log penalty statistics
        
        # Metrics for logging
        self.total_pairs = 0
        self.penalized_pairs = 0
        self.avg_penalty = 0.0
        self.max_penalty = 0.0

    def forward(self, outputs, targets, embeddings=None):
        # Standard MSE Loss
        mse = self.mse_loss(outputs, targets)
        
        # Increment step counter
        self.step_counter += 1
        
        # --- Penalty Term Calculation ---
        penalty = 0.0
        if embeddings is not None and self.step_counter >= self.activate_after and self.step_counter % self.skipping_steps == 0:
            batch_size = embeddings.size(0)
            
            if batch_size > 1 and self.lambda_penalty > 0:
                # Get shape information
                if len(embeddings.shape) > 2:  # If embeddings are multi-dimensional
                    # Flatten embeddings - assumes shape [batch, sequence_len, features]
                    emb_flat = embeddings.reshape(batch_size, -1)
                else:
                    emb_flat = embeddings
                
                # Flatten targets for consistent distance calculation
                tgt_flat = targets.reshape(batch_size, -1)
                
                # Normalize embeddings for cosine similarity calculation
                emb_norm = emb_flat / (torch.norm(emb_flat, dim=1, keepdim=True) + self.eps)
                
                # Calculate pairwise cosine similarity (batch_size x batch_size)
                sim_matrix = torch.matmul(emb_norm, emb_norm.t())
                
                # Calculate pairwise target distance (MSE between targets)
                # We expand dimensions to compute pairwise distances efficiently
                tgt_expanded_1 = tgt_flat.unsqueeze(1)  # [batch, 1, dim]
                tgt_expanded_2 = tgt_flat.unsqueeze(0)  # [1, batch, dim]
                dist_matrix_sq = ((tgt_expanded_1 - tgt_expanded_2)**2).mean(dim=2)  # [batch, batch]
                
                # --- Identify pairs exceeding thresholds ---
                # Create a mask to ignore diagonal elements (self-similarity)
                mask = ~torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
                
                # Apply thresholds
                sim_over_thresh = sim_matrix > self.sim_threshold
                dist_over_thresh = dist_matrix_sq > self.dist_threshold**2  # Compare squared distance
                
                # Combine conditions and mask
                penalize_mask = sim_over_thresh & dist_over_thresh & mask
                
                # For logging and threshold tuning
                self.total_pairs = mask.sum().item()
                self.penalized_pairs = penalize_mask.sum().item()
                
                if penalize_mask.any():
                    # Calculate penalty for identified pairs
                    penalty_pairs = (sim_matrix[penalize_mask] - self.sim_threshold) * \
                                   (torch.sqrt(dist_matrix_sq[penalize_mask] + self.eps) - self.dist_threshold)
                    
                    # Average penalty over the pairs that meet the criteria
                    penalty = penalty_pairs.mean()
                    
                    # Store metrics for logging
                    self.avg_penalty = penalty.item()
                    self.max_penalty = penalty_pairs.max().item() if penalty_pairs.numel() > 0 else 0
                
                # Log statistics for monitoring and threshold tuning
                if self.step_counter % self.logging_freq == 0:
                    penalty_ratio = self.penalized_pairs / max(1, self.total_pairs)
                    print(f"\nPenalty Stats: Step {self.step_counter}")
                    print(f"  Pairs satisfying conditions: {self.penalized_pairs}/{self.total_pairs} ({penalty_ratio:.2%})")
                    print(f"  Avg penalty value: {self.avg_penalty:.4f}, Max: {self.max_penalty:.4f}")
                    print(f"  Thresholds - Similarity: {self.sim_threshold:.2f}, Distance: {self.dist_threshold:.2f}")
                    print(f"  Current lambda_penalty: {self.lambda_penalty:.4f}")
        
        # Combine losses
        total_loss = mse + self.lambda_penalty * penalty
        return total_loss


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
        # Pass the hyperparameters from args with fallbacks for missing parameters
        lambda_penalty = getattr(self.args, 'lambda_penalty', 1)  # Default to 1 if not provided
        sim_threshold = getattr(self.args, 'sim_threshold', 0.7)    # Default to 0.7 if not provided
        dist_threshold = getattr(self.args, 'dist_threshold', 0.5)  # Default to 0.5 if not provided
        
        print(f"Setting up CombinedLoss with lambda_penalty={lambda_penalty}, " 
              f"sim_threshold={sim_threshold}, dist_threshold={dist_threshold}")
              
        criterion = CombinedLoss(
            lambda_penalty=lambda_penalty,
            sim_threshold=sim_threshold,
            dist_threshold=dist_threshold
        )
        return criterion

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
                        # Get embeddings for validation
                        try:
                            outputs, embeddings = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_embeddings=True)
                        except Exception as e:
                            print(f"Warning: Could not get embeddings in validation: {e}")
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            embeddings = None
                else:
                    # Get embeddings for validation
                    try:
                        outputs, embeddings = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_embeddings=True)
                    except Exception as e:
                        print(f"Warning: Could not get embeddings in validation: {e}")
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        embeddings = None
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Pass embeddings to criterion during validation
                try:
                    if embeddings is not None:
                        loss = criterion(outputs, batch_y, embeddings)
                    else:
                        loss = criterion(outputs, batch_y)
                except Exception as e:
                    print(f"Warning: Error in validation loss calculation: {e}")
                    loss = nn.MSELoss()(outputs, batch_y)  # Fallback to standard MSE
                
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

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
                try:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            # Get embeddings for training
                            try:
                                outputs, embeddings = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_embeddings=True)
                            except Exception as e:
                                print(f"Warning: Could not get embeddings in training: {e}")
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                embeddings = None

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            
                            # Modified to pass embeddings to loss function
                            loss = criterion(outputs, batch_y, embeddings)
                    else:
                        # Get embeddings for training
                        try:
                            outputs, embeddings = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_embeddings=True)
                        except Exception as e:
                            print(f"Warning: Could not get embeddings in training: {e}")
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            embeddings = None

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        # Modified to pass embeddings to loss function
                        loss = criterion(outputs, batch_y, embeddings)
                except Exception as e:
                    print(f"Error during forward/loss calculation: {e}")
                    # Fallback to standard processing without embeddings
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = nn.MSELoss()(outputs, batch_y)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = nn.MSELoss()(outputs, batch_y)

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
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
            train_loss = np.average(train_loss)
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
