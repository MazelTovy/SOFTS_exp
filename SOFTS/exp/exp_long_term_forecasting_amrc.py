from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
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
        model_optim = optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate,
        )
        return model_optim

    def _select_criterion(self):
        # Basic MSE loss - we'll handle the embedding penalty separately in the train loop
        criterion = nn.MSELoss()
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
                
                # Use only MSE for validation
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            penalty_values = []

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

                # target dimension
                f_dim = -1 if self.args.features == 'MS' else 0
                target_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Run model with embedding computation
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # Run the model to get outputs
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        
                        # Get input embeddings directly from encoder
                        input_emb = self.model.module.enc_embedding(batch_x, batch_x_mark) if self.args.use_multi_gpu else self.model.enc_embedding(batch_x, batch_x_mark)
                        input_emb = self.model.module.encoder(input_emb, attn_mask=None)
                        
                        # First pass MSE loss
                        mse_loss = criterion(outputs, target_y)
                        
                        # Calculate value difference between predictions and targets
                        value_diff = torch.abs(outputs - target_y).mean(dim=2)  # [B, L]
                        value_diff_norm = value_diff / (value_diff.mean() + 1e-8)  # Normalize
                        
                        # Calculate embedding difference for corresponding input time steps
                        # Focus on last seq_len steps that contributed most to the prediction
                        input_emb_last = input_emb[:, -self.args.seq_len:, :]
                        
                        # Calculate sample-wise embedding similarity within each batch
                        # For each sample i, compute embedding similarity with all other samples j
                        batch_size = input_emb_last.size(0)
                        emb_sim_matrix = torch.zeros(batch_size, batch_size, device=self.device)
                        
                        for j in range(batch_size):
                            # Calculate MSE distance between this sample's embedding and all others
                            sample_emb = input_emb_last[j:j+1]  # [1, seq_len, dim]
                            emb_diff = F.mse_loss(
                                sample_emb.expand(batch_size, -1, -1),
                                input_emb_last,
                                reduction='none'
                            ).mean(dim=(1, 2))  # [batch_size]
                            emb_sim_matrix[j] = emb_diff
                        
                        # Normalize similarity scores
                        emb_sim_matrix = emb_sim_matrix / (emb_sim_matrix.mean() + 1e-8)
                        
                        # Calculate output difference matrix
                        output_diff_matrix = torch.zeros(batch_size, batch_size, device=self.device)
                        
                        for j in range(batch_size):
                            # Calculate MSE between this sample's target and all others' targets
                            sample_target = target_y[j:j+1]  # [1, pred_len, dim]
                            target_diff = F.mse_loss(
                                sample_target.expand(batch_size, -1, -1),
                                target_y,
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
                        
                        # Total loss with penalty
                        loss = mse_loss + self.emb_penalty_weight * penalty
                        penalty_values.append(penalty.item())
                
                else:
                    # Run the model to get outputs
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    
                    # Get input embeddings directly from encoder
                    input_emb = self.model.module.enc_embedding(batch_x, batch_x_mark) if self.args.use_multi_gpu else self.model.enc_embedding(batch_x, batch_x_mark)
                    if self.args.encoder:
                        input_emb, attns = self.model.encoder(input_emb, attn_mask=None)


                    # output_emb = self.model.module.enc_embedding(target_y, batch_x_mark) if self.args.use_multi_gpu else self.model.enc_embedding(target_y, batch_x_mark)
                    # output_emb, attns = self.model.encoder(output_emb, attn_mask=None)


                    # First pass MSE loss
                    mse_loss = criterion(outputs, target_y)
                    
                    # Calculate value difference between predictions and targets
                    value_diff = torch.abs(outputs - target_y).mean(dim=2)  # [B, L]
                    value_diff_norm = value_diff / (value_diff.mean() + 1e-8)  # Normalize
                    
                    # Calculate embedding difference for corresponding input time steps
                    # Focus on last seq_len steps that contributed most to the prediction
                    input_emb_last = input_emb[:, -self.args.seq_len:, :]
                    
                    # Calculate sample-wise embedding similarity within each batch
                    # For each sample i, compute embedding similarity with all other samples j
                    batch_size = input_emb_last.size(0)
                    emb_sim_matrix = torch.zeros(batch_size, batch_size, device=self.device)
                    
                    for j in range(batch_size):
                        # Calculate MSE distance between this sample's embedding and all others
                        sample_emb = input_emb_last[j:j+1]  # [1, seq_len, dim]
                        emb_diff = F.mse_loss(
                            sample_emb.expand(batch_size, -1, -1),
                            input_emb_last,
                            reduction='none'
                        ).mean(dim=(1, 2))  # [batch_size]
                        emb_sim_matrix[j] = emb_diff
                    
                    # Normalize similarity scores
                    emb_sim_matrix = emb_sim_matrix / (emb_sim_matrix.mean() + 1e-8)
                    
                    # Calculate output difference matrix
                    output_diff_matrix = torch.zeros(batch_size, batch_size, device=self.device)
                    
                    for j in range(batch_size):
                        # Calculate MSE between this sample's target and all others' targets
                        sample_target = target_y[j:j+1]  # [1, pred_len, dim]
                        target_diff = F.mse_loss(
                            sample_target.expand(batch_size, -1, -1),
                            target_y,
                            reduction='none'
                        ).mean(dim=(1, 2))  # [batch_size]
                        output_diff_matrix[j] = target_diff
                    
                    # for j in range(batch_size):
                    #     # Calculate MSE between this sample's target and all others' targets
                    #     sample_target = output_emb[j:j+1]  # [1, pred_len, dim]
                    #     target_diff = F.mse_loss(
                    #         sample_target.expand(batch_size, -1, -1),
                    #         output_emb,
                    #         reduction='none'
                    #     ).mean(dim=(1, 2))  # [batch_size]
                    #     output_diff_matrix[j] = target_diff
                    
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
                    penalty = penalty_matrix1.mean()
                    
                    # Total loss with penalty
                    loss = mse_loss + self.emb_penalty_weight * penalty
                    penalty_values.append(penalty.item())

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | mse: {3:.7f} | penalty: {4:.7f}".format(
                        i + 1, epoch + 1, loss_float, mse_loss.item(), penalty.item()))
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
            penalty_avg = np.average(penalty_values) if penalty_values else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Penalty: {3:.7f} | Vali Loss: {4:.7f} | Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, penalty_avg, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if not self.args.no_patience:
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
