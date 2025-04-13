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
import random

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.beta = getattr(args, 'beta', 1.0)  # ZeroMask loss coefficient
        self.gamma = getattr(args, 'gamma', 0.1)  # Consistency loss coefficient
        self.mask_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # 掩码比例，从序列长度的5%到50%

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
        
    def _apply_zero_mask(self, x, mask_len):
        """Apply a zero mask to the first mask_len steps of the sequence"""
        x_masked = x.clone()
        x_masked[:, :mask_len, :] = 0.0
        return x_masked

    def _multi_window_mask_analysis(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y, criterion):
        """
        对多个窗口长度的掩码进行系统分析，找出问题数据点
        返回：
        1. 问题得分 - 指示每个时间点的问题严重性
        2. 最优掩码长度 - 导致最大性能改善的掩码长度
        3. 最优掩码后的嵌入 - 用于一致性约束
        """
        batch_size, seq_len, _ = batch_x.shape
        
        # 存储每个掩码尝试的结果
        original_outputs, original_embeddings = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        original_outputs = original_outputs[:, -self.args.pred_len:, f_dim:]
        original_mse = criterion(original_outputs, batch_y)
        
        # 问题得分矩阵 - 初始化为0
        problem_scores = torch.zeros((batch_size, seq_len), device=self.device)
        
        # 存储所有掩码结果
        all_masked_mse = []
        all_masked_embeddings = []
        
        # 系统尝试不同的掩码长度
        best_improvement = 0
        best_mask_len = 0
        best_embeddings = None
        
        for ratio in self.mask_ratios:
            mask_len = int(seq_len * ratio)
            if mask_len == 0:
                continue
                
            # 应用掩码
            batch_x_masked = self._apply_zero_mask(batch_x, mask_len)
            masked_outputs, masked_embeddings = self.model(batch_x_masked, batch_x_mark, dec_inp, batch_y_mark)
            masked_outputs = masked_outputs[:, -self.args.pred_len:, f_dim:]
            masked_mse = criterion(masked_outputs, batch_y)
            
            # 计算改进程度
            improvement = original_mse - masked_mse
            
            # 存储结果
            all_masked_mse.append(masked_mse)
            all_masked_embeddings.append(masked_embeddings)
            
            # 如果掩码后性能更好，更新问题得分
            if improvement > 0:
                # 对被掩码的时间步，增加其问题得分
                # 问题得分增加量与性能改进成正比
                improvement_ratio = improvement / original_mse
                for t in range(mask_len):
                    # 更近的时间点问题影响更大
                    problem_scores[:, t] += improvement_ratio * (1.0 - t/mask_len)
                
                # 更新最佳掩码
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_mask_len = mask_len
                    best_embeddings = masked_embeddings
        
        # 如果没有找到改进，使用原始嵌入
        if best_embeddings is None:
            best_embeddings = original_embeddings
            best_mask_len = 0
            
        # 正则化问题得分
        max_scores = torch.max(problem_scores, dim=1, keepdim=True)[0]
        if torch.any(max_scores > 0):
            problem_scores = problem_scores / (max_scores + 1e-7)
            
        return problem_scores, best_mask_len, best_embeddings, original_mse

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
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0][0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0][0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
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

                # 确定目标维度
                f_dim = -1 if self.args.features == 'MS' else 0
                target_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # 多窗口掩码分析
                        problem_scores, best_mask_len, best_masked_embeddings, original_mse = self._multi_window_mask_analysis(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, target_y, criterion
                        )
                        
                        # 获取原始输出和嵌入
                        outputs, original_embeddings = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention:
                            outputs = outputs[0]
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        
                        # 基础MSE损失
                        mse_loss = criterion(outputs, target_y)
                        
                        # 计算Zero Mask Loss - 对问题数据点的惩罚
                        # 如果问题得分高，增加惩罚
                        zero_mask_loss = torch.mean(problem_scores) * (original_mse - mse_loss).abs()
                        
                        # 计算一致性损失 - 一致性损失对问题区域要求较低
                        # 用(1-problem_scores)作为权重，问题区域权重低
                        weights = 1.0 - problem_scores.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
                        weighted_diff = (original_embeddings - best_masked_embeddings) * weights
                        consistency_loss = torch.mean(weighted_diff ** 2)
                        
                        # 总损失
                        loss = mse_loss + self.beta * zero_mask_loss + self.gamma * consistency_loss

                else:
                    # 多窗口掩码分析
                    problem_scores, best_mask_len, best_masked_embeddings, original_mse = self._multi_window_mask_analysis(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, target_y, criterion
                    )
                    
                    # 获取原始输出和嵌入
                    outputs, original_embeddings = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention:
                        outputs = outputs[0]
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    
                    # 基础MSE损失
                    mse_loss = criterion(outputs, target_y)
                    
                    # 计算Zero Mask Loss - 对问题数据点的惩罚
                    # 如果问题得分高，增加惩罚
                    zero_mask_loss = torch.mean(problem_scores) * (original_mse - mse_loss).abs()
                    
                    # 计算一致性损失 - 一致性损失对问题区域要求较低
                    # 用(1-problem_scores)作为权重，问题区域权重低
                    weights = 1.0 - problem_scores.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
                    weighted_diff = (original_embeddings - best_masked_embeddings) * weights
                    consistency_loss = torch.mean(weighted_diff ** 2)
                    
                    # 总损失
                    loss = mse_loss + self.beta * zero_mask_loss + self.gamma * consistency_loss

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | mse: {3:.7f} | zero_mask: {4:.7f} | consistency: {5:.7f} | best_mask_len: {6}".format(
                        i + 1, epoch + 1, loss_float, mse_loss.item(), 
                        zero_mask_loss.item(), consistency_loss.item(), best_mask_len))
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
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0][0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0][0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

        mse = mse.avg
        mae = mae.avg
        print('mse:{}, mae:{}'.format(mse, mae))

        return
