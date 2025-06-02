"""
Example training script for TSMixer with AML/AMRC on ETTm1 dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from torchtsmixer import TSMixer, TSMixerAML, TSMixerAMRC
import argparse
import os
from sklearn.preprocessing import StandardScaler


class ETTDataset(Dataset):
    """ETT Dataset for time series forecasting."""
    
    def __init__(self, data_path, seq_len, pred_len, split='train', scale=True):
        # Read data
        df = pd.read_csv(data_path)
        
        # Use date column as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Split data
        n = len(df)
        if split == 'train':
            start, end = 0, int(n * 0.7)
        elif split == 'val':
            start, end = int(n * 0.7), int(n * 0.9)
        else:  # test
            start, end = int(n * 0.9), n
        
        # Select subset
        df = df.iloc[start:end]
        self.data = df.values.astype(np.float32)
        
        # Scale data
        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()
            if split == 'train':
                self.data = self.scaler.fit_transform(self.data)
            else:
                # Use training statistics
                train_df = pd.read_csv(data_path)
                if 'date' in train_df.columns:
                    train_df = train_df.set_index('date')
                train_data = train_df.iloc[:int(len(train_df) * 0.7)].values
                self.scaler.fit(train_data)
                self.data = self.scaler.transform(self.data)
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = self.data.shape[1]
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def train_model(args):
    """Train TSMixer model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading data...")
    train_dataset = ETTDataset(args.data_path, args.seq_len, args.pred_len, 'train')
    val_dataset = ETTDataset(args.data_path, args.seq_len, args.pred_len, 'val')
    test_dataset = ETTDataset(args.data_path, args.seq_len, args.pred_len, 'test')
    
    n_channels = train_dataset.n_channels
    print(f"Number of channels: {n_channels}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    if args.model_type == 'base':
        model = TSMixer(
            sequence_length=args.seq_len,
            prediction_length=args.pred_len,
            input_channels=n_channels,
            output_channels=n_channels,
            num_blocks=args.num_blocks,
            ff_dim=args.ff_dim,
            dropout_rate=args.dropout,
        )
    elif args.model_type == 'aml':
        model = TSMixerAML(
            sequence_length=args.seq_len,
            prediction_length=args.pred_len,
            input_channels=n_channels,
            output_channels=n_channels,
            num_blocks=args.num_blocks,
            ff_dim=args.ff_dim,
            dropout_rate=args.dropout,
            mask_penalty_weight=args.mask_penalty_weight,
        )
    else:  # amrc
        model = TSMixerAMRC(
            sequence_length=args.seq_len,
            prediction_length=args.pred_len,
            input_channels=n_channels,
            output_channels=n_channels,
            num_blocks=args.num_blocks,
            ff_dim=args.ff_dim,
            dropout_rate=args.dropout,
            mask_penalty_weight=args.mask_penalty_weight,
            emb_penalty_weight=args.emb_penalty_weight,
        )
    
    model = model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    base_criterion = nn.MSELoss()
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mask_penalty = 0.0
        train_emb_penalty = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.model_type == 'base':
                outputs = model(inputs)
                base_loss = base_criterion(outputs, targets)
                total_loss = base_loss
                mask_penalty = torch.tensor(0.)
                emb_penalty = torch.tensor(0.)
            elif args.model_type == 'aml':
                outputs, mask_penalty = model(inputs, targets)
                base_loss = base_criterion(outputs, targets)
                total_loss = base_loss + mask_penalty
                emb_penalty = torch.tensor(0.)
            else:  # amrc
                outputs, mask_penalty, emb_penalty = model(inputs, targets)
                base_loss = base_criterion(outputs, targets)
                total_loss = base_loss + mask_penalty + emb_penalty
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += base_loss.item()
            train_mask_penalty += mask_penalty.item()
            train_emb_penalty += emb_penalty.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {base_loss.item():.4f}, Mask P: {mask_penalty.item():.4f}, "
                      f"Emb P: {emb_penalty.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass without penalties
                if args.model_type == 'base':
                    outputs = model(inputs)
                elif args.model_type == 'aml':
                    outputs, _ = model(inputs)
                else:  # amrc
                    outputs, _, _ = model(inputs)
                
                loss = base_criterion(outputs, targets)
                val_loss += loss.item()
        
        # Print epoch summary
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mask_penalty = train_mask_penalty / len(train_loader)
        avg_train_emb_penalty = train_emb_penalty / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        if args.model_type != 'base':
            print(f"  Train Mask Penalty: {avg_train_mask_penalty:.4f}")
        if args.model_type == 'amrc':
            print(f"  Train Emb Penalty: {avg_train_emb_penalty:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print("-" * 50)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'tsmixer_{args.model_type}_best.pth')
            print(f"  Saved best model with val loss: {best_val_loss:.4f}")
    
    # Test phase
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(f'tsmixer_{args.model_type}_best.pth'))
    model.eval()
    
    test_loss = 0.0
    test_mae = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if args.model_type == 'base':
                outputs = model(inputs)
            elif args.model_type == 'aml':
                outputs, _ = model(inputs)
            else:  # amrc
                outputs, _, _ = model(inputs)
            
            loss = base_criterion(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets))
            
            test_loss += loss.item()
            test_mae += mae.item()
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_mae = test_mae / len(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Test MSE: {avg_test_loss:.4f}")
    print(f"  Test MAE: {avg_test_mae:.4f}")
    print(f"  Test RMSE: {np.sqrt(avg_test_loss):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train TSMixer on ETTm1')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/path/to/ETTm1.csv', 
                        help='Path to ETTm1 dataset')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction length')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='base', 
                        choices=['base', 'aml', 'amrc'], help='Model type')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of TSMixer blocks')
    parser.add_argument('--ff_dim', type=int, default=64, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--mask_penalty_weight', type=float, default=0.1, 
                        help='Mask penalty weight')
    parser.add_argument('--emb_penalty_weight', type=float, default=0.1, 
                        help='Embedding penalty weight')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main() 