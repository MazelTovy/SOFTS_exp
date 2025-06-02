"""
Example training script for TSMixer with AMRC (Adaptive Mask with Representation Consistency).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchtsmixer import TSMixerAMRC
import argparse


def create_dummy_data(n_samples=1000, seq_len=96, pred_len=24, n_channels=7):
    """Create dummy time series data for demonstration."""
    # Generate synthetic time series data
    time = np.arange(n_samples + seq_len + pred_len)
    data = []
    
    for i in range(n_channels):
        # Create different patterns for each channel
        freq = 0.1 + i * 0.05
        amplitude = 1.0 + i * 0.2
        signal = amplitude * np.sin(2 * np.pi * freq * time)
        signal += 0.1 * np.random.randn(len(time))  # Add noise
        data.append(signal)
    
    data = np.array(data).T  # Shape: (time, channels)
    
    # Create sequences
    X, y = [], []
    for i in range(n_samples):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return X, y


def train_model(args):
    """Train TSMixer with AMRC."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    print("Creating dummy data...")
    X_train, y_train = create_dummy_data(n_samples=1000, seq_len=args.seq_len, 
                                         pred_len=args.pred_len, n_channels=args.n_channels)
    X_val, y_val = create_dummy_data(n_samples=200, seq_len=args.seq_len, 
                                     pred_len=args.pred_len, n_channels=args.n_channels)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = TSMixerAMRC(
        sequence_length=args.seq_len,
        prediction_length=args.pred_len,
        input_channels=args.n_channels,
        output_channels=args.n_channels,
        num_blocks=args.num_blocks,
        ff_dim=args.ff_dim,
        dropout_rate=args.dropout,
        mask_penalty_weight=args.mask_penalty_weight,
        emb_penalty_weight=args.emb_penalty_weight,
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    base_criterion = nn.MSELoss()
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mask_penalty = 0.0
        train_emb_penalty = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with penalties
            outputs, mask_penalty, emb_penalty = model(inputs, targets)
            
            # Calculate base loss
            base_loss = base_criterion(outputs, targets)
            
            # Total loss
            total_loss = base_loss + mask_penalty + emb_penalty
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += base_loss.item()
            train_mask_penalty += mask_penalty.item()
            train_emb_penalty += emb_penalty.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {base_loss.item():.4f}, Mask Penalty: {mask_penalty.item():.4f}, "
                      f"Emb Penalty: {emb_penalty.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass without penalties (eval mode)
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
        print(f"  Train Mask Penalty: {avg_train_mask_penalty:.4f}")
        print(f"  Train Emb Penalty: {avg_train_emb_penalty:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print("-" * 50)
    
    print("\nTraining completed!")
    
    # Save model
    torch.save(model.state_dict(), 'tsmixer_amrc_model.pth')
    print("Model saved to 'tsmixer_amrc_model.pth'")


def main():
    parser = argparse.ArgumentParser(description='Train TSMixer with AMRC')
    
    # Data parameters
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction length')
    parser.add_argument('--n_channels', type=int, default=7, help='Number of channels')
    
    # Model parameters
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of TSMixer blocks')
    parser.add_argument('--ff_dim', type=int, default=64, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--mask_penalty_weight', type=float, default=0.1, help='Mask penalty weight')
    parser.add_argument('--emb_penalty_weight', type=float, default=0.1, help='Embedding penalty weight')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main() 