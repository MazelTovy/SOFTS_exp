from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .tsmixer import TSMixer


class TSMixerAMRC(TSMixer):
    """TSMixer with Adaptive Mask with Representation Consistency (AMRC).
    
    This version includes both mask penalty and embedding consistency penalty.
    """
    
    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        output_channels: int = None,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        ff_dim: int = 64,
        normalize_before: bool = True,
        norm_type: str = "batch",
        mask_penalty_weight: float = 0.1,
        emb_penalty_weight: float = 0.1,
    ):
        super().__init__(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            input_channels=input_channels,
            output_channels=output_channels,
            activation_fn=activation_fn,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.mask_penalty_weight = mask_penalty_weight
        self.emb_penalty_weight = emb_penalty_weight
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Store intermediate embeddings
        self._embeddings = None
        
    def apply_mask(self, x: torch.Tensor, mask_length: int) -> torch.Tensor:
        """Apply masking to the beginning of sequence up to mask_length."""
        masked_x = x.clone()
        masked_x[:, :mask_length, :] = 0.
        return masked_x
    
    def compute_mask_penalty(self, x_hist: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mask penalty loss when masked input performs better than original."""
        if not self.training:
            return torch.tensor(0., device=x_hist.device)
            
        criterion = nn.MSELoss(reduction='none')
        
        # Original loss per sample
        original_loss = criterion(outputs, targets).mean(dim=[1, 2])
        
        # Sample random mask lengths
        population = list(range(1, min(48, self.sequence_length)))
        if len(population) == 0:
            return torch.tensor(0., device=x_hist.device)
            
        local_rng = random.Random()
        local_rng.seed()
        num_masks = min(12, len(population))
        mask_lengths = local_rng.sample(population, num_masks)
        
        mask_losses = []
        max_mask_len = int(self.sequence_length * 2 / 3)
        
        # Compute loss for each mask length
        for mask_len in mask_lengths:
            if mask_len > max_mask_len:
                continue
                
            # Apply mask and forward pass
            masked_x = self.apply_mask(x_hist, mask_len)
            with torch.no_grad():
                masked_outputs = self.forward_base(masked_x)
            
            masked_loss = criterion(masked_outputs, targets).mean(dim=[1, 2])
            mask_losses.append(masked_loss)
        
        if not mask_losses:
            return torch.tensor(0., device=x_hist.device)
        
        # Check if any mask performs better
        mask_losses = torch.stack(mask_losses).permute(1, 0)
        improvement = original_loss.unsqueeze(1) - mask_losses
        improvement = torch.relu(improvement)
        
        # Find best mask for each sample
        best_improvements = improvement.max(-1).values
        best_indices = improvement.max(-1).indices
        
        # Create batch with best masks
        mask_batch = []
        for i in range(x_hist.size(0)):
            if best_improvements[i] > 0:
                best_mask_idx = best_indices[i].item()
                mask_len = mask_lengths[best_mask_idx]
                mask_x = self.apply_mask(x_hist[i:i+1], mask_len)
                mask_batch.append(mask_x)
            else:
                mask_batch.append(x_hist[i:i+1])
        
        if mask_batch:
            mask_batch = torch.cat(mask_batch, dim=0)
            mask_outputs = self.forward_base(mask_batch)
            penalty = F.mse_loss(outputs, mask_outputs)
        else:
            penalty = torch.tensor(0., device=x_hist.device)
        
        return penalty * self.mask_penalty_weight
    
    def compute_emb_penalty(self, embeddings: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute embedding consistency penalty."""
        if not self.training or embeddings is None:
            return torch.tensor(0., device=outputs.device)
        
        # Flatten embeddings and outputs for correlation computation
        B = embeddings.size(0)
        emb_flat = embeddings.reshape(B, -1)
        out_flat = outputs.reshape(B, -1)
        
        # Compute correlation matrices
        emb_norm = F.normalize(emb_flat, p=2, dim=1)
        out_norm = F.normalize(out_flat, p=2, dim=1)
        
        emb_sim = torch.matmul(emb_norm, emb_norm.T)
        out_sim = torch.matmul(out_norm, out_norm.T)
        
        # Consistency penalty
        penalty = F.mse_loss(emb_sim, out_sim)
        
        return penalty * self.emb_penalty_weight
    
    def forward_base(self, x_hist: torch.Tensor) -> torch.Tensor:
        """Base forward pass without penalties."""
        return super().forward(x_hist)
    
    def forward(self, x_hist: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional penalty computation.
        
        Args:
            x_hist: Input time series tensor
            targets: Target tensor for training (optional)
            
        Returns:
            outputs: Prediction tensor
            mask_penalty: Mask penalty loss (0 if targets not provided)
            emb_penalty: Embedding consistency penalty loss
        """
        # Store input embeddings for consistency penalty
        self._embeddings = x_hist.clone()
        
        # Process through mixer layers
        x = self.mixer_layers(x_hist)
        
        # Apply temporal projection
        from .layers import feature_to_time, time_to_feature
        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        outputs = time_to_feature(x_temp)
        
        # Compute penalties if training and targets provided
        if self.training and targets is not None:
            mask_penalty = self.compute_mask_penalty(x_hist, outputs, targets)
            emb_penalty = self.compute_emb_penalty(self._embeddings, outputs)
        else:
            mask_penalty = torch.tensor(0., device=x_hist.device)
            emb_penalty = torch.tensor(0., device=x_hist.device)
        
        return outputs, mask_penalty, emb_penalty 