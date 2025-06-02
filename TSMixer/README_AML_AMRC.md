# TSMixer with AML and AMRC Extensions

This repository contains two enhanced versions of TSMixer with adaptive mask regularization:

1. **TSMixerAML** - Adaptive Mask Learning (ablation with mask penalty only)
2. **TSMixerAMRC** - Adaptive Mask with Representation Consistency (full method with both penalties)

## Installation

First install the TSMixer package:

```bash
cd /scratch/sx2490/SOFTS_exp/TSMixer
pip install -e .
```

## Key Features

### TSMixerAML
- Inherits from the base TSMixer model
- Adds mask penalty loss during training
- Penalizes when masked input sequences perform better than original sequences
- Controlled by `mask_penalty_weight` parameter

### TSMixerAMRC
- Includes all features of TSMixerAML
- Additionally includes embedding consistency penalty
- Ensures representation consistency between embeddings and outputs
- Controlled by both `mask_penalty_weight` and `emb_penalty_weight` parameters

## Usage Example

### Using TSMixerAML

```python
from torchtsmixer import TSMixerAML

# Initialize model
model = TSMixerAML(
    sequence_length=96,
    prediction_length=24,
    input_channels=7,
    output_channels=7,
    mask_penalty_weight=0.1,  # Weight for mask penalty
)

# Training loop
outputs, mask_penalty = model(inputs, targets)  # During training
outputs, _ = model(inputs)  # During evaluation
```

### Using TSMixerAMRC

```python
from torchtsmixer import TSMixerAMRC

# Initialize model  
model = TSMixerAMRC(
    sequence_length=96,
    prediction_length=24,
    input_channels=7,
    output_channels=7,
    mask_penalty_weight=0.1,  # Weight for mask penalty
    emb_penalty_weight=0.1,   # Weight for embedding consistency penalty
)

# Training loop
outputs, mask_penalty, emb_penalty = model(inputs, targets)  # During training
outputs, _, _ = model(inputs)  # During evaluation
```

## Training Scripts

Two example training scripts are provided:

### Train TSMixerAML

```bash
python train_tsmixer_aml.py \
    --seq_len 96 \
    --pred_len 24 \
    --n_channels 7 \
    --num_blocks 2 \
    --ff_dim 64 \
    --mask_penalty_weight 0.1 \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 0.001
```

### Train TSMixerAMRC

```bash
python train_tsmixer_amrc.py \
    --seq_len 96 \
    --pred_len 24 \
    --n_channels 7 \
    --num_blocks 2 \
    --ff_dim 64 \
    --mask_penalty_weight 0.1 \
    --emb_penalty_weight 0.1 \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 0.001
```

## Key Differences from Base TSMixer

1. **Forward Method Returns**: The forward method now returns tuples including penalty losses
2. **Training Mode**: Penalties are only computed during training when targets are provided
3. **Eval Mode**: During evaluation, penalties are returned as zero tensors

## Implementation Details

### Mask Penalty
- Randomly samples mask lengths (up to 2/3 of sequence length)
- Tests if masked sequences perform better than original
- Applies penalty when masks improve performance

### Embedding Consistency Penalty (AMRC only)
- Computes similarity matrices for embeddings and outputs
- Penalizes inconsistencies between representation similarities
- Helps maintain representation quality

## Tips for Usage

1. Start with small penalty weights (0.01-0.1) and adjust based on validation performance
2. Monitor both base loss and penalty losses during training
3. Use validation loss (without penalties) for model selection
4. The penalties are designed to regularize training, not for evaluation

## Integration with Existing Code

To integrate with existing TSMixer code:

```python
# Replace
from torchtsmixer import TSMixer
model = TSMixer(...)

# With
from torchtsmixer import TSMixerAML  # or TSMixerAMRC
model = TSMixerAML(...)  # or TSMixerAMRC(...)

# Update training loop to handle returned penalties
if isinstance(model, (TSMixerAML, TSMixerAMRC)):
    if isinstance(model, TSMixerAML):
        outputs, mask_penalty = model(inputs, targets)
        total_loss = base_loss + mask_penalty
    else:  # TSMixerAMRC
        outputs, mask_penalty, emb_penalty = model(inputs, targets)
        total_loss = base_loss + mask_penalty + emb_penalty
else:
    outputs = model(inputs)
    total_loss = base_loss
``` 