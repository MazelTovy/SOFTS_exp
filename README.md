# Adaptive Masking Loss with Representation Consistency (AMRC): Time Series Forecasting with Redundancy Suppression

This repository contains the implementation of **AMRC (Adaptive Masking Loss with Representation Consistency)**, a novel optimization framework for time series forecasting that addresses the fundamental issue of redundant feature learning. The method is presented in our paper *"Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency"*.

## Abstract

Time series forecasting plays a pivotal role in critical domains such as energy management and financial markets. Through systematic experimentation, we reveal a counterintuitive phenomenon: **appropriately truncating historical data can paradoxically enhance prediction accuracy**, indicating that existing models learn substantial redundant features during training. Building upon information bottleneck theory, we propose AMRC, which features two core components:

1. **Adaptive Masking Loss (AML)**: Dynamically identifies highly discriminative temporal segments to guide gradient descent
2. **Embedding Similarity Penalty (ESP)**: Stabilizes mapping relationships among inputs, labels, and predictions

## Key Theoretical Insights

### 1. The Redundancy Learning Phenomenon

Our analysis challenges the prevailing "long-sequence information gain hypothesis" in time series forecasting. Through extensive experiments across multiple benchmarks (Table 1 in paper), we demonstrate that:

- Over **50%** of samples exhibit improved predictive performance when input sequences are optimally masked
- This phenomenon is **architecture-agnostic**, manifesting in both Transformer-based (iTransformer, PatchTST) and MLP-based models (TSMixer, SOFTS)
- The improvement is consistent across diverse datasets with varying temporal characteristics

### 2. Information Bottleneck Perspective

According to Information Bottleneck (IB) Theory, a neural network functions as a bottleneck that compresses input information during feature extraction. For time series forecasting, the objective can be formulated as:

```
max I(Z; Y) - β I(Z; X)
```

Where:
- `Z`: Latent representation
- `Y`: Prediction target
- `X`: Input sequence
- `I(·,·)`: Mutual information
- `β`: Trade-off parameter

Current models focus primarily on maximizing `I(Z; Y)` but fail to explicitly minimize `I(Z; X)`, leading to redundant feature retention.

### 3. Representation Similarity Paradox

Through t-SNE visualization analysis, we observe that:
- **Input embeddings** maintain natural dispersion patterns
- **Model representations** exhibit abnormal clustering despite diverse labels
- This concentration indicates encoding of task-irrelevant features that distort input-output mappings

## Methodology

### Adaptive Masking Loss (AML)

AML addresses redundancy by guiding the encoder toward minimal sufficient representations:

1. **Stochastic Mask Sampling**: Generate m masked variants by randomly sampling mask indices
2. **Optimal Selection**: Identify the mask that minimizes prediction loss
3. **Representation Alignment**: Minimize distance between original and optimal masked representations

```
L_AML = β · ||Z - Z_k*||²
```

Where `Z_k*` represents the encoder output from the optimally masked input.

### Embedding Similarity Penalty (ESP)

ESP enforces geometric consistency between embedding and output spaces:

```
L_ESP = (1/n²) ∑∑ |Δ_E^ij - Δ_O^ij|
```

Where:
- `Δ_E^ij`: Normalized pairwise distances in embedding space
- `Δ_O^ij`: Normalized pairwise distances in output space

### Combined Objective

The final training objective integrates both components:

```
L_total = L_pred + λ_AML · L_AML + λ_ESP · L_ESP
```

## Experimental Validation

### Datasets
- **ETT** (Electricity Transformer Temperature): ETTh1, ETTh2, ETTm1, ETTm2
- **Solar-Energy**: 137-channel solar power production data
- **Electricity**: Hourly electricity consumption
- **Weather**: 21-channel meteorological data

### Key Results

1. **Consistent Performance Gains**: AMRC achieves improvements across all tested architectures
   - Average MSE reduction: 3-7% across different models
   - More pronounced on datasets with strong temporal redundancy

2. **Architecture Agnostic**: Effective on diverse model families
   - Transformer-based: iTransformer, PatchTST
   - MLP-based: TSMixer, SOFTS, TimeMixer

3. **Redundancy Reduction**: Post-training analysis shows
   - Decreased susceptibility to input masking (Ratio* < Ratio)
   - More robust feature representations

## Implementation Notes

### Requirements
- PyTorch >= 1.10
- NumPy, Pandas, scikit-learn
- Model-specific dependencies (see individual model directories)

### Integration

AMRC is designed as a **plug-and-play** training framework that can be integrated into existing time series forecasting models without architectural modifications. The implementation follows these principles:

1. **Non-invasive**: No changes to model architecture required
2. **Flexible**: Hyperparameters λ_AML and λ_ESP can be tuned per dataset
3. **Efficient**: Minimal computational overhead during training

### Repository Structure

```
SOFTS_exp/
├── iTransformer/    # iTransformer with AMRC integration
├── PatchTST/        # PatchTST with AMRC integration  
├── TimeMixer/       # TimeMixer with AMRC integration
├── TSMixer/         # TSMixer with AMRC integration
└── SOFTS/           # SOFTS baseline implementation
```

## Limitations and Future Work

1. **Computational Overhead**: AML requires m additional forward passes per batch
2. **High-Dimensional Challenges**: ESP effectiveness diminishes in very high-dimensional embedding spaces
3. **Approximation Bounds**: Optimal mask selection is limited by sampling size m

## Acknowledgments

This research investigates a fundamental but overlooked aspect of time series forecasting: the detrimental effects of redundant feature learning. By introducing AMRC, we provide both theoretical insights and practical solutions for improving forecasting accuracy through redundancy suppression.

---

**Note**: Due to ongoing code refinement and validation, specific implementation details and usage instructions will be updated upon publication. The theoretical framework and experimental results presented here represent the core contributions of our work.
