"""
Test script to verify TSMixerAML and TSMixerAMRC implementations.
"""

import torch
from torchtsmixer import TSMixer, TSMixerAML, TSMixerAMRC


def test_forward_pass():
    """Test forward pass for all models."""
    batch_size = 8
    seq_len = 96
    pred_len = 24
    n_channels = 7
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, n_channels)
    targets = torch.randn(batch_size, pred_len, n_channels)
    
    print("Testing TSMixer base model...")
    model_base = TSMixer(
        sequence_length=seq_len,
        prediction_length=pred_len,
        input_channels=n_channels,
        output_channels=n_channels,
    )
    
    # Test base model
    output_base = model_base(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_base.shape}")
    print(f"  Expected output shape: ({batch_size}, {pred_len}, {n_channels})")
    assert output_base.shape == (batch_size, pred_len, n_channels), "Base model output shape mismatch!"
    print("  ✓ Base model test passed!")
    
    print("\nTesting TSMixerAML...")
    model_aml = TSMixerAML(
        sequence_length=seq_len,
        prediction_length=pred_len,
        input_channels=n_channels,
        output_channels=n_channels,
        mask_penalty_weight=0.1,
    )
    
    # Test AML in eval mode
    model_aml.eval()
    output_aml, mask_penalty = model_aml(x)
    print(f"  Eval mode - Output shape: {output_aml.shape}, Mask penalty: {mask_penalty.item()}")
    assert output_aml.shape == (batch_size, pred_len, n_channels), "AML model output shape mismatch!"
    assert mask_penalty.item() == 0., "Mask penalty should be 0 in eval mode!"
    
    # Test AML in train mode
    model_aml.train()
    output_aml, mask_penalty = model_aml(x, targets)
    print(f"  Train mode - Output shape: {output_aml.shape}, Mask penalty: {mask_penalty.item()}")
    assert output_aml.shape == (batch_size, pred_len, n_channels), "AML model output shape mismatch!"
    print("  ✓ AML model test passed!")
    
    print("\nTesting TSMixerAMRC...")
    model_amrc = TSMixerAMRC(
        sequence_length=seq_len,
        prediction_length=pred_len,
        input_channels=n_channels,
        output_channels=n_channels,
        mask_penalty_weight=0.1,
        emb_penalty_weight=0.1,
    )
    
    # Test AMRC in eval mode
    model_amrc.eval()
    output_amrc, mask_penalty, emb_penalty = model_amrc(x)
    print(f"  Eval mode - Output shape: {output_amrc.shape}, "
          f"Mask penalty: {mask_penalty.item()}, Emb penalty: {emb_penalty.item()}")
    assert output_amrc.shape == (batch_size, pred_len, n_channels), "AMRC model output shape mismatch!"
    assert mask_penalty.item() == 0., "Mask penalty should be 0 in eval mode!"
    assert emb_penalty.item() == 0., "Embedding penalty should be 0 in eval mode!"
    
    # Test AMRC in train mode
    model_amrc.train()
    output_amrc, mask_penalty, emb_penalty = model_amrc(x, targets)
    print(f"  Train mode - Output shape: {output_amrc.shape}, "
          f"Mask penalty: {mask_penalty.item()}, Emb penalty: {emb_penalty.item()}")
    assert output_amrc.shape == (batch_size, pred_len, n_channels), "AMRC model output shape mismatch!"
    print("  ✓ AMRC model test passed!")


def test_backward_pass():
    """Test backward pass for penalty computation."""
    print("\nTesting backward pass...")
    
    batch_size = 4
    seq_len = 48
    pred_len = 12
    n_channels = 3
    
    x = torch.randn(batch_size, seq_len, n_channels, requires_grad=True)
    targets = torch.randn(batch_size, pred_len, n_channels)
    
    # Test AML backward
    model_aml = TSMixerAML(
        sequence_length=seq_len,
        prediction_length=pred_len,
        input_channels=n_channels,
        output_channels=n_channels,
    )
    model_aml.train()
    
    output, mask_penalty = model_aml(x, targets)
    loss = torch.nn.MSELoss()(output, targets) + mask_penalty
    loss.backward()
    
    assert x.grad is not None, "Gradients not computed for AML!"
    print("  ✓ AML backward pass successful!")
    
    # Test AMRC backward
    x.grad = None  # Reset gradients
    model_amrc = TSMixerAMRC(
        sequence_length=seq_len,
        prediction_length=pred_len,
        input_channels=n_channels,
        output_channels=n_channels,
    )
    model_amrc.train()
    
    output, mask_penalty, emb_penalty = model_amrc(x, targets)
    loss = torch.nn.MSELoss()(output, targets) + mask_penalty + emb_penalty
    loss.backward()
    
    assert x.grad is not None, "Gradients not computed for AMRC!"
    print("  ✓ AMRC backward pass successful!")


if __name__ == "__main__":
    print("Running TSMixer implementation tests...")
    print("=" * 50)
    
    test_forward_pass()
    test_backward_pass()
    
    print("\n" + "=" * 50)
    print("All tests passed successfully! ✓") 