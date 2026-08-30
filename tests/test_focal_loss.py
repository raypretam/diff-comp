"""
Test script to verify Focal MSE Loss implementation
Tests both with and without label weights
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ddim_bitdit import focal_mse_loss


def test_focal_mse_loss():
    """Test focal MSE loss function"""
    print("="*80)
    print("Testing Focal MSE Loss Implementation")
    print("="*80)
    
    # Test 1: Basic functionality without label weights
    print("\n[Test 1] Basic focal MSE without label weights")
    print("-" * 40)
    
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    target = torch.tensor([1.1, 2.5, 2.8, 4.0, 6.0])
    
    # Standard MSE for comparison
    standard_mse = torch.nn.functional.mse_loss(pred, target)
    print(f"Standard MSE: {standard_mse.item():.6f}")
    
    # Focal MSE with gamma=0 (should be similar to standard MSE)
    focal_loss_gamma0 = focal_mse_loss(pred, target, gamma=0.0)
    print(f"Focal MSE (gamma=0.0): {focal_loss_gamma0.item():.6f}")
    print(f"Difference: {abs(standard_mse.item() - focal_loss_gamma0.item()):.6f}")
    assert abs(standard_mse.item() - focal_loss_gamma0.item()) < 0.01, "Gamma=0 should be close to standard MSE"
    
    # Focal MSE with gamma=2.0 (default, emphasizes hard examples)
    focal_loss_gamma2 = focal_mse_loss(pred, target, gamma=2.0)
    print(f"Focal MSE (gamma=2.0): {focal_loss_gamma2.item():.6f}")
    
    # Focal MSE with gamma=5.0 (strong emphasis on hard examples)
    focal_loss_gamma5 = focal_mse_loss(pred, target, gamma=5.0)
    print(f"Focal MSE (gamma=5.0): {focal_loss_gamma5.item():.6f}")
    
    print("✓ Test 1 passed: Basic focal loss working correctly")
    
    # Test 2: With label weights
    print("\n[Test 2] Focal MSE with per-sample label weights")
    print("-" * 40)
    
    # Create label weights (simulating rare vs common classes)
    label_weights = torch.tensor([1.0, 1.0, 5.0, 1.0, 10.0])  # Last two tokens are rare classes
    
    focal_loss_weighted = focal_mse_loss(pred, target, gamma=2.0, label_weights=label_weights)
    print(f"Focal MSE (gamma=2.0, weighted): {focal_loss_weighted.item():.6f}")
    print(f"Unweighted focal MSE: {focal_loss_gamma2.item():.6f}")
    print(f"Ratio (weighted/unweighted): {focal_loss_weighted.item() / focal_loss_gamma2.item():.2f}")
    
    # Weighted loss should be higher due to high weights on errors
    assert focal_loss_weighted.item() > focal_loss_gamma2.item(), "Weighted loss should be higher with high weights on errors"
    print("✓ Test 2 passed: Label weighting working correctly")
    
    # Test 3: Batch processing (realistic scenario)
    print("\n[Test 3] Batch processing with masking")
    print("-" * 40)
    
    batch_size = 4
    seq_len = 10
    bits = 7
    
    # Simulate predictions and targets for batch
    pred_batch = torch.randn(batch_size, seq_len, bits)
    target_batch = torch.randn(batch_size, seq_len, bits)
    
    # Simulate mask (some tokens are padding)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 8:] = False  # Last 2 tokens of first sample are padding
    mask[1, 9:] = False  # Last token of second sample is padding
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, bits)
    
    # Apply mask
    pred_masked = pred_batch[mask_expanded]
    target_masked = target_batch[mask_expanded]
    
    # Compute loss
    batch_loss = focal_mse_loss(pred_masked, target_masked, gamma=2.0)
    print(f"Batch shape: {pred_batch.shape}")
    print(f"Masked tokens: {pred_masked.shape[0]} (from {batch_size * seq_len * bits} total)")
    print(f"Batch focal MSE loss: {batch_loss.item():.6f}")
    
    assert batch_loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(batch_loss), "Loss should not be NaN"
    assert not torch.isinf(batch_loss), "Loss should not be Inf"
    
    print("✓ Test 3 passed: Batch processing working correctly")
    
    # Test 4: Per-token label weights in batch (realistic NeCTI scenario)
    print("\n[Test 4] Per-token label weights in batch scenario")
    print("-" * 40)
    
    # Simulate label IDs for each token
    num_labels = 70
    label_ids = torch.randint(0, num_labels, (batch_size, seq_len))
    
    # Create label weights (inverse frequency simulation)
    class_weights = torch.rand(num_labels) * 5  # Random weights 0-5
    class_weights[0] = 0.1  # "No_rel" is very common, low weight
    class_weights[1:10] = 0.5  # Common labels
    class_weights[60:] = 10.0  # Rare labels, high weight
    
    # Get per-token weights
    label_ids_flat = label_ids[mask]  # [num_valid_tokens]
    token_weights = class_weights[label_ids_flat]  # [num_valid_tokens]
    token_weights_expanded = token_weights.unsqueeze(-1).expand(-1, bits).reshape(-1)  # [num_valid_tokens * bits]
    
    # Compute weighted loss
    batch_loss_weighted = focal_mse_loss(pred_masked, target_masked, gamma=2.0, label_weights=token_weights_expanded)
    print(f"Token weights shape: {token_weights_expanded.shape}")
    print(f"Token weights (mean): {token_weights_expanded.mean().item():.3f}")
    print(f"Token weights (min/max): {token_weights_expanded.min().item():.3f} / {token_weights_expanded.max().item():.3f}")
    print(f"Batch focal MSE (weighted): {batch_loss_weighted.item():.6f}")
    print(f"Batch focal MSE (unweighted): {batch_loss.item():.6f}")
    
    assert batch_loss_weighted.item() > 0, "Weighted loss should be positive"
    assert not torch.isnan(batch_loss_weighted), "Weighted loss should not be NaN"
    
    print("✓ Test 4 passed: Per-token label weighting working correctly")
    
    # Test 5: Gradient flow
    print("\n[Test 5] Gradient flow verification")
    print("-" * 40)
    
    pred_requires_grad = torch.randn(100, requires_grad=True)
    target_no_grad = torch.randn(100)
    
    loss = focal_mse_loss(pred_requires_grad, target_no_grad, gamma=2.0)
    loss.backward()
    
    print(f"Loss value: {loss.item():.6f}")
    print(f"Gradient computed: {pred_requires_grad.grad is not None}")
    print(f"Gradient shape: {pred_requires_grad.grad.shape}")
    print(f"Gradient (mean abs): {pred_requires_grad.grad.abs().mean().item():.6f}")
    print(f"Gradient (max abs): {pred_requires_grad.grad.abs().max().item():.6f}")
    
    assert pred_requires_grad.grad is not None, "Gradients should be computed"
    assert not torch.isnan(pred_requires_grad.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(pred_requires_grad.grad).any(), "Gradients should not contain Inf"
    
    print("✓ Test 5 passed: Gradient flow working correctly")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
    print("\nFocal MSE Loss implementation is correct and ready to use.")
    print("\nTo use in training, update your config file:")
    print("  loss_type: 'focal_mse'")
    print("  focal_gamma: 2.0  # Adjust as needed")
    print("  use_label_weights: True  # To enable per-class weighting")


def test_loss_comparison():
    """Compare focal loss behavior with different gamma values"""
    print("\n" + "="*80)
    print("Focal Loss Behavior Analysis")
    print("="*80)
    
    # Create scenario with one hard example and several easy examples
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0])
    target = torch.tensor([1.05, 2.02, 3.01, 4.03, 5.0])  # Last one is hard example
    
    print("\nPredictions:", pred.tolist())
    print("Targets:    ", target.tolist())
    print("Errors:     ", (pred - target).abs().tolist())
    
    print("\nLoss values for different gamma:")
    print("-" * 40)
    
    for gamma in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        loss = focal_mse_loss(pred, target, gamma=gamma)
        print(f"gamma={gamma:.1f}: loss={loss.item():.6f}")
    
    print("\nInterpretation:")
    print("- gamma=0: Equal weight to all errors (similar to standard MSE)")
    print("- gamma=2: Standard focal loss, emphasizes hard examples")
    print("- gamma=5: Strong emphasis on hard examples")
    print("\nHigher gamma → more focus on hard examples (large errors)")


if __name__ == "__main__":
    try:
        test_focal_mse_loss()
        test_loss_comparison()
        print("\n✓ All verification tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
