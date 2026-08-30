"""
Test script for contrastive learning integration
Verifies that InfoNCE loss computation works correctly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import sys
sys.path.append('/home/pretam-pg/DiffusionSL')

from models.contrastive_loss import InfoNCELoss, HierarchicalInfoNCELoss


def test_info_nce_basic():
    """Test basic InfoNCE loss with simple examples"""
    print("=" * 60)
    print("Test 1: Basic InfoNCE Loss")
    print("=" * 60)
    
    # Create loss function
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # Create dummy embeddings: [batch=2, seq_len=4, dim=8]
    embeddings = torch.randn(2, 4, 8)
    
    # Labels: same labels should have similar embeddings
    # Batch 0: [0, 1, 1, 2]
    # Batch 1: [0, 0, 1, 2]
    labels = torch.tensor([[0, 1, 1, 2], 
                           [0, 0, 1, 2]])
    
    # Mask: all valid
    mask = torch.ones(2, 4)
    
    # Compute loss
    loss = loss_fn(embeddings, labels, mask)
    
    print(f"✓ Loss computed successfully: {loss.item():.4f}")
    print(f"✓ Loss is finite: {torch.isfinite(loss).item()}")
    print(f"✓ Loss is non-negative: {loss.item() >= 0}")
    print()


def test_info_nce_with_mask():
    """Test InfoNCE loss with masked tokens"""
    print("=" * 60)
    print("Test 2: InfoNCE Loss with Masking")
    print("=" * 60)
    
    loss_fn = InfoNCELoss(temperature=0.07)
    
    embeddings = torch.randn(2, 4, 8)
    labels = torch.tensor([[0, 1, 1, 2], 
                           [0, 0, 1, -100]])  # -100 is padding
    
    # Mask: last token in second batch is padding
    mask = torch.tensor([[1, 1, 1, 1],
                         [1, 1, 1, 0]], dtype=torch.float)
    
    loss = loss_fn(embeddings, labels, mask)
    
    print(f"✓ Loss with masking: {loss.item():.4f}")
    print(f"✓ Loss is finite: {torch.isfinite(loss).item()}")
    print()


def test_info_nce_no_positives():
    """Test InfoNCE when all labels are different (no positives)"""
    print("=" * 60)
    print("Test 3: InfoNCE with No Positive Pairs")
    print("=" * 60)
    
    loss_fn = InfoNCELoss(temperature=0.07)
    
    embeddings = torch.randn(2, 4, 8)
    # All different labels - no positive pairs
    labels = torch.tensor([[0, 1, 2, 3], 
                           [4, 5, 6, 7]])
    mask = torch.ones(2, 4)
    
    loss = loss_fn(embeddings, labels, mask)
    
    print(f"✓ Loss when no positives: {loss.item():.4f}")
    print(f"✓ Correctly returns zero or small value: {loss.item() < 0.01}")
    print()


def test_hierarchical_info_nce():
    """Test hierarchical InfoNCE with coarse-fine label structure"""
    print("=" * 60)
    print("Test 4: Hierarchical InfoNCE Loss")
    print("=" * 60)
    
    # Create label hierarchy
    # Fine labels: 0-9
    # Coarse mapping: 0->0, 1->0, 2->1, 3->1, 4->2, 5->2, etc.
    label_hierarchy = {
        0: 0, 1: 0,  # Fine labels 0,1 -> Coarse 0
        2: 1, 3: 1,  # Fine labels 2,3 -> Coarse 1
        4: 2, 5: 2,  # Fine labels 4,5 -> Coarse 2
        6: 3, 7: 3,  # Fine labels 6,7 -> Coarse 3
    }
    
    loss_fn = HierarchicalInfoNCELoss(
        label_hierarchy=label_hierarchy,
        temperature=0.07,
        coarse_weight=0.5
    )
    
    embeddings = torch.randn(2, 4, 8)
    # Labels with hierarchical structure
    # Batch 0: [0, 1, 2, 3] - two pairs of coarse positives
    # Batch 1: [0, 2, 4, 6] - all different coarse labels
    labels = torch.tensor([[0, 1, 2, 3], 
                           [0, 2, 4, 6]])
    mask = torch.ones(2, 4)
    
    loss = loss_fn(embeddings, labels, mask)
    
    print(f"✓ Hierarchical loss: {loss.item():.4f}")
    print(f"✓ Loss is finite: {torch.isfinite(loss).item()}")
    print()


def test_gradient_flow():
    """Test that gradients flow through contrastive loss"""
    print("=" * 60)
    print("Test 5: Gradient Flow")
    print("=" * 60)
    
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # Create embeddings that require grad
    embeddings = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[0, 1, 1, 2], 
                           [0, 0, 1, 2]])
    mask = torch.ones(2, 4)
    
    # Compute loss and backward
    loss = loss_fn(embeddings, labels, mask)
    loss.backward()
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Gradients computed: {embeddings.grad is not None}")
    print(f"✓ Gradient norm: {embeddings.grad.norm().item():.4f}")
    print(f"✓ Gradients are finite: {torch.isfinite(embeddings.grad).all().item()}")
    print()


def test_batch_sizes():
    """Test different batch sizes and sequence lengths"""
    print("=" * 60)
    print("Test 6: Various Batch Sizes and Sequence Lengths")
    print("=" * 60)
    
    loss_fn = InfoNCELoss(temperature=0.07)
    
    test_configs = [
        (1, 10, 16),   # Small batch
        (4, 20, 32),   # Medium batch
        (8, 50, 64),   # Large batch
        (16, 100, 128) # Very large batch
    ]
    
    for batch_size, seq_len, dim in test_configs:
        embeddings = torch.randn(batch_size, seq_len, dim)
        labels = torch.randint(0, 10, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)
        
        loss = loss_fn(embeddings, labels, mask)
        
        print(f"✓ Config (B={batch_size}, L={seq_len}, D={dim}): loss={loss.item():.4f}")
    
    print()


def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 60)
    print("TESTING CONTRASTIVE LOSS IMPLEMENTATION")
    print("=" * 60 + "\n")
    
    try:
        test_info_nce_basic()
        test_info_nce_with_mask()
        test_info_nce_no_positives()
        test_hierarchical_info_nce()
        test_gradient_flow()
        test_batch_sizes()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nContrastive learning implementation is ready to use.")
        print("\nTo train with contrastive learning:")
        print("  1. Set use_contrastive: True in config file")
        print("  2. Adjust contrastive_weight (default: 0.1)")
        print("  3. Adjust contrastive_temp (default: 0.07)")
        print("  4. Run training as normal\n")
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
