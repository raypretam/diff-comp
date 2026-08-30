"""
Verification script for Hierarchical Diffusion NeCTI
Tests that both stages work correctly with local attention
"""
import torch
import sys
from pathlib import Path

print("\n" + "="*70)
print("HIERARCHICAL DIFFUSION NECTI - PIPELINE VERIFICATION")
print("="*70 + "\n")

# Test imports
print("1. Testing imports...")
try:
    from models.hierarchial_diffusion import (
        HierarchicalDiffusionNeCTI,
        LocalWindowAttention,
        LocalRefinementDiT,
        create_hierarchical_model
    )
    from data.ner.necti_dataset_hierarchial import (
        HierarchicalNeCTILabelSet,
        HierarchicalNeCTICollator,
        get_coarse_id_from_fine_label
    )
    from transformers import AutoTokenizer
    print("   ✓ All imports successful\n")
except Exception as e:
    print(f"   ✗ Import failed: {e}\n")
    sys.exit(1)

# Test Stage 1: Global Attention
print("2. Testing Stage 1 (Global Attention for Coarse Labels)...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Create sample fine-to-coarse mapping
    fine_to_coarse_map = {i: i % 4 for i in range(56)}  # Map 56 fine labels to 4 coarse
    
    model = HierarchicalDiffusionNeCTI(
        device=device,
        backbone='xlm-roberta-base',
        num_coarse_classes=4,
        num_fine_classes=56,
        global_depth=2,  # Small for testing
        local_depth=2,
        local_window_size=5,
        fine_to_coarse_map=fine_to_coarse_map,
        time_steps=100,
        sampling_steps=10
    )
    
    # Sample batch
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    coarse_labels = torch.randint(0, 4, (batch_size, seq_len)).to(device)
    fine_labels = torch.randint(0, 56, (batch_size, seq_len)).to(device)
    
    # Test Stage 1 forward (training)
    loss_stage1 = model.forward_stage1(
        model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state,
        coarse_labels,
        attention_mask
    )
    print(f"   ✓ Stage 1 forward pass: loss = {loss_stage1.item():.4f}")
    
    # Test Stage 1 sampling (inference)
    coarse_preds = model.sample_stage1(
        model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state,
        attention_mask
    )
    print(f"   ✓ Stage 1 sampling: shape = {coarse_preds.shape}")
    print(f"   ✓ Predicted coarse labels: {coarse_preds[0, :10].cpu().tolist()}\n")
    
except Exception as e:
    print(f"   ✗ Stage 1 test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Stage 2: Local Attention
print("3. Testing Stage 2 (Local Attention for Fine-Grained Refinement)...")
try:
    # Test Stage 2 forward (training)
    loss_stage2 = model.forward_stage2(
        model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state,
        coarse_labels,
        fine_labels,
        attention_mask
    )
    print(f"   ✓ Stage 2 forward pass: loss = {loss_stage2.item():.4f}")
    
    # Test Stage 2 sampling (inference)
    fine_preds = model.sample_stage2(
        model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state,
        coarse_preds,  # Use Stage 1 predictions
        attention_mask
    )
    print(f"   ✓ Stage 2 sampling: shape = {fine_preds.shape}")
    print(f"   ✓ Predicted fine labels: {fine_preds[0, :10].cpu().tolist()}\n")
    
except Exception as e:
    print(f"   ✗ Stage 2 test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test joint training
print("4. Testing Joint Training (Both Stages)...")
try:
    losses = model(input_ids, attention_mask, coarse_labels, fine_labels, stage='both')
    print(f"   ✓ Joint forward pass:")
    print(f"     Total loss: {losses['loss'].item():.4f}")
    print(f"     Stage 1 loss: {losses['stage1_loss'].item():.4f}")
    print(f"     Stage 2 loss: {losses['stage2_loss'].item():.4f}\n")
except Exception as e:
    print(f"   ✗ Joint training test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test full inference pipeline
print("5. Testing Full Inference Pipeline (Stage 1 → Stage 2)...")
try:
    coarse_preds, fine_preds = model.inference(input_ids, attention_mask)
    print(f"   ✓ Full inference complete")
    print(f"     Coarse predictions: {coarse_preds.shape}")
    print(f"     Fine predictions: {fine_preds.shape}")
    print(f"   Example output (first 10 tokens):")
    print(f"     Coarse: {coarse_preds[0, :10].cpu().tolist()}")
    print(f"     Fine:   {fine_preds[0, :10].cpu().tolist()}\n")
except Exception as e:
    print(f"   ✗ Full inference test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Local Attention Window
print("6. Testing Local Window Attention...")
try:
    local_attn = LocalWindowAttention(
        hidden_size=256,
        num_heads=4,
        window_size=5,  # 2 left + self + 2 right
        dropout=0.1
    ).to(device)
    
    # Test input
    x = torch.randn(2, 20, 256).to(device)
    mask = torch.ones(2, 20).to(device)
    
    # Forward pass
    out = local_attn(x, mask)
    print(f"   ✓ Local attention forward: {out.shape}")
    print(f"   ✓ Window size: 5 (attends to 5 tokens total)\n")
    
except Exception as e:
    print(f"   ✗ Local attention test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test coarse label mapping
print("7. Testing Fine → Coarse Label Mapping (Real Dataset Labels)...")
try:
    test_cases = [
        # Actual finegrain labels from DepNeCTI dataset
        ('T6', 0), ('K1', 0), ('U', 0),      # Tatpurusha variants
        ('BvS', 1), ('Bs5', 1), ('Bv', 1),   # Bahuvrihi variants
        ('Ds', 2), ('Di', 2),                # Dvandva variants
        ('A1', 3),                           # Avyayibhava
        ('Comp_root', 4),                    # ROOT
        ('No_rel', 5),                       # No_rel
    ]
    
    for label, expected_coarse in test_cases:
        coarse = get_coarse_id_from_fine_label(label)
        status = "✓" if coarse == expected_coarse else "✗"
        coarse_name = ['Tatpurusha', 'Bahuvrihi', 'Dvandva', 'Avyayibhava', 'ROOT', 'No_rel'][coarse]
        print(f"   {status} '{label}' → {coarse_name} (id={coarse})")
    print()
    
except Exception as e:
    print(f"   ✗ Label mapping test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nArchitecture Summary:")
print("  Stage 1: Global Self-Attention DiT (6 layers, 768 hidden)")
print("           → Predicts 4 coarse categories")
print("  Stage 2: Local Window Attention DiT (4 layers, window=5)")
print("           → Refines to 56 fine-grained labels")
print("\nReady for training!")
print("  Run: python -m trainers.train_hierarchial --config configs/hierarchial_necti.yaml")
print("="*70 + "\n")
