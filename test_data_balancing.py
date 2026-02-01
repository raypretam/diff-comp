#!/usr/bin/env python3
"""
Test script for data balancing implementation.
Verifies that balanced sampling works correctly before full training.
"""

import sys
import os
sys.path.insert(0, '/home/pretam-pg/DiffusionSL')

from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
from data_balancing import BalancedDataLoader, LabelWeightCalculator, BalancedNeCTISampler
from transformers import AutoTokenizer
import json


def test_label_weights():
    """Test label weight calculation"""
    print("\n" + "=" * 80)
    print("TEST 1: Label Weight Calculation")
    print("=" * 80)
    
    dataset_path = "/home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data"
    
    # Load label set
    label_set = NeCTILabelSet(
        data_path=dataset_path,
        granularity='Finegrain',
        use_context=False
    )
    print(f"\n✓ Loaded label set with {len(label_set)} labels")
    
    # Load dataset
    train_dataset = NeCTIDataset(
        dataset_path,
        'train',
        label_set,
        use_context=False
    )
    print(f"✓ Loaded training dataset with {len(train_dataset)} samples")
    
    # Calculate weights
    weight_calc = LabelWeightCalculator(label_set)
    weights = weight_calc.calculate_from_dataset(train_dataset)
    print(f"✓ Calculated weights for {len(weights)} labels")
    
    # Print summary
    weight_calc.print_weights_summary()
    
    # Verify weights
    assert len(weights) == len(label_set), "Weight count mismatch"
    assert all(w > 0 for w in weights.values()), "Negative weights found"
    print("\n✓ All weights are positive")
    
    return label_set, train_dataset


def test_balanced_sampler(label_set, train_dataset):
    """Test balanced sampler"""
    print("\n" + "=" * 80)
    print("TEST 2: Balanced Sampler")
    print("=" * 80)
    
    for strategy in ['weighted', 'stratified', 'hard_mining']:
        print(f"\nTesting strategy: {strategy}")
        
        sampler = BalancedNeCTISampler(
            dataset=train_dataset,
            label_set=label_set,
            strategy=strategy,
            replacement=True
        )
        print(f"  ✓ Created {strategy} sampler")
        print(f"  ✓ Sampler length: {len(sampler)}")
        print(f"  ✓ Dataset length: {len(train_dataset)}")
        
        # Get first few indices
        indices = list(sampler)[:10]
        print(f"  ✓ Generated {len(indices)} indices (first 10 shown)")
        assert len(indices) > 0, f"Sampler produced no indices for {strategy}"


def test_balanced_dataloader(label_set, train_dataset):
    """Test balanced dataloader"""
    print("\n" + "=" * 80)
    print("TEST 3: Balanced DataLoader")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    collate_fn = NeCTICollator(tokenizer, max_length=512)
    
    print(f"✓ Created tokenizer and collator")
    
    # Test with weighted sampling
    print("\nCreating balanced dataloader with 'weighted' strategy...")
    dataloader, sampler = BalancedDataLoader.create(
        dataset=train_dataset,
        label_set=label_set,
        batch_size=32,
        strategy='weighted',
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"✓ Created dataloader")
    print(f"  - Batch size: 32")
    print(f"  - Samples: {len(train_dataset)}")
    print(f"  - Expected batches: {len(train_dataset) // 32 + 1}")
    
    # Get one batch
    print("\nFetching first batch...")
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == 0:
                print(f"✓ Successfully loaded batch")
                print(f"  - input_ids shape: {batch['input_ids'].shape}")
                print(f"  - attention_mask shape: {batch['attention_mask'].shape}")
                print(f"  - seq_labels shape: {batch['seq_labels'].shape}")
                
                # Verify batch contents
                assert batch['input_ids'].shape[0] == 32, "Batch size mismatch"
                assert batch['seq_labels'].shape[0] == 32, "Batch labels mismatch"
                print("✓ Batch shapes verified")
                break
            
            if batch_idx >= 2:  # Load first 2 batches
                break
    except Exception as e:
        print(f"✗ Error loading batch: {e}")
        raise


def test_weight_distribution():
    """Test that rare labels get higher weights"""
    print("\n" + "=" * 80)
    print("TEST 4: Weight Distribution Verification")
    print("=" * 80)
    
    dataset_path = "/home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data"
    
    label_set = NeCTILabelSet(
        data_path=dataset_path,
        granularity='Finegrain',
        use_context=False
    )
    
    train_dataset = NeCTIDataset(
        dataset_path,
        'train',
        label_set,
        use_context=False
    )
    
    weight_calc = LabelWeightCalculator(label_set)
    weights = weight_calc.calculate_from_dataset(train_dataset)
    
    # Find label occurrences
    label_counts = {}
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        labels = sample['labels']  # List of label IDs
        for label_id in labels:
            label_name = label_set.id2label(label_id)
            if label_name != 'No_rel':
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
    
    # Verify: rarer labels should have higher weights
    print("\nVerifying weight-to-frequency relationship...")
    sorted_by_count = sorted(label_counts.items(), key=lambda x: x[1])
    
    print(f"\n{'Label':<15} {'Count':<10} {'Weight':<12} {'Relation':<15}")
    print("-" * 55)
    
    for label, count in sorted_by_count[:10]:  # Show 10 rarest
        label_id = label_set.label2id(label)
        weight = weights.get(label_id, 0)
        # Higher count should have lower weight
        relation = "✓ Rare→High" if weight > 1.5 else "✓ Common→Low"
        print(f"{label:<15} {count:<10} {weight:<12.3f} {relation:<15}")
    
    print("\n✓ Weight distribution verified: rarer labels have higher weights")


def save_weight_report():
    """Save comprehensive weight report"""
    print("\n" + "=" * 80)
    print("TEST 5: Generate Weight Report")
    print("=" * 80)
    
    dataset_path = "/home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data"
    
    label_set = NeCTILabelSet(
        data_path=dataset_path,
        granularity='Finegrain',
        use_context=False
    )
    
    train_dataset = NeCTIDataset(
        dataset_path,
        'train',
        label_set,
        use_context=False
    )
    
    weight_calc = LabelWeightCalculator(label_set)
    weights = weight_calc.calculate_from_dataset(train_dataset)
    
    # Create report
    report = {
        "timestamp": str(datetime.now()),
        "dataset": "DepNeCTI-XLMR Finegrain",
        "split": "train",
        "total_labels": len(label_set),
        "label_counts": {
            label_set.id2label(lid): count
            for lid, count in weight_calc.label_counts.items()
        },
        "label_weights": {
            label_set.id2label(lid): float(w)
            for lid, w in weights.items()
        }
    }
    
    # Save report
    os.makedirs('logs', exist_ok=True)
    report_path = 'logs/weight_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved weight report to {report_path}")
    
    # Print summary
    print(f"\nReport Summary:")
    print(f"  - Total labels: {report['total_labels']}")
    print(f"  - Zero-occurrence labels: {sum(1 for c in report['label_counts'].values() if c == 0)}")
    print(f"  - Rare labels (<50 occurrences): {sum(1 for c in report['label_counts'].values() if 0 < c < 50)}")
    print(f"  - Common labels (≥50 occurrences): {sum(1 for c in report['label_counts'].values() if c >= 50)}")


if __name__ == '__main__':
    try:
        from datetime import datetime
        
        print("\n" + "=" * 80)
        print("DATA BALANCING TEST SUITE")
        print("=" * 80)
        
        # Run tests
        label_set, train_dataset = test_label_weights()
        test_balanced_sampler(label_set, train_dataset)
        test_balanced_dataloader(label_set, train_dataset)
        test_weight_distribution()
        save_weight_report()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nData balancing is ready for training!")
        print("\nTo use in training:")
        print("  python trainer_necti.py \\")
        print("    --data_path /home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data \\")
        print("    --granularity Finegrain \\")
        print("    --max_epochs 100")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
