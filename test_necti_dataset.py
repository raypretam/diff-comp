"""
Test script to verify NeCTI dataset loading and processing
"""

import os
import sys
sys.path.append('/home/pretam-pg/DiffusionSL')

from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def test_label_set():
    """Test label set loading"""
    print("=" * 60)
    print("Testing Label Set Loading")
    print("=" * 60)
    
    data_path = "/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data"
    
    # Test Coarse
    print("\n--- Coarse Granularity ---")
    coarse_labels = NeCTILabelSet(data_path, granularity='Coarse')
    print(f"Number of labels: {len(coarse_labels)}")
    print(f"Labels: {coarse_labels._labelset}")
    print(f"\nLabel mappings:")
    print(coarse_labels)
    
    # Test Finegrain
    print("\n--- Finegrain Granularity ---")
    fine_labels = NeCTILabelSet(data_path, granularity='Finegrain')
    print(f"Number of labels: {len(fine_labels)}")
    print(f"Labels: {fine_labels._labelset}")
    
    return coarse_labels, fine_labels

def test_dataset(label_set, granularity='Coarse'):
    """Test dataset loading"""
    print("\n" + "=" * 60)
    print(f"Testing {granularity} Dataset Loading")
    print("=" * 60)
    
    data_path = "/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data"
    
    for split in ['train', 'dev', 'test']:
        print(f"\n--- {split.upper()} Split ---")
        try:
            dataset = NeCTIDataset(data_path, split, label_set)
            print(f"Number of sentences: {len(dataset)}")
            
            # Show first example
            example = dataset[0]
            tokens = example['tokens']
            labels = example['labels']
            compounds = example['compounds']
            
            print(f"\nFirst example:")
            print(f"Tokens: {' '.join(tokens[:15])}...")  # Show first 15 tokens
            print(f"Labels: {labels[:15]}...")
            print(f"Token count: {len(tokens)}")
            print(f"\nCompounds found: {len(compounds)}")
            
            for i, comp in enumerate(compounds[:3]):  # Show first 3 compounds
                print(f"\n  Compound {i+1}:")
                print(f"    Span: [{comp['start']}:{comp['end']+1}]")
                print(f"    Tokens (WX): {' '.join(comp['tokens'])}")
                print(f"    Tokens (Devanagari): {' '.join(comp['tokens_devanagari'])}")
                print(f"    Type: {comp['type']}")
                print(f"    Level: {comp['level']}")
                print(f"    Internal types: {comp['internal_types']}")
            
        except FileNotFoundError as e:
            print(f"File not found: {e}")

def test_collator(label_set):
    """Test data collation"""
    print("\n" + "=" * 60)
    print("Testing Data Collation")
    print("=" * 60)
    
    data_path = "/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data"
    
    # Load dataset
    dataset = NeCTIDataset(data_path, 'train', label_set)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    collator = NeCTICollator(tokenizer, max_length=256)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
        shuffle=False
    )
    
    # Get first batch
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    seq_labels = batch['seq_labels']
    compounds = batch['compounds']
    
    print(f"\nBatch shapes:")
    print(f"Input IDs: {input_ids.shape}")
    print(f"Attention Mask: {attention_mask.shape}")
    print(f"Sequence Labels: {seq_labels.shape}")
    print(f"Compounds: {len(compounds)} sentences")
    
    print(f"\nFirst sequence in batch:")
    print(f"Input IDs: {input_ids[0][:20]}")
    print(f"Attention Mask: {attention_mask[0][:20]}")
    print(f"Labels: {seq_labels[0][:20]}")
    
    print(f"\nLabel statistics:")
    valid_labels = seq_labels[seq_labels != -100]
    print(f"Valid labels: {valid_labels.shape[0]}")
    print(f"Padding labels (-100): {(seq_labels == -100).sum().item()}")
    
    print(f"\nCompounds in first sentence:")
    for i, comp in enumerate(compounds[0]):
        print(f"  Compound {i+1}:")
        print(f"    WX: {comp['tokens']}")
        print(f"    Devanagari: {comp['tokens_devanagari']}")
        print(f"    Type: {comp['type']}, Span: [{comp['start']}:{comp['end']+1}]")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NeCTI Dataset Test Suite")
    print("=" * 60)
    
    try:
        # Test label sets
        coarse_labels, fine_labels = test_label_set()
        
        # Test datasets
        test_dataset(coarse_labels, 'Coarse')
        test_dataset(fine_labels, 'Finegrain')
        
        # Test collator
        test_collator(coarse_labels)
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
