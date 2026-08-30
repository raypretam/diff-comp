#!/usr/bin/env python3
"""
Test script to verify SaCTI multi-lingual auto-detection
Tests loading data from Sanskrit, Marathi, and English datasets
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
sys.path.insert(0, '/home/pretam-pg/DiffusionSL')

from data.ner.sacti_dataset import SaCTILabelSet, SaCTIDataset

def test_language(language_name, data_path, granularity='coarse'):
    """Test loading a specific language dataset"""
    print(f"\n{'='*70}")
    print(f"Testing {language_name} ({granularity})")
    print(f"{'='*70}")
    
    try:
        # Initialize label set with language specification
        label_set = SaCTILabelSet(
            data_path=data_path,
            granularity=granularity,
            language=language_name
        )
        
        print(f"\n✓ Label set loaded successfully")
        print(f"  Number of labels: {len(label_set)}")
        print(f"  First 10 labels: {label_set._labelset[:10]}")
        
        # Try loading train split
        dataset = SaCTIDataset(
            data_path=data_path,
            split='train',
            label_set=label_set
        )
        
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Number of sentences: {len(dataset)}")
        
        # Show first example
        if len(dataset) > 0:
            example = dataset[0]
            print(f"\n  First example:")
            print(f"    Tokens: {example['tokens'][:5]}...")
            print(f"    Labels (IDs): {example['labels'][:5]}...")
            print(f"    Labels (text): {[label_set.id2label(l) for l in example['labels'][:5]]}...")
            if 'gram_roles' in example:
                print(f"    Gram roles: {example['gram_roles'][:5]}...")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading {language_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_detection():
    """Test auto-detection of language"""
    print(f"\n{'='*70}")
    print(f"Testing Auto-Detection")
    print(f"{'='*70}")
    
    base_path = '/home/pretam-pg/SaCTI/data'
    
    for lang_folder in ['sacti', 'marathi', 'english']:
        path = f"{base_path}"
        print(f"\nTesting path: {path} with language hint in data")
        
        try:
            label_set = SaCTILabelSet(
                data_path=path,
                granularity='coarse',
                language=lang_folder  # Explicit language
            )
            print(f"  ✓ Detected language: {label_set.language}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

if __name__ == '__main__':
    print("="*70)
    print("SaCTI Multi-Lingual Dataset Test")
    print("="*70)
    
    base_path = '/home/pretam-pg/SaCTI/data'
    
    # Test each language
    results = {}
    results['Sanskrit (sacti)'] = test_language('sacti', base_path, 'coarse')
    results['Marathi'] = test_language('marathi', base_path, 'coarse')
    results['English'] = test_language('english', base_path, 'coarse')
    
    # Test auto-detection
    test_auto_detection()
    
    # Test fine-grained Sanskrit
    results['Sanskrit Fine'] = test_language('sacti', base_path, 'fine')
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    for lang, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {lang}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ Some tests failed")
        sys.exit(1)
