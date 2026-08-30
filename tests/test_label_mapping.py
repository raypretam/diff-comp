"""
Test script to verify fine-to-coarse label mapping with real NeCTI data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys
sys.path.insert(0, '/home/pretam-pg/DiffusionSL')

from data.ner.necti_dataset_hierarchial import (
    get_coarse_id_from_fine_label,
    COARSE_CATEGORIES,
    FINEGRAIN_TO_COARSE_MAP
)

print("\n" + "="*70)
print("NECTI LABEL MAPPING VERIFICATION")
print("="*70 + "\n")

# Test with actual finegrain labels from the dataset
test_cases = [
    # Tatpurusha variants
    ('T6', 0, 'Tatpurusha'),
    ('K1', 0, 'Tatpurusha'),
    ('U', 0, 'Tatpurusha'),
    ('T3', 0, 'Tatpurusha'),
    ('K7', 0, 'Tatpurusha'),
    ('Tm', 0, 'Tatpurusha'),
    
    # Bahuvrihi variants
    ('BvS', 1, 'Bahuvrihi'),
    ('Bs5', 1, 'Bahuvrihi'),
    ('Bv', 1, 'Bahuvrihi'),
    ('Bb', 1, 'Bahuvrihi'),
    
    # Dvandva variants
    ('Ds', 2, 'Dvandva'),
    ('Di', 2, 'Dvandva'),
    ('D', 2, 'Dvandva'),
    
    # Avyayibhava variants
    ('A1', 3, 'Avyayibhava'),
    ('A2', 3, 'Avyayibhava'),
    
    # Special cases
    ('Comp_root', 4, 'ROOT'),
    ('No_rel', 5, 'No_rel'),
    ('root', 4, 'ROOT'),
]

print("1. Testing abbreviated codes (actual dataset labels):\n")
passed = 0
failed = 0

for fine_label, expected_coarse, coarse_name in test_cases:
    actual_coarse = get_coarse_id_from_fine_label(fine_label)
    status = "✓" if actual_coarse == expected_coarse else "✗"
    
    if actual_coarse == expected_coarse:
        passed += 1
        print(f"   {status} '{fine_label}' → {coarse_name} (id={actual_coarse})")
    else:
        failed += 1
        print(f"   {status} '{fine_label}' → Expected: {coarse_name} (id={expected_coarse}), Got: {actual_coarse}")

print(f"\nResults: {passed}/{len(test_cases)} passed, {failed} failed")

# Show complete mapping
print("\n" + "="*70)
print("2. Complete Fine-to-Coarse Mapping:")
print("="*70 + "\n")

print("Tatpurusha (0):")
print("  ", [k for k, v in FINEGRAIN_TO_COARSE_MAP.items() if v == 0])
print("\nBahuvrihi (1):")
print("  ", [k for k, v in FINEGRAIN_TO_COARSE_MAP.items() if v == 1])
print("\nDvandva (2):")
print("  ", [k for k, v in FINEGRAIN_TO_COARSE_MAP.items() if v == 2])
print("\nAvyayibhava (3):")
print("  ", [k for k, v in FINEGRAIN_TO_COARSE_MAP.items() if v == 3])
print("\nROOT (4):")
print("  ", [k for k, v in FINEGRAIN_TO_COARSE_MAP.items() if v == 4])
print("\nNo_rel (5):")
print("  ", [k for k, v in FINEGRAIN_TO_COARSE_MAP.items() if v == 5])

print("\n" + "="*70)
if failed == 0:
    print("✓ ALL TESTS PASSED - Label mapping is correct!")
else:
    print(f"✗ {failed} TESTS FAILED - Label mapping needs fixes!")
print("="*70 + "\n")
