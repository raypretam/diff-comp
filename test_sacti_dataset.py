"""
Test script to verify SaCTI dataset loading
"""
import sys
sys.path.append('/home/pretam-pg/DiffusionSL')

from data.ner.sacti_dataset import SaCTILabelSet, SaCTIDataset
from collections import Counter

print("=" * 80)
print("Testing SaCTI Dataset Loading")
print("=" * 80)

# Test Coarse-grained
print("\n### Testing Coarse-grained Labels ###\n")
data_path = '/home/pretam-pg/SaCTI/data/sacti'
label_set_coarse = SaCTILabelSet(data_path, granularity='coarse')

print(f"\nLabel set size: {len(label_set_coarse)}")
print(f"Labels: {label_set_coarse._labelset}")

# Load train data
train_data = SaCTIDataset(data_path, 'train', label_set_coarse)
print(f"\nTrain dataset size: {len(train_data)}")

# Check first sample
sample = train_data[0]
print(f"\nFirst sample:")
print(f"  Tokens: {sample['tokens']}")
print(f"  Labels (IDs): {sample['labels']}")
print(f"  Labels (names): {sample['compound_types']}")

# Label distribution
label_counter = Counter()
for i in range(len(train_data)):
    sample = train_data[i]
    for ct in sample['compound_types']:
        label_counter[ct] += 1

print(f"\nLabel distribution in train set:")
for label, count in label_counter.most_common():
    pct = 100 * count / sum(label_counter.values())
    print(f"  {label:20s}: {count:6d} ({pct:5.2f}%)")

# Test Fine-grained
print("\n\n### Testing Fine-grained Labels ###\n")
label_set_fine = SaCTILabelSet(data_path, granularity='fine')

print(f"\nLabel set size: {len(label_set_fine)}")
print(f"Labels: {label_set_fine._labelset}")

# Load train data
train_data_fine = SaCTIDataset(data_path, 'train', label_set_fine)
print(f"\nTrain dataset size: {len(train_data_fine)}")

# Check first sample
sample = train_data_fine[0]
print(f"\nFirst sample:")
print(f"  Tokens: {sample['tokens']}")
print(f"  Labels (IDs): {sample['labels']}")
print(f"  Labels (names): {sample['compound_types']}")

# Label distribution
label_counter_fine = Counter()
for i in range(len(train_data_fine)):
    sample = train_data_fine[i]
    for ct in sample['compound_types']:
        label_counter_fine[ct] += 1

print(f"\nLabel distribution in train set:")
for label, count in label_counter_fine.most_common():
    pct = 100 * count / sum(label_counter_fine.values())
    print(f"  {label:20s}: {count:6d} ({pct:5.2f}%)")

print("\n" + "=" * 80)
print("✓ Dataset loading test completed successfully!")
print("=" * 80)
