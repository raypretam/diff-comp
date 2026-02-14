#!/usr/bin/env python3
"""Quick test to verify path construction is correct"""

from data.ner.sacti_dataset import SaCTILabelSet

# Test path construction
print("Testing path construction...")
label_set = SaCTILabelSet(
    data_path='/home/pretam-pg/SaCTI/data',
    granularity='coarse',
    language='sacti'
)

print(f'\nConstructed path: {label_set.data_path}')
print(f'Expected path: /home/pretam-pg/SaCTI/data/sacti/coarse')
print(f'Path correct: {label_set.data_path == "/home/pretam-pg/SaCTI/data/sacti/coarse"}')
print(f'\nNumber of labels: {len(label_set)}')
print(f'Labels: {list(label_set._labelset)}')
