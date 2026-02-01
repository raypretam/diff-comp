"""
Data Balancing & Sampling for NeCTI
Implements weighted sampling strategies to handle severe class imbalance
"""

import torch
import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler, DataLoader
from collections import Counter
from typing import List, Dict, Tuple
import json


class LabelWeightCalculator:
    """Calculate class weights based on label frequency and difficulty"""
    
    def __init__(self, label_set, min_occurrences=None):
        """
        Args:
            label_set: NeCTILabelSet object
            min_occurrences: If provided, labels with fewer occurrences get boosted weight
        """
        self.label_set = label_set
        self.min_occurrences = min_occurrences or 50
        self.label_counts = {}
        self.label_weights = {}
    
    def calculate_from_dataset(self, dataset, per_compound=False):
        """
        Calculate class weights from a dataset.
        
        Args:
            dataset: NeCTIDataset object
            per_compound: If True, weight by compound level; if False, by token level
        
        Returns:
            Dict mapping label_id -> weight
        """
        self.label_counts = Counter()
        
        # Count label occurrences
        for idx in range(len(dataset)):
            sample = dataset[idx]
            # sample['labels'] contains label IDs
            labels = sample['labels']  # List of label IDs
            
            if per_compound:
                # Count only non-No_rel labels (label_id > 1 typically)
                for label_id in labels:
                    label_name = self.label_set.id2label(label_id)
                    if label_name != 'No_rel':
                        self.label_counts[label_name] += 1
            else:
                # Count all labels
                for label_id in labels:
                    label_name = self.label_set.id2label(label_id)
                    self.label_counts[label_name] += 1
        
        # Calculate inverse frequency weights
        total_labels = sum(self.label_counts.values())
        
        # Initialize weights for all labels in label_set
        for label_id in range(len(self.label_set)):
            label_name = self.label_set.id2label(label_id)
            count = self.label_counts.get(label_name, 0)
            
            if count == 0:
                # Zero-occurrence label: give high weight
                weight = max(self.label_counts.values()) / self.min_occurrences if self.label_counts else 1.0
            else:
                # Inverse frequency: rare labels get higher weight
                weight = total_labels / (count * len(self.label_counts))
            
            self.label_weights[label_id] = weight
        
        return self.label_weights
    
    def get_per_sample_weight(self, sample_labels: List[str]) -> float:
        """
        Get weight for a single sample (sentence) based on rarest label.
        
        This ensures that samples with rare labels get higher probability during training.
        
        Args:
            sample_labels: List of label strings OR label IDs in the sample
        
        Returns:
            Weight score (0-1)
        """
        if not sample_labels:
            return 1.0
        
        # Find rarest label in this sample
        label_weights = []
        for label in sample_labels:
            if isinstance(label, str):
                # If label is string
                if label != 'No_rel':
                    label_id = self.label_set.label2id(label)
                    weight = self.label_weights.get(label_id, 1.0)
                    label_weights.append(weight)
            else:
                # If label is ID
                label_name = self.label_set.id2label(label)
                if label_name != 'No_rel':
                    weight = self.label_weights.get(label, 1.0)
                    label_weights.append(weight)
        
        if not label_weights:
            return 1.0
        
        # Use maximum weight (prioritize samples with rare labels)
        return max(label_weights)
    
    def print_weights_summary(self):
        """Print summary of label weights"""
        print("\n" + "=" * 80)
        print("LABEL WEIGHTS SUMMARY (for training)")
        print("=" * 80)
        
        weights_by_count = []
        for label_id, weight in sorted(self.label_weights.items(), key=lambda x: -x[1]):
            label_name = self.label_set.id2label(label_id)
            count = self.label_counts.get(label_name, 0)
            weights_by_count.append((label_name, count, weight))
        
        # Print top 20
        print(f"\n{'Label':<20} {'Count':<10} {'Weight':<10}")
        print("-" * 40)
        for label, count, weight in weights_by_count[:20]:
            print(f"{label:<20} {count:<10} {weight:<10.3f}")
        
        if len(weights_by_count) > 20:
            print(f"\n... and {len(weights_by_count) - 20} more labels")
        
        print("=" * 80 + "\n")


class BalancedNeCTISampler(Sampler):
    """
    Balanced sampler for NeCTI dataset.
    
    Implements multiple sampling strategies:
    1. Random weighted sampling based on label frequency
    2. Stratified sampling to ensure all labels represented in each batch
    3. Hard example mining (samples with rare/hard labels)
    """
    
    def __init__(self, dataset, label_set, strategy='weighted', replacement=True):
        """
        Args:
            dataset: NeCTIDataset
            label_set: NeCTILabelSet
            strategy: 'weighted' (WeightedRandomSampler), 'stratified', or 'hard_mining'
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset
        self.label_set = label_set
        self.strategy = strategy
        self.replacement = replacement
        self.num_samples = len(dataset)
        
        # Calculate weights
        self.weight_calc = LabelWeightCalculator(label_set)
        self.weight_calc.calculate_from_dataset(dataset, per_compound=False)
        
        if strategy == 'weighted':
            # Calculate per-sample weights based on rarest label
            self.sample_weights = []
            for idx in range(len(dataset)):
                sample = dataset[idx]
                labels = sample['labels']  # List of label IDs
                weight = self.weight_calc.get_per_sample_weight(labels)
                self.sample_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(self.sample_weights)
            self.sample_weights = [w / total_weight for w in self.sample_weights] if total_weight > 0 else [1.0 / len(dataset)] * len(dataset)
        
        elif strategy == 'stratified':
            # For stratified sampling, group samples by their rarest label
            self.label_to_samples = {}
            for idx in range(len(dataset)):
                sample = dataset[idx]
                labels = sample['labels']
                rarest_label = self._get_rarest_label_from_ids(labels)
                if rarest_label not in self.label_to_samples:
                    self.label_to_samples[rarest_label] = []
                self.label_to_samples[rarest_label].append(idx)
        
        elif strategy == 'hard_mining':
            # Hard mining: prioritize samples with zero-occurrence or very rare labels
            self.sample_weights = []
            for idx in range(len(dataset)):
                sample = dataset[idx]
                labels = sample['labels']
                weight = self._calculate_hard_mining_weight_from_ids(labels)
                self.sample_weights.append(weight)
            
            # Normalize
            total_weight = sum(self.sample_weights)
            self.sample_weights = [w / total_weight for w in self.sample_weights] if total_weight > 0 else [1.0 / len(dataset)] * len(dataset)
    
    def _get_rarest_label(self, labels: List[str]) -> str:
        """Get the rarest label in a list of label strings"""
        rarest_label = None
        min_count = float('inf')
        
        for label in labels:
            if label != 'No_rel':
                count = self.weight_calc.label_counts.get(label, 0)
                if count < min_count:
                    min_count = count
                    rarest_label = label
        
        return rarest_label or 'Comp_root'
    
    def _get_rarest_label_from_ids(self, label_ids: List[int]) -> str:
        """Get the rarest label in a list of label IDs"""
        rarest_label = None
        min_count = float('inf')
        
        for label_id in label_ids:
            label_name = self.label_set.id2label(label_id)
            if label_name != 'No_rel':
                count = self.weight_calc.label_counts.get(label_name, 0)
                if count < min_count:
                    min_count = count
                    rarest_label = label_name
        
        return rarest_label or 'Comp_root'
    
    def _calculate_hard_mining_weight(self, labels: List[str]) -> float:
        """Calculate weight for hard mining (harder = higher weight)"""
        has_zero_occurrence = False
        has_rare = False
        
        for label in labels:
            if label != 'No_rel':
                count = self.weight_calc.label_counts.get(label, 0)
                if count == 0:
                    has_zero_occurrence = True
                elif count < 50:
                    has_rare = True
        
        if has_zero_occurrence:
            return 3.0
        elif has_rare:
            return 2.0
        else:
            return 1.0
    
    def _calculate_hard_mining_weight_from_ids(self, label_ids: List[int]) -> float:
        """Calculate weight for hard mining from label IDs"""
        has_zero_occurrence = False
        has_rare = False
        
        for label_id in label_ids:
            label_name = self.label_set.id2label(label_id)
            if label_name != 'No_rel':
                count = self.weight_calc.label_counts.get(label_name, 0)
                if count == 0:
                    has_zero_occurrence = True
                elif count < 50:
                    has_rare = True
        
        if has_zero_occurrence:
            return 3.0
        elif has_rare:
            return 2.0
        else:
            return 1.0
    
    def __iter__(self):
        if self.strategy == 'weighted':
            # Use PyTorch's WeightedRandomSampler
            sampler = WeightedRandomSampler(
                self.sample_weights,
                num_samples=self.num_samples,
                replacement=self.replacement
            )
            return iter(sampler)
        
        elif self.strategy == 'stratified':
            # Stratified sampling: sample from each label group proportionally
            indices = []
            
            # Determine samples per label
            samples_per_label = self.num_samples // len(self.label_to_samples)
            
            for label, sample_indices in self.label_to_samples.items():
                if len(sample_indices) <= samples_per_label:
                    # Take all samples from this label
                    indices.extend(sample_indices)
                    # Pad with repetition if needed
                    while len([i for i in indices if i in sample_indices]) < samples_per_label:
                        indices.extend(sample_indices)
                else:
                    # Sample with replacement from this label
                    selected = np.random.choice(
                        sample_indices,
                        size=samples_per_label,
                        replace=True
                    )
                    indices.extend(selected)
            
            # Shuffle
            np.random.shuffle(indices)
            return iter(indices[:self.num_samples])
        
        elif self.strategy == 'hard_mining':
            sampler = WeightedRandomSampler(
                self.sample_weights,
                num_samples=self.num_samples,
                replacement=self.replacement
            )
            return iter(sampler)
    
    def __len__(self):
        return self.num_samples


class BalancedDataLoader:
    """Helper class to create balanced DataLoaders"""
    
    @staticmethod
    def create(dataset, label_set, batch_size, 
               strategy='weighted', num_workers=0, 
               collate_fn=None, shuffle_in_sampler=True):
        """
        Create a DataLoader with balanced sampling.
        
        Args:
            dataset: NeCTIDataset
            label_set: NeCTILabelSet
            batch_size: Batch size
            strategy: 'weighted', 'stratified', or 'hard_mining'
            num_workers: Number of workers
            collate_fn: Collate function
            shuffle_in_sampler: If True, shuffle is handled by sampler, not DataLoader
        
        Returns:
            DataLoader with balanced sampling
        """
        sampler = BalancedNeCTISampler(
            dataset,
            label_set,
            strategy=strategy,
            replacement=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False  # Sampling is handled by sampler
        )
        
        return dataloader, sampler


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    """
    Example usage in trainer_necti.py:
    
    from data_balancing import BalancedDataLoader, LabelWeightCalculator
    
    # In NeCTITrainer.__init__():
    
    # Create balanced sampler for training
    self.train_dataloader, train_sampler = BalancedDataLoader.create(
        dataset=train_dataset,
        label_set=self.label_set,
        batch_size=self.args.batch_size,
        strategy='weighted',  # or 'stratified', 'hard_mining'
        num_workers=self.args.num_workers,
        collate_fn=self.collate_fn
    )
    
    # Print weight summary
    train_sampler.weight_calc.print_weights_summary()
    
    # Save weights for reference
    with open(f'{output_dir}/label_weights.json', 'w') as f:
        json.dump({
            str(label_id): weight
            for label_id, weight in train_sampler.weight_calc.label_weights.items()
        }, f, indent=2)
    """
    print("Data balancing module ready for integration into trainer_necti.py")
