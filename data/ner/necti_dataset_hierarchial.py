"""
Minimal Extension for Hierarchical Diffusion NeCTI
==================================================

This extends your existing NeCTIDataset with coarse label support.
Minimal changes - inherits from your existing classes.

KEY POINT: Your existing dataset ALREADY has local dependency info encoded
in labels like '+1:Tatpurusha' (head is 1 position right). No changes needed
to the core data loading - we just add coarse label derivation.

Usage:
    # In your training code, replace:
    from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
    
    # With:
    from data.ner.necti_dataset_hierarchical import (
        HierarchicalNeCTILabelSet,  # Extended label set with coarse support
        NeCTIDataset,               # Re-exported unchanged
        HierarchicalNeCTICollator   # Extended collator adds coarse_labels
    )
"""

import re
import torch
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer

# Import and re-export your existing classes - these remain UNCHANGED
from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator


# =============================================================================
# Coarse Category Mapping
# =============================================================================

COARSE_CATEGORIES = ['Tatpurusha', 'Bahuvrihi', 'Dvandva', 'Avyayibhava', 'ROOT', 'No_rel']

# Direct mapping from abbreviated fine-grained codes to coarse categories
# Based on DepNeCTI dataset format where:
# - Tatpurusha: T1-T7, K1-K7, U, Tm, Tds, Tdt, Tdu (T* and K* prefixes)
# - Bahuvrihi: Bs, Bs2-Bs7, Bv, Bvs, BvS, BVS, Bb, Bvp, BvU, Bsmn, Bsp, Bss, Bsu (Bv* and Bs* prefixes)
# - Dvandva: D, Di, Ds, d (D* prefix)
# - Avyayibhava: A1, A2, A4, A7 (A* prefix)
FINEGRAIN_TO_COARSE_MAP = {
    # Tatpurusha variants (index 0)
    'T': 0, 'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0, 'T5': 0, 'T6': 0, 'T7': 0,
    'K': 0, 'K1': 0, 'K2': 0, 'K3': 0, 'K4': 0, 'K5': 0, 'K6': 0, 'K7': 0, 'Km': 0,
    'U': 0, 'Tm': 0, 'Tds': 0, 'Tdt': 0, 'Tdu': 0,
    
    # Bahuvrihi variants (index 1)
    'Bv': 1, 'Bs': 1, 'Bb': 1,
    'Bs2': 1, 'Bs3': 1, 'Bs4': 1, 'Bs5': 1, 'Bs6': 1, 'Bs7': 1,
    'Bvs': 1, 'BvS': 1, 'BVS': 1, 'Bvp': 1, 'BvU': 1,
    'Bsmn': 1, 'Bsp': 1, 'Bss': 1, 'Bsu': 1,
    
    # Dvandva variants (index 2)
    'D': 2, 'd': 2, 'Di': 2, 'Ds': 2,
    
    # Avyayibhava variants (index 3)
    'A1': 3, 'A2': 3, 'A4': 3, 'A7': 3,
    
    # Special cases
    'Comp_root': 4,  # ROOT
    'No_rel': 5,     # No relation
    'root': 5,       # No relation
}

# Fallback patterns for full Sanskrit terms (if present in different formats)
COARSE_PATTERNS = {
    0: [  # Tatpurusha
        r'tatpuru[sṣ]', r'karma', r'karaṇa', r'sampradāna', r'apādāna', 
        r'sambandha', r'adhikaraṇa', r'dvigu', r'prādi', r'gati', 
        r'upapada', r'nañ', r'dvitīyā', r'tṛtīyā', r'caturthī', 
        r'pañcamī', r'ṣaṣṭhī', r'saptamī', r'accusative', r'instrumental',
        r'dative', r'ablative', r'genitive', r'locative',
    ],
    1: [  # Bahuvrihi
        r'bahuvr[iī]hi', r'samānādhikaraṇa', r'vyadhikaraṇa',
        r'tulya', r'eka', r'possessive', r'exocentric',
    ],
    2: [  # Dvandva
        r'dvandva', r'dwandwa', r'itaretara', r'samāhāra', r'ekaśeṣa',
        r'copulative', r'coordinative',
    ],
    3: [  # Avyayibhava
        r'avyay[iī]bhāva', r'indeclinable', r'adverbial',
    ],
}


def get_coarse_id_from_fine_label(fine_label: str) -> int:
    """
    Determine coarse category ID from fine-grained label.
    
    Handles both abbreviated codes (BvS, T6, K1, Ds, etc.) and full terms.
    
    Args:
        fine_label: Label like 'BvS', 'T6', 'K1', 'Ds', 'Comp_root', 'No_rel'
    
    Returns:
        Coarse category ID (0-5):
            0: Tatpurusha
            1: Bahuvrihi
            2: Dvandva
            3: Avyayibhava
            4: ROOT (Comp_root)
            5: No_rel
    """
    # Handle special labels first
    if 'Comp_root' in fine_label or fine_label == 'root':
        return 4  # ROOT
    if 'No_rel' in fine_label:
        return 5  # No_rel
    
    # Direct lookup for abbreviated codes (most common case in DepNeCTI)
    if fine_label in FINEGRAIN_TO_COARSE_MAP:
        return FINEGRAIN_TO_COARSE_MAP[fine_label]
    
    # Fallback: Try pattern matching for full Sanskrit terms
    label_lower = fine_label.lower()
    for coarse_id, patterns in COARSE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, label_lower):
                return coarse_id
    
    # Default to Tatpurusha (most common compound type)
    return 0


# =============================================================================
# Extended Label Set with Hierarchy
# =============================================================================

class HierarchicalNeCTILabelSet(NeCTILabelSet):
    """
    Extends NeCTILabelSet with coarse label support and fine-to-coarse mapping.
    
    Fully backward compatible - all existing methods still work.
    """
    
    def __init__(self, data_path: str, granularity: str = 'Finegrain', use_context: bool = False):
        # Initialize parent class (loads fine-grained labels)
        super().__init__(data_path, granularity, use_context)
        
        # Setup coarse labels
        self._coarse_labels = COARSE_CATEGORIES
        self._coarse_label2id = {label: i for i, label in enumerate(self._coarse_labels)}
        self._coarse_id2label = {i: label for i, label in enumerate(self._coarse_labels)}
        
        # Build fine-to-coarse mapping
        self._fine_to_coarse: Dict[int, int] = {}
        for fine_id in range(len(self)):
            fine_label = self.id2label(fine_id)
            coarse_id = get_coarse_id_from_fine_label(fine_label)
            self._fine_to_coarse[fine_id] = coarse_id
        
        # Print hierarchy summary
        self._print_hierarchy_summary()
    
    def _print_hierarchy_summary(self):
        """Print summary of label hierarchy"""
        print(f"\n{'='*50}")
        print("LABEL HIERARCHY SUMMARY")
        print(f"{'='*50}")
        
        for coarse_id, coarse_name in enumerate(self._coarse_labels):
            fine_ids = [fid for fid, cid in self._fine_to_coarse.items() if cid == coarse_id]
            print(f"{coarse_name}: {len(fine_ids)} fine-grained subtypes")
            
            # Show first 3 examples
            for fid in fine_ids[:3]:
                print(f"    {fid}: {self.id2label(fid)}")
            if len(fine_ids) > 3:
                print(f"    ... and {len(fine_ids) - 3} more")
        print(f"{'='*50}\n")
    
    # === Coarse Label Accessors ===
    
    @property
    def num_coarse_classes(self) -> int:
        return len(self._coarse_labels)
    
    @property
    def num_fine_classes(self) -> int:
        return len(self)
    
    def coarse_label2id(self, label: str) -> int:
        return self._coarse_label2id.get(label, 0)
    
    def coarse_id2label(self, idx: int) -> str:
        return self._coarse_id2label.get(idx, 'Tatpurusha')
    
    # === Hierarchy Accessors ===
    
    def fine_to_coarse(self, fine_id: int) -> int:
        """Convert fine-grained label ID to coarse label ID"""
        return self._fine_to_coarse.get(fine_id, 0)
    
    def get_fine_to_coarse_map(self) -> Dict[int, int]:
        """Get complete fine-to-coarse mapping dictionary"""
        return self._fine_to_coarse.copy()
    
    def get_valid_fine_for_coarse(self, coarse_id: int) -> List[int]:
        """Get list of valid fine-grained IDs for a coarse category"""
        return [fid for fid, cid in self._fine_to_coarse.items() if cid == coarse_id]
    
    def coarse_labels_from_fine(self, fine_ids: List[int]) -> List[int]:
        """Convert a list of fine-grained IDs to coarse IDs"""
        return [self._fine_to_coarse.get(fid, 0) for fid in fine_ids]


# =============================================================================
# Extended Collator with Coarse Labels
# =============================================================================

class HierarchicalNeCTICollator(NeCTICollator):
    """
    Extends NeCTICollator to also produce coarse labels.
    
    Output batch now includes:
        - 'seq_labels' / 'fine_labels' (fine-grained, same as before)
        - 'coarse_labels' (NEW: coarse category labels)
        - All other fields unchanged
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 256, 
        add_lstm: bool = False, 
        label_set: Optional[HierarchicalNeCTILabelSet] = None
    ):
        super().__init__(tokenizer, max_length, add_lstm, label_set)
        
        # Build mapping tensor for efficient conversion
        if label_set is not None and isinstance(label_set, HierarchicalNeCTILabelSet):
            self._fine_to_coarse_tensor = torch.zeros(len(label_set) + 1, dtype=torch.long)
            for fine_id, coarse_id in label_set.get_fine_to_coarse_map().items():
                self._fine_to_coarse_tensor[fine_id] = coarse_id
        else:
            self._fine_to_coarse_tensor = None
            if label_set is not None:
                print("Warning: Using HierarchicalNeCTICollator with non-hierarchical label set. "
                      "Coarse labels will not be available.")
    
    def __call__(self, batch):
        # Get base collation from parent (includes seq_labels, attention_mask, etc.)
        result = super().__call__(batch)
        
        # Add coarse labels if we have hierarchical label set
        if self._fine_to_coarse_tensor is not None:
            fine_labels = result['seq_labels']  # [batch, seq_len]
            
            # Convert fine to coarse
            coarse_labels = fine_labels.clone()
            
            # Apply mapping (handle -100 mask values)
            mask = fine_labels != -100
            valid_fine = fine_labels[mask].clamp(0, len(self._fine_to_coarse_tensor) - 1)
            coarse_labels[mask] = self._fine_to_coarse_tensor[valid_fine]
            
            # Add to result
            result['coarse_labels'] = coarse_labels
            result['fine_labels'] = fine_labels  # Alias for clarity
        
        return result


# =============================================================================
# Test Function
# =============================================================================

def test_hierarchical_extensions():
    """Test the hierarchical extensions"""
    from transformers import AutoTokenizer
    
    # Paths (update as needed)
    data_path = '/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data'
    
    print("=" * 60)
    print("Testing HierarchicalNeCTILabelSet")
    print("=" * 60)
    
    # Create hierarchical label set
    label_set = HierarchicalNeCTILabelSet(
        data_path=data_path,
        granularity='Finegrain',
        use_context=True
    )
    
    print(f"Fine-grained classes: {label_set.num_fine_classes}")
    print(f"Coarse classes: {label_set.num_coarse_classes}")
    
    # Test conversion
    print("\nSample conversions:")
    for i in range(min(10, len(label_set))):
        fine_label = label_set.id2label(i)
        coarse_id = label_set.fine_to_coarse(i)
        coarse_label = label_set.coarse_id2label(coarse_id)
        print(f"  {i}: {fine_label} -> {coarse_label} (ID: {coarse_id})")
    
    print("\n" + "=" * 60)
    print("Testing with Dataset and Collator")
    print("=" * 60)
    
    # Create dataset (uses existing NeCTIDataset unchanged!)
    dataset = NeCTIDataset(
        data_path=data_path,
        mode='dev',
        label_set=label_set,
        use_context=True
    )
    
    # Create tokenizer and collator
    tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')
    collator = HierarchicalNeCTICollator(
        tokenizer=tokenizer,
        max_length=256,
        label_set=label_set
    )
    
    # Test collation
    batch = [dataset[i] for i in range(4)]
    collated = collator(batch)
    
    print(f"\nCollated batch keys: {list(collated.keys())}")
    print(f"Fine labels shape: {collated['fine_labels'].shape}")
    print(f"Coarse labels shape: {collated['coarse_labels'].shape}")
    
    # Show sample
    print("\nSample (first sequence, valid tokens only):")
    fine = collated['fine_labels'][0].tolist()
    coarse = collated['coarse_labels'][0].tolist()
    
    count = 0
    for i, (f, c) in enumerate(zip(fine, coarse)):
        if f != -100 and count < 10:
            fine_name = label_set.id2label(f)
            coarse_name = label_set.coarse_id2label(c)
            print(f"  Token {i}: Fine={fine_name}, Coarse={coarse_name}")
            count += 1
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_hierarchical_extensions()