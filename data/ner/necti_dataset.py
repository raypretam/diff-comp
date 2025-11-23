"""
Dataset module for DepNeCTI nested compound identification
Adapted for DiffusionSL framework with XLM-R encoder
"""

import torch
from torch.utils.data import Dataset
import json
import os
from transformers import PreTrainedTokenizer
from typing import List, Dict, Tuple
from indic_transliteration import sanscript


class NeCTILabelSet:
    """Label set for nested compound identification"""
    
    def __init__(self, data_path: str, granularity: str = 'Coarse'):
        """
        Args:
            data_path: Base path to NeCTIS Model Data
            granularity: 'Coarse' or 'Finegrain'
        """
        assert granularity in ['Coarse', 'Finegrain'], "granularity must be 'Coarse' or 'Finegrain'"
        
        self.granularity = granularity
        self.data_path = os.path.join(data_path, 'With Context', granularity)
        
        # Read all splits to collect labels
        self._labelset = set()
        
        for split in ['train', 'dev', 'test']:
            filename = f"{granularity}_{split}_san"
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                labels = self._extract_labels_from_file(filepath)
                self._labelset.update(labels)
        
        # Sort labels: No_rel and Comp_root first, then actual compound types
        self._labelset = sorted(list(self._labelset), key=lambda x: (
            0 if x == 'No_rel' else (1 if x == 'Comp_root' else 2),
            x
        ))
        
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}
        
        print(f"Loaded {len(self._labelset)} labels for {granularity} granularity")
        print(f"Labels: {self._labelset}")
    
    def _extract_labels_from_file(self, filepath: str) -> set:
        """Extract unique compound type labels from CoNLL-U format file"""
        labels = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 6:
                        relation = parts[5]  # Sixth column is the relation/compound type
                        labels.add(relation)
        return labels
    
    def label2id(self, label: str) -> int:
        return self._label2id.get(label, self._label2id.get('No_rel', 0))
    
    def id2label(self, idx: int) -> str:
        return self._id2label.get(idx, 'No_rel')
    
    def __str__(self):
        string = [f"{v}:\t{k}" for k, v in self._label2id.items()]
        return '\n'.join(string)
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self._labelset)


class NeCTIDataset(Dataset):
    """Dataset for nested compound identification"""
    
    def __init__(self, data_path: str, mode: str, label_set: NeCTILabelSet):
        """
        Args:
            data_path: Base path to NeCTIS Model Data
            mode: 'train', 'dev', 'test', or 'ood'
            label_set: NeCTILabelSet instance
        """
        super(NeCTIDataset, self).__init__()
        assert mode in ['train', 'dev', 'test', 'ood'], "mode must be train/dev/test/ood"
        
        self.label_set = label_set
        self.mode = mode
        
        # Construct file path
        filename = f"{label_set.granularity}_{mode}_san"
        filepath = os.path.join(data_path, 'With Context', label_set.granularity, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Parse CoNLL-U format
        self.data = self._parse_conllu(filepath)
        print(f"Loaded {len(self.data)} sentences for {mode} split")
    
    def _parse_conllu(self, filepath: str) -> List[Dict]:
        """Parse CoNLL-U format file into list of sentences with compound spans"""
        sentences = []
        current_data = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:  # Empty line indicates sentence boundary
                    if current_data:
                        sentence_info = self._process_sentence(current_data)
                        if sentence_info:
                            sentences.append(sentence_info)
                        current_data = []
                elif not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 6:
                        idx = int(parts[0])
                        token = parts[1]
                        comp_label = parts[2]  # Comp2, Comp3, CompNo, etc.
                        head_idx = int(parts[4])
                        relation = parts[5]  # Compound type or No_rel/Comp_root
                        
                        current_data.append({
                            'idx': idx,
                            'token': token,
                            'comp_label': comp_label,
                            'head_idx': head_idx,
                            'relation': relation
                        })
        
        # Handle last sentence if file doesn't end with empty line
        if current_data:
            sentence_info = self._process_sentence(current_data)
            if sentence_info:
                sentences.append(sentence_info)
        
        return sentences
    
    def _process_sentence(self, sentence_data: List[Dict]) -> Dict:
        """Process a sentence to extract tokens, labels (relation types), and compound spans"""
        # Remove DUMMY token if present
        if sentence_data and sentence_data[-1]['token'] == 'DUMMY':
            sentence_data = sentence_data[:-1]
        
        if not sentence_data:
            return None
        
        tokens = [item['token'] for item in sentence_data]
        # Labels are the relation types (compound types to predict)
        relation_labels = [item['relation'] for item in sentence_data]
        
        # Extract compound spans and types
        compounds = self._extract_compounds(sentence_data)
        
        return {
            'tokens': tokens,
            'relation_labels': relation_labels,
            'compounds': compounds
        }
    
    def _extract_compounds(self, sentence_data: List[Dict]) -> List[Dict]:
        """
        Extract compound spans and their types from dependency structure.
        Returns list of compounds with their start/end positions and types.
        """
        compounds = []
        
        # Find all compound roots (tokens with Comp_root relation)
        compound_roots = [item for item in sentence_data if item['relation'] == 'Comp_root']
        
        for root in compound_roots:
            # Find all tokens that are part of this compound
            # (tokens with the same head as root or transitively connected)
            compound_tokens = self._find_compound_members(sentence_data, root)
            
            if len(compound_tokens) > 1:  # Valid compound has multiple members
                # Sort by index to get span
                compound_tokens.sort(key=lambda x: x['idx'])
                
                start_idx = compound_tokens[0]['idx'] - 1  # 0-indexed
                end_idx = compound_tokens[-1]['idx'] - 1   # 0-indexed
                
                # Determine compound type from the relations
                comp_types = set()
                for token in compound_tokens:
                    if token['relation'] not in ['Comp_root', 'No_rel']:
                        comp_types.add(token['relation'])
                
                # For nested compounds, the relation to the parent compound
                compound_type = root['relation'] if root['relation'] != 'Comp_root' else 'Compound'
                
                # Get the compound level (Comp2, Comp3, etc.)
                comp_level = root['comp_label']
                
                # Get tokens in WX and transliterate to Devanagari
                wx_tokens = [t['token'] for t in compound_tokens]
                devanagari_tokens = [sanscript.transliterate(token, sanscript.WX, sanscript.DEVANAGARI) 
                                    for token in wx_tokens]
                
                compounds.append({
                    'start': start_idx,
                    'end': end_idx,
                    'type': compound_type,
                    'level': comp_level,
                    'internal_types': list(comp_types),
                    'tokens': wx_tokens,
                    'tokens_devanagari': devanagari_tokens
                })
        
        return compounds
    
    def _find_compound_members(self, sentence_data: List[Dict], root: Dict) -> List[Dict]:
        """
        Find all tokens that are members of a compound headed by root.
        Uses the dependency structure to identify compound members.
        """
        members = [root]
        root_idx = root['idx']
        
        # Find all tokens whose head is this root and are part of a compound
        for item in sentence_data:
            if item['idx'] != root_idx:
                # Check if this token points to the root and is a compound member
                if item['head_idx'] == root_idx and item['comp_label'].startswith('Comp'):
                    members.append(item)
                    # Recursively find members pointing to this token
                    sub_members = self._find_transitive_members(sentence_data, item)
                    members.extend(sub_members)
        
        # Remove duplicates
        seen = set()
        unique_members = []
        for m in members:
            if m['idx'] not in seen:
                seen.add(m['idx'])
                unique_members.append(m)
        
        return unique_members
    
    def _find_transitive_members(self, sentence_data: List[Dict], parent: Dict) -> List[Dict]:
        """Find all tokens transitively connected to parent in the compound"""
        members = []
        parent_idx = parent['idx']
        
        for item in sentence_data:
            if item['head_idx'] == parent_idx and item['comp_label'].startswith('Comp'):
                members.append(item)
                # Recursively find more members
                sub_members = self._find_transitive_members(sentence_data, item)
                members.extend(sub_members)
        
        return members
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        sentence = self.data[item]['tokens']
        relation_labels = self.data[item]['relation_labels']
        compounds = self.data[item]['compounds']
        
        # Convert relation_labels to label IDs
        labels = [self.label_set.label2id(l) for l in relation_labels]
        
        return {
            'tokens': sentence,
            'labels': labels,
            'compounds': compounds
        }


class NeCTICollator:
    """Collator for NeCTI dataset compatible with DiffusionSL"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        # Extract data from batch items
        sentences = [item['tokens'] for item in batch]
        labels = [item['labels'] for item in batch]
        compounds = [item['compounds'] for item in batch]
        
        # Tokenize with word-level alignment
        inputs_encoding = self.tokenizer(
            sentences,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs_encoding.input_ids
        attention_mask = inputs_encoding.attention_mask
        word_ids = [inputs_encoding.word_ids(i) for i in range(len(batch))]
        
        # Align labels with subword tokens
        seq_labels = []
        for ids, las in zip(word_ids, labels):
            temp = [i if i is not None else -100 for i in ids]
            seql = []
            for i in range(len(temp)):
                if temp[i] == -100:
                    seql.append(-100)
                else:
                    if i == 0 or temp[i] != temp[i - 1]:
                        # First subword of a word gets the label
                        if temp[i] < len(las):
                            seql.append(las[temp[i]])
                        else:
                            seql.append(-100)
                    else:
                        # Subsequent subwords are ignored in loss
                        seql.append(-100)
            seq_labels.append(seql)
        
        assert len(seq_labels) == len(sentences)
        assert len(seq_labels[0]) == len(input_ids[0])
        
        return {
            'input_ids': torch.as_tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.as_tensor(attention_mask, dtype=torch.long),
            'seq_labels': torch.as_tensor(seq_labels, dtype=torch.long),
            'compounds': compounds,  # Keep compound span information
            'word_ids': word_ids  # Keep word-to-token mapping
        }
