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
        
        # Sort labels: CompNo first, then Comp tags, then relation types
        self._labelset = sorted(list(self._labelset), key=lambda x: (
            0 if x == 'CompNo' else (1 if x.startswith('Comp') else 2),
            x
        ))
        
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}
        
        print(f"Loaded {len(self._labelset)} labels for {granularity} granularity")
        print(f"Labels: {self._labelset}")
    
    def _extract_labels_from_file(self, filepath: str) -> set:
        """Extract unique labels from CoNLL-U format file"""
        labels = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        label = parts[2]  # Third column is the compound label
                        labels.add(label)
        return labels
    
    def label2id(self, label: str) -> int:
        return self._label2id.get(label, self._label2id.get('CompNo', 0))
    
    def id2label(self, idx: int) -> str:
        return self._id2label.get(idx, 'CompNo')
    
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
        """Parse CoNLL-U format file into list of sentences"""
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:  # Empty line indicates sentence boundary
                    if current_tokens:
                        # Remove DUMMY token if present
                        if current_tokens and current_tokens[-1] == 'DUMMY':
                            current_tokens = current_tokens[:-1]
                            current_labels = current_labels[:-1]
                        
                        if current_tokens:  # Only add non-empty sentences
                            sentences.append({
                                'tokens': current_tokens,
                                'labels': current_labels
                            })
                        current_tokens = []
                        current_labels = []
                elif not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        token = parts[1]  # Second column is the token
                        label = parts[2]  # Third column is the compound label
                        current_tokens.append(token)
                        current_labels.append(label)
        
        # Handle last sentence if file doesn't end with empty line
        if current_tokens:
            if current_tokens and current_tokens[-1] == 'DUMMY':
                current_tokens = current_tokens[:-1]
                current_labels = current_labels[:-1]
            if current_tokens:
                sentences.append({
                    'tokens': current_tokens,
                    'labels': current_labels
                })
        
        return sentences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        sentence = self.data[item]['tokens']
        labels = [self.label_set.label2id(l) for l in self.data[item]['labels']]
        return sentence, labels


class NeCTICollator:
    """Collator for NeCTI dataset compatible with DiffusionSL"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        sentences, labels = map(list, zip(*batch))
        
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
        
        return torch.as_tensor(input_ids, dtype=torch.long), \
               torch.as_tensor(attention_mask, dtype=torch.long), \
               torch.as_tensor(seq_labels, dtype=torch.long)
