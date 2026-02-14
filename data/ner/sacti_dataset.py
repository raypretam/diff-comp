"""
Dataset module for SaCTI (Sanskrit Compound Type Identification)
Adapted for DiffusionSL framework with XLM-R encoder
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict
import os


class SaCTILabelSet:
    """Label set for Sanskrit Compound Type Identification (multi-lingual support)"""
    
    def __init__(self, data_path: str, granularity: str = 'coarse', language: str = 'auto'):
        """
        Args:
            data_path: Base path to SaCTI data (e.g., /path/to/SaCTI/data)
            granularity: 'coarse' or 'fine' (case-insensitive)
            language: 'sacti' (Sanskrit), 'marathi', 'english', or 'auto' for detection
        """
        # Normalize granularity to lowercase
        granularity = granularity.lower()
        assert granularity in ['coarse', 'fine'], "granularity must be 'coarse' or 'fine'"
        
        self.granularity = granularity
        self.language = self._detect_language(data_path, language)
        
        # Build path based on language
        if self.language == 'sacti':
            self.data_path = os.path.join(data_path, 'sacti', granularity)
        else:
            self.data_path = os.path.join(data_path, self.language, granularity)
        
        print(f"Initializing SaCTILabelSet")
        print(f"  Language: {self.language}")
        print(f"  Data path: {self.data_path}")
        
        # Read all splits to collect labels
        self._labelset = set()
        self._labelset.add('No_rel')  # For tokens without compound relations
        
        for split in ['train', 'dev', 'test']:
            filepath = os.path.join(self.data_path, f"{split}.conll")
            if os.path.exists(filepath):
                labels = self._extract_labels_from_file(filepath)
                self._labelset.update(labels)
        
        # Sort labels: No_rel first, then alphabetically
        self._labelset = sorted(list(self._labelset), key=lambda x: (
            0 if x == 'No_rel' else 1,
            x
        ))
        
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}
        
        print(f"Loaded {len(self._labelset)} labels for {granularity} granularity")
        print(f"Labels: {self._labelset}")
    
    def _detect_language(self, data_path: str, language: str) -> str:
        """Auto-detect language from path or validate specified language"""
        if language != 'auto':
            return language
        
        # Try to infer from path
        if 'sacti' in data_path.lower():
            return 'sacti'
        elif 'marathi' in data_path.lower():
            return 'marathi'
        elif 'english' in data_path.lower():
            return 'english'
        
        # Check which subdirectories exist
        for lang in ['sacti', 'marathi', 'english']:
            if os.path.exists(os.path.join(data_path, lang)):
                print(f"Auto-detected language: {lang}")
                return lang
        
        # Default to Sanskrit
        print("Could not auto-detect language, defaulting to 'sacti'")
        return 'sacti'
    
    def _extract_labels_from_file(self, filepath: str) -> set:
        """Extract compound type labels from SaCTI CoNLL file (language-aware)"""
        labels = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 8:
                    compound_type = self._extract_label_from_line(parts)
                    if compound_type and compound_type != '_':
                        labels.add(compound_type)
        
        return labels
    
    def _extract_label_from_line(self, parts: List[str]) -> str:
        """Extract label based on language format"""
        if self.language == 'english':
            # English: compound_word marker in column 8, actual type needs special handling
            # Look for 'compound' in column 8 (dependency relation)
            dep_rel = parts[8] if len(parts) > 8 else '_'
            if 'compound' in dep_rel.lower():
                return 'Compound'
            # Also check column 7 for semantic role
            sem_role = parts[7] if len(parts) > 7 else '_'
            if sem_role in ['ARG0', 'ARG1', 'ARG2']:
                return sem_role
            return 'No_rel'
        
        elif self.language == 'marathi':
            # Marathi: Column 7 has single letter codes (T, A, etc.)
            compound_type = parts[7] if len(parts) > 7 else '_'
            # Expand single letter codes to full names for consistency
            type_mapping = {
                'T': 'Tatpurusha',
                'A': 'Avyayibhava',
                'B': 'Bahuvrihi',
                'D': 'Dvandva',
                'K': 'Karmadharaya',
                'Dv': 'Dvigu'
            }
            return type_mapping.get(compound_type, compound_type)
        
        else:  # sacti (Sanskrit)
            # Sanskrit: Column 7 has full compound type names
            return parts[7] if len(parts) > 7 else '_'
    
    def label2id(self, label: str) -> int:
        return self._label2id.get(label, self._label2id.get('No_rel', 0))
    
    def id2label(self, idx: int) -> str:
        return self._id2label.get(idx, 'No_rel')
    
    def __len__(self):
        return len(self._labelset)
    
    def __str__(self):
        string = [f"{v}:\t{k}" for k, v in self._label2id.items()]
        return '\n'.join(string)


class SaCTIDataset(Dataset):
    """Dataset for Sanskrit Compound Type Identification (multi-lingual support)"""
    
    def __init__(self, data_path: str, split: str, label_set: SaCTILabelSet):
        """
        Args:
            data_path: Base path to SaCTI data (will use label_set's language)
            split: 'train', 'dev', or 'test'
            label_set: SaCTILabelSet instance (contains language info)
        """
        self.data_path = data_path
        self.split = split
        self.label_set = label_set
        self.language = label_set.language
        
        self.data = []
        self._load_data()
        
        print(f"Loaded {len(self.data)} sentences for {split} split ({self.language})")
    
    def _load_data(self):
        """Load SaCTI data from CoNLL file"""
        # label_set already has correct path with language folder
        filepath = os.path.join(self.label_set.data_path, f"{self.split}.conll")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        current_sent = []
        sent_tokens = []
        sent_labels = []
        sent_gram_roles = []  # Store grammatical roles (column 8)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    if sent_tokens:
                        self.data.append({
                            'tokens': sent_tokens,
                            'compound_types': sent_labels,
                            'gram_roles': sent_gram_roles,
                            'raw_data': current_sent
                        })
                        sent_tokens = []
                        sent_labels = []
                        sent_gram_roles = []
                        current_sent = []
                    continue
                
                if line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 9:
                    token_id = parts[0]
                    word = parts[1]
                    
                    # Use language-aware label extraction
                    compound_type = self.label_set._extract_label_from_line(parts)
                    if not compound_type or compound_type == '_':
                        compound_type = 'No_rel'
                    
                    gram_role = parts[8] if parts[8] != '_' else 'No_rel'
                    
                    sent_tokens.append(word)
                    sent_labels.append(compound_type)
                    sent_gram_roles.append(gram_role)
                    current_sent.append(parts)
            
            # Handle last sentence
            if sent_tokens:
                self.data.append({
                    'tokens': sent_tokens,
                    'compound_types': sent_labels,
                    'gram_roles': sent_gram_roles,
                    'raw_data': current_sent
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        sentence = self.data[item]['tokens']
        compound_types = self.data[item]['compound_types']
        
        # Convert labels to IDs
        labels = [self.label_set.label2id(ct) for ct in compound_types]
        
        return {
            'tokens': sentence,
            'labels': labels,
            'compound_types': compound_types  # Keep original for reference
        }


class SaCTICollator:
    """Collator for SaCTI dataset compatible with DiffusionSL"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 256, label_set=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_set = label_set
    
    def __call__(self, batch):
        # Extract data from batch items
        sentences = [item['tokens'] for item in batch]
        labels = [item['labels'] for item in batch]
        
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
                            label_id = las[temp[i]]
                            seql.append(label_id)
                        else:
                            seql.append(-100)
                    else:
                        # Subsequent subwords are ignored in loss
                        seql.append(-100)
            seq_labels.append(seql)
        
        assert len(seq_labels) == len(sentences)
        assert len(seq_labels[0]) == len(input_ids[0])
        
        result = {
            'input_ids': torch.as_tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.as_tensor(attention_mask, dtype=torch.long),
            'seq_labels': torch.as_tensor(seq_labels, dtype=torch.long),
            'word_ids': word_ids,
            'sentences': sentences
        }
        
        return result
