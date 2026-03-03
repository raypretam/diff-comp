"""
Bracket-augmented dataset for NeCTI.

Each sample produces two label sequences per token:
  seq_bracket_labels  : bracket structure labels  (int tensor)
  seq_comptype_labels : compound type labels       (int tensor)

These can be fed to a dual-head diffusion model or concatenated for a joint head.
"""

import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional

from data.ner.necti_bracket import (
    NeCTIBracketLabelSet, encode_sentence, NULL_BRACKET
)


class NeCTIBracketDataset(Dataset):
    """Dataset that emits bracket + compound-type labels per token."""

    def __init__(
        self,
        data_path: str,
        mode: str,
        label_set: NeCTIBracketLabelSet,
        use_context: bool = False,
    ):
        super().__init__()
        assert mode in ('train', 'dev', 'test', 'ood'), \
            "mode must be train/dev/test/ood"

        self.label_set   = label_set
        self.mode        = mode
        self.use_context = use_context
        self.k           = label_set.k

        context_dir = 'With Context' if use_context else 'Without Context'
        granularity = label_set.granularity
        base_path = label_set.data_path   # already includes context_dir/granularity

        # Map modes to filenames
        _mode_map = {'train': 'train', 'dev': 'dev', 'test': 'test', 'ood': 'ood'}
        filename = f"{granularity}_{_mode_map[mode]}_san"
        filepath = os.path.join(base_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        raw_sentences = self.label_set._parse_conllu(filepath)
        self.samples = self._build_samples(raw_sentences)
        print(f"[BracketDataset] {mode}: {len(self.samples)} sentences")

    def _build_samples(self, raw_sentences: List[List[Dict]]) -> List[Dict]:
        samples = []
        for sent_data in raw_sentences:
            if not sent_data:
                continue
            # Strip DUMMY token if present
            if sent_data[-1]['token'] == 'DUMMY':
                sent_data = sent_data[:-1]
            if not sent_data:
                continue
            # Clamp head indices that pointed to the stripped DUMMY token (head > n → root=0)
            n = len(sent_data)
            sent_data = [{**item, 'head_idx': item['head_idx'] if item['head_idx'] <= n else 0}
                         for item in sent_data]

            brackets, comptypes = encode_sentence(sent_data, k=self.k)
            tokens = [item['token'] for item in sent_data]

            bracket_ids  = [self.label_set.br2id(b) for b in brackets]
            comptype_ids = [self.label_set.ct2id(c) for c in comptypes]

            samples.append({
                'tokens'       : tokens,
                'brackets'     : brackets,
                'comptypes'    : comptypes,
                'bracket_ids'  : bracket_ids,
                'comptype_ids' : comptype_ids,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class NeCTIBracketCollator:
    """
    Collator for NeCTIBracketDataset.

    Tokenises with word-piece tokeniser and aligns two label sequences:
      seq_bracket_labels  : (B, seq_len)  — bracket label ids
      seq_comptype_labels : (B, seq_len)  — compound-type label ids

    Positions corresponding to sub-word continuation pieces or padding are
    marked with IGNORE_INDEX (-100, compatible with CrossEntropyLoss).

    If `joint_head=True`, a single `seq_labels` tensor is produced with
    joint bracket×comptype ids (for use with existing single-head BitDit).
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_set: NeCTIBracketLabelSet,
        max_length: int = 256,
        joint_head: bool = False,
    ):
        self.tokenizer   = tokenizer
        self.label_set   = label_set
        self.max_length  = max_length
        self.joint_head  = joint_head

    def __call__(self, batch: List[Dict]):
        tokens_batch   = [s['tokens']       for s in batch]
        br_ids_batch   = [s['bracket_ids']  for s in batch]
        ct_ids_batch   = [s['comptype_ids'] for s in batch]

        # Tokenise while preserving word boundaries
        encoded = self.tokenizer(
            tokens_batch,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids      = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        B, L           = input_ids.shape

        seq_bracket  = torch.full((B, L), self.IGNORE_INDEX, dtype=torch.long)
        seq_comptype = torch.full((B, L), self.IGNORE_INDEX, dtype=torch.long)

        for b in range(B):
            word_ids = encoded.word_ids(batch_index=b)
            prev_word = None
            for pos, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid == prev_word:
                    # sub-word continuation → ignore
                    continue
                prev_word = wid
                if wid < len(br_ids_batch[b]):
                    seq_bracket[b, pos]  = br_ids_batch[b][wid]
                    seq_comptype[b, pos] = ct_ids_batch[b][wid]

        out = {
            'input_ids'          : input_ids,
            'attention_mask'     : attention_mask,
            'seq_bracket_labels' : seq_bracket,
            'seq_comptype_labels': seq_comptype,
            'compounds'          : [s.get('tokens', []) for s in batch],
        }

        if self.joint_head:
            # Merge into single seq_labels tensor for compatibility with BitDit
            seq_labels = torch.full((B, L), self.IGNORE_INDEX, dtype=torch.long)
            valid_mask = (seq_bracket != self.IGNORE_INDEX)
            seq_labels[valid_mask] = (
                seq_bracket[valid_mask] * self.label_set.num_comptype_classes
                + seq_comptype[valid_mask]
            )
            out['seq_labels'] = seq_labels

        return out
