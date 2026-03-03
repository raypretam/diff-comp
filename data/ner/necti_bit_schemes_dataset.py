"""
Dataset + Collator for Bit4 / Bit7 / HexaTagging NeCTI experiments.

Produces per-token JOINT label ids compatible with DiffusionSL's BitDit:
    seq_labels[b, pos] = struct_id * num_comptypes + comptype_id

A separate seq_struct_labels tensor carries only the structural class
(used for the optional auxiliary structural head).
"""

import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict

from data.ner.necti_bit_schemes import NeCTIBitSchemeLabelSet


class NeCTIBitSchemeDataset(Dataset):
    """Dataset for Bit4 / Bit7 / HexaTagging structural encodings."""

    def __init__(
        self,
        data_path: str,
        mode: str,
        label_set: NeCTIBitSchemeLabelSet,
        use_context: bool = False,
    ):
        super().__init__()
        assert mode in ('train', 'dev', 'test', 'ood')
        self.label_set   = label_set
        self.mode        = mode
        self.use_context = use_context

        fp = os.path.join(label_set.data_path, f"{label_set.granularity}_{mode}_san")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Data file not found: {fp}")

        raw = label_set._parse_conllu(fp)
        self.samples = self._build(raw)
        print(f"[{label_set.scheme.upper()}Dataset] {mode}: {len(self.samples)} sentences")

    def _build(self, raw_sentences: List[List[Dict]]) -> List[Dict]:
        samples = []
        for sent in raw_sentences:
            if not sent:
                continue
            # Strip DUMMY token
            if sent[-1]['token'] == 'DUMMY':
                sent = sent[:-1]
            if not sent:
                continue
            # After stripping DUMMY, clamp any head_idx that pointed to it
            n = len(sent)
            sent = [{**item, 'head_idx': item['head_idx'] if item['head_idx'] <= n else 0}
                    for item in sent]

            struct_ids, ct_ids = self.label_set.encode_sentence(sent)
            joint_ids = [self.label_set.joint_id(s, c)
                         for s, c in zip(struct_ids, ct_ids)]
            tokens    = [item['token'] for item in sent]

            samples.append({
                'tokens'     : tokens,
                'struct_ids' : struct_ids,
                'ct_ids'     : ct_ids,
                'joint_ids'  : joint_ids,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NeCTIBitSchemeCollator:
    """
    Collates NeCTIBitSchemeDataset samples.

    Outputs:
        input_ids            : (B, L)
        attention_mask       : (B, L)
        seq_labels           : (B, L)  joint ids — fed to BitDit
        seq_struct_labels    : (B, L)  structural ids — for aux head
        seq_ct_labels        : (B, L)  comptype ids  — for aux head
    """

    IGNORE = -100

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_set: NeCTIBitSchemeLabelSet,
        max_length: int = 256,
    ):
        self.tokenizer  = tokenizer
        self.label_set  = label_set
        self.max_length = max_length

    def __call__(self, batch: List[Dict]):
        tokens_batch = [s['tokens']     for s in batch]
        joint_batch  = [s['joint_ids']  for s in batch]
        struct_batch = [s['struct_ids'] for s in batch]
        ct_batch     = [s['ct_ids']     for s in batch]

        enc = self.tokenizer(
            tokens_batch,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        B, L = enc['input_ids'].shape
        IGN  = self.IGNORE

        seq_joint  = torch.full((B, L), IGN, dtype=torch.long)
        seq_struct = torch.full((B, L), IGN, dtype=torch.long)
        seq_ct     = torch.full((B, L), IGN, dtype=torch.long)

        for b in range(B):
            prev = None
            for pos, wid in enumerate(enc.word_ids(batch_index=b)):
                if wid is None or wid == prev:
                    continue
                prev = wid
                if wid < len(joint_batch[b]):
                    seq_joint[b, pos]  = joint_batch[b][wid]
                    seq_struct[b, pos] = struct_batch[b][wid]
                    seq_ct[b, pos]     = ct_batch[b][wid]

        return {
            'input_ids'         : enc['input_ids'],
            'attention_mask'    : enc['attention_mask'],
            'seq_labels'        : seq_joint,
            'seq_struct_labels' : seq_struct,
            'seq_ct_labels'     : seq_ct,
            'tokens'            : [s['tokens'] for s in batch],
        }
