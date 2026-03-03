"""
Inference Script for Single-Stage Bi-Affine Diffusion NeCTI
============================================================

Loads a trained checkpoint from train_single_stage_biaffine.py and runs
inference on test (or dev/ood) split.

Computes:
    - Token-level: Coarse Accuracy, Fine Accuracy
    - Span-level:  USS, LSS, EM
    - Per-class fine-grained F1 breakdown
    - Optional: export predictions to CoNLL-U format

Usage:
    # Basic evaluation on test set
    python inference_single_stage_biaffine.py \\
        --checkpoint saved_models/single_stage_biaffine_necti/best.pt \\
        --config configs/single_stage_biaffine.yaml \\
        --split test

    # With post-hoc constraint
    python inference_single_stage_biaffine.py \\
        --checkpoint saved_models/single_stage_biaffine_necti/best.pt \\
        --config configs/single_stage_biaffine.yaml \\
        --split test --use_constrained_eval

    # Multi-sample majority voting (runs DDIM N times, takes majority)
    python inference_single_stage_biaffine.py \\
        --checkpoint saved_models/single_stage_biaffine_necti/best.pt \\
        --config configs/single_stage_biaffine.yaml \\
        --split test --num_samples 5

    # Export predictions to CoNLL-U file
    python inference_single_stage_biaffine.py \\
        --checkpoint saved_models/single_stage_biaffine_necti/best.pt \\
        --config configs/single_stage_biaffine.yaml \\
        --split test --export_conllu predictions.conllu

    # Evaluate on dev set
    python inference_single_stage_biaffine.py \\
        --checkpoint saved_models/single_stage_biaffine_necti/best.pt \\
        --config configs/single_stage_biaffine.yaml \\
        --split dev
"""

import os
import sys
import json
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from models.single_stage_biaffine_diffusion import (
    SingleStageBiAffineDiffusion,
    create_single_stage_biaffine_model,
)
from data.ner.necti_dataset import NeCTIDataset, NeCTILabelSet, NeCTICollator
from data.ner.necti_dataset_hierarchial import get_coarse_id_from_fine_label, COARSE_CATEGORIES


class SingleStageBiAffineInference:
    """
    Inference engine for Single-Stage Bi-Affine Diffusion NeCTI.
    
    Supports:
        - Single-pass inference
        - Multi-sample majority voting
        - Post-hoc coarse→fine constraint
        - CoNLL-U export
        - Per-class metric breakdown
    """

    def __init__(self, checkpoint_path: str, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config['gpu']['use_gpu'] else 'cpu'
        )

        # Label set
        data_config = config['data']
        self.fine_label_set = NeCTILabelSet(
            data_path=data_config['data_path'],
            granularity='Finegrain',
            use_context=data_config['use_context'],
        )

        # Fine-to-coarse mapping
        self.fine_to_coarse_map = {}
        for fine_id in range(len(self.fine_label_set)):
            fine_name = self.fine_label_set.id2label(fine_id)
            coarse_id = get_coarse_id_from_fine_label(fine_name)
            self.fine_to_coarse_map[fine_id] = coarse_id

        # Build model
        cfg = config
        biaffine_cfg = cfg.get('biaffine', {})
        self.model = create_single_stage_biaffine_model(
            device=str(self.device),
            backbone=cfg['backbone'],
            dim_model=cfg['dim_model'],
            freeze_bert=cfg.get('freeze_bert', False),
            time_steps=cfg['time_steps'],
            sampling_steps=cfg['sampling_steps'],
            snr_scale=cfg['snr_scale'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            num_fine_classes=cfg.get('num_fine_classes', len(self.fine_label_set)),
            num_coarse_classes=cfg.get('num_coarse_classes', len(COARSE_CATEGORIES)),
            fine_to_coarse_map=self.fine_to_coarse_map,
            objective=cfg['objective'],
            loss_type=cfg['loss_type'],
            aux_coarse_weight=biaffine_cfg.get('aux_coarse_weight', 0.1),
        )

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")
        print(f"  Best USS: {ckpt.get('best_uss', '?')}")
        print(f"  Best LSS: {ckpt.get('best_lss', '?')}")

        # Tokenizer and data
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['backbone'])
        self.collator = NeCTICollator(
            tokenizer=self.tokenizer,
            max_length=data_config['max_length'],
        )

        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {total:,}")
        print(f"  Device: {self.device}")

    def _get_dataloader(self, split: str, batch_size: int = 8) -> DataLoader:
        data_config = self.config['data']
        dataset = NeCTIDataset(
            data_path=data_config['data_path'],
            mode=split,
            label_set=self.fine_label_set,
            use_context=data_config['use_context'],
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_config['num_workers'],
            collate_fn=self.collator,
            pin_memory=True,
        )

    # =========================================================================
    # Core Inference
    # =========================================================================

    @torch.no_grad()
    def run_inference(
        self,
        split: str = 'test',
        batch_size: int = 8,
        use_constraint: bool = False,
        num_samples: int = 1,
    ) -> Dict:
        """
        Run inference on a dataset split.
        
        Args:
            split: 'test', 'dev', or 'ood'
            batch_size: Batch size for inference
            use_constraint: Apply post-hoc coarse→fine constraint
            num_samples: Number of DDIM samples for majority voting (1 = single pass)
        
        Returns:
            Dict with metrics and per-sentence predictions
        """
        dataloader = self._get_dataloader(split, batch_size)

        all_coarse_preds = []
        all_coarse_labels = []
        all_fine_preds = []
        all_fine_labels = []
        true_compounds_all = []
        pred_compounds_all = []
        sentence_results = []  # For export

        desc = f"Inference [{split}]"
        if num_samples > 1:
            desc += f" (majority vote x{num_samples})"

        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['seq_labels'].to(self.device)

            # Derive coarse ground truth
            coarse_labels = self._get_coarse_labels(fine_labels)

            if num_samples > 1:
                # Multi-sample majority voting
                coarse_preds, fine_preds = self._majority_vote_inference(
                    input_ids, attention_mask, use_constraint, num_samples
                )
            else:
                # Single-pass inference
                coarse_preds, fine_preds = self.model.inference(
                    input_ids, attention_mask, apply_constraint=use_constraint,
                )

            mask = fine_labels != -100

            for i in range(input_ids.shape[0]):
                sent_mask = mask[i]
                if sent_mask.sum() == 0:
                    continue

                pred_ids = fine_preds[i][sent_mask].cpu().tolist()
                true_ids = fine_labels[i][sent_mask].cpu().tolist()
                coarse_pred_ids = coarse_preds[i][sent_mask].cpu().tolist()
                coarse_true_ids = coarse_labels[i][sent_mask].cpu().tolist()

                all_fine_preds.extend(pred_ids)
                all_fine_labels.extend(true_ids)
                all_coarse_preds.extend(coarse_pred_ids)
                all_coarse_labels.extend(coarse_true_ids)

                pred_names = [
                    self.fine_label_set.id2label(pid)
                    if 0 <= pid < len(self.fine_label_set) else "No_rel"
                    for pid in pred_ids
                ]
                true_names = [
                    self.fine_label_set.id2label(tid)
                    if 0 <= tid < len(self.fine_label_set) else "No_rel"
                    for tid in true_ids
                ]

                pred_compounds = self._extract_compounds_from_labels(pred_names)
                true_compounds = self._extract_compounds_from_labels(true_names)
                pred_compounds_all.append(pred_compounds)
                true_compounds_all.append(true_compounds)

                # Store per-sentence for export
                sentence_results.append({
                    'pred_ids': pred_ids,
                    'true_ids': true_ids,
                    'pred_names': pred_names,
                    'true_names': true_names,
                    'pred_compounds': pred_compounds,
                    'true_compounds': true_compounds,
                })

        # Compute metrics
        metrics = self._compute_all_metrics(
            all_fine_preds, all_fine_labels,
            all_coarse_preds, all_coarse_labels,
            true_compounds_all, pred_compounds_all,
        )

        return {
            'metrics': metrics,
            'sentence_results': sentence_results,
            'config': {
                'split': split,
                'num_samples': num_samples,
                'use_constraint': use_constraint,
            },
        }

    @torch.no_grad()
    def _majority_vote_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_constraint: bool,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run DDIM num_samples times and take per-token majority vote."""
        all_fine = []
        all_coarse = []

        # Extract features once (shared across samples)
        bert_features = self.model.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        enhanced_features, biaffine_scores = self.model.biaffine_encoder(
            bert_features, attention_mask
        )

        for _ in range(num_samples):
            fine_preds = self.model.sample(enhanced_features, biaffine_scores, attention_mask)
            coarse_preds = self.model.coarse_head(enhanced_features).argmax(dim=-1)

            if use_constraint and self.model.coarse_to_fine_map is not None:
                fine_preds = self.model._apply_constraint(fine_preds, coarse_preds)

            all_fine.append(fine_preds)
            all_coarse.append(coarse_preds)

        # Majority vote per token
        fine_stack = torch.stack(all_fine, dim=0)  # [S, B, N]
        coarse_stack = torch.stack(all_coarse, dim=0)

        B, N = fine_stack.shape[1], fine_stack.shape[2]

        fine_voted = torch.zeros(B, N, dtype=torch.long, device=fine_stack.device)
        coarse_voted = torch.zeros(B, N, dtype=torch.long, device=coarse_stack.device)

        for b in range(B):
            for n in range(N):
                fine_votes = fine_stack[:, b, n].cpu().tolist()
                coarse_votes = coarse_stack[:, b, n].cpu().tolist()
                fine_voted[b, n] = Counter(fine_votes).most_common(1)[0][0]
                coarse_voted[b, n] = Counter(coarse_votes).most_common(1)[0][0]

        return coarse_voted, fine_voted

    # =========================================================================
    # Metrics
    # =========================================================================

    def _compute_all_metrics(
        self,
        fine_preds, fine_labels,
        coarse_preds, coarse_labels,
        true_compounds_all, pred_compounds_all,
    ) -> Dict[str, float]:
        """Compute all metrics."""
        fine_preds_arr = np.array(fine_preds)
        fine_labels_arr = np.array(fine_labels)
        coarse_preds_arr = np.array(coarse_preds)
        coarse_labels_arr = np.array(coarse_labels)

        coarse_acc = float(np.mean(coarse_preds_arr == coarse_labels_arr)) if len(coarse_labels) else 0.0
        fine_acc = float(np.mean(fine_preds_arr == fine_labels_arr)) if len(fine_labels) else 0.0

        uss = self._compute_span_f1(true_compounds_all, pred_compounds_all, labeled=False)
        lss = self._compute_span_f1(true_compounds_all, pred_compounds_all, labeled=True)
        em = self._compute_exact_match(true_compounds_all, pred_compounds_all)

        # Binary P/R/F1 (compound vs non-compound)
        binary = self._compute_binary_f1(fine_preds_arr, fine_labels_arr)

        # Per-class fine F1
        per_class = self._compute_per_class_f1(fine_preds_arr, fine_labels_arr)

        return {
            'coarse_acc': coarse_acc,
            'fine_acc': fine_acc,
            'USS': uss,
            'LSS': lss,
            'EM': em,
            'binary_precision': binary['precision'],
            'binary_recall': binary['recall'],
            'binary_f1': binary['f1'],
            'per_class_f1': per_class,
        }

    def _compute_binary_f1(self, preds, labels):
        """Binary compound vs non-compound F1."""
        no_rel_id = self.fine_label_set.label2id('No_rel')
        root_id = self.fine_label_set.label2id('root')

        pred_compound = (preds != no_rel_id) & (preds != root_id)
        label_compound = (labels != no_rel_id) & (labels != root_id)

        tp = int(np.sum(pred_compound & label_compound))
        fp = int(np.sum(pred_compound & ~label_compound))
        fn = int(np.sum(~pred_compound & label_compound))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {'precision': p, 'recall': r, 'f1': f1}

    def _compute_per_class_f1(self, preds, labels) -> Dict[str, Dict[str, float]]:
        """Per-class precision/recall/F1."""
        classes = sorted(set(labels.tolist()) | set(preds.tolist()))
        results = {}
        for cls_id in classes:
            if cls_id < 0 or cls_id >= len(self.fine_label_set):
                continue
            cls_name = self.fine_label_set.id2label(cls_id)
            tp = int(np.sum((preds == cls_id) & (labels == cls_id)))
            fp = int(np.sum((preds == cls_id) & (labels != cls_id)))
            fn = int(np.sum((preds != cls_id) & (labels == cls_id)))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            results[cls_name] = {'precision': p, 'recall': r, 'f1': f1, 'support': tp + fn}
        return results

    def _get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        coarse_labels = fine_labels.clone()
        valid = fine_labels != -100
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            coarse_labels[(fine_labels == fine_id) & valid] = coarse_id
        return coarse_labels

    # =========================================================================
    # Span / Compound Utilities
    # =========================================================================

    def _extract_compounds_from_labels(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        NON_COMPOUND = {'No_rel', 'root', '_'}
        compounds = []
        i = 0
        n = len(labels)
        while i < n:
            if labels[i] not in NON_COMPOUND:
                region_start = i
                while i < n and labels[i] not in NON_COMPOUND:
                    i += 1
                region_end = i - 1
                comp_start = region_start
                member_labels = []
                for j in range(region_start, region_end + 1):
                    if labels[j] == 'Comp_root':
                        if member_labels:
                            label = Counter(member_labels).most_common(1)[0][0]
                        else:
                            label = 'Comp_root'
                        compounds.append((comp_start, j, label))
                        comp_start = j + 1
                        member_labels = []
                    else:
                        member_labels.append(labels[j])
                if comp_start <= region_end and member_labels:
                    label = Counter(member_labels).most_common(1)[0][0]
                    compounds.append((comp_start, region_end, label))
            else:
                i += 1
        return compounds

    def _compute_span_f1(self, true_all, pred_all, labeled=False):
        total_correct = total_pred = total_true = 0
        for true_comps, pred_comps in zip(true_all, pred_all):
            if labeled:
                true_set = set(true_comps)
                pred_set = set(pred_comps)
            else:
                true_set = set((s, e) for s, e, _ in true_comps)
                pred_set = set((s, e) for s, e, _ in pred_comps)
            total_correct += len(true_set & pred_set)
            total_pred += len(pred_set)
            total_true += len(true_set)
        p = total_correct / total_pred if total_pred > 0 else 0.0
        r = total_correct / total_true if total_true > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _compute_exact_match(self, true_all, pred_all):
        total_true = matched = 0
        for true_comps, pred_comps in zip(true_all, pred_all):
            if not true_comps:
                continue
            total_true += len(true_comps)
            true_set = set(true_comps)
            for comp in pred_comps:
                if comp in true_set:
                    matched += 1
        return matched / total_true if total_true > 0 else 0.0

    # =========================================================================
    # CoNLL-U Export
    # =========================================================================

    def export_conllu(self, sentence_results: List[Dict], output_file: str):
        """
        Export predictions to CoNLL-U format compatible with Eval_USS_LSS.py.
        
        Format per sentence:
            # sent_id = N
            token_idx  _  _  _  _  _  head_idx  label  _  _
        """
        with open(output_file, 'w') as f:
            for sent_idx, sent in enumerate(sentence_results):
                f.write(f"# sent_id = {sent_idx + 1}\n")
                for tok_idx, (pred_name, true_name) in enumerate(
                    zip(sent['pred_names'], sent['true_names'])
                ):
                    # CoNLL-U: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                    # We fill HEAD=0 and DEPREL=pred_name for the prediction file
                    f.write(f"{tok_idx + 1}\t_\t_\t_\t_\t_\t0\t{pred_name}\t_\t_\n")
                f.write("\n")
        print(f"Exported {len(sentence_results)} sentences to {output_file}")

    def export_conllu_gold(self, sentence_results: List[Dict], output_file: str):
        """Export gold labels in CoNLL-U format."""
        with open(output_file, 'w') as f:
            for sent_idx, sent in enumerate(sentence_results):
                f.write(f"# sent_id = {sent_idx + 1}\n")
                for tok_idx, true_name in enumerate(sent['true_names']):
                    f.write(f"{tok_idx + 1}\t_\t_\t_\t_\t_\t0\t{true_name}\t_\t_\n")
                f.write("\n")
        print(f"Exported {len(sentence_results)} gold sentences to {output_file}")


# =============================================================================
# CLI
# =============================================================================

def print_results(results: Dict):
    """Pretty-print results."""
    metrics = results['metrics']
    cfg = results['config']

    print(f"\n{'='*65}")
    print(f"INFERENCE RESULTS — {cfg['split'].upper()} SET")
    if cfg['num_samples'] > 1:
        print(f"  Majority voting: {cfg['num_samples']} samples")
    if cfg['use_constraint']:
        print(f"  Post-hoc constraint: ON")
    print(f"{'='*65}")

    print(f"\n  Coarse Accuracy:  {metrics['coarse_acc'] * 100:.2f}%")
    print(f"  Fine Accuracy:    {metrics['fine_acc'] * 100:.2f}%")
    print(f"\n  USS (Unlabeled):  {metrics['USS'] * 100:.2f}")
    print(f"  LSS (Labeled):    {metrics['LSS'] * 100:.2f}")
    print(f"  EM (Exact Match): {metrics['EM'] * 100:.2f}")
    print(f"\n  Binary Compound Detection:")
    print(f"    Precision: {metrics['binary_precision'] * 100:.2f}")
    print(f"    Recall:    {metrics['binary_recall'] * 100:.2f}")
    print(f"    F1:        {metrics['binary_f1'] * 100:.2f}")

    # Per-class breakdown (sorted by support, top 20)
    per_class = metrics['per_class_f1']
    if per_class:
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['support'], reverse=True)
        print(f"\n  Per-Class F1 (top 20 by support):")
        print(f"  {'Label':<25} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-'*57}")
        for name, m in sorted_classes[:20]:
            print(f"  {name:<25} {m['precision']*100:>7.2f} {m['recall']*100:>7.2f} "
                  f"{m['f1']*100:>7.2f} {m['support']:>8}")

    print(f"\n{'='*65}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference for Single-Stage Bi-Affine Diffusion NeCTI'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained checkpoint (.pt)',
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config YAML (same as used for training)',
    )
    parser.add_argument(
        '--split', type=str, default='test', choices=['test', 'dev', 'ood'],
        help='Dataset split to evaluate on',
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for inference',
    )
    parser.add_argument(
        '--use_constrained_eval', action='store_true',
        help='Apply post-hoc coarse→fine constraint',
    )
    parser.add_argument(
        '--num_samples', type=int, default=1,
        help='Number of DDIM samples for majority voting (1 = single pass)',
    )
    parser.add_argument(
        '--export_conllu', type=str, default=None,
        help='If set, export predictions to this CoNLL-U file',
    )
    parser.add_argument(
        '--export_gold_conllu', type=str, default=None,
        help='If set, export gold labels to this CoNLL-U file',
    )
    parser.add_argument(
        '--save_json', type=str, default=None,
        help='If set, save full results (metrics + per-sentence) to this JSON file',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Build inference engine
    engine = SingleStageBiAffineInference(args.checkpoint, config, args)

    # Run inference
    results = engine.run_inference(
        split=args.split,
        batch_size=args.batch_size,
        use_constraint=args.use_constrained_eval,
        num_samples=args.num_samples,
    )

    # Print results
    print_results(results)

    # Export CoNLL-U
    if args.export_conllu:
        engine.export_conllu(results['sentence_results'], args.export_conllu)
    if args.export_gold_conllu:
        engine.export_conllu_gold(results['sentence_results'], args.export_gold_conllu)

    # Save full JSON
    if args.save_json:
        # Make per_class_f1 serializable
        save_data = {
            'metrics': {
                k: v for k, v in results['metrics'].items()
                if k != 'per_class_f1'
            },
            'per_class_f1': results['metrics']['per_class_f1'],
            'config': results['config'],
            'num_sentences': len(results['sentence_results']),
        }
        with open(args.save_json, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved results to {args.save_json}")


if __name__ == '__main__':
    main()
