"""
Inference script for Hierarchical Diffusion NeCTI
==================================================

Loads trained hierarchical model and evaluates on test/OOD datasets.

Two-stage inference:
  Stage 1: Global attention → coarse label prediction
  Stage 2: Local attention → fine-grained label refinement

Usage:
    # Evaluate on test set
    python inference_hierarchial.py --config configs/hierarchial_necti.yaml \
        --checkpoint saved_models/hierarchical_necti/best.pt --splits test

    # Evaluate on test + OOD, save predictions
    python inference_hierarchial.py --config configs/hierarchial_necti.yaml \
        --checkpoint saved_models/hierarchical_necti/best.pt \
        --splits test ood --save_predictions --output_dir ./inference_results

    # Evaluate on all splits
    python inference_hierarchial.py --config configs/hierarchial_necti.yaml \
        --checkpoint saved_models/hierarchical_necti/best.pt \
        --splits train dev test ood
"""

import os
import argparse
import yaml
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from models.hierarchial_diffusion import HierarchicalDiffusionNeCTI, create_hierarchical_model
from data.ner.necti_dataset import NeCTIDataset, NeCTILabelSet, NeCTICollator
from data.ner.necti_dataset_hierarchial import get_coarse_id_from_fine_label, COARSE_CATEGORIES


class HierarchicalInference:
    """Inference for hierarchical diffusion compound identification"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda',
        batch_size: int = 8,
    ):
        """
        Args:
            config_path: Path to the YAML config used during training
            checkpoint_path: Path to saved model checkpoint (best.pt)
            device: 'cuda' or 'cpu'
            batch_size: Batch size for inference
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.batch_size = batch_size

        # Device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Data config
        data_config = self.config['data']
        self.data_path = data_config['data_path']
        self.use_context = data_config['use_context']
        self.max_length = data_config['max_length']
        self.num_workers = data_config.get('num_workers', 4)

        # Label sets
        self.fine_label_set = NeCTILabelSet(
            data_path=self.data_path,
            granularity='Finegrain',
            use_context=self.use_context,
        )
        self.coarse_label_set = NeCTILabelSet(
            data_path=self.data_path,
            granularity='Coarse',
            use_context=self.use_context,
        )

        # Fine-to-coarse map
        self.fine_to_coarse_map = self._build_label_hierarchy()

        # Tokenizer & collator
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['backbone'])
        self.collate_fn = NeCTICollator(self.tokenizer, max_length=self.max_length)

        # Build model
        self._setup_model()

        # Load checkpoint
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        epoch = checkpoint.get('epoch', '?')
        best_metric = checkpoint.get('best_metric', '?')
        print(f"Checkpoint from epoch {epoch}, best metric: {best_metric}")
        print(f"Fine-grained classes: {len(self.fine_label_set)}")
        print(f"Coarse classes: {len(COARSE_CATEGORIES)}")
        print("Model loaded successfully!\n")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_label_hierarchy(self) -> Dict[int, int]:
        """Build fine-label-id → coarse-label-id mapping."""
        fine_to_coarse = {}
        for fine_id in range(len(self.fine_label_set)):
            fine_name = self.fine_label_set.id2label(fine_id)
            coarse_id = get_coarse_id_from_fine_label(fine_name)
            fine_to_coarse[fine_id] = coarse_id
        return fine_to_coarse

    def _setup_model(self):
        """Instantiate the hierarchical diffusion model (weights loaded later)."""
        cfg = self.config
        self.model = create_hierarchical_model(
            device=str(self.device),
            backbone=cfg['backbone'],
            dim_model=cfg['dim_model'],
            freeze_bert=True,  # No training, freeze backbone
            time_steps=cfg['time_steps'],
            sampling_steps=cfg['sampling_steps'],
            noise_schedule=cfg['noise_schedule'],
            snr_scale=cfg['snr_scale'],
            global_depth=cfg['stage1']['depth'],
            global_num_heads=cfg['stage1']['num_heads'],
            num_coarse_classes=cfg['stage1']['num_classes'],
            local_depth=cfg['stage2']['depth'],
            local_num_heads=cfg['stage2']['num_heads'],
            local_window_size=cfg['stage2']['window_size'],
            num_fine_classes=cfg['stage2']['num_classes'],
            fine_to_coarse_map=self.fine_to_coarse_map,
            objective=cfg['objective'],
            loss_type=cfg['loss_type'],
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

    def _get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        """Convert fine-grained labels to coarse labels."""
        coarse_labels = fine_labels.clone()
        valid_mask = fine_labels != -100
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            mask = (fine_labels == fine_id) & valid_mask
            coarse_labels[mask] = coarse_id
        return coarse_labels

    def _get_dataloader(self, mode: str) -> DataLoader:
        """Create dataloader for a given split."""
        dataset = NeCTIDataset(
            self.data_path, mode, self.fine_label_set, use_context=self.use_context
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        print(f"Loaded {len(dataset)} {mode} samples ({len(loader)} batches)")
        return loader

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, split_name: str = "test",
                 save_predictions: bool = False) -> Dict:
        """
        Evaluate the hierarchical model on a dataloader.

        Returns dict with:
            coarse_acc, fine_acc, USS, LSS, EM,
            uss_precision, uss_recall, lss_precision, lss_recall,
            and optionally 'detailed_predictions'.
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {split_name} set...")
        print(f"{'=' * 60}\n")

        self.model.eval()

        all_coarse_preds: List[int] = []
        all_coarse_labels: List[int] = []
        all_fine_preds: List[int] = []
        all_fine_labels: List[int] = []

        true_compounds_all: List[List[Tuple]] = []
        pred_compounds_all: List[List[Tuple]] = []

        detailed_predictions: List[Dict] = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['seq_labels'].to(self.device)
            coarse_labels = self._get_coarse_labels(fine_labels)

            # Two-stage inference
            coarse_preds, fine_preds = self.model.inference(input_ids, attention_mask)

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

                # Convert IDs → label names
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

                if save_predictions:
                    coarse_pred_names = [
                        COARSE_CATEGORIES[cid] if 0 <= cid < len(COARSE_CATEGORIES) else "No_rel"
                        for cid in coarse_pred_ids
                    ]
                    coarse_true_names = [
                        COARSE_CATEGORIES[cid] if 0 <= cid < len(COARSE_CATEGORIES) else "No_rel"
                        for cid in coarse_true_ids
                    ]
                    detailed_predictions.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'fine_predictions': pred_names,
                        'fine_true_labels': true_names,
                        'coarse_predictions': coarse_pred_names,
                        'coarse_true_labels': coarse_true_names,
                        'pred_compounds': [
                            {'start': s, 'end': e, 'label': l} for s, e, l in pred_compounds
                        ],
                        'true_compounds': [
                            {'start': s, 'end': e, 'label': l} for s, e, l in true_compounds
                        ],
                    })

        # -- Flat accuracy --
        coarse_acc = (
            np.mean(np.array(all_coarse_preds) == np.array(all_coarse_labels))
            if all_coarse_labels else 0.0
        )
        fine_acc = (
            np.mean(np.array(all_fine_preds) == np.array(all_fine_labels))
            if all_fine_labels else 0.0
        )

        # -- Span-based metrics --
        uss_p, uss_r, uss_f1 = self._compute_span_prf(
            true_compounds_all, pred_compounds_all, labeled=False
        )
        lss_p, lss_r, lss_f1 = self._compute_span_prf(
            true_compounds_all, pred_compounds_all, labeled=True
        )
        em = self._compute_exact_match(true_compounds_all, pred_compounds_all)

        # -- Print results --
        print(f"\n{split_name.upper()} Results:")
        print("-" * 50)
        print(f"  Coarse Acc:     {coarse_acc:.4f}")
        print(f"  Fine Acc:       {fine_acc:.4f}")
        print(f"  USS Precision:  {uss_p:.4f}")
        print(f"  USS Recall:     {uss_r:.4f}")
        print(f"  USS F1:         {uss_f1:.4f}")
        print(f"  LSS Precision:  {lss_p:.4f}")
        print(f"  LSS Recall:     {lss_r:.4f}")
        print(f"  LSS F1:         {lss_f1:.4f}")
        print(f"  Exact Match:    {em:.4f}")
        print("-" * 50)

        results: Dict = {
            'coarse_acc': float(coarse_acc),
            'fine_acc': float(fine_acc),
            'USS': float(uss_f1),
            'USS_precision': float(uss_p),
            'USS_recall': float(uss_r),
            'LSS': float(lss_f1),
            'LSS_precision': float(lss_p),
            'LSS_recall': float(lss_r),
            'EM': float(em),
        }

        if save_predictions:
            results['detailed_predictions'] = detailed_predictions

        return results

    # ------------------------------------------------------------------
    # Compound extraction & metric helpers (same as training script)
    # ------------------------------------------------------------------

    def _extract_compounds_from_labels(
        self, labels: List[str]
    ) -> List[Tuple[int, int, str]]:
        """
        Extract compound spans from a word-level label sequence.

        A compound is a contiguous run of non-No_rel/non-root tokens.
        Comp_root marks the end of each compound within a run.
        The compound label is the majority vote of member labels (excluding Comp_root).
        """
        NON_COMPOUND = {'No_rel', 'root', '_'}
        compounds: List[Tuple[int, int, str]] = []
        i = 0
        n = len(labels)

        while i < n:
            if labels[i] not in NON_COMPOUND:
                region_start = i
                while i < n and labels[i] not in NON_COMPOUND:
                    i += 1
                region_end = i - 1

                comp_start = region_start
                member_labels: List[str] = []

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

    def _compute_span_prf(
        self,
        true_all: List[List[Tuple]],
        pred_all: List[List[Tuple]],
        labeled: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Compute Precision, Recall, F1 over compound spans.

        USS (labeled=False): matches on (start, end) boundaries only
        LSS (labeled=True): matches on (start, end, label) tuples
        """
        total_correct = 0
        total_pred = 0
        total_true = 0

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
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def _compute_exact_match(
        self,
        true_all: List[List[Tuple]],
        pred_all: List[List[Tuple]],
    ) -> float:
        """
        Exact Match: fraction of true compounds exactly matched
        (correct span + correct label) in predictions.
        """
        total_true = 0
        matched = 0

        for true_comps, pred_comps in zip(true_all, pred_all):
            if not true_comps:
                continue
            total_true += len(true_comps)
            true_set = set(true_comps)
            for comp in pred_comps:
                if comp in true_set:
                    matched += 1

        return matched / total_true if total_true > 0 else 0.0

    # ------------------------------------------------------------------
    # Run inference on multiple splits
    # ------------------------------------------------------------------

    def run_inference(
        self,
        splits: List[str] = ['test'],
        save_predictions: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Run inference on the specified splits.

        Args:
            splits: List of splits ('train', 'dev', 'test', 'ood')
            save_predictions: Whether to save per-sample predictions
            output_dir: Directory to write prediction/metric JSON files

        Returns:
            Dict mapping split name → metrics dict
        """
        all_results: Dict[str, Dict] = {}

        for split in splits:
            try:
                loader = self._get_dataloader(split)
            except FileNotFoundError:
                print(f"\n{split.upper()} dataset not found, skipping...")
                continue

            split_results = self.evaluate(loader, split, save_predictions)

            # Save to disk
            if save_predictions and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                detailed = split_results.pop('detailed_predictions', [])

                metrics_path = os.path.join(output_dir, f'{split}_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(split_results, f, indent=2)
                print(f"Metrics saved to: {metrics_path}")

                preds_path = os.path.join(output_dir, f'{split}_predictions.json')
                with open(preds_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed, f, indent=2, ensure_ascii=False)
                print(f"Predictions saved to: {preds_path}")

            all_results[split] = split_results

        return all_results


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Inference for Hierarchical Diffusion NeCTI'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to the YAML config used during training',
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to saved model checkpoint (e.g. saved_models/hierarchical_necti/best.pt)',
    )
    parser.add_argument(
        '--splits', type=str, nargs='+', default=['test'],
        choices=['train', 'dev', 'test', 'ood'],
        help='Splits to evaluate on',
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for inference',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on',
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Save per-sample predictions to JSON',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./inference_results/hierarchical',
        help='Directory to save prediction and metric files',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inference = HierarchicalInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )

    results = inference.run_inference(
        splits=args.splits,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("INFERENCE SUMMARY")
    print(f"{'=' * 60}\n")

    for split, metrics in results.items():
        print(f"{split.upper()}:")
        print(f"  Coarse Acc:  {metrics['coarse_acc']:.4f}")
        print(f"  Fine Acc:    {metrics['fine_acc']:.4f}")
        print(f"  USS (F1):    {metrics['USS']:.4f}  (P={metrics['USS_precision']:.4f}, R={metrics['USS_recall']:.4f})")
        print(f"  LSS (F1):    {metrics['LSS']:.4f}  (P={metrics['LSS_precision']:.4f}, R={metrics['LSS_recall']:.4f})")
        print(f"  Exact Match: {metrics['EM']:.4f}")
        print()


if __name__ == '__main__':
    main()
