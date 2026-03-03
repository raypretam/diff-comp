"""
Improved Inference for Hierarchical Diffusion NeCTI
====================================================

Key improvements:
1. Constrained Decoding - Fine labels constrained to coarse category
2. MBR Decoding - Multiple samples, select most consistent
3. Span-Level Post-Processing - Ensure consistency within compound spans

Usage:
    # Standard inference with constrained decoding
    python inference_improved.py --config configs/hierarchial_necti.yaml \
        --checkpoint saved_models/hierarchical_necti/best.pt \
        --use_constrained

    # With MBR decoding (slower but more robust)
    python inference_improved.py --config configs/hierarchial_necti.yaml \
        --checkpoint saved_models/hierarchical_necti/best.pt \
        --use_constrained --use_mbr --mbr_samples 8

    # With span-level post-processing
    python inference_improved.py --config configs/hierarchial_necti.yaml \
        --checkpoint saved_models/hierarchical_necti/best.pt \
        --use_constrained --use_span_voting
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


class ImprovedHierarchicalInference:
    """
    Improved inference with constrained decoding, MBR, and span voting.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda',
        batch_size: int = 8,
    ):
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
        self.coarse_to_fine_map = self._build_reverse_hierarchy()

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

    def _build_label_hierarchy(self) -> Dict[int, int]:
        """Build fine-label-id → coarse-label-id mapping."""
        fine_to_coarse = {}
        for fine_id in range(len(self.fine_label_set)):
            fine_name = self.fine_label_set.id2label(fine_id)
            coarse_id = get_coarse_id_from_fine_label(fine_name)
            fine_to_coarse[fine_id] = coarse_id
        return fine_to_coarse

    def _build_reverse_hierarchy(self) -> Dict[int, List[int]]:
        """Build coarse-label-id → list of fine-label-ids mapping."""
        coarse_to_fine = {}
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            if coarse_id not in coarse_to_fine:
                coarse_to_fine[coarse_id] = []
            coarse_to_fine[coarse_id].append(fine_id)
        
        print("\nCoarse-to-Fine mapping:")
        for coarse_id, fine_ids in coarse_to_fine.items():
            coarse_name = COARSE_CATEGORIES[coarse_id] if coarse_id < len(COARSE_CATEGORIES) else f"ID:{coarse_id}"
            print(f"  {coarse_name}: {len(fine_ids)} fine labels")
        
        return coarse_to_fine

    def _setup_model(self):
        """Instantiate the hierarchical diffusion model."""
        cfg = self.config
        self.model = create_hierarchical_model(
            device=str(self.device),
            backbone=cfg['backbone'],
            dim_model=cfg['dim_model'],
            freeze_bert=True,
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

    # =========================================================================
    # CONSTRAINED DECODING
    # =========================================================================
    
    @torch.no_grad()
    def constrained_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-stage inference with fine labels constrained to coarse category.
        """
        # Stage 1: Coarse predictions
        features = self.model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        coarse_preds = self.model.sample_stage1(features, attention_mask)
        
        # Stage 2: Fine predictions (standard)
        fine_preds = self.model.sample_stage2(features, coarse_preds, attention_mask)
        
        # Apply hard constraint: replace invalid fine predictions
        fine_preds = self._apply_coarse_constraint(fine_preds, coarse_preds)
        
        return coarse_preds, fine_preds
    
    def _apply_coarse_constraint(
        self,
        fine_preds: torch.Tensor,
        coarse_preds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Constrain fine predictions to be valid within predicted coarse category.
        If a fine prediction doesn't match its coarse, replace with most common
        valid fine label for that coarse.
        """
        batch_size, seq_len = fine_preds.shape
        constrained = fine_preds.clone()
        
        for b in range(batch_size):
            for s in range(seq_len):
                coarse_id = coarse_preds[b, s].item()
                fine_id = fine_preds[b, s].item()
                
                # Check if fine is valid for coarse
                if coarse_id in self.coarse_to_fine_map:
                    valid_fine = self.coarse_to_fine_map[coarse_id]
                    if fine_id not in valid_fine and len(valid_fine) > 0:
                        # Replace with first valid fine label
                        # (Could be improved to use frequency-based selection)
                        constrained[b, s] = valid_fine[0]
        
        return constrained

    # =========================================================================
    # MBR DECODING
    # =========================================================================
    
    @torch.no_grad()
    def mbr_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_samples: int = 8,
        use_constrained: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MBR decoding: generate multiple samples, select most consistent.
        """
        features = self.model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Stage 1: Single coarse prediction
        coarse_preds = self.model.sample_stage1(features, attention_mask)
        
        # Stage 2: Multiple fine samples
        samples = []
        for _ in range(num_samples):
            fine_pred = self.model.sample_stage2(features, coarse_preds, attention_mask)
            if use_constrained:
                fine_pred = self._apply_coarse_constraint(fine_pred, coarse_preds)
            samples.append(fine_pred)
        
        # Compute agreement scores
        agreement_scores = torch.zeros(num_samples, device=self.device)
        for i in range(num_samples):
            for j in range(num_samples):
                if i != j:
                    agreement_scores[i] += (samples[i] == samples[j]).float().sum()
        
        # Select best
        best_idx = agreement_scores.argmax().item()
        
        return coarse_preds, samples[best_idx]

    # =========================================================================
    # SPAN-LEVEL VOTING
    # =========================================================================
    
    def apply_span_voting(
        self,
        fine_preds: List[int],
        compound_mask: List[int],
    ) -> List[int]:
        """
        Apply majority voting within each compound span.
        All tokens in a span get the same (majority) label.
        """
        result = fine_preds.copy()
        n = len(fine_preds)
        
        i = 0
        while i < n:
            if compound_mask[i] == 1:
                # Start of compound span
                start = i
                span_labels = []
                
                while i < n and compound_mask[i] == 1:
                    span_labels.append(fine_preds[i])
                    i += 1
                
                end = i  # exclusive
                
                # Majority vote (excluding special labels)
                non_special = [l for l in span_labels if l not in [-100]]
                if non_special:
                    majority = Counter(non_special).most_common(1)[0][0]
                    # Apply to all positions in span
                    for j in range(start, end):
                        if result[j] != -100:
                            result[j] = majority
            else:
                i += 1
        
        return result

    # =========================================================================
    # MAIN EVALUATION
    # =========================================================================

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        split_name: str = "test",
        use_constrained: bool = True,
        use_mbr: bool = False,
        mbr_samples: int = 8,
        use_span_voting: bool = False,
        save_predictions: bool = False,
    ) -> Dict:
        """
        Evaluate with selected improvements.
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {split_name} set...")
        print(f"  Constrained decoding: {use_constrained}")
        print(f"  MBR decoding: {use_mbr} (samples={mbr_samples})")
        print(f"  Span voting: {use_span_voting}")
        print(f"{'=' * 60}\n")

        self.model.eval()

        all_coarse_preds = []
        all_coarse_labels = []
        all_fine_preds = []
        all_fine_labels = []
        true_compounds_all = []
        pred_compounds_all = []
        detailed_predictions = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['seq_labels'].to(self.device)
            coarse_labels = self._get_coarse_labels(fine_labels)
            
            # Get compound masks if available
            compound_masks = batch.get('seq_span_labels', None)

            # Select inference method
            if use_mbr:
                coarse_preds, fine_preds = self.mbr_inference(
                    input_ids, attention_mask, mbr_samples, use_constrained
                )
            elif use_constrained:
                coarse_preds, fine_preds = self.constrained_inference(input_ids, attention_mask)
            else:
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
                
                # Apply span voting if enabled
                if use_span_voting and compound_masks is not None:
                    comp_mask = compound_masks[i][sent_mask].cpu().tolist()
                    pred_ids = self.apply_span_voting(pred_ids, comp_mask)

                all_fine_preds.extend(pred_ids)
                all_fine_labels.extend(true_ids)
                all_coarse_preds.extend(coarse_pred_ids)
                all_coarse_labels.extend(coarse_true_ids)

                # Convert to names
                pred_names = [
                    self.fine_label_set.id2label(pid) if 0 <= pid < len(self.fine_label_set) else "No_rel"
                    for pid in pred_ids
                ]
                true_names = [
                    self.fine_label_set.id2label(tid) if 0 <= tid < len(self.fine_label_set) else "No_rel"
                    for tid in true_ids
                ]

                pred_compounds = self._extract_compounds_from_labels(pred_names)
                true_compounds = self._extract_compounds_from_labels(true_names)

                pred_compounds_all.append(pred_compounds)
                true_compounds_all.append(true_compounds)

                if save_predictions:
                    detailed_predictions.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'fine_predictions': pred_names,
                        'fine_true_labels': true_names,
                        'pred_compounds': [{'start': s, 'end': e, 'label': l} for s, e, l in pred_compounds],
                        'true_compounds': [{'start': s, 'end': e, 'label': l} for s, e, l in true_compounds],
                    })

        # Compute metrics
        coarse_acc = np.mean(np.array(all_coarse_preds) == np.array(all_coarse_labels)) if all_coarse_labels else 0.0
        fine_acc = np.mean(np.array(all_fine_preds) == np.array(all_fine_labels)) if all_fine_labels else 0.0

        uss_p, uss_r, uss_f1 = self._compute_span_prf(true_compounds_all, pred_compounds_all, labeled=False)
        lss_p, lss_r, lss_f1 = self._compute_span_prf(true_compounds_all, pred_compounds_all, labeled=True)
        em = self._compute_exact_match(true_compounds_all, pred_compounds_all)

        # Print results
        print(f"\n{split_name.upper()} Results:")
        print("-" * 50)
        print(f"  Coarse Acc:     {coarse_acc:.4f}")
        print(f"  Fine Acc:       {fine_acc:.4f}")
        print(f"  USS F1:         {uss_f1:.4f}  (P={uss_p:.4f}, R={uss_r:.4f})")
        print(f"  LSS F1:         {lss_f1:.4f}  (P={lss_p:.4f}, R={lss_r:.4f})")
        print(f"  Exact Match:    {em:.4f}")
        print("-" * 50)

        results = {
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

    # =========================================================================
    # Helper methods (same as original)
    # =========================================================================

    def _extract_compounds_from_labels(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        """Extract compound spans from label sequence."""
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

    def _compute_span_prf(self, true_all, pred_all, labeled=False):
        """Compute P/R/F1 for spans."""
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
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def _compute_exact_match(self, true_all, pred_all):
        """Compute exact match rate."""
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

    def run_inference(self, splits, use_constrained, use_mbr, mbr_samples, 
                      use_span_voting, save_predictions, output_dir):
        """Run inference on multiple splits."""
        all_results = {}

        for split in splits:
            try:
                loader = self._get_dataloader(split)
            except FileNotFoundError:
                print(f"{split} dataset not found, skipping...")
                continue

            results = self.evaluate(
                loader, split,
                use_constrained=use_constrained,
                use_mbr=use_mbr,
                mbr_samples=mbr_samples,
                use_span_voting=use_span_voting,
                save_predictions=save_predictions,
            )

            if save_predictions and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                detailed = results.pop('detailed_predictions', [])

                with open(os.path.join(output_dir, f'{split}_metrics.json'), 'w') as f:
                    json.dump(results, f, indent=2)

                with open(os.path.join(output_dir, f'{split}_predictions.json'), 'w', encoding='utf-8') as f:
                    json.dump(detailed, f, indent=2, ensure_ascii=False)

            all_results[split] = results

        return all_results


def parse_args():
    parser = argparse.ArgumentParser(description='Improved Inference for Hierarchical Diffusion NeCTI')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--splits', type=str, nargs='+', default=['test'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Improvement flags
    parser.add_argument('--use_constrained', action='store_true',
                        help='Use constrained decoding (fine within coarse)')
    parser.add_argument('--use_mbr', action='store_true',
                        help='Use MBR decoding (multiple samples)')
    parser.add_argument('--mbr_samples', type=int, default=8,
                        help='Number of samples for MBR')
    parser.add_argument('--use_span_voting', action='store_true',
                        help='Apply majority voting within compound spans')
    
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./inference_results/improved')
    
    return parser.parse_args()


def main():
    args = parse_args()

    inference = ImprovedHierarchicalInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )

    results = inference.run_inference(
        splits=args.splits,
        use_constrained=args.use_constrained,
        use_mbr=args.use_mbr,
        mbr_samples=args.mbr_samples,
        use_span_voting=args.use_span_voting,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for split, metrics in results.items():
        print(f"\n{split.upper()}:")
        print(f"  USS: {metrics['USS']:.4f}  LSS: {metrics['LSS']:.4f}  EM: {metrics['EM']:.4f}")


if __name__ == '__main__':
    main()