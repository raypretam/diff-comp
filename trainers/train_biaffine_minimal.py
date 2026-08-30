"""
Training script for BiAffine-Enhanced BitDit (minimal change from baseline).

Based on trainer_necti.py with MINIMAL modifications:
    - Uses BiaffineBitDit instead of BitDit
    - Adds USS/LSS/EM evaluation alongside baseline binary P/R/F1
    - Logs biaffine gate value

Everything else is identical to the baseline trainer.
"""

import os
import sys
from argparse import Namespace
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from models.biaffine_bitdit import BiaffineBitDit
from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
from utils import get_lr_scheduler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class BiaffineMinimalTrainer:
    """
    Trainer that is nearly identical to NeCTITrainer from trainer_necti.py,
    but uses BiaffineBitDit and adds USS/LSS/EM metrics.
    """

    def __init__(self, args: Namespace):
        self.args = args

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Logger
        use_wandb = (getattr(args, 'logger', 'None') == 'wandb') and HAS_WANDB
        context_mode = "with_ctx" if getattr(args, 'use_context', False) else "no_ctx"
        if use_wandb:
            run_name = (
                f"necti-biaffine-minimal-{args.granularity}-{context_mode}"
                f"--rank_{getattr(args, 'biaffine_rank', 64)}"
                f"--gate_{getattr(args, 'biaffine_gate_init', -3.0)}"
            )
            wandb.init(project="DiffusionSL-NeCTI", name=run_name)
            wandb.config.update(vars(args))
            wandb.define_metric("f1", summary="max")
            wandb.define_metric("USS", summary="max")
            wandb.define_metric("LSS", summary="max")
        self.use_wandb = use_wandb

        # Label set
        self.label_set = NeCTILabelSet(
            data_path=args.data_path,
            granularity=args.granularity,
            use_context=getattr(args, 'use_context', False),
        )
        if args.num_classes != len(self.label_set):
            print(f"Adjusting num_classes {args.num_classes} → {len(self.label_set)}")
            args.num_classes = len(self.label_set)

        # Model — BiaffineBitDit (same as BitDit + biaffine conditioner)
        self.model = BiaffineBitDit(
            device=self.device,
            num_classes=args.num_classes,
            backbone=args.backbone,
            time_steps=args.time_steps,
            sampling_steps=args.sampling_steps,
            noise_schedule=args.noise_schedule,
            ddim_sampling_eta=args.ddim_sampling_eta,
            self_condition=args.self_condition,
            snr_scale=args.snr_scale,
            dataset=f"necti_{args.granularity}",
            dim_model=args.dim_model,
            dim_time=args.dim_time,
            objective=args.objective,
            loss_type=args.loss_type,
            add_lstm=args.add_lstm,
            freeze_bert=args.freeze_bert,
            max_length=args.max_length,
            depth=args.depth,
            num_labels=len(self.label_set),
            # BiAffine-specific
            biaffine_rank=getattr(args, 'biaffine_rank', 64),
            biaffine_num_heads=getattr(args, 'biaffine_num_heads', 8),
            biaffine_dropout=getattr(args, 'biaffine_dropout', 0.1),
            biaffine_gate_init=getattr(args, 'biaffine_gate_init', -3.0),
        )

        # Tokenizer and data
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        self.collate_fn = NeCTICollator(self.tokenizer, max_length=args.max_length)
        self.train_loader = self._get_dataloader('train', args.batch_size)
        self.dev_loader = self._get_dataloader('dev', args.batch_size)
        self.test_loader = self._get_dataloader('test', args.batch_size)

        # Optimizer — same as baseline
        bert_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    bert_params.append(param)
                else:
                    other_params.append(param)
        self.optimizer = AdamW(
            [
                {'params': bert_params, 'lr': float(args.lr_bert)},
                {'params': other_params, 'lr': float(args.lr_other)},
            ],
            weight_decay=float(args.weight_decay),
        )

        self.steps = int(args.max_steps)
        self.lr_scheduler = get_lr_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=self.steps,
            num_cycles=getattr(args, 'num_cycles', None),
        )

        # Print model info
        num_total = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        biaffine_params = sum(
            p.numel() for n, p in self.model.named_parameters() if 'biaffine' in n
        )
        print(f"\nTotal parameters: {num_total:,}")
        print(f"Trainable parameters: {num_trainable:,}")
        print(f"BiAffine parameters: {biaffine_params:,} ({biaffine_params/num_total*100:.2f}%)")

    def _get_dataloader(self, mode: str, bsz: int):
        dataset = NeCTIDataset(
            self.args.data_path, mode, self.label_set,
            use_context=getattr(self.args, 'use_context', False),
        )
        return DataLoader(
            dataset,
            batch_size=bsz,
            num_workers=self.args.num_workers,
            drop_last=False,
            shuffle=(mode == 'train'),
            collate_fn=self.collate_fn,
        )

    # =========================================================================
    # Training Loop — identical structure to trainer_necti.py
    # =========================================================================

    def train(self):
        print("\n" + "=" * 60)
        print("BiAffine-Enhanced BitDit Training (Minimal Change)")
        print("=" * 60 + "\n")

        best_f1 = 0.0
        best_uss = 0.0
        best_lss = 0.0
        patience_counter = 0
        patience = getattr(self.args, 'patience', 10)
        global_step = 0

        for epoch in range(1, self.args.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.max_epochs}")
            print("-" * 50)

            # --- Train ---
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            pbar = tqdm(self.train_loader, desc="Training")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_labels = batch['seq_labels'].to(self.device)

                loss = self.model(input_ids, attention_mask, seq_labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
                self.optimizer.step()
                self.lr_scheduler.step()

                train_loss += loss.item()
                train_steps += 1
                global_step += 1

                # Log gate value
                gate_val = self.model.biaffine.get_gate_value()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'gate': f'{gate_val:.4f}',
                })

                if self.use_wandb:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/biaffine_gate': gate_val,
                        'train/lr_bert': self.optimizer.param_groups[0]['lr'],
                        'train/lr_other': self.optimizer.param_groups[1]['lr'],
                        'global_step': global_step,
                    }
                    # Log per-component losses if available
                    if hasattr(self.model, 'last_losses'):
                        for k, v in self.model.last_losses.items():
                            log_dict[f'train/{k}'] = v
                    wandb.log(log_dict)

            avg_loss = train_loss / max(train_steps, 1)
            print(f"Avg training loss: {avg_loss:.4f}")

            # --- Evaluate ---
            print("\nEvaluating on dev set...")
            dev_results = self.evaluate(self.dev_loader, "dev")

            # Print results
            print(f"  Binary P/R/F1: {dev_results['precision']:.4f} / "
                  f"{dev_results['recall']:.4f} / {dev_results['f1']:.4f}")
            print(f"  USS: {dev_results['USS']:.4f}  |  LSS: {dev_results['LSS']:.4f}  |  "
                  f"EM: {dev_results['EM']:.4f}")
            print(f"  Gate: {dev_results['gate']:.4f}")

            if self.use_wandb:
                wandb.log({
                    'dev/f1': dev_results['f1'],
                    'dev/precision': dev_results['precision'],
                    'dev/recall': dev_results['recall'],
                    'dev/USS': dev_results['USS'],
                    'dev/LSS': dev_results['LSS'],
                    'dev/EM': dev_results['EM'],
                    'dev/gate': dev_results['gate'],
                    'epoch': epoch,
                })

            # Save best model (using F1 as primary metric, same as baseline)
            improved = False
            if dev_results['f1'] > best_f1:
                best_f1 = dev_results['f1']
                improved = True
            if dev_results['USS'] > best_uss:
                best_uss = dev_results['USS']
            if dev_results['LSS'] > best_lss:
                best_lss = dev_results['LSS']

            if improved:
                self._save_model(epoch, dev_results)
                print(f"  *** New best F1: {best_f1:.4f} — saved! ***")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # --- Final test evaluation ---
        print("\n" + "=" * 60)
        print("Training complete. Evaluating on test set...")
        print("=" * 60)
        self._load_best_model()

        test_results = self.evaluate(self.test_loader, "test")
        print(f"\nTest Results:")
        print(f"  Binary F1: {test_results['f1']:.4f}")
        print(f"  USS: {test_results['USS']:.4f}")
        print(f"  LSS: {test_results['LSS']:.4f}")
        print(f"  EM: {test_results['EM']:.4f}")
        print(f"\nBest Dev: F1={best_f1:.4f}, USS={best_uss:.4f}, LSS={best_lss:.4f}")

        if self.use_wandb:
            wandb.log({
                'test/f1': test_results['f1'],
                'test/USS': test_results['USS'],
                'test/LSS': test_results['LSS'],
                'test/EM': test_results['EM'],
            })

    # =========================================================================
    # Evaluation — baseline binary P/R/F1 + USS/LSS/EM
    # =========================================================================

    @torch.no_grad()
    def evaluate(self, dataloader, split_name="dev") -> Dict[str, float]:
        self.model.eval()

        all_predictions = []
        all_labels = []
        true_compounds_all = []
        pred_compounds_all = []

        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels = batch['seq_labels'].to(self.device)

            predictions, _ = self.model(input_ids, attention_mask, seq_labels)

            mask = (seq_labels != -100)

            for i in range(input_ids.shape[0]):
                sent_mask = mask[i]
                if sent_mask.sum() == 0:
                    continue

                pred_ids = predictions[i][sent_mask].cpu().tolist()
                true_ids = seq_labels[i][sent_mask].cpu().tolist()

                all_predictions.extend(pred_ids)
                all_labels.extend(true_ids)

                # Convert to label names for compound extraction
                pred_names = [
                    self.label_set.id2label(int(pid))
                    if 0 <= int(pid) < len(self.label_set) else "No_rel"
                    for pid in pred_ids
                ]
                true_names = [
                    self.label_set.id2label(int(tid))
                    if 0 <= int(tid) < len(self.label_set) else "No_rel"
                    for tid in true_ids
                ]

                pred_compounds = self._extract_compounds_from_labels(pred_names)
                true_compounds = self._extract_compounds_from_labels(true_names)
                pred_compounds_all.append(pred_compounds)
                true_compounds_all.append(true_compounds)

        # Binary P/R/F1 (same as baseline trainer_necti.py)
        binary_metrics = self._calculate_binary_metrics(all_predictions, all_labels)

        # USS/LSS/EM
        uss = self._compute_span_f1(true_compounds_all, pred_compounds_all, labeled=False)
        lss = self._compute_span_f1(true_compounds_all, pred_compounds_all, labeled=True)
        em = self._compute_exact_match(true_compounds_all, pred_compounds_all)

        return {
            **binary_metrics,
            'USS': uss,
            'LSS': lss,
            'EM': em,
            'gate': self.model.biaffine.get_gate_value(),
        }

    def _calculate_binary_metrics(self, predictions, labels):
        """Binary P/R/F1 — identical to trainer_necti.py"""
        predictions = np.array(predictions)
        labels = np.array(labels)

        no_rel_id = self.label_set.label2id('No_rel')
        root_id = self.label_set.label2id('root')

        pred_is_compound = (predictions != no_rel_id) & (predictions != root_id)
        label_is_compound = (labels != no_rel_id) & (labels != root_id)

        tp = np.sum(pred_is_compound & label_is_compound)
        fp = np.sum(pred_is_compound & ~label_is_compound)
        fn = np.sum(~pred_is_compound & label_is_compound)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'precision': precision, 'recall': recall, 'f1': f1}

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
    # Checkpointing — same as baseline
    # =========================================================================

    def _save_model(self, epoch, results):
        context_mode = "with_ctx" if getattr(self.args, 'use_context', False) else "no_ctx"
        save_dir = os.path.join(
            os.getcwd(), 'saved_models',
            f'necti_biaffine_minimal_{self.args.granularity}_{context_mode}',
        )
        os.makedirs(save_dir, exist_ok=True)

        import glob
        for old_ckpt in glob.glob(os.path.join(save_dir, '*.pt')):
            os.remove(old_ckpt)

        save_path = os.path.join(save_dir, 'best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'results': results,
            'args': self.args,
        }, save_path)

    def _load_best_model(self):
        context_mode = "with_ctx" if getattr(self.args, 'use_context', False) else "no_ctx"
        save_dir = os.path.join(
            os.getcwd(), 'saved_models',
            f'necti_biaffine_minimal_{self.args.granularity}_{context_mode}',
        )
        model_path = os.path.join(save_dir, 'best_model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            print("No saved model found, using current model")


def main():
    """Load config and start training."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str,
        default='configs/biaffine_minimal_necti.yaml',
        help='Path to config YAML',
    )
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    cli_args = parser.parse_args()

    # Load YAML config
    with open(cli_args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Build Namespace (same approach as options.py)
    args = Namespace(**cfg)

    # CLI overrides
    if cli_args.use_context:
        args.use_context = True
    if cli_args.no_wandb:
        args.logger = 'None'

    # Defaults for any missing keys
    for key, default in [
        ('use_context', False),
        ('num_workers', 4),
        ('patience', 10),
        ('min_delta', 0.0001),
        ('biaffine_rank', 64),
        ('biaffine_num_heads', 8),
        ('biaffine_dropout', 0.1),
        ('biaffine_gate_init', -3.0),
    ]:
        if not hasattr(args, key):
            setattr(args, key, default)

    trainer = BiaffineMinimalTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
