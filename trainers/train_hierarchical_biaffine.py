"""
Training Script for Hierarchical Biaffine Diffusion NeCTI
==========================================================

Uses the HierarchicalBiaffineDiffusion model which adds:
1. Biaffine pairwise context enriching BERT features for Stage 2
2. Biaffine attention bias in every DiT block for fine-grained prediction
3. Coarse label conditioning from Stage 1 (preserved from original)

The biaffine module addresses the USS→LSS gap by providing O(1) pairwise
signal for fine-grained label discrimination (vs O(1/n) from global attention).

Usage:
    python train_hierarchical_biaffine.py --config configs/hierarchical_biaffine_necti.yaml --stage both
    python train_hierarchical_biaffine.py --config configs/hierarchical_biaffine_necti.yaml --stage both --use_constrained_eval
"""

import os
import sys
import argparse
import yaml
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Import modules
from models.hierarchical_biaffine_diffusion import (
    HierarchicalBiaffineDiffusion,
    create_hierarchical_biaffine_model,
)
from data.ner.necti_dataset import NeCTIDataset, NeCTILabelSet, NeCTICollator
from data.ner.necti_dataset_hierarchial import get_coarse_id_from_fine_label, COARSE_CATEGORIES


class HierarchicalBiaffineTrainer:
    """
    Trainer for Hierarchical Biaffine Diffusion NeCTI.
    
    Key difference from HierarchicalTrainer: uses biaffine-conditioned Stage 2
    and has a separate param group for the biaffine conditioner.
    """

    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config['gpu']['use_gpu'] else 'cpu'
        )
        self.use_constrained_eval = args.use_constrained_eval

        # Setup output
        self.output_dir = Path(config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logging()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_lss = 0.0

        # Mixed precision
        if config['gpu'].get('precision', 32) == 16:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        print(f"\n{'='*60}")
        print("HIERARCHICAL BIAFFINE DIFFUSION TRAINER")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Constrained Evaluation: {self.use_constrained_eval}")
        print(f"  Mixed Precision: {self.scaler is not None}")
        print(f"{'='*60}\n")

    # =========================================================================
    # Data Setup
    # =========================================================================

    def _setup_data(self):
        data_config = self.config['data']

        self.coarse_label_set = NeCTILabelSet(
            data_path=data_config['data_path'],
            granularity='Coarse',
            use_context=data_config['use_context'],
        )
        self.fine_label_set = NeCTILabelSet(
            data_path=data_config['data_path'],
            granularity='Finegrain',
            use_context=data_config['use_context'],
        )

        self.fine_to_coarse_map = self._build_label_hierarchy()
        self.coarse_to_fine_map = self._build_reverse_hierarchy()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['backbone'])

        self.train_dataset = NeCTIDataset(
            data_path=data_config['data_path'],
            mode='train',
            label_set=self.fine_label_set,
            use_context=data_config['use_context'],
        )
        self.dev_dataset = NeCTIDataset(
            data_path=data_config['data_path'],
            mode='dev',
            label_set=self.fine_label_set,
            use_context=data_config['use_context'],
        )

        collator = NeCTICollator(
            tokenizer=self.tokenizer,
            max_length=data_config['max_length'],
        )

        train_config = self.config['training']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            collate_fn=collator,
            pin_memory=True,
        )
        self.dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            collate_fn=collator,
            pin_memory=True,
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Dev samples: {len(self.dev_dataset)}")
        print(f"Coarse classes: {len(self.coarse_label_set)}")
        print(f"Fine classes: {len(self.fine_label_set)}")

    def _build_label_hierarchy(self) -> Dict[int, int]:
        fine_to_coarse = {}
        for fine_id in range(len(self.fine_label_set)):
            fine_name = self.fine_label_set.id2label(fine_id)
            coarse_id = get_coarse_id_from_fine_label(fine_name)
            fine_to_coarse[fine_id] = coarse_id
        print(f"\nBuilt fine-to-coarse mapping for {len(fine_to_coarse)} labels")
        return fine_to_coarse

    def _build_reverse_hierarchy(self) -> Dict[int, List[int]]:
        coarse_to_fine: Dict[int, List[int]] = {}
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            coarse_to_fine.setdefault(coarse_id, []).append(fine_id)
        print("\nCoarse-to-Fine mapping:")
        for cid, fids in sorted(coarse_to_fine.items()):
            name = COARSE_CATEGORIES[cid] if cid < len(COARSE_CATEGORIES) else f"ID:{cid}"
            print(f"  {name}: {len(fids)} fine labels")
        return coarse_to_fine

    # =========================================================================
    # Model Setup
    # =========================================================================

    def _setup_model(self):
        cfg = self.config
        biaffine_cfg = cfg.get('biaffine', {})

        self.model = create_hierarchical_biaffine_model(
            device=str(self.device),
            backbone=cfg['backbone'],
            dim_model=cfg['dim_model'],
            freeze_bert=cfg['freeze_bert'],
            time_steps=cfg['time_steps'],
            sampling_steps=cfg['sampling_steps'],
            noise_schedule=cfg['noise_schedule'],
            snr_scale=cfg['snr_scale'],
            global_depth=cfg['stage1']['depth'],
            global_num_heads=cfg['stage1']['num_heads'],
            num_coarse_classes=cfg['stage1']['num_classes'],
            fine_depth=cfg['stage2']['depth'],
            fine_num_heads=cfg['stage2']['num_heads'],
            num_fine_classes=cfg['stage2']['num_classes'],
            biaffine_dropout=biaffine_cfg.get('dropout', 0.1),
            fine_to_coarse_map=self.fine_to_coarse_map,
            objective=cfg['objective'],
            loss_type=cfg['loss_type'],
        )

        if self.args.resume:
            print(f"Resuming from: {self.args.resume}")
            ckpt = torch.load(self.args.resume, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        biaffine_params = sum(p.numel() for p in self.model.biaffine_conditioner.parameters())
        print(f"\nTotal parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Biaffine conditioner parameters: {biaffine_params:,}")

    # =========================================================================
    # Optimizer Setup
    # =========================================================================

    def _setup_optimizer(self):
        train_config = self.config['training']

        # 4 parameter groups: backbone, stage1, biaffine, stage2+coarse_embed
        param_groups = [
            {
                'params': list(self.model.backbone.parameters()),
                'lr': float(train_config['lr_bert']),
                'name': 'backbone',
            },
            {
                'params': list(self.model.global_dit.parameters()),
                'lr': float(train_config['lr_stage1']),
                'name': 'stage1',
            },
            {
                'params': list(self.model.biaffine_conditioner.parameters()),
                'lr': float(train_config.get('lr_biaffine', train_config['lr_stage2'])),
                'name': 'biaffine',
            },
            {
                'params': (
                    list(self.model.fine_dit.parameters())
                    + list(self.model.coarse_embed.parameters())
                ),
                'lr': float(train_config['lr_stage2']),
                'name': 'stage2',
            },
        ]
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]

        self.optimizer = AdamW(param_groups, weight_decay=train_config['weight_decay'])

        total_steps = len(self.train_loader) * train_config['max_epochs']
        warmup_steps = train_config.get('warmup_steps', 500)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    # =========================================================================
    # Logging
    # =========================================================================

    def _setup_logging(self):
        log_config = self.config.get('logging', {})
        self.logger = log_config.get('logger', 'None')

        if self.logger == 'wandb':
            import wandb
            wandb.init(
                project=log_config.get('project_name', 'hierarchical-biaffine-necti'),
                name=log_config.get(
                    'experiment_name',
                    f'biaffine_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                ),
                config=self.config,
            )
            self.wandb = wandb
        elif self.logger == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.output_dir / 'tensorboard')

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        coarse_labels = fine_labels.clone()
        valid_mask = fine_labels != -100
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            mask = (fine_labels == fine_id) & valid_mask
            coarse_labels[mask] = coarse_id
        return coarse_labels

    # =========================================================================
    # Training
    # =========================================================================

    def train_epoch(self, stage: str = 'both') -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        stage1_loss_sum = 0.0
        stage2_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['seq_labels'].to(self.device)
            coarse_labels = self._get_coarse_labels(fine_labels)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    losses = self.model(
                        input_ids, attention_mask, coarse_labels, fine_labels,
                        stage=stage, epoch=self.current_epoch,
                    )
                    loss = losses['loss'] if isinstance(losses, dict) else losses

                if torch.isnan(loss):
                    print(f"Warning: NaN loss at step {self.global_step}, skipping")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['training']['max_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model(
                    input_ids, attention_mask, coarse_labels, fine_labels,
                    stage=stage, epoch=self.current_epoch,
                )
                loss = losses['loss'] if isinstance(losses, dict) else losses

                if torch.isnan(loss):
                    print(f"Warning: NaN loss at step {self.global_step}, skipping")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['training']['max_grad_norm']
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            if isinstance(losses, dict):
                stage1_loss_sum += losses['stage1_loss'].item()
                stage2_loss_sum += losses['stage2_loss'].item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })

            if self.logger == 'wandb' and self.global_step % 50 == 0:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                }
                if isinstance(losses, dict):
                    log_dict['train/stage1_loss'] = losses['stage1_loss'].item()
                    log_dict['train/stage2_loss'] = losses['stage2_loss'].item()
                self.wandb.log(log_dict, step=self.global_step)

        return {
            'loss': total_loss / max(num_batches, 1),
            'stage1_loss': stage1_loss_sum / max(num_batches, 1),
            'stage2_loss': stage2_loss_sum / max(num_batches, 1),
        }

    # =========================================================================
    # Evaluation
    # =========================================================================

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()

        all_coarse_preds = []
        all_coarse_labels = []
        all_fine_preds = []
        all_fine_labels = []
        true_compounds_all = []
        pred_compounds_all = []

        for batch in tqdm(self.dev_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['seq_labels'].to(self.device)
            coarse_labels = self._get_coarse_labels(fine_labels)

            coarse_preds, fine_preds = self.model.inference(
                input_ids, attention_mask,
                apply_constraint=self.use_constrained_eval,
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

        coarse_acc = (
            np.mean(np.array(all_coarse_preds) == np.array(all_coarse_labels))
            if all_coarse_labels else 0.0
        )
        fine_acc = (
            np.mean(np.array(all_fine_preds) == np.array(all_fine_labels))
            if all_fine_labels else 0.0
        )
        uss = self._compute_span_f1(true_compounds_all, pred_compounds_all, labeled=False)
        lss = self._compute_span_f1(true_compounds_all, pred_compounds_all, labeled=True)
        em = self._compute_exact_match(true_compounds_all, pred_compounds_all)

        return {
            'coarse_acc': coarse_acc,
            'fine_acc': fine_acc,
            'USS': uss,
            'LSS': lss,
            'EM': em,
        }

    # =========================================================================
    # Span / Compound Extraction & Metrics
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
    # Checkpointing
    # =========================================================================

    def save_checkpoint(self, is_best: bool = False):
        if not is_best:
            return
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'best_lss': self.best_lss,
            'config': self.config,
        }
        for old in glob.glob(str(self.output_dir / '*.pt')):
            os.remove(old)
        torch.save(checkpoint, self.output_dir / 'best.pt')
        print(f"  ✓ Saved best model - USS: {self.best_metric:.4f}, LSS: {self.best_lss:.4f}")

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def train(self):
        train_config = self.config['training']
        stage = self.args.stage or train_config.get('strategy', 'both')
        if stage == 'joint':
            stage = 'both'

        print(f"\n{'='*60}")
        print("TRAINING — HIERARCHICAL BIAFFINE DIFFUSION")
        print(f"{'='*60}")
        print(f"  Strategy: {stage}")
        print(f"  Max epochs: {train_config['max_epochs']}")
        print(f"  Patience: {train_config.get('patience', 15)}")
        print(f"{'='*60}\n")

        patience_counter = 0

        for epoch in range(train_config['max_epochs']):
            self.current_epoch = epoch

            if stage == 'sequential':
                stage1_epochs = train_config.get('stage1_epochs', 30)
                if epoch < stage1_epochs:
                    current_stage = 'stage1'
                    print(f"\n[Sequential] Training Stage 1 (epoch {epoch}/{stage1_epochs})")
                else:
                    current_stage = 'stage2'
                    print(f"\n[Sequential] Training Stage 2 (epoch {epoch - stage1_epochs})")
                    if train_config.get('freeze_stage1_for_stage2', False) and epoch == stage1_epochs:
                        print("  Freezing Stage 1 parameters")
                        for param in self.model.global_dit.parameters():
                            param.requires_grad = False
            else:
                current_stage = stage

            # Train
            train_metrics = self.train_epoch(current_stage)
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            if current_stage == 'both':
                print(f"  Stage 1 Loss: {train_metrics['stage1_loss']:.4f}")
                print(f"  Stage 2 Loss: {train_metrics['stage2_loss']:.4f}")

            # Evaluate
            eval_metrics = self.evaluate()
            print(f"Epoch {epoch} - Coarse Acc: {eval_metrics['coarse_acc']:.4f}, "
                  f"Fine Acc: {eval_metrics['fine_acc']:.4f}")
            print(f"Epoch {epoch} - USS: {eval_metrics['USS']:.4f}, "
                  f"LSS: {eval_metrics['LSS']:.4f}, EM: {eval_metrics['EM']:.4f}")

            if self.logger == 'wandb':
                self.wandb.log(
                    {'epoch': epoch, **{f'val/{k}': v for k, v in eval_metrics.items()}},
                    step=self.global_step,
                )

            # Check improvement (monitor LSS as primary — the metric we want to fix)
            current_lss = eval_metrics['LSS']
            current_uss = eval_metrics['USS']
            is_best = current_lss > self.best_lss

            if is_best:
                self.best_lss = current_lss
                self.best_metric = current_uss
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(is_best=is_best)

            if patience_counter >= train_config.get('patience', 15):
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Best USS: {self.best_metric:.4f}")
        print(f"  Best LSS: {self.best_lss:.4f}")
        print(f"{'='*60}\n")

        if self.logger == 'wandb':
            self.wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training for Hierarchical Biaffine Diffusion NeCTI'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument(
        '--stage', type=str, default=None,
        choices=['stage1', 'stage2', 'both', 'sequential'],
        help='Training stage (default: both)',
    )
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--use_constrained_eval', action='store_true',
        help='Use constrained decoding during evaluation',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = HierarchicalBiaffineTrainer(config, args)
    trainer.train()


if __name__ == '__main__':
    main()
