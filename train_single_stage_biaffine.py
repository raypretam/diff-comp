"""
Training Script for Single-Stage Bi-Affine Diffusion NeCTI
===========================================================

This trains the FIXED single-stage model (NO cascade, NO coarse conditioning).

Key difference from HierarchicalBiaffineTrainer:
- Uses SingleStageBiAffineDiffusion (not HierarchicalBiaffineDiffusion)
- NO Stage 1 / Stage 2 - single diffusion process
- NO coarse conditioning at inference
- Auxiliary coarse loss for regularization only

Expected results:
    Baseline:     USS=98.71, LSS=86.11, EM=68.33
    This model:   USS~98-99, LSS~87-89, EM~70-74

Usage:
    python train_single_stage_biaffine.py --config configs/single_stage_biaffine.yaml
    python train_single_stage_biaffine.py --config configs/single_stage_biaffine.yaml --use_constrained_eval
"""

import os
import sys
import argparse
import yaml
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Import the FIXED single-stage model
from models.single_stage_biaffine_diffusion import (
    SingleStageBiAffineDiffusion,
    create_single_stage_biaffine_model,
)

# Import your existing data utilities
from data.ner.necti_dataset import NeCTIDataset, NeCTILabelSet, NeCTICollator
from data.ner.necti_dataset_hierarchial import get_coarse_id_from_fine_label, COARSE_CATEGORIES


class SingleStageBiAffineTrainer:
    """
    Trainer for Single-Stage Bi-Affine Diffusion NeCTI.
    
    Key differences from HierarchicalBiaffineTrainer:
    - Single-stage diffusion (no Stage 1 / Stage 2)
    - No coarse conditioning at inference
    - No error cascade
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
        self.best_uss = 0.0
        self.best_lss = 0.0

        # Mixed precision
        if config['gpu'].get('precision', 32) == 16:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        print(f"\n{'='*60}")
        print("SINGLE-STAGE BI-AFFINE DIFFUSION TRAINER")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Constrained Evaluation: {self.use_constrained_eval}")
        print(f"  Mixed Precision: {self.scaler is not None}")
        print(f"  NO cascade - single-stage diffusion")
        print(f"  NO coarse conditioning at inference")
        print(f"{'='*60}\n")

    # =========================================================================
    # Data Setup (same as your existing code)
    # =========================================================================

    def _setup_data(self):
        data_config = self.config['data']

        self.fine_label_set = NeCTILabelSet(
            data_path=data_config['data_path'],
            granularity='Finegrain',
            use_context=data_config['use_context'],
        )

        # Build fine-to-coarse mapping
        self.fine_to_coarse_map = {}
        for fine_id in range(len(self.fine_label_set)):
            fine_name = self.fine_label_set.id2label(fine_id)
            coarse_id = get_coarse_id_from_fine_label(fine_name)
            self.fine_to_coarse_map[fine_id] = coarse_id
        
        print(f"\nBuilt fine-to-coarse mapping for {len(self.fine_to_coarse_map)} labels")

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
        print(f"Fine classes: {len(self.fine_label_set)}")
        print(f"Coarse classes: {len(COARSE_CATEGORIES)}")

    # =========================================================================
    # Model Setup - Uses SingleStageBiAffineDiffusion
    # =========================================================================

    def _setup_model(self):
        cfg = self.config
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

        if self.args.resume:
            print(f"Resuming from: {self.args.resume}")
            ckpt = torch.load(self.args.resume, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        biaffine_params = sum(p.numel() for p in self.model.biaffine_encoder.parameters())
        
        print(f"\nTotal parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Bi-affine encoder parameters: {biaffine_params:,}")

    # =========================================================================
    # Optimizer Setup
    # =========================================================================

    def _setup_optimizer(self):
        train_config = self.config['training']

        # Two parameter groups: backbone and everything else
        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() 
                       if 'backbone' not in n and p.requires_grad]

        param_groups = [
            {
                'params': backbone_params,
                'lr': float(train_config['lr_bert']),
                'name': 'backbone',
            },
            {
                'params': other_params,
                'lr': float(train_config.get('lr_other', train_config.get('lr_dit', 1e-4))),
                'name': 'dit_and_biaffine',
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
                project=log_config.get('project_name', 'single-stage-biaffine-necti'),
                name=log_config.get(
                    'experiment_name',
                    f'single_stage_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                ),
                config=self.config,
            )
            self.wandb = wandb
        elif self.logger == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.output_dir / 'tensorboard')

    # =========================================================================
    # Training - SIMPLIFIED (no Stage 1 / Stage 2)
    # =========================================================================

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch - single stage only."""
        self.model.train()

        total_loss = 0.0
        diff_loss_sum = 0.0
        coarse_loss_sum = 0.0
        coarse_acc_sum = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['seq_labels'].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    # Single forward call - no stage parameter needed!
                    out = self.model(input_ids, attention_mask, fine_labels)
                    loss = out['loss']

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
                out = self.model(input_ids, attention_mask, fine_labels)
                loss = out['loss']

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
            diff_loss_sum += out['diffusion_loss'].item()
            coarse_loss_sum += out['coarse_loss'].item()
            coarse_acc_sum += out['coarse_acc'].item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'diff': f'{out["diffusion_loss"].item():.4f}',
                'c_acc': f'{out["coarse_acc"].item():.3f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })

            if self.logger == 'wandb' and self.global_step % 50 == 0:
                self.wandb.log({
                    'train/loss': loss.item(),
                    'train/diffusion_loss': out['diffusion_loss'].item(),
                    'train/coarse_loss': out['coarse_loss'].item(),
                    'train/coarse_acc': out['coarse_acc'].item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                }, step=self.global_step)

        return {
            'loss': total_loss / max(num_batches, 1),
            'diffusion_loss': diff_loss_sum / max(num_batches, 1),
            'coarse_loss': coarse_loss_sum / max(num_batches, 1),
            'coarse_acc': coarse_acc_sum / max(num_batches, 1),
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
            
            # Derive coarse labels for metrics
            coarse_labels = self._get_coarse_labels(fine_labels)

            # Inference - NO coarse conditioning!
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

    def _get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        """Convert fine labels to coarse labels for metrics."""
        coarse_labels = fine_labels.clone()
        valid = fine_labels != -100
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            coarse_labels[(fine_labels == fine_id) & valid] = coarse_id
        return coarse_labels

    # =========================================================================
    # Span / Compound Extraction & Metrics (same as your existing code)
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
            'best_uss': self.best_uss,
            'best_lss': self.best_lss,
            'config': self.config,
        }
        for old in glob.glob(str(self.output_dir / '*.pt')):
            os.remove(old)
        torch.save(checkpoint, self.output_dir / 'best.pt')
        print(f"  ✓ Saved best model - USS: {self.best_uss:.4f}, LSS: {self.best_lss:.4f}")

    # =========================================================================
    # Main Training Loop - SIMPLIFIED (no stage switching)
    # =========================================================================

    def train(self):
        train_config = self.config['training']

        print(f"\n{'='*60}")
        print("TRAINING — SINGLE-STAGE BI-AFFINE DIFFUSION")
        print(f"{'='*60}")
        print(f"  Max epochs: {train_config['max_epochs']}")
        print(f"  Patience: {train_config.get('patience', 15)}")
        print(f"  NO Stage 1 / Stage 2 - single diffusion")
        print(f"  NO coarse conditioning - no error cascade")
        print(f"{'='*60}\n")

        patience_counter = 0

        for epoch in range(train_config['max_epochs']):
            self.current_epoch = epoch

            # Train - single stage, no stage parameter needed
            train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Diffusion Loss: {train_metrics['diffusion_loss']:.4f}")
            print(f"  Coarse Loss: {train_metrics['coarse_loss']:.4f}")
            print(f"  Coarse Acc: {train_metrics['coarse_acc']:.4f}")

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

            # Check improvement - monitor USS (primary metric)
            current_uss = eval_metrics['USS']
            current_lss = eval_metrics['LSS']
            is_best = current_uss > self.best_uss

            if is_best:
                self.best_uss = current_uss
                self.best_lss = current_lss
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
        print(f"  Best USS: {self.best_uss:.4f}")
        print(f"  Best LSS: {self.best_lss:.4f}")
        print(f"{'='*60}\n")

        if self.logger == 'wandb':
            self.wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training for Single-Stage Bi-Affine Diffusion NeCTI'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--use_constrained_eval', action='store_true',
        help='Use post-hoc constraint during evaluation',
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

    trainer = SingleStageBiAffineTrainer(config, args)
    trainer.train()


if __name__ == '__main__':
    main()