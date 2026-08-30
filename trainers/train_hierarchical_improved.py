"""
Training Script for Hierarchical Diffusion NeCTI (FIXED)
=========================================================

This is a CLEAN training script for the fixed hierarchical model.

Key changes from the broken "improved" version:
1. REMOVED span consistency loss (was computing 0.0, broken)
2. REMOVED contrastive loss (was hurting performance)
3. REMOVED MBR evaluation (too slow, not helping)
4. KEPT constrained decoding as optional post-processing

The core fix is in hierarchial_diffusion.py:
- Stage 2 no longer conditions on Stage 1's coarse predictions
- This prevents error propagation that was killing USS/LSS

Usage:
    # Standard training (recommended)
    python train_hierarchical.py --config configs/hierarchial_necti.yaml --stage both
    
    # With constrained post-processing at eval time
    python train_hierarchical.py --config configs/hierarchial_necti.yaml --stage both --use_constrained_eval
    
    # Sequential training (Stage 1 first, then Stage 2)
    python train_hierarchical.py --config configs/hierarchial_necti.yaml --stage sequential
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

# Import your modules
from models.hierarchial_diffusion import HierarchicalDiffusionNeCTI, create_hierarchical_model
from data.ner.necti_dataset import NeCTIDataset, NeCTILabelSet, NeCTICollator
from data.ner.necti_dataset_hierarchial import get_coarse_id_from_fine_label, COARSE_CATEGORIES


class HierarchicalTrainer:
    """
    Clean trainer for Hierarchical Diffusion NeCTI.
    
    Uses the FIXED model where Stage 2 does NOT condition on Stage 1.
    This prevents error propagation while preserving local attention benefits.
    """
    
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['gpu']['use_gpu'] else 'cpu')
        
        # Only keep constrained eval (proven to help, no training overhead)
        self.use_constrained_eval = args.use_constrained_eval
        
        # Setup output directory
        self.output_dir = Path(config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
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
        print("HIERARCHICAL DIFFUSION TRAINER (FIXED)")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Constrained Evaluation: {self.use_constrained_eval}")
        print(f"  Mixed Precision: {self.scaler is not None}")
        print(f"{'='*60}\n")
    
    def _setup_data(self):
        """Setup datasets and dataloaders"""
        data_config = self.config['data']
        
        # Label sets
        self.coarse_label_set = NeCTILabelSet(
            data_path=data_config['data_path'],
            granularity='Coarse',
            use_context=data_config['use_context']
        )
        
        self.fine_label_set = NeCTILabelSet(
            data_path=data_config['data_path'],
            granularity='Finegrain',
            use_context=data_config['use_context']
        )
        
        # Build fine-to-coarse mapping
        self.fine_to_coarse_map = self._build_label_hierarchy()
        self.coarse_to_fine_map = self._build_reverse_hierarchy()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['backbone'])
        
        # Datasets
        self.train_dataset = NeCTIDataset(
            data_path=data_config['data_path'],
            mode='train',
            label_set=self.fine_label_set,
            use_context=data_config['use_context']
        )
        
        self.dev_dataset = NeCTIDataset(
            data_path=data_config['data_path'],
            mode='dev',
            label_set=self.fine_label_set,
            use_context=data_config['use_context']
        )
        
        # Collator
        collator = NeCTICollator(
            tokenizer=self.tokenizer,
            max_length=data_config['max_length'],
        )
        
        # Dataloaders
        train_config = self.config['training']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            collate_fn=collator,
            pin_memory=True
        )
        
        self.dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            collate_fn=collator,
            pin_memory=True
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Dev samples: {len(self.dev_dataset)}")
        print(f"Coarse classes: {len(self.coarse_label_set)}")
        print(f"Fine classes: {len(self.fine_label_set)}")
    
    def _build_label_hierarchy(self) -> Dict[int, int]:
        """Build mapping from fine-grained label IDs to coarse label IDs."""
        fine_to_coarse = {}
        
        for fine_id in range(len(self.fine_label_set)):
            fine_name = self.fine_label_set.id2label(fine_id)
            coarse_id = get_coarse_id_from_fine_label(fine_name)
            fine_to_coarse[fine_id] = coarse_id
        
        print(f"\nBuilt fine-to-coarse mapping for {len(fine_to_coarse)} labels")
        return fine_to_coarse
    
    def _build_reverse_hierarchy(self) -> Dict[int, List[int]]:
        """Build coarse-label-id → list of fine-label-ids mapping."""
        coarse_to_fine = {}
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            if coarse_id not in coarse_to_fine:
                coarse_to_fine[coarse_id] = []
            coarse_to_fine[coarse_id].append(fine_id)
        
        print("\nCoarse-to-Fine mapping:")
        for coarse_id, fine_ids in sorted(coarse_to_fine.items()):
            coarse_name = COARSE_CATEGORIES[coarse_id] if coarse_id < len(COARSE_CATEGORIES) else f"ID:{coarse_id}"
            print(f"  {coarse_name}: {len(fine_ids)} fine labels")
        
        return coarse_to_fine
    
    def _setup_model(self):
        """Setup the hierarchical diffusion model"""
        model_config = self.config
        
        self.model = create_hierarchical_model(
            device=str(self.device),
            backbone=model_config['backbone'],
            dim_model=model_config['dim_model'],
            freeze_bert=model_config['freeze_bert'],
            time_steps=model_config['time_steps'],
            sampling_steps=model_config['sampling_steps'],
            noise_schedule=model_config['noise_schedule'],
            snr_scale=model_config['snr_scale'],
            global_depth=model_config['stage1']['depth'],
            global_num_heads=model_config['stage1']['num_heads'],
            num_coarse_classes=model_config['stage1']['num_classes'],
            fine_depth=model_config['stage2']['depth'],
            fine_num_heads=model_config['stage2']['num_heads'],
            num_fine_classes=model_config['stage2']['num_classes'],
            fine_to_coarse_map=self.fine_to_coarse_map,
            objective=model_config['objective'],
            loss_type=model_config['loss_type']
        )
        
        # Load checkpoint if provided
        if self.args.resume:
            print(f"Resuming from: {self.args.resume}")
            checkpoint = torch.load(self.args.resume, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        train_config = self.config['training']
        
        # Parameter groups with different learning rates
        param_groups = [
            {
                'params': list(self.model.backbone.parameters()),
                'lr': float(train_config['lr_bert']),
                'name': 'backbone'
            },
            {
                'params': list(self.model.global_dit.parameters()),
                'lr': float(train_config['lr_stage1']),
                'name': 'stage1'
            },
            {
                'params': list(self.model.fine_dit.parameters()) + list(self.model.coarse_embed.parameters()),
                'lr': float(train_config['lr_stage2']),
                'name': 'stage2'
            }
        ]
        
        # Filter empty groups
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=train_config['weight_decay']
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * train_config['max_epochs']
        warmup_steps = train_config.get('warmup_steps', 500)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    
    def _setup_logging(self):
        """Setup logging"""
        log_config = self.config.get('logging', {})
        self.logger = log_config.get('logger', 'None')
        
        if self.logger == 'wandb':
            import wandb
            wandb.init(
                project=log_config.get('project_name', 'hierarchical-necti'),
                name=log_config.get('experiment_name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                config=self.config
            )
            self.wandb = wandb
        elif self.logger == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.output_dir / 'tensorboard')
    
    def _get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        """Convert fine-grained labels to coarse labels"""
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
        """Train for one epoch."""
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
                        input_ids, attention_mask, coarse_labels, fine_labels, stage=stage
                    )
                    
                    if isinstance(losses, dict):
                        loss = losses['loss']
                    else:
                        loss = losses
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss at step {self.global_step}, skipping")
                    continue
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model(
                    input_ids, attention_mask, coarse_labels, fine_labels, stage=stage
                )
                
                if isinstance(losses, dict):
                    loss = losses['loss']
                else:
                    loss = losses
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss at step {self.global_step}, skipping")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            if isinstance(losses, dict):
                stage1_loss_sum += losses['stage1_loss'].item()
                stage2_loss_sum += losses['stage2_loss'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log
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
            'loss': total_loss / num_batches,
            'stage1_loss': stage1_loss_sum / num_batches,
            'stage2_loss': stage2_loss_sum / num_batches,
        }
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on dev set."""
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
            
            # Inference (with optional constraint)
            coarse_preds, fine_preds = self.model.inference(
                input_ids, attention_mask, 
                apply_constraint=self.use_constrained_eval
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
                
                # Convert to names for compound extraction
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
        
        # Compute metrics
        coarse_acc = np.mean(np.array(all_coarse_preds) == np.array(all_coarse_labels)) if all_coarse_labels else 0.0
        fine_acc = np.mean(np.array(all_fine_preds) == np.array(all_fine_labels)) if all_fine_labels else 0.0
        
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
        """Compute F1 for spans."""
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
        return f1
    
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
    
    # =========================================================================
    # Checkpointing
    # =========================================================================
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint (only keeps best)."""
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
            'config': self.config
        }
        
        # Remove old checkpoints
        for old_ckpt in glob.glob(str(self.output_dir / '*.pt')):
            os.remove(old_ckpt)
        
        torch.save(checkpoint, self.output_dir / 'best.pt')
        print(f"  ✓ Saved best model - USS: {self.best_metric:.4f}, LSS: {self.best_lss:.4f}")
    
    # =========================================================================
    # Main Training Loop
    # =========================================================================
    
    def train(self):
        """Main training loop."""
        train_config = self.config['training']
        stage = self.args.stage or train_config.get('strategy', 'both')
        
        # Map 'joint' to 'both' for compatibility
        if stage == 'joint':
            stage = 'both'
        
        print(f"\n{'='*60}")
        print(f"TRAINING")
        print(f"{'='*60}")
        print(f"  Strategy: {stage}")
        print(f"  Max epochs: {train_config['max_epochs']}")
        print(f"  Patience: {train_config.get('patience', 10)}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(train_config['max_epochs']):
            self.current_epoch = epoch
            
            # Handle sequential training
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
            print(f"Epoch {epoch} - Coarse Acc: {eval_metrics['coarse_acc']:.4f}, Fine Acc: {eval_metrics['fine_acc']:.4f}")
            print(f"Epoch {epoch} - USS: {eval_metrics['USS']:.4f}, LSS: {eval_metrics['LSS']:.4f}, EM: {eval_metrics['EM']:.4f}")
            
            # Log
            if self.logger == 'wandb':
                self.wandb.log({
                    'epoch': epoch,
                    **{f'val/{k}': v for k, v in eval_metrics.items()}
                }, step=self.global_step)
            
            # Check improvement (monitor USS as primary)
            current_metric = eval_metrics['USS']
            current_lss = eval_metrics['LSS']
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_lss = current_lss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if patience_counter >= train_config.get('patience', 10):
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
    parser = argparse.ArgumentParser(description='Training for Hierarchical Diffusion NeCTI')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--stage', type=str, default=None,
                        choices=['stage1', 'stage2', 'both', 'sequential'],
                        help='Training stage (default: both)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Evaluation options
    parser.add_argument('--use_constrained_eval', action='store_true',
                        help='Use constrained decoding during evaluation (optional post-processing)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and train
    trainer = HierarchicalTrainer(config, args)
    trainer.train()


if __name__ == '__main__':
    main()