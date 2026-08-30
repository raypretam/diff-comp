"""
Training Script for Hierarchical Diffusion NeCTI
=================================================

Two-stage training:
- Stage 1: Global attention for coarse labels
- Stage 2: Local attention for fine-grained refinement

Usage:
    python train_hierarchical.py --config configs/hierarchical_necti.yaml
    
    # Train only Stage 1
    python train_hierarchical.py --config configs/hierarchical_necti.yaml --stage stage1
    
    # Train only Stage 2 (requires pretrained Stage 1)
    python train_hierarchical.py --config configs/hierarchical_necti.yaml --stage stage2 --stage1_checkpoint path/to/stage1.pt
"""

import os
import sys
import argparse
import yaml
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
# GradScaler and autocast are used via torch.amp API (non-deprecated)

from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Import your modules
from models.hierarchial_diffusion import HierarchicalDiffusionNeCTI, create_hierarchical_model
from data.ner.necti_dataset import NeCTIDataset, NeCTILabelSet, NeCTICollator
from data.ner.necti_dataset_hierarchial import get_coarse_id_from_fine_label


class HierarchicalTrainer:
    """
    Trainer for Hierarchical Diffusion NeCTI model.
    
    Supports:
    - Joint training of both stages
    - Sequential training (Stage 1 → Stage 2)
    - Individual stage training
    """
    
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['gpu']['use_gpu'] else 'cpu')
        
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
        
        # Mixed precision
        if config['gpu'].get('precision', 32) == 16:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
    
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
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['backbone'])
        
        # Datasets
        self.train_dataset = NeCTIDataset(
            data_path=data_config['data_path'],
            mode='train',
            label_set=self.fine_label_set,  # Use fine-grained for full info
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
            # label_set=self.fine_label_set
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
        """
        Build mapping from fine-grained label IDs to coarse label IDs.
        
        Uses the correct mapping function from necti_dataset_hierarchial.py
        that handles abbreviated codes like 'BvS', 'T6', 'K1', 'Ds', etc.
        """
        fine_to_coarse = {}
        
        for fine_id in range(len(self.fine_label_set)):
            fine_name = self.fine_label_set.id2label(fine_id)
            
            # Use the correct mapping function for abbreviated codes
            coarse_id = get_coarse_id_from_fine_label(fine_name)
            fine_to_coarse[fine_id] = coarse_id
        
        # Print some examples for verification
        print(f"\nBuilt fine-to-coarse mapping for {len(fine_to_coarse)} labels")
        print("Sample mappings:")
        for fine_id in range(min(10, len(fine_to_coarse))):
            fine_name = self.fine_label_set.id2label(fine_id)
            coarse_id = fine_to_coarse[fine_id]
            coarse_names = ['Tatpurusha', 'Bahuvrihi', 'Dvandva', 'Avyayibhava', 'ROOT', 'No_rel']
            print(f"  {fine_name} (id={fine_id}) → {coarse_names[coarse_id]} (id={coarse_id})")
        
        return fine_to_coarse
    
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
            global_num_heads=model_config['stage1']['num_heads'],  # Add this
            num_coarse_classes=model_config['stage1']['num_classes'],
            local_depth=model_config['stage2']['depth'],
            local_num_heads=model_config['stage2']['num_heads'],  # Add this
            local_window_size=model_config['stage2']['window_size'],
            num_fine_classes=model_config['stage2']['num_classes'],
            fine_to_coarse_map=self.fine_to_coarse_map,
            objective=model_config['objective'],
            loss_type=model_config['loss_type']
        )
        
        # Load Stage 1 checkpoint if provided (for Stage 2 only training)
        if self.args.stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {self.args.stage1_checkpoint}")
            checkpoint = torch.load(self.args.stage1_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        train_config = self.config['training']
        
        # Group parameters with different learning rates
        # Note: When using param groups with different LRs, we need to ensure
        # they're properly formatted as floats
        lr_bert = float(train_config['lr_bert'])
        lr_stage1 = float(train_config['lr_stage1'])
        lr_stage2 = float(train_config['lr_stage2'])
        
        param_groups = [
            {
                'params': list(self.model.backbone.parameters()),
                'lr': lr_bert,
                'name': 'backbone'
            },
            {
                'params': list(self.model.global_dit.parameters()),
                'lr': lr_stage1,
                'name': 'stage1'
            },
            {
                'params': list(self.model.local_dit.parameters()),
                'lr': lr_stage2,
                'name': 'stage2'
            }
        ]
        
        # Filter out empty parameter groups
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
        """Setup logging (wandb, tensorboard, etc.)"""
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
        
        # Only map valid labels (not -100 padding)
        valid_mask = fine_labels != -100
        
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            mask = (fine_labels == fine_id) & valid_mask
            coarse_labels[mask] = coarse_id
        
        return coarse_labels
    
    def train_epoch(self, stage: str = 'both') -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        stage1_loss_sum = 0.0
        stage2_loss_sum = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['seq_labels'].to(self.device)  # Changed from 'labels' to 'seq_labels'
            
            # Get coarse labels
            coarse_labels = self._get_coarse_labels(fine_labels)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    if stage == 'both':
                        losses = self.model(
                            input_ids, attention_mask,
                            coarse_labels, fine_labels,
                            stage='both'
                        )
                        loss = losses['loss']
                        stage1_loss_sum += losses['stage1_loss'].item()
                        stage2_loss_sum += losses['stage2_loss'].item()
                    elif stage == 'stage1':
                        loss = self.model(
                            input_ids, attention_mask,
                            coarse_labels, None,
                            stage='stage1'
                        )
                    elif stage == 'stage2':
                        loss = self.model(
                            input_ids, attention_mask,
                            coarse_labels, fine_labels,
                            stage='stage2'
                        )
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at step {self.global_step}, skipping batch")
                    continue
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                
                # scaler.step() automatically skips optimizer.step() if grads contain inf/nan
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
            else:
                if stage == 'both':
                    losses = self.model(
                        input_ids, attention_mask,
                        coarse_labels, fine_labels,
                        stage='both'
                    )
                    loss = losses['loss']
                    stage1_loss_sum += losses['stage1_loss'].item()
                    stage2_loss_sum += losses['stage2_loss'].item()
                elif stage == 'stage1':
                    loss = self.model(
                        input_ids, attention_mask,
                        coarse_labels, None,
                        stage='stage1'
                    )
                elif stage == 'stage2':
                    loss = self.model(
                        input_ids, attention_mask,
                        coarse_labels, fine_labels,
                        stage='stage2'
                    )
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at step {self.global_step}, skipping batch")
                    continue
                
                loss.backward()
                
                # Check gradient norms for debugging
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Invalid gradient norm {grad_norm.item():.4f} at step {self.global_step}, skipping batch")
                    continue
                
                self.optimizer.step()
                self.scheduler.step()  # Called after optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb
            if self.logger == 'wandb' and self.global_step % 50 == 0:
                log_dict = {'train/loss': loss.item(), 'train/lr': self.scheduler.get_last_lr()[0]}
                if stage == 'both':
                    log_dict['train/stage1_loss'] = losses['stage1_loss'].item()
                    log_dict['train/stage2_loss'] = losses['stage2_loss'].item()
                self.wandb.log(log_dict, step=self.global_step)
        
        metrics = {
            'loss': total_loss / num_batches,
        }
        if stage == 'both':
            metrics['stage1_loss'] = stage1_loss_sum / num_batches
            metrics['stage2_loss'] = stage2_loss_sum / num_batches
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on dev set using USS, LSS, and EM metrics.
        
        Labels are plain relation codes (T6, BvS, Comp_root, No_rel) — no distance prefix.
        Compounds are contiguous runs of non-No_rel tokens, each ending at Comp_root.
        
        USS = F1 on compound span boundaries (start, end)
        LSS = F1 on compound span tuples (start, end, label)
        EM  = fraction of true compounds exactly matched
        """
        from collections import Counter
        
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
            
            # Inference
            coarse_preds, fine_preds = self.model.inference(input_ids, attention_mask)
            
            # Valid mask: non-padded, non-special-token positions (one per word)
            mask = fine_labels != -100
            
            # Process sentence-by-sentence
            for i in range(input_ids.shape[0]):
                sent_mask = mask[i]  # [seq_len] boolean
                if sent_mask.sum() == 0:
                    continue
                
                # Use mask to get exactly one prediction per word (correct alignment)
                pred_ids = fine_preds[i][sent_mask].cpu().tolist()
                true_ids = fine_labels[i][sent_mask].cpu().tolist()
                coarse_pred_ids = coarse_preds[i][sent_mask].cpu().tolist()
                coarse_true_ids = coarse_labels[i][sent_mask].cpu().tolist()
                
                # Flat accuracy tracking
                all_fine_preds.extend(pred_ids)
                all_fine_labels.extend(true_ids)
                all_coarse_preds.extend(coarse_pred_ids)
                all_coarse_labels.extend(coarse_true_ids)
                
                # Convert IDs to label names
                pred_names = [self.fine_label_set.id2label(pid) if 0 <= pid < len(self.fine_label_set) else "No_rel"
                              for pid in pred_ids]
                true_names = [self.fine_label_set.id2label(tid) if 0 <= tid < len(self.fine_label_set) else "No_rel"
                              for tid in true_ids]
                
                # Extract compound spans from contiguous label sequences
                pred_compounds = self._extract_compounds_from_labels(pred_names)
                true_compounds = self._extract_compounds_from_labels(true_names)
                
                pred_compounds_all.append(pred_compounds)
                true_compounds_all.append(true_compounds)
        
        # Flat accuracy
        coarse_acc = np.mean(np.array(all_coarse_preds) == np.array(all_coarse_labels)) if all_coarse_labels else 0.0
        fine_acc = np.mean(np.array(all_fine_preds) == np.array(all_fine_labels)) if all_fine_labels else 0.0
        
        # Span-based metrics
        uss = self._compute_span_f1(true_compounds_all, pred_compounds_all, labeled=False)
        lss = self._compute_span_f1(true_compounds_all, pred_compounds_all, labeled=True)
        em = self._compute_exact_match(true_compounds_all, pred_compounds_all)
        
        metrics = {
            'coarse_acc': coarse_acc,
            'fine_acc': fine_acc,
            'USS': uss,
            'LSS': lss,
            'EM': em,
        }
        
        return metrics
    
    def _extract_compounds_from_labels(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        """
        Extract compound spans from a word-level label sequence.
        
        A compound is a contiguous run of non-No_rel/non-root tokens.
        Comp_root marks the end of each compound within a run.
        The compound label is the majority vote of member labels (excluding Comp_root).
        
        Example:
            labels = ['T6', 'T6', 'Comp_root', 'No_rel', 'Bs6', 'K1', 'Comp_root']
            → [(0, 2, 'T6'), (4, 6, 'K1')]  (or Bs6 depending on count)
        """
        from collections import Counter
        
        NON_COMPOUND = {'No_rel', 'root', '_'}
        compounds = []
        i = 0
        n = len(labels)
        
        while i < n:
            if labels[i] not in NON_COMPOUND:
                # Start of a compound region
                region_start = i
                
                # Collect tokens until we hit No_rel/root or end
                while i < n and labels[i] not in NON_COMPOUND:
                    i += 1
                region_end = i - 1  # inclusive, last non-No_rel token
                
                # Split this region into individual compounds at Comp_root boundaries
                comp_start = region_start
                member_labels = []
                
                for j in range(region_start, region_end + 1):
                    if labels[j] == 'Comp_root':
                        # This Comp_root ends a compound
                        if member_labels:
                            label = Counter(member_labels).most_common(1)[0][0]
                        else:
                            label = 'Comp_root'
                        compounds.append((comp_start, j, label))
                        comp_start = j + 1
                        member_labels = []
                    else:
                        member_labels.append(labels[j])
                
                # Handle remaining tokens after last Comp_root (partial compound)
                if comp_start <= region_end and member_labels:
                    label = Counter(member_labels).most_common(1)[0][0]
                    compounds.append((comp_start, region_end, label))
            else:
                i += 1
        
        return compounds
    
    def _compute_span_f1(self, true_all: List[List[Tuple]], pred_all: List[List[Tuple]], labeled: bool = False) -> float:
        """
        Compute F1 score over compound spans.
        
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
        return f1
    
    def _compute_exact_match(self, true_all: List[List[Tuple]], pred_all: List[List[Tuple]]) -> float:
        """
        Compute Exact Match: fraction of true compounds that are exactly matched.
        Matches Eval_USS_LSS.py semantics: counts how many predicted compounds
        appear in the true set, divided by total true compounds.
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
    
    def _coarse_id_to_name(self, coarse_id: int) -> str:
        """Convert coarse ID back to category name"""
        from data.ner.necti_dataset_hierarchial import COARSE_CATEGORIES
        if 0 <= coarse_id < len(COARSE_CATEGORIES):
            return COARSE_CATEGORIES[coarse_id]
        return 'No_rel'
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.output_dir / 'last.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
            print(f"Saved best model with metric: {self.best_metric:.4f}")
    
    def train(self):
        """Main training loop"""
        train_config = self.config['training']
        strategy = train_config.get('strategy', 'joint')
        stage = self.args.stage or strategy
        
        print(f"\n{'='*60}")
        print(f"Training Strategy: {stage}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(train_config['max_epochs']):
            self.current_epoch = epoch
            
            # Determine which stage to train
            if stage == 'sequential':
                # First train Stage 1, then Stage 2
                if epoch < train_config.get('stage1_epochs', 30):
                    current_stage = 'stage1'
                else:
                    current_stage = 'stage2'
                    # Optionally freeze Stage 1
                    if train_config.get('freeze_stage1_for_stage2', False):
                        for param in self.model.global_dit.parameters():
                            param.requires_grad = False
            else:
                current_stage = stage
            
            # Train
            train_metrics = self.train_epoch(current_stage)
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Evaluate
            eval_metrics = self.evaluate()
            print(f"Epoch {epoch} - Coarse Acc: {eval_metrics['coarse_acc']:.4f}, Fine Acc: {eval_metrics['fine_acc']:.4f}")
            print(f"Epoch {epoch} - USS: {eval_metrics['USS']:.4f}, LSS: {eval_metrics['LSS']:.4f}, EM: {eval_metrics['EM']:.4f}")
            
            # Log
            if self.logger == 'wandb':
                self.wandb.log({
                    'epoch': epoch,
                    'val/coarse_acc': eval_metrics['coarse_acc'],
                    'val/fine_acc': eval_metrics['fine_acc'],
                    'val/USS': eval_metrics['USS'],
                    'val/LSS': eval_metrics['LSS'],
                    'val/EM': eval_metrics['EM'],
                }, step=self.global_step)
            
            # Check for improvement (monitor USS as primary metric)
            current_metric = eval_metrics['USS']
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if patience_counter >= train_config.get('patience', 10):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\nTraining complete! Best USS: {self.best_metric:.4f}")
        
        if self.logger == 'wandb':
            self.wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Hierarchical Diffusion NeCTI')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--stage', type=str, default=None,
                        choices=['stage1', 'stage2', 'both', 'sequential'],
                        help='Which stage(s) to train (overrides config)')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help='Path to Stage 1 checkpoint (for Stage 2 only training)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = HierarchicalTrainer(config, args)
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.best_metric = checkpoint['best_metric']
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()