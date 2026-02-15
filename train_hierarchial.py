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
from torch.cuda.amp import GradScaler, autocast

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
                
                # Check gradient norms for debugging
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Invalid gradient norm {grad_norm.item():.4f} at step {self.global_step}, skipping batch")
                    continue
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()  # Moved after optimizer.step()
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
        """Evaluate on dev set"""
        self.model.eval()
        
        all_coarse_preds = []
        all_coarse_labels = []
        all_fine_preds = []
        all_fine_labels = []
        
        for batch in tqdm(self.dev_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['seq_labels'].to(self.device)  # Changed from 'labels' to 'seq_labels'
            coarse_labels = self._get_coarse_labels(fine_labels)
            
            # Inference
            coarse_preds, fine_preds = self.model.inference(input_ids, attention_mask)
            
            # Collect predictions (only valid positions)
            mask = fine_labels != -100
            
            all_coarse_preds.extend(coarse_preds[mask].cpu().numpy().tolist())
            all_coarse_labels.extend(coarse_labels[mask].cpu().numpy().tolist())
            all_fine_preds.extend(fine_preds[mask].cpu().numpy().tolist())
            all_fine_labels.extend(fine_labels[mask].cpu().numpy().tolist())
        
        # Calculate metrics
        coarse_acc = np.mean(np.array(all_coarse_preds) == np.array(all_coarse_labels))
        fine_acc = np.mean(np.array(all_fine_preds) == np.array(all_fine_labels))
        
        # USS: Unlabeled Span Score (structure only)
        # LSS: Labeled Span Score (structure + label)
        # For simplicity, using accuracy as proxy. Replace with your actual USS/LSS computation.
        
        metrics = {
            'coarse_acc': coarse_acc,
            'fine_acc': fine_acc,
            'coarse_lss': coarse_acc,  # Placeholder - use actual LSS computation
            'fine_lss': fine_acc,  # Placeholder - use actual LSS computation
        }
        
        return metrics
    
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
            
            # Log
            if self.logger == 'wandb':
                self.wandb.log({
                    'epoch': epoch,
                    'val/coarse_acc': eval_metrics['coarse_acc'],
                    'val/fine_acc': eval_metrics['fine_acc'],
                    'val/coarse_lss': eval_metrics['coarse_lss'],
                    'val/fine_lss': eval_metrics['fine_lss'],
                }, step=self.global_step)
            
            # Check for improvement
            current_metric = eval_metrics['fine_lss']  # Monitor fine-grained performance
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
        
        print(f"\nTraining complete! Best fine-grained LSS: {self.best_metric:.4f}")
        
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