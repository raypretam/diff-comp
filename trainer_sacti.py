"""
Trainer for SaCTI (Sanskrit Compound Type Identification) using DiffusionSL
Simpler than NeCTI - this is token-level classification, not span extraction
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json
import wandb
from prettytable import PrettyTable
from typing import Dict, List
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

from data.ner.sacti_dataset import SaCTILabelSet, SaCTIDataset, SaCTICollator
from models.ddim_bitdit import BitDit
from options import get_args


class SaCTITrainer:
    """Trainer for Sanskrit Compound Type Identification"""
    
    def __init__(self, args):
        self.args = args
        self._print_hyperparameters()
        self.device = self._configure_device()
        
        # Initialize label set and datasets
        print("\n" + "="*80)
        print("Initializing SaCTI Dataset...")
        print("="*80)
        
        # Get language setting (SaCTILabelSet will handle path construction)
        language = getattr(args, 'language', 'auto')
        
        # SaCTILabelSet constructs the full path internally: data_path/language/granularity
        self.label_set = SaCTILabelSet(
            data_path=args.data_path,  # Just pass base path: /home/pretam-pg/SaCTI/data
            granularity=args.granularity,  # 'coarse' or 'fine'
            language=language  # 'sacti', 'marathi', 'english', or 'auto'
        )
        
        # Get the constructed path from label_set
        self.dataset_path = self.label_set.data_path
        
        print(f"Language: {self.label_set.language}")
        print(f"Dataset path: {self.dataset_path}")
        
        # Initialize WandB if enabled
        if self.args.logger == 'wandb':
            run_name = f"sacti-{self.label_set.language}-{args.granularity}-lr_bert_{args.lr_bert}-lr_other_{args.lr_other}-epochs_{args.max_epochs}"
            wandb.init(project="DiffusionSL-SaCTI", name=run_name)
            wandb.config.update(self.args)
            wandb.define_metric("f1", summary="max")
        
        args.num_labels = len(self.label_set)
        print(f"\nTotal labels: {args.num_labels}")
        print(f"Granularity: {args.granularity}")
        
        # Initialize model
        print("\n" + "="*80)
        print("Initializing BitDit Model...")
        print("="*80)
        
        self.model = BitDit(
            device=self.device,
            num_classes=args.num_labels,
            backbone=args.backbone,
            time_steps=args.time_steps,
            sampling_steps=args.sampling_steps,
            depth=args.depth,
            ddim_sampling_eta=args.ddim_sampling_eta,
            self_condition=args.self_condition,
            snr_scale=args.snr_scale,
            dataset=args.dataset,
            dim_model=args.dim_model,
            dim_time=args.dim_time,
            max_length=args.max_length,
            num_labels=args.num_labels,
            noise_schedule=args.noise_schedule,
            objective=args.objective,
            loss_type=args.loss_type,
            add_lstm=args.add_lstm,
            freeze_bert=args.freeze_bert,
            label_set=self.label_set
        ).to(self.device)
        
        self._print_num_parameters()
        
        if self.args.logger == "wandb":
            wandb.watch(self.model, log_freq=1000)
        
        # Initialize tokenizer and collator
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        self.collate_fn = SaCTICollator(
            self.tokenizer,
            max_length=self.args.max_length,
            label_set=self.label_set
        )
        
        # Create data loaders
        self.train_dataloader = self._get_dataloader('train', self.args.batch_size)
        self.dev_dataloader = self._get_dataloader('dev', self.args.batch_size)
        self.test_dataloader = self._get_dataloader('test', self.args.batch_size)
        
        # Try to load OOD data if available
        try:
            self.ood_dataloader = self._get_dataloader('ood', self.args.batch_size)
            print(f"  OOD: {len(self.ood_dataloader.dataset)} samples, {len(self.ood_dataloader)} batches")
        except FileNotFoundError:
            print("  OOD dataset not found, skipping...")
            self.ood_dataloader = None
        
        # Training setup
        self.best_dev_f1 = 0.0
        self.patience_counter = 0
        
        # Calculate total steps
        self.total_steps = len(self.train_dataloader) * self.args.max_epochs
        self.warmup_steps = int(self.total_steps * self.args.warmup_ratio)
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_dataloader.dataset)} samples, {len(self.train_dataloader)} batches")
        print(f"  Dev: {len(self.dev_dataloader.dataset)} samples, {len(self.dev_dataloader)} batches")
        print(f"  Test: {len(self.test_dataloader.dataset)} samples, {len(self.test_dataloader)} batches")
        print(f"\nTraining schedule:")
        print(f"Total epochs: {self.args.max_epochs}")
        print(f"Total training steps: {self.total_steps}")
        print(f"Warmup steps: {self.warmup_steps} ({100*self.warmup_steps/self.total_steps:.1f}% of total)")
        print(f"{'='*80}\n")
        
        self.optimizer, self.lr_scheduler = \
            self._configure_optimizer_and_scheduler(self.args.optimizer_type, self.args.lr_scheduler_type)
    
    def _get_dataloader(self, mode: str, bsz: int):
        """Create dataloader for specified split"""
        dataset = SaCTIDataset(self.dataset_path, mode, self.label_set)
        
        shuffle = (mode == 'train')
        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            num_workers=self.args.num_workers,
            drop_last=False,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
        return dataloader
    
    def _configure_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def _print_hyperparameters(self):
        """Print hyperparameters in a formatted table"""
        hparams = PrettyTable()
        hparams.title = 'Hyper Parameters'
        hparams.field_names = ["Name", "Value"]
        hparams.add_rows([[k, v] for k, v in self.args.__dict__.items()])
        print(hparams)
    
    def _print_num_parameters(self):
        num_para = sum(p.numel() for p in self.model.parameters())
        num_trainable_para = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of all parameters: {num_para:,}")
        print(f"Number of trainable parameters: {num_trainable_para:,}")
    
    def _configure_optimizer_and_scheduler(self, optimizer_type: str, scheduler_type: str):
        """Configure optimizer and learning rate scheduler"""
        from models.ddim_bitdit import default
        
        # Separate parameters for different learning rates
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': bert_params, 'lr': self.args.lr_bert},
            {'params': other_params, 'lr': self.args.lr_other}
        ]
        
        # Optimizer
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(param_groups)
        
        # Scheduler
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps,
                eta_min=1e-7
            )
        elif scheduler_type == 'linear':
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
            )
        else:  # constant
            from torch.optim.lr_scheduler import LambdaLR
            scheduler = LambdaLR(optimizer, lambda _: 1.0)
        
        return optimizer, scheduler
    
    @torch.no_grad()
    def evaluate(self, dataloader, split_name="dev"):
        """Evaluate model on given dataloader"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels = batch['seq_labels'].to(self.device)
            
            # Predictions
            predictions, _ = self.model(input_ids, attention_mask, seq_labels, None)
            
            # Get valid predictions (not padding)
            mask = (seq_labels != -100)
            
            for i in range(input_ids.shape[0]):
                valid_preds = predictions[i][mask[i]].cpu().tolist()
                valid_labels = seq_labels[i][mask[i]].cpu().tolist()
                
                all_preds.extend(valid_preds)
                all_labels.extend(valid_labels)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        accuracy = np.mean([p == l for p, l in zip(all_preds, all_labels)])
        
        # Print classification report
        # Get unique labels present in the data
        unique_labels = sorted(set(all_labels) | set(all_preds))
        label_names = [self.label_set.id2label(i) for i in unique_labels]
        
        print(f"\n{'='*60}")
        print(f"{split_name.upper()} Classification Report:")
        print(classification_report(
            all_labels, all_preds,
            labels=unique_labels,
            target_names=label_names,
            digits=4,
            zero_division=0
        ))
        print(f"{'='*60}\n")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _save_model(self, epoch, f1_score):
        """Save model checkpoint"""
        save_dir = os.path.join(os.getcwd(), 'saved_models', f'sacti_{self.label_set.language}_{self.args.granularity}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with epoch and F1 in filename
        save_path = os.path.join(save_dir, f'best_model_epoch{epoch}_f1{f1_score:.4f}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'args': self.args
        }, save_path)
        
        # Save a "latest" version for easy loading
        latest_path = os.path.join(save_dir, 'best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'args': self.args
        }, latest_path)
    
    def _load_best_model(self):
        """Load best model for final evaluation"""
        save_dir = os.path.join(os.getcwd(), 'saved_models', f'sacti_{self.label_set.language}_{self.args.granularity}')
        model_path = os.path.join(save_dir, 'best_model.pt')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from epoch {checkpoint['epoch']} with F1 {checkpoint['f1_score']:.4f}")
        else:
            print("⚠ No saved model found, using current model")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        best_f1 = 0.0
        global_step = 0
        
        for epoch in range(1, self.args.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.max_epochs}")
            print("-" * 50)
            
            self.model.train()
            epoch_loss = 0.0
            train_steps = 0
            
            pbar = tqdm(self.train_dataloader, desc='Training')
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_labels = batch['seq_labels'].to(self.device)
                
                # Forward pass
                loss = self.model(input_ids, attention_mask, seq_labels, None)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                
                self.optimizer.step()
                self.lr_scheduler.step()
                
                epoch_loss += loss.item()
                train_steps += 1
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr_bert': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'lr_other': f'{self.optimizer.param_groups[1]["lr"]:.2e}'
                })
                
                # Log to WandB
                if self.args.logger == 'wandb':
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr_bert': self.optimizer.param_groups[0]['lr'],
                        'train/lr_other': self.optimizer.param_groups[1]['lr'],
                        'global_step': global_step
                    })
            
            avg_loss = epoch_loss / train_steps
            print(f"Average training loss: {avg_loss:.4f}")
            print(f"Learning rates - BERT: {self.optimizer.param_groups[0]['lr']:.2e}, "
                  f"Other: {self.optimizer.param_groups[1]['lr']:.2e}")
            
            # Evaluation
            print(f"\nEvaluating on dev set...")
            dev_results = self.evaluate(self.dev_dataloader, split_name="dev")
            
            print(f"\nDev Accuracy: {dev_results['accuracy']:.4f}")
            print(f"Dev Precision: {dev_results['precision']:.4f}")
            print(f"Dev Recall: {dev_results['recall']:.4f}")
            print(f"Dev F1: {dev_results['f1']:.4f}")
            
            # Log to WandB
            if self.args.logger == 'wandb':
                wandb.log({
                    'dev/accuracy': dev_results['accuracy'],
                    'dev/precision': dev_results['precision'],
                    'dev/recall': dev_results['recall'],
                    'dev/f1': dev_results['f1'],
                    'epoch': epoch
                })
            
            # Save best model
            if dev_results['f1'] > best_f1:
                best_f1 = dev_results['f1']
                self.patience_counter = 0
                self._save_model(epoch, dev_results['f1'])
                print(f"\n✓ New best F1: {best_f1:.4f} - Model saved!")
            else:
                self.patience_counter += 1
                print(f"Early stopping counter: {self.patience_counter}/{self.args.patience}")
                
                if self.patience_counter >= self.args.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80 + "\n")
        
        # Load best model for final evaluation
        print("Loading best model for final evaluation...")
        self._load_best_model()
        
        # Final test evaluation
        print("\nEvaluating on test set...")
        test_results = self.evaluate(self.test_dataloader, split_name="test")
        print(f"\nFinal Test Results:")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"F1: {test_results['f1']:.4f}")
        
        # Evaluate on OOD if available
        if self.ood_dataloader is not None:
            print("\nEvaluating on OOD set...")
            ood_results = self.evaluate(self.ood_dataloader, split_name="ood")
            print(f"\nOOD Results:")
            print(f"Accuracy: {ood_results['accuracy']:.4f}")
            print(f"Precision: {ood_results['precision']:.4f}")
            print(f"Recall: {ood_results['recall']:.4f}")
            print(f"F1: {ood_results['f1']:.4f}")
            
            if self.args.logger == 'wandb':
                wandb.log({
                    'ood/accuracy': ood_results['accuracy'],
                    'ood/precision': ood_results['precision'],
                    'ood/recall': ood_results['recall'],
                    'ood/f1': ood_results['f1']
                })
        
        # Log final test results to WandB
        if self.args.logger == 'wandb':
            wandb.log({
                'test/accuracy': test_results['accuracy'],
                'test/precision': test_results['precision'],
                'test/recall': test_results['recall'],
                'test/f1': test_results['f1']
            })
            wandb.finish()


def main():
    """Main entry point"""
    args = get_args()
    trainer = SaCTITrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
