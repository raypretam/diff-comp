"""
Training script for nested compound identification using DiffusionSL with Cross-Attention DiT
Adapted for DepNeCTI data with XLM-R encoder
"""

import os
from argparse import Namespace
import torch
from models.ddim_bitdit_cross_attn import BitDitCrossAttn
from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
import wandb
from tqdm import tqdm

from prettytable import PrettyTable
from utils import get_lr_scheduler
from typing import Dict, List
import numpy as np


class NeCTITrainerCrossAttn:
    """Trainer for nested compound identification using cross-attention DiT"""
    
    def __init__(self, args: Namespace):
        self.args = args
        self._print_hyperparameters()
        
        # Set default value for use_context if not present
        if not hasattr(self.args, 'use_context'):
            self.args.use_context = False
        
        # Set default value for save_limit if not present
        if not hasattr(self.args, 'save_limit'):
            self.args.save_limit = 3
        
        # Track saved checkpoint files for cleanup
        self.saved_checkpoints = []
        
        # Early stopping
        self.patience = getattr(self.args, 'patience', 5)
        self.min_delta = getattr(self.args, 'min_delta', 0.0001)
        self.early_stopping_counter = 0
        self.best_f1_for_early_stopping = 0.0
        print(f"Early stopping enabled with patience={self.patience}, min_delta={self.min_delta}")
        
        context_mode = "with_ctx" if self.args.use_context else "no_ctx"
        model_variant = "cross_attn"  # This trainer always uses cross-attention
        
        if self.args.logger == 'wandb':
            run_name = f"necti-{args.granularity}-{context_mode}-{model_variant}--lr_bert_{args.lr_bert}--lr_other_{args.lr_other}--epochs_{args.max_epochs}"
            wandb.init(project="DiffusionSL-NeCTI", name=run_name)
            wandb.config.update(self.args)
            wandb.define_metric("f1", summary="max")
        
        self.device = self._configure_device()
        
        # DepNeCTI data path
        self.dataset_path = self.args.data_path
        
        # Load label set
        self.label_set = NeCTILabelSet(
            data_path=self.dataset_path,
            granularity=self.args.granularity,
            use_context=self.args.use_context
        )
        
        if self.args.num_classes != len(self.label_set):
            print(f"Number of classes ({self.args.num_classes}) adjusted to {len(self.label_set)} based on dataset")
            self.args.num_classes = len(self.label_set)
        
        # Initialize model with XLM-R backbone and cross-attention DiT
        self.model = BitDitCrossAttn(
            device=self.device,
            num_classes=self.args.num_classes,
            backbone=self.args.backbone,
            time_steps=self.args.time_steps,
            sampling_steps=self.args.sampling_steps,
            noise_schedule=self.args.noise_schedule,
            ddim_sampling_eta=self.args.ddim_sampling_eta,
            self_condition=self.args.self_condition,
            snr_scale=self.args.snr_scale,
            dataset=f"necti_{self.args.granularity}",
            dim_model=self.args.dim_model,
            dim_time=self.args.dim_time,
            objective=self.args.objective,
            loss_type=self.args.loss_type,
            add_lstm=self.args.add_lstm,
            freeze_bert=self.args.freeze_bert,
            max_length=self.args.max_length,
            depth=self.args.depth,
            num_labels=len(self.label_set)
        ).to(self.device)
        
        if self.args.logger == "wandb":
            wandb.watch(self.model, log_freq=1000)
        
        # Initialize tokenizer for XLM-R
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        self.collate_fn = NeCTICollator(self.tokenizer, max_length=self.args.max_length)
        
        # Create data loaders
        self.train_dataloader = self._get_dataloader('train', self.args.batch_size)
        self.dev_dataloader = self._get_dataloader('dev', self.args.batch_size)
        self.test_dataloader = self._get_dataloader('test', self.args.batch_size)
        
        # Try to load OOD data if available
        try:
            self.ood_dataloader = self._get_dataloader('ood', self.args.batch_size)
        except FileNotFoundError:
            print("OOD dataset not found, skipping...")
            self.ood_dataloader = None
        
        self.steps = self.args.max_steps
        
        self.optimizer, self.lr_scheduler = \
            self._configure_optimizer_and_scheduler(self.args.optimizer_type, self.args.lr_scheduler_type)
    
    def _get_dataloader(self, mode: str, bsz: int):
        """Create dataloader for specified split"""
        dataset = NeCTIDataset(self.dataset_path, mode, self.label_set, use_context=self.args.use_context)
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
    
    def _print_hyperparameters(self):
        hparams = PrettyTable()
        hparams.title = 'Hyper Parameters (Cross-Attention DiT)'
        hparams.field_names = ["Name", "Value"]
        hparams.add_rows([[k, v] for k, v in self.args.__dict__.items()])
        print(hparams)
    
    def _print_num_parameters(self):
        num_para = sum(p.numel() for p in self.model.parameters())
        num_trainable_para = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of all parameters: {num_para:,}")
        print(f"Number of trainable parameters: {num_trainable_para:,}")
    
    def _configure_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def _configure_optimizer_and_scheduler(self, optimizer_type: str, lr_scheduler_type: str):
        """Configure optimizer and learning rate scheduler"""
        # Separate parameters for BERT backbone and other components
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    bert_params.append(param)
                else:
                    other_params.append(param)
        
        optimizer_params = [
            {'params': bert_params, 'lr': self.args.lr_bert},
            {'params': other_params, 'lr': self.args.lr_other}
        ]
        
        if optimizer_type.lower() in ['adam', 'adamw']:
            optimizer = AdamW(optimizer_params, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented")
        
        lr_scheduler = get_lr_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.steps,
            num_cycles=self.args.num_cycles if hasattr(self.args, 'num_cycles') else None
        )
        
        return optimizer, lr_scheduler
    
    def train(self):
        """Main training loop"""
        self._print_num_parameters()
        print("\n" + "=" * 50)
        print("Starting training with Cross-Attention DiT...")
        print("=" * 50 + "\n")
        
        best_f1 = 0.0
        global_step = 0
        
        for epoch in range(1, self.args.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.max_epochs}")
            print("-" * 50)
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            pbar = tqdm(self.train_dataloader, desc="Training")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_labels = batch['seq_labels'].to(self.device)
                
                loss = self.model(input_ids, attention_mask, seq_labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                global_step += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if self.args.logger == 'wandb':
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr_bert': self.optimizer.param_groups[0]['lr'],
                        'train/lr_other': self.optimizer.param_groups[1]['lr'],
                        'global_step': global_step
                    })
            
            avg_train_loss = train_loss / train_steps
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            # Evaluation on dev set
            print("\nEvaluating on dev set...")
            dev_results = self.evaluate(self.dev_dataloader, "dev")
            
            print(f"Dev F1: {dev_results['f1']:.4f}")
            print(f"Dev Precision: {dev_results['precision']:.4f}")
            print(f"Dev Recall: {dev_results['recall']:.4f}")
            
            if self.args.logger == 'wandb':
                wandb.log({
                    'dev/f1': dev_results['f1'],
                    'dev/precision': dev_results['precision'],
                    'dev/recall': dev_results['recall'],
                    'epoch': epoch
                })
            
            # Save best model
            if dev_results['f1'] > best_f1:
                best_f1 = dev_results['f1']
                self._save_model(epoch, dev_results['f1'])
                print(f"New best F1: {best_f1:.4f} - Model saved!")
            
            # Early stopping check
            if dev_results['f1'] > self.best_f1_for_early_stopping + self.min_delta:
                # Significant improvement
                self.best_f1_for_early_stopping = dev_results['f1']
                self.early_stopping_counter = 0
            else:
                # No significant improvement
                self.early_stopping_counter += 1
                print(f"Early stopping counter: {self.early_stopping_counter}/{self.patience}")
                
                if self.early_stopping_counter >= self.patience:
                    print(f"\n{'='*50}")
                    print(f"Early stopping triggered after {epoch} epochs!")
                    print(f"Best F1 score: {best_f1:.4f}")
                    print(f"{'='*50}\n")
                    break
        
        # Final evaluation on test set
        print("\n" + "=" * 50)
        print("Training completed! Evaluating on test set...")
        print("=" * 50 + "\n")
        
        # Load best model
        self._load_best_model()
        
        test_results = self.evaluate(self.test_dataloader, "test")
        print(f"\nTest Results:")
        print(f"F1: {test_results['f1']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        
        if self.ood_dataloader is not None:
            print("\nEvaluating on OOD set...")
            ood_results = self.evaluate(self.ood_dataloader, "ood")
            print(f"OOD F1: {ood_results['f1']:.4f}")
        
        if self.args.logger == 'wandb':
            wandb.log({
                'test/f1': test_results['f1'],
                'test/precision': test_results['precision'],
                'test/recall': test_results['recall']
            })
    
    @torch.no_grad()
    def evaluate(self, dataloader, split_name="dev"):
        """Evaluate model on given dataloader"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_compounds = []
        
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels = batch['seq_labels'].to(self.device)
            compounds = batch['compounds']
            
            # Forward returns (predictions, path) during inference
            predictions, _ = self.model(input_ids, attention_mask, seq_labels)
            
            # Collect predictions and labels (excluding padding)
            mask = (seq_labels != -100)
            
            batch_preds = predictions[mask].cpu().numpy()
            batch_labels = seq_labels[mask].cpu().numpy()
            
            all_predictions.extend(batch_preds.tolist())
            all_labels.extend(batch_labels.tolist())
            all_compounds.extend(compounds)
        
        # Calculate metrics
        results = self._calculate_metrics(all_predictions, all_labels)
        
        return results
    
    def _calculate_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # For compound identification: treat No_rel and root as negative class
        no_rel_id = self.label_set.label2id('No_rel')
        root_id = self.label_set.label2id('Comp_root')
        
        # Binary classification: compound vs non-compound
        pred_is_compound = (predictions != no_rel_id) & (predictions != root_id)
        label_is_compound = (labels != no_rel_id) & (labels != root_id)
        
        tp = np.sum(pred_is_compound & label_is_compound)
        fp = np.sum(pred_is_compound & ~label_is_compound)
        fn = np.sum(~pred_is_compound & label_is_compound)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _save_model(self, epoch, f1_score):
        """Save model checkpoint"""
        context_mode = "with_ctx" if self.args.use_context else "no_ctx"
        save_dir = os.path.join(os.getcwd(), 'saved_models', f'necti_{self.args.granularity}_{context_mode}_cross_attn')
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'args': self.args
        }
        
        save_path = os.path.join(save_dir, f'best_model_epoch{epoch}_f1{f1_score:.4f}.pt')
        torch.save(checkpoint_data, save_path)
        
        # Save a "latest" version for easy loading
        latest_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(checkpoint_data, latest_path)
        
        # Track this checkpoint with its F1 score
        self.saved_checkpoints.append((save_path, f1_score))
        
        # Keep only top 3 models by F1 score
        if len(self.saved_checkpoints) > self.args.save_limit:
            # Sort by F1 score (descending)
            self.saved_checkpoints.sort(key=lambda x: x[1], reverse=True)
            # Remove the worst model
            old_checkpoint_path, old_f1 = self.saved_checkpoints.pop()
            if os.path.exists(old_checkpoint_path):
                os.remove(old_checkpoint_path)
                print(f"Removed old checkpoint: {os.path.basename(old_checkpoint_path)} (F1: {old_f1:.4f})")
    
    def _load_best_model(self):
        """Load best model for final evaluation"""
        context_mode = "with_ctx" if self.args.use_context else "no_ctx"
        save_dir = os.path.join(os.getcwd(), 'saved_models', f'necti_{self.args.granularity}_{context_mode}_cross_attn')
        model_path = os.path.join(save_dir, 'best_model.pt')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with F1 {checkpoint['f1_score']:.4f}")
        else:
            print("No saved model found, using current model")


def main():
    """Main entry point"""
    from options import get_args
    args = get_args()
    
    trainer = NeCTITrainerCrossAttn(args)
    trainer.train()


if __name__ == '__main__':
    main()
