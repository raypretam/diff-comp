"""
Training script for nested compound identification using DiffusionSL
Adapted for DepNeCTI data with XLM-R encoder
"""

import os
from argparse import Namespace
import torch
from models.ddim_bitdit import BitDit
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


class NeCTITrainer:
    """Trainer for nested compound identification"""
    
    def __init__(self, args: Namespace):
        self.args = args
        self._print_hyperparameters()
        
        # Set default value for use_context if not present
        if not hasattr(self.args, 'use_context'):
            self.args.use_context = False
        
        context_mode = "with_ctx" if self.args.use_context else "no_ctx"
        
        if self.args.logger == 'wandb':
            run_name = f"necti-{args.granularity}-{context_mode}--lr_bert_{args.lr_bert}--lr_other_{args.lr_other}--epochs_{args.max_epochs}"
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
        
        # Initialize model with XLM-R backbone
        self.model = BitDit(
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
        )
        
        if self.args.logger == "wandb":
            wandb.watch(self.model, log_freq=1000)
        
        # Initialize tokenizer for XLM-R
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        self.collate_fn = NeCTICollator(self.tokenizer, max_length=self.args.max_length, add_lstm=self.args.add_lstm)
        
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
        hparams.title = 'Hyper Parameters'
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
        print("Starting training...")
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
                words2pieces = batch.get('words2pieces', None)  # Get words2pieces if present
                if words2pieces is not None:
                    words2pieces = words2pieces.to(self.device)
                
                # Forward pass with optional words2pieces
                loss = self.model(input_ids, attention_mask, seq_labels, words2pieces)
                
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
            words2pieces = batch.get('words2pieces', None)
            if words2pieces is not None:
                words2pieces = words2pieces.to(self.device)
            
            compounds = batch['compounds']
            
            with torch.no_grad():
                # Forward pass with optional words2pieces
                predictions, path_x = self.model(input_ids, attention_mask, seq_labels, words2pieces)
            
            # When LSTM is used, we need to convert seq_labels to word-level to match predictions
            if self.args.add_lstm and words2pieces is not None:
                # Convert piece-level labels to word-level
                bsz, num_words, max_pieces = words2pieces.shape
                word_seq_labels = torch.full((bsz, num_words), -100, dtype=seq_labels.dtype, device=seq_labels.device)
                
                valid_pieces_mask = words2pieces.gt(0)
                for b in range(bsz):
                    for w in range(num_words):
                        valid_indices = words2pieces[b, w][valid_pieces_mask[b, w]]
                        if len(valid_indices) > 0:
                            seq_len = seq_labels.shape[1]
                            valid_indices = valid_indices.clamp(0, seq_len - 1)
                            # Get label for first valid piece of this word
                            first_piece_label = seq_labels[b, valid_indices[0]]
                            word_seq_labels[b, w] = first_piece_label
                
                # Use word-level labels for evaluation
                eval_labels = word_seq_labels
            else:
                eval_labels = seq_labels
            
            # Collect predictions and labels (excluding padding)
            mask = (eval_labels != -100)
            
            batch_preds = predictions[mask].cpu().numpy()
            batch_labels = eval_labels[mask].cpu().numpy()
            
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
        root_id = self.label_set.label2id('root')
        
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
        save_dir = os.path.join(os.getcwd(), 'saved_models', f'necti_{self.args.granularity}_{context_mode}')
        os.makedirs(save_dir, exist_ok=True)
        
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
        context_mode = "with_ctx" if self.args.use_context else "no_ctx"
        save_dir = os.path.join(os.getcwd(), 'saved_models', f'necti_{self.args.granularity}_{context_mode}')
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
    
    trainer = NeCTITrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
