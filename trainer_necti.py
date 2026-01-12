"""
Training script for nested compound identification using DiffusionSL
Adapted for DepNeCTI data with XLM-R encoder — UPDATED FOR BIT_DIM SUPPORT
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
    """Trainer for nested compound identification (UPDATED FOR BIT_DIM)"""
    
    def __init__(self, args: Namespace):
        self.args = args
        self._print_hyperparameters()

        # Ensure bit_dim exists (new argument)
        if not hasattr(self.args, "bit_dim"):
            self.args.bit_dim = 16

        # CONTEXT
        if not hasattr(self.args, 'use_context'):
            self.args.use_context = False
        
        context_mode = "with_ctx" if self.args.use_context else "no_ctx"
        
        # wandb
        if self.args.logger == 'wandb':
            run_name = (
                f"necti-{args.granularity}-{context_mode}"
                f"--bitdim_{args.bit_dim}"
                f"--lr_bert_{args.lr_bert}--lr_other_{args.lr_other}"
                f"--epochs_{args.max_epochs}"
            )
            wandb.init(project="DiffusionSL-NeCTI", name=run_name)
            wandb.config.update(self.args)
            wandb.define_metric("f1", summary="max")
        
        self.device = self._configure_device()
        
        # Load label set
        self.dataset_path = self.args.data_path
        self.label_set = NeCTILabelSet(
            data_path=self.dataset_path,
            granularity=self.args.granularity,
            use_context=self.args.use_context
        )
        
        if self.args.num_classes != len(self.label_set):
            print(f"Adjusting num_classes to: {len(self.label_set)}")
            self.args.num_classes = len(self.label_set)
        

        # *** BITDIT v2 INITIALIZATION ***
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
            num_labels=len(self.label_set),
            bit_dim=self.args.bit_dim
        )

        if self.args.logger == "wandb":
            wandb.watch(self.model, log_freq=1000)
        
        # Tokenizer + collator
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        self.collate_fn = NeCTICollator(
            self.tokenizer,
            max_length=self.args.max_length,
            add_lstm=self.args.add_lstm
        )
        
        # Dataloaders
        self.train_dataloader = self._get_dataloader('train', self.args.batch_size)
        self.dev_dataloader = self._get_dataloader('dev', self.args.batch_size)
        self.test_dataloader = self._get_dataloader('test', self.args.batch_size)

        try:
            self.ood_dataloader = self._get_dataloader('ood', self.args.batch_size)
        except:
            print("No OOD dataset found")
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
        self._print_num_parameters()
        print("\n======== Starting Training ========\n")
        
        best_f1 = 0.0
        global_step = 0
        
        for epoch in range(1, self.args.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.max_epochs}")
            print("-" * 50)

            # ---------------------- TRAIN ----------------------
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            for batch in tqdm(self.train_dataloader, desc="Training"):
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_labels = batch['seq_labels'].to(self.device)

                words2pieces = batch.get('words2pieces')
                if words2pieces is not None:
                    words2pieces = words2pieces.to(self.device)

                # BitDit v2 returns LOSS directly
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    seq_labels=seq_labels,
                    words2pieces=words2pieces
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()

                train_loss += loss.item()
                train_steps += 1
                global_step += 1

                if self.args.logger == 'wandb':
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr_bert': self.optimizer.param_groups[0]['lr'],
                        'train/lr_other': self.optimizer.param_groups[1]['lr'],
                        'global_step': global_step
                    })
            
            print(f"Avg Training Loss: {train_loss / train_steps:.4f}")
            
            # ---------------------- DEV EVAL ----------------------
            dev_results = self.evaluate(self.dev_dataloader, "dev")
            print(f"DEV F1 = {dev_results['f1']:.4f}")

            if self.args.logger == 'wandb':
                wandb.log({
                    'dev/f1': dev_results['f1'],
                    'dev/precision': dev_results['precision'],
                    'dev/recall': dev_results['recall'],
                    'epoch': epoch
                })
            
            # save best
            if dev_results['f1'] > best_f1:
                best_f1 = dev_results['f1']
                self._save_model(epoch, best_f1)
                print("New BEST model saved!")

        # ---------------------- FINAL TEST ----------------------
        print("\n======== FINAL TEST ========\n")
        self._load_best_model()

        test_results = self.evaluate(self.test_dataloader, "test")
        print("Test F1 =", test_results['f1'])

        if self.ood_dataloader:
            ood_results = self.evaluate(self.ood_dataloader, "ood")
            print("OOD F1 =", ood_results['f1'])
        
        if self.args.logger == 'wandb':
            wandb.log({
                'test/f1': test_results['f1'],
                'test/precision': test_results['precision'],
                'test/recall': test_results['recall']
            })
    
    @torch.no_grad()
    def evaluate(self, dataloader, split_name="dev"):

        self.model.eval()

        all_preds = []
        all_labels = []

        # majority voting enabled only for test/ood
        use_voting = split_name in ["test", "ood"]
        votes = 5 if use_voting else 1

        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels = batch['seq_labels'].to(self.device)

            words2pieces = batch.get("words2pieces")
            if words2pieces is not None:
                words2pieces = words2pieces.to(self.device)

            # get BERT features once (optimization)
            bert_features = self.model.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

            shape = (input_ids.size(0), input_ids.size(1))

            # BITDIT v2 SAMPLING
            if use_voting:
                # majority voting
                vote_pool = []
                for _ in range(votes):
                    preds, _ = self.model.sample(shape, bert_features, attention_mask)
                    vote_pool.append(preds.unsqueeze(0))  # [1, bsz, seq]
                vote_pool = torch.cat(vote_pool, dim=0)     # [votes, bsz, seq]
                preds = vote_pool.mode(dim=0).values        # majority vote
            else:
                # single pass
                preds, _ = self.model.sample(shape, bert_features, attention_mask)

            # MASKING & METRICS
            mask = seq_labels != -100
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(seq_labels[mask].cpu().tolist())

        return self._calculate_metrics(all_preds, all_labels)
    
    def _calculate_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score for compound identification"""
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