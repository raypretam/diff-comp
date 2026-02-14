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
from typing import Dict, List, Tuple
import numpy as np
import json

from data_balancing import BalancedDataLoader, LabelWeightCalculator


class NeCTITrainer:
    """Trainer for nested compound identification"""
    
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
        
        # Get compound-aware parameters
        compound_aware = getattr(self.args, 'compound_aware', False)
        compound_pooling = getattr(self.args, 'compound_pooling', 'mean')
        use_graph_encoder = getattr(self.args, 'use_graph_encoder', False)
        num_gnn_layers = getattr(self.args, 'num_gnn_layers', 2)
        
        # Get contrastive learning parameters
        use_contrastive = getattr(self.args, 'use_contrastive', False)
        contrastive_weight = getattr(self.args, 'contrastive_weight', 0.1)
        contrastive_config = self.args if use_contrastive else None
        
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
            num_labels=len(self.label_set),
            compound_aware=compound_aware,
            compound_pooling=compound_pooling,
            use_graph_encoder=use_graph_encoder,
            num_gnn_layers=num_gnn_layers,
            label_set=self.label_set if compound_aware else None,
            use_contrastive=use_contrastive,
            contrastive_weight=contrastive_weight,
            contrastive_config=contrastive_config
        )
        
        if self.args.logger == "wandb":
            wandb.watch(self.model, log_freq=1000)
        
        # Initialize tokenizer for XLM-R
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        # CRITICAL: Pass label_set to collator so it can mask ROOT:Comp_root
        self.collate_fn = NeCTICollator(
            self.tokenizer, 
            max_length=self.args.max_length, 
            add_lstm=self.args.add_lstm,
            label_set=self.label_set
        )
        
        # Create data loaders (using standard sampling for all splits)
        self.train_dataloader = self._get_dataloader('train', self.args.batch_size, balanced=False)
        self.dev_dataloader = self._get_dataloader('dev', self.args.batch_size, balanced=False)
        self.test_dataloader = self._get_dataloader('test', self.args.batch_size, balanced=False)
        
        # Try to load OOD data if available
        try:
            self.ood_dataloader = self._get_dataloader('ood', self.args.batch_size, balanced=False)
        except FileNotFoundError:
            print("OOD dataset not found, skipping...")
            self.ood_dataloader = None
        
        # Calculate actual total training steps based on dataloader size and epochs
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.args.max_epochs * self.steps_per_epoch
        
        # Calculate warmup steps as a percentage of total steps if not explicitly set or zero
        if self.args.warmup_steps == 0:
            self.warmup_steps = int(0.1 * self.total_steps)  # 10% warmup by default
        else:
            self.warmup_steps = self.args.warmup_steps
        
        print(f"\n{'='*80}")
        print("Training Configuration:")
        print(f"{'='*80}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Total epochs: {self.args.max_epochs}")
        print(f"Total training steps: {self.total_steps}")
        print(f"Warmup steps: {self.warmup_steps} ({100*self.warmup_steps/self.total_steps:.1f}% of total)")
        print(f"{'='*80}\n")
        
        self.optimizer, self.lr_scheduler = \
            self._configure_optimizer_and_scheduler(self.args.optimizer_type, self.args.lr_scheduler_type)
    
    def _get_dataloader(self, mode: str, bsz: int, balanced: bool = False):
        """
        Create dataloader for specified split.
        
        Args:
            mode: 'train', 'dev', 'test', or 'ood'
            bsz: Batch size
            balanced: If True, use balanced sampling (for training data)
        
        Returns:
            If balanced: (dataloader, sampler)
            If not balanced: dataloader
        """
        dataset = NeCTIDataset(self.dataset_path, mode, self.label_set, use_context=self.args.use_context)
        
        if balanced and mode == 'train':
            # Use balanced sampling for training
            dataloader, sampler = BalancedDataLoader.create(
                dataset=dataset,
                label_set=self.label_set,
                batch_size=bsz,
                strategy='weighted',  # Options: 'weighted', 'stratified', 'hard_mining'
                num_workers=self.args.num_workers,
                collate_fn=self.collate_fn
            )
            
            # Print and save weight summary
            sampler.weight_calc.print_weights_summary()
            
            # Save weights to file
            os.makedirs('logs', exist_ok=True)
            with open('logs/label_weights.json', 'w') as f:
                json.dump({
                    str(self.label_set.id2label(label_id)): weight
                    for label_id, weight in sampler.weight_calc.label_weights.items()
                }, f, indent=2)
            
            return dataloader, sampler
        else:
            # Standard dataloader for eval data
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
    
    def _check_gradient_flow(self, step: int):
        """Check if gradients are flowing properly"""
        total_norm = 0.0
        num_params_with_grad = 0
        num_params_without_grad = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    num_params_with_grad += 1
                else:
                    num_params_without_grad += 1
        
        total_norm = total_norm ** 0.5
        
        if step % 50 == 0:  # Log every 50 steps
            print(f"  [Step {step}] Gradient norm: {total_norm:.4f}, Params with grad: {num_params_with_grad}, without: {num_params_without_grad}")
            
        return total_norm
    
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
        
        # Use calculated total_steps and warmup_steps instead of args values
        lr_scheduler = get_lr_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
            num_cycles=self.args.num_cycles if hasattr(self.args, 'num_cycles') else None
        )

        return optimizer, lr_scheduler

    def decode_relative_to_spans(self, token_ids: List[int], relative_tags: List[str]) -> List[Tuple[int, int, str]]:
        """
        Decodes Relative Tags (e.g. '+1:Tatpurusha') into Span Tuples (start, end, label).
        Includes cycle detection to prevent recursion errors.
        """
        # 1. Build Adjacency List (Head -> Children)
        children = {tid: [] for tid in token_ids}
        node_relations = {} 
        
        for i, tag in enumerate(relative_tags):
            if i >= len(token_ids): break
            current_id = token_ids[i]
            
            if tag in ["ROOT:root", "ROOT:Comp_root"]:
                node_relations[current_id] = "Comp_root"
                continue
            
            if ":" not in tag: continue 
                
            try:
                dist_str, rel = tag.split(":", 1)
                dist = int(dist_str)
                head_list_idx = i + dist
                
                if 0 <= head_list_idx < len(token_ids):
                    head_id = token_ids[head_list_idx]
                    
                    # Prevent immediate self-loops
                    if head_id == current_id:
                        continue
                        
                    children[head_id].append(current_id)
                    node_relations[current_id] = rel
            except ValueError:
                continue

        # 2. Extract Spans (Cycle-Safe)
        spans = []
        subtree_ranges = {} 

        def get_subtree_range(node_id, current_path):
            # If already computed, return it
            if node_id in subtree_ranges: 
                return subtree_ranges[node_id]
            
            # CYCLE DETECTION: If we see a node currently in our recursion stack, stop.
            if node_id in current_path:
                return (node_id, node_id)
            
            # Add to current path
            current_path.add(node_id)
            
            indices = [node_id]
            for child in children[node_id]:
                # Recursive call with path tracking
                c_min, c_max = get_subtree_range(child, current_path)
                indices.extend([c_min, c_max])
            
            # Remove from path before returning (backtracking)
            current_path.remove(node_id)
            
            res = (min(indices), max(indices))
            subtree_ranges[node_id] = res
            return res

        for head_id in token_ids:
            kids = children[head_id]
            if not kids: continue
            
            # Collect relations of direct children
            child_rels = [node_relations.get(k, '') for k in kids]
            valid_rels = [r for r in child_rels if r not in ['No_rel', 'root', 'Comp_root', '']]
            
            if valid_rels:
                # Start traversal with an empty path set
                s_min, s_max = get_subtree_range(head_id, set())
                primary_label = valid_rels[0]
                spans.append((s_min, s_max, primary_label))

        return spans
    
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
                words2pieces = batch.get('words2pieces', None)
                if words2pieces is not None:
                    words2pieces = words2pieces.to(self.device)
                
                loss = self.model(input_ids, attention_mask, seq_labels, words2pieces)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Check gradient flow for first epoch to detect issues early
                if epoch == 1 and train_steps % 50 == 0:
                    grad_norm = self._check_gradient_flow(train_steps)
                    if grad_norm == 0:
                        print(f"  WARNING: Zero gradient norm detected at step {train_steps}!")
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                global_step += 1
                
                # Update progress bar with loss and learning rate
                current_lr_bert = self.optimizer.param_groups[0]['lr']
                current_lr_other = self.optimizer.param_groups[1]['lr']
                
                # Prepare postfix dict with main metrics
                postfix_dict = {
                    'loss': f'{loss.item():.4f}',
                    'lr_bert': f'{current_lr_bert:.2e}',
                    'lr_other': f'{current_lr_other:.2e}'
                }
                
                # Add separate loss components if using contrastive learning
                if hasattr(self.model, 'last_losses') and self.model.last_losses:
                    postfix_dict['diff'] = f"{self.model.last_losses['diffusion']:.4f}"
                    postfix_dict['cont'] = f"{self.model.last_losses['contrastive']:.4f}"
                    postfix_dict['t_avg'] = f"{self.model.last_losses['ts_mean']:.0f}"
                
                pbar.set_postfix(postfix_dict)
                
                # Log to wandb (include contrastive loss tracking)
                if self.args.logger == 'wandb':
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/lr_bert': current_lr_bert,
                        'train/lr_other': current_lr_other,
                        'global_step': global_step
                    }
                    # Add separate loss components if available
                    if hasattr(self.model, 'last_losses') and self.model.last_losses:
                        log_dict['train/diffusion_loss'] = self.model.last_losses['diffusion']
                        log_dict['train/contrastive_loss'] = self.model.last_losses['contrastive']
                        log_dict['train/ts_mean'] = self.model.last_losses['ts_mean']
                        # Track contrastive application rate
                        contrastive_active = 1.0 if self.model.last_losses['contrastive'] > 0 else 0.0
                        log_dict['train/contrastive_active'] = contrastive_active
                    wandb.log(log_dict)
            
            avg_train_loss = train_loss / train_steps
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Learning rates - BERT: {self.optimizer.param_groups[0]['lr']:.2e}, Other: {self.optimizer.param_groups[1]['lr']:.2e}")
            
            # Evaluation on dev set
            print("\nEvaluating on dev set...")
            dev_results = self.evaluate(self.dev_dataloader, "dev")
            
            print(f"Dev USS: {dev_results['USS']:.4f}")
            print(f"Dev LSS: {dev_results['LSS']:.4f}")
            print(f"Dev EM: {dev_results['EM']:.4f}")
            
            if self.args.logger == 'wandb':
                wandb.log({
                    'dev/USS': dev_results['USS'],
                    'dev/LSS': dev_results['LSS'],
                    'dev/EM': dev_results['EM'],
                    'epoch': epoch
                })
            
            # Save only best model
            if dev_results['USS'] > best_f1:
                best_f1 = dev_results['USS']
                self._save_model(epoch, dev_results['USS'], is_best=True)
                print(f"New best USS: {best_f1:.4f} - Model saved!")
            
            # Early stopping check
            if dev_results['USS'] > self.best_f1_for_early_stopping + self.min_delta:
                # Significant improvement
                self.best_f1_for_early_stopping = dev_results['USS']
                self.early_stopping_counter = 0
            else:
                # No significant improvement
                self.early_stopping_counter += 1
                print(f"Early stopping counter: {self.early_stopping_counter}/{self.patience}")
                
                if self.early_stopping_counter >= self.patience:
                    print(f"\n{'='*50}")
                    print(f"Early stopping triggered after {epoch} epochs!")
                    print(f"Best USS score: {best_f1:.4f}")
                    print(f"{'='*50}\n")
                    break
            
            # Save checkpoint every 20 epochs
            if epoch % 20 == 0:
                self._save_model(epoch, dev_results['EM'], is_best=False)
                print(f"Checkpoint saved at epoch {epoch}")
        
        # Final evaluation on test set
        print("\n" + "=" * 50)
        print("Training completed! Evaluating on test set...")
        print("=" * 50 + "\n")
        
        # Load best model
        self._load_best_model()
        
        test_results = self.evaluate(self.test_dataloader, "test")
        print(f"\nTest Results:")
        print(f"USS: {test_results['USS']:.4f}")
        print(f"LSS: {test_results['LSS']:.4f}")
        print(f"EM: {test_results['EM']:.4f}")
        
        if self.ood_dataloader is not None:
            print("\nEvaluating on OOD set...")
            ood_results = self.evaluate(self.ood_dataloader, "ood")
            print(f"OOD USS: {ood_results['USS']:.4f}")
        
        if self.args.logger == 'wandb':
            wandb.log({
                'test/USS': test_results['USS'],
                'test/LSS': test_results['LSS'],
                'test/EM': test_results['EM']
            })
    
    def infer_and_add_roots(self, token_indices: List[int], tags: List[str]) -> List[str]:
        """
        Infer ROOT:Comp_root positions from dependency structure and add them.
        Roots are tokens that have NO incoming edges from compound relations.
        """
        # Build set of tokens that are pointed to by compound relations
        has_incoming = set()
        
        for i, tag in enumerate(tags):
            if tag in ["ROOT:root", "ROOT:Comp_root", "No_rel"]:
                continue
            
            if ":" in tag:
                try:
                    dist_str, rel = tag.split(":", 1)
                    # Skip No_rel relations
                    if rel == "No_rel":
                        continue
                    dist = int(dist_str)
                    head_idx = i + dist
                    
                    if 0 <= head_idx < len(token_indices):
                        # This token (head_idx) receives an edge
                        has_incoming.add(head_idx)
                except:
                    pass
        
        # Tokens without incoming compound edges are roots
        result_tags = []
        for i, tag in enumerate(tags):
            # If this token has compound children but no incoming compound edges -> it's a root
            if i not in has_incoming and tag != "No_rel":
                # Check if it has outgoing edges (i.e., it's part of a compound)
                has_outgoing = False
                for j, other_tag in enumerate(tags):
                    if ":" in other_tag:
                        try:
                            dist_str, rel = other_tag.split(":", 1)
                            if rel != "No_rel":
                                dist = int(dist_str)
                                if j + dist == i:
                                    has_outgoing = True
                                    break
                        except:
                            pass
                
                # If it has compound children pointing to it, mark as root
                if has_outgoing:
                    result_tags.append("ROOT:Comp_root")
                else:
                    result_tags.append(tag)
            else:
                result_tags.append(tag)
        
        return result_tags

    @torch.no_grad()
    def evaluate(self, dataloader, split_name="dev"):
        self.model.eval()
        
        # Use the specific Span evaluator
        from span_based_evaluator import SpanBasedEvaluator
        
        true_spans_all = []
        pred_spans_all = []
        
        # Debug counters
        debug_empty_preds = 0
        debug_empty_true = 0
        debug_total = 0
        debug_samples = []
        
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels = batch['seq_labels'].to(self.device)
            
            # Predictions (Bit Diffusion)
            predictions, _ = self.model(input_ids, attention_mask, seq_labels, None)
            
            mask = (seq_labels != -100)
            
            # Process sentence by sentence
            for i in range(input_ids.shape[0]):
                # 1. Get Length
                valid_len = mask[i].sum().item()
                
                # 2. Get Raw Predictions and Ground Truth (IDs)
                pred_ids = predictions[i, :valid_len].cpu().tolist()
                true_ids = seq_labels[i, :valid_len].cpu().tolist()
                
                # 3. Convert IDs -> Relative Tags (e.g. "+1:Tat")
                # Note: -100 (masked) tokens will map back to some label, but we filter by valid_len
                pred_tags = [self.label_set.id2label(pid) if pid != -100 else "No_rel" for pid in pred_ids]
                true_tags = [self.label_set.id2label(tid) if tid != -100 else "No_rel" for tid in true_ids]
                
                # 4. CRITICAL: Infer ROOT:Comp_root positions from dependency structure
                # (since we masked it from training, model can't predict it)
                token_indices = list(range(1, valid_len + 1))
                pred_tags_with_roots = self.infer_and_add_roots(token_indices, pred_tags)
                true_tags_with_roots = self.infer_and_add_roots(token_indices, true_tags)
                
                # 5. Decode Tags -> Spans using dependency structure
                pred_spans_sent = self.decode_relative_to_spans(token_indices, pred_tags_with_roots)
                true_spans_sent = self.decode_relative_to_spans(token_indices, true_tags_with_roots)
                
                true_spans_all.append(true_spans_sent)
                pred_spans_all.append(pred_spans_sent)
                
                # Debug tracking
                debug_total += 1
                if not pred_spans_sent:
                    debug_empty_preds += 1
                if not true_spans_sent:
                    debug_empty_true += 1
                if len(debug_samples) < 3:
                    debug_samples.append({
                        'pred_tags': pred_tags_with_roots,
                        'true_tags': true_tags_with_roots,
                        'pred_spans': pred_spans_sent,
                        'true_spans': true_spans_sent
                    })
        
        # Calculate Metrics using the strict SpanBasedEvaluator
        # This gives you USS (Unlabeled), LSS (Labeled), and EM (Exact Match)
        results = SpanBasedEvaluator.evaluate_batch(true_spans_all, pred_spans_all)
        
        # Debug output
        print(f"\n{'='*60}")
        print(f"DEBUG EVALUATION STATS ({split_name}):")
        print(f"Total sentences: {debug_total}")
        print(f"Empty predictions: {debug_empty_preds} ({100*debug_empty_preds/debug_total:.1f}%)")
        print(f"Empty ground truth: {debug_empty_true} ({100*debug_empty_true/debug_total:.1f}%)")
        print(f"\nSample predictions (first 3):")
        for idx, sample in enumerate(debug_samples):
            print(f"\n  Sample {idx+1}:")
            print(f"    Pred tags: {sample['pred_tags'][:10]}...")
            print(f"    True tags: {sample['true_tags'][:10]}...")
            print(f"    Pred spans: {sample['pred_spans']}")
            print(f"    True spans: {sample['true_spans']}")
        print(f"{'='*60}\n")
        
        return results
    
    # def _calculate_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
    #     """Calculate precision, recall, and F1 score"""
    #     predictions = np.array(predictions)
    #     labels = np.array(labels)
        
    #     # For compound identification: treat No_rel and root as negative class
    #     no_rel_id = self.label_set.label2id('No_rel')
    #     root_id = self.label_set.label2id('root')
        
    #     # Binary classification: compound vs non-compound
    #     pred_is_compound = (predictions != no_rel_id) & (predictions != root_id)
    #     label_is_compound = (labels != no_rel_id) & (labels != root_id)
        
    #     tp = np.sum(pred_is_compound & label_is_compound)
    #     fp = np.sum(pred_is_compound & ~label_is_compound)
    #     fn = np.sum(~pred_is_compound & label_is_compound)
        
    #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    #     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
    #     return {
    #         'precision': precision,
    #         'recall': recall,
    #         'f1': f1
    #     }
    
    def _save_model(self, epoch, f1_score, is_best=False):
        """Save model checkpoint"""
        context_mode = "with_ctx" if self.args.use_context else "no_ctx"
        save_dir = os.path.join(os.getcwd(), 'saved_models', f'necti_{self.args.granularity}_{context_mode}')
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'args': self.args
        }
        
        if is_best:
            save_path = os.path.join(save_dir, f'best_model_epoch{epoch}_f1{f1_score:.4f}.pt')
            # Save a "latest best" version for easy loading
            latest_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint_data, save_path)
            torch.save(checkpoint_data, latest_path)
        else:
            # Save periodic checkpoint
            save_path = os.path.join(save_dir, f'checkpoint_epoch{epoch}_f1{f1_score:.4f}.pt')
            torch.save(checkpoint_data, save_path)
            
            # Track this checkpoint
            self.saved_checkpoints.append(save_path)
            
            # Remove old checkpoints if limit exceeded
            if len(self.saved_checkpoints) > self.args.save_limit:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    print(f"Removed old checkpoint: {os.path.basename(old_checkpoint)}")
    
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