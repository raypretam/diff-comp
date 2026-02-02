"""
Inference script for nested compound identification using DiffusionSL
Loads best trained model and evaluates on test/OOD datasets
"""

import os
import argparse
import torch
from models.ddim_bitdit import BitDit
from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, List
import numpy as np
import json
import time


class NeCTIInference:
    """Inference for nested compound identification"""
    
    def __init__(self, model_path: str, data_path: str, granularity: str, use_context: bool = False, device: str = 'cuda'):
        """
        Initialize inference
        
        Args:
            model_path: Path to saved model checkpoint
            data_path: Path to DepNeCTI data directory
            granularity: 'Coarse' or 'Finegrain'
            use_context: Whether to use 'With Context' data
            device: Device to run inference on
        """
        self.model_path = model_path
        self.data_path = data_path
        self.granularity = granularity
        self.use_context = use_context
        
        # Setup device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        # Load checkpoint
        print(f"\nLoading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.args = checkpoint['args']
        
        print(f"Model trained for {checkpoint['epoch']} epochs")
        print(f"Best dev F1: {checkpoint['f1_score']:.4f}")
        
        # Load label set
        self.label_set = NeCTILabelSet(
            data_path=self.data_path,
            granularity=self.granularity,
            use_context=self.use_context
        )
        context_mode = "With Context" if self.use_context else "Without Context"
        print(f"\nUsing data mode: {context_mode}")
        print(f"Number of classes: {len(self.label_set)}")
        
        # Initialize model
        self.model = BitDit(
            device=self.device,
            num_classes=len(self.label_set),
            backbone=self.args.backbone,
            time_steps=self.args.time_steps,
            sampling_steps=self.args.sampling_steps,
            noise_schedule=self.args.noise_schedule,
            ddim_sampling_eta=self.args.ddim_sampling_eta,
            self_condition=self.args.self_condition,
            snr_scale=self.args.snr_scale,
            dataset=f"necti_{self.granularity}",
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
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"\nModel loaded successfully!")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        self.collate_fn = NeCTICollator(self.tokenizer, max_length=self.args.max_length)
    
    def _get_dataloader(self, mode: str, batch_size: int = 8):
        """Create dataloader for specified split"""
        dataset = NeCTIDataset(self.data_path, mode, self.label_set, use_context=self.use_context)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        print(f"Loaded {len(dataset)} {mode} samples")
        return dataloader
    
    @torch.no_grad()
    def evaluate(self, dataloader, split_name="test", save_predictions=False, 
                use_mbr=False, num_mbr_samples=5, mbr_temperature=0.8, mbr_metric='uss'):
        """
        Evaluate model on given dataloader
        
        Args:
            dataloader: DataLoader for evaluation
            split_name: Name of the split (for logging)
            save_predictions: Whether to save predictions to file
            use_mbr: Whether to use MBR decoding
            num_mbr_samples: Number of samples for MBR
            mbr_temperature: Temperature for MBR sampling
            mbr_metric: Metric for MBR ('uss', 'lss', or 'voting')
        
        Returns:
            Dictionary with metrics and predictions
        """
        print(f"\n{'='*50}")
        print(f"Evaluating on {split_name} set...")
        if use_mbr:
            print(f"Using MBR decoding: num_samples={num_mbr_samples}, temperature={mbr_temperature}, metric={mbr_metric}")
        print(f"{'='*50}\n")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        detailed_predictions = []
        all_pred_relations = []
        all_true_relations = []
        
        # Reset timing counters at start of evaluation
        if use_mbr:
            self._mbr_timing_counter = 0
            self._selection_timings = {
                'conversion': [],
                'scoring': [],
                'total': []
            }
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels = batch['seq_labels'].to(self.device)
            words2pieces = batch.get('words2pieces', None)
            if words2pieces is not None:
                words2pieces = words2pieces.to(self.device)
            
            # Generate predictions with MBR if enabled
            if use_mbr:
                predictions = self._mbr_decode(
                    input_ids, attention_mask, seq_labels, words2pieces,
                    num_samples=num_mbr_samples,
                    temperature=mbr_temperature,
                    metric=mbr_metric
                )
            else:
                # Standard inference
                if self.model.add_lstm:
                    predictions, _ = self.model(input_ids, attention_mask, seq_labels, words2pieces)
                else:
                    predictions, _ = self.model(input_ids, attention_mask, seq_labels)
            
            # Process batch
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # Get valid positions (non-padding)
                valid_mask = (seq_labels[i] != -100)
                
                pred_seq = predictions[i][valid_mask].cpu().numpy()
                label_seq = seq_labels[i][valid_mask].cpu().numpy()
                
                all_predictions.extend(pred_seq.tolist())
                all_labels.extend(label_seq.tolist())
                
                # Convert to relations format for USS/LSS calculation
                pred_labels_decoded = [self.label_set.id2label(p) for p in pred_seq]
                true_labels_decoded = [self.label_set.id2label(l) for l in label_seq]
                
                # Create relations: [token_idx, head_idx, relation_type]
                pred_relations = []
                true_relations = []
                for j, (pred_label, true_label) in enumerate(zip(pred_labels_decoded, true_labels_decoded)):
                    token_idx = j + 1  # 1-indexed
                    pred_relations.append([token_idx, 0, pred_label])
                    true_relations.append([token_idx, 0, true_label])
                
                all_pred_relations.append(pred_relations)
                all_true_relations.append(true_relations)
                
                # Store detailed predictions if requested
                if save_predictions:
                    # Get original sentence tokens directly from batch
                    original_tokens = batch['sentences'][i]
                    original_text = ' '.join(original_tokens)
                    
                    # Use original tokens aligned with predictions
                    # Predictions are already aligned with original tokens (one label per token)
                    
                    # Annotated sentences: token/LABEL style
                    annotated_pred = " ".join([f"{t}/{l}" for t, l in zip(original_tokens, pred_labels_decoded)])
                    annotated_true = " ".join([f"{t}/{l}" for t, l in zip(original_tokens, true_labels_decoded)])
                    
                    detailed_predictions.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'original_text': original_text,
                        'original_tokens': original_tokens,
                        'predictions': pred_labels_decoded,
                        'true_labels': true_labels_decoded,
                        'annotated_pred': annotated_pred,
                        'annotated_true': annotated_true,
                        'comparison': {
                            'prediction': pred_labels_decoded,
                            'ground_truth': true_labels_decoded,
                            'tokens': original_tokens,
                            'matches': [pred == true for pred, true in zip(pred_labels_decoded, true_labels_decoded)]
                        }
                    })
        
        # Calculate metrics
        results = self._calculate_metrics(all_predictions, all_labels, all_pred_relations, all_true_relations)
        
        # Print results
        print(f"\n{split_name.upper()} Results:")
        print("-" * 50)
        print(f"Precision:    {results['precision']:.4f}")
        print(f"Recall:       {results['recall']:.4f}")
        print(f"F1 Score:     {results['f1']:.4f}")
        print(f"Accuracy:     {results['accuracy']:.4f}")
        print(f"USS (F1):     {results['uss']:.4f}")
        print(f"LSS (F1):     {results['lss']:.4f}")
        print(f"LSS Prec:     {results['lss_precision']:.4f}")
        print(f"LSS Recall:   {results['lss_recall']:.4f}")
        print(f"Exact Match:  {results['exact_match']:.4f}")
        print("-" * 50)
        
        # Print per-label metrics for diagnosis
        if 'per_label_metrics' in results and results['per_label_metrics']:
            print(f"\nPER-LABEL METRICS (sorted by F1):")
            print("-" * 80)
            sorted_labels = sorted(
                results['per_label_metrics'].items(),
                key=lambda x: x[1]['f1'],
                reverse=False  # Low F1 first to identify problem areas
            )
            
            for label, metrics in sorted_labels:
                print(f"{label:20s} | F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f} | "
                      f"Rec: {metrics['recall']:.4f} | TP: {metrics['tp']:4d} | "
                      f"FP: {metrics['fp']:4d} | FN: {metrics['fn']:4d} | Total: {metrics['total_occurrences']:4d}")
            print("-" * 80)
        
        # Add detailed predictions to results
        if save_predictions:
            results['detailed_predictions'] = detailed_predictions
        
        # Add MBR config if used
        if use_mbr:
            results['used_mbr'] = True
            results['mbr_config'] = {
                'num_samples': num_mbr_samples,
                'temperature': mbr_temperature,
                'metric': mbr_metric
            }
        
        # Print final MBR timing summary
        if use_mbr and hasattr(self, '_selection_timings'):
            print(f"\n{'='*50}")
            print("MBR TIMING SUMMARY")
            print(f"{'='*50}")
            avg_conversion = np.mean(self._selection_timings['conversion'])
            avg_scoring = np.mean(self._selection_timings['scoring'])
            avg_total = np.mean(self._selection_timings['total'])
            total_selection_time = sum(self._selection_timings['total'])
            print(f"Per-instance averages:")
            print(f"  Conversion to relations: {avg_conversion:.4f}s ({avg_conversion/avg_total*100:.1f}%)")
            print(f"  Pairwise scoring:        {avg_scoring:.4f}s ({avg_scoring/avg_total*100:.1f}%)")
            print(f"  Total selection:         {avg_total:.4f}s")
            print(f"Total selection time:      {total_selection_time:.2f}s")
            print(f"Number of samples:         {num_mbr_samples}")
            print(f"Pairwise comparisons/inst: {num_mbr_samples * (num_mbr_samples - 1)}")
            print(f"{'='*50}\n")
        
        return results
    
    def _calculate_metrics(self, predictions: List[int], labels: List[int], 
                          all_pred_relations: List[List], all_true_relations: List[List]) -> Dict[str, float]:
        """Calculate precision, recall, F1, accuracy, USS, and LSS"""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Overall accuracy
        accuracy = np.mean(predictions == labels)
        
        # For compound identification: treat No_rel and root as negative class
        no_rel_id = self.label_set.label2id('No_rel')
        root_id = self.label_set.label2id('root')
        
        # Binary classification: compound vs non-compound
        pred_is_compound = (predictions != no_rel_id) & (predictions != root_id)
        label_is_compound = (labels != no_rel_id) & (labels != root_id)
        
        tp = np.sum(pred_is_compound & label_is_compound)
        fp = np.sum(pred_is_compound & ~label_is_compound)
        fn = np.sum(~pred_is_compound & label_is_compound)
        tn = np.sum(~pred_is_compound & ~label_is_compound)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate USS and LSS
        uss = self._calculate_uss(all_true_relations, all_pred_relations)
        lss_metrics = self._calculate_lss(all_true_relations, all_pred_relations)
        
        # Calculate per-label metrics for diagnosis
        label_metrics = self._calculate_per_label_metrics(predictions, labels)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'uss': uss,
            'lss': lss_metrics['f1'],
            'lss_precision': lss_metrics['precision'],
            'lss_recall': lss_metrics['recall'],
            'exact_match': lss_metrics['exact_match'],
            'per_label_metrics': label_metrics
        }
    
    def _calculate_uss(self, true_relations: List[List], pred_relations: List[List]) -> float:
        """Calculate Unlabeled Span Score (USS) - only on relations with labels != 'No_rel'"""
        correct = 0
        true_count = 0
        pred_count = 0
        
        for true_rels, pred_rels in zip(true_relations, pred_relations):
            # Filter: only consider relations where label != 'No_rel'
            true_filtered = [r for r in true_rels if r[2] != 'No_rel']
            pred_filtered = [r for r in pred_rels if r[2] != 'No_rel']
            
            # Compare spans: [token_idx, head_idx]
            for pred_rel in pred_filtered:
                pred_span = [pred_rel[0], pred_rel[1]]
                for true_rel in true_filtered:
                    true_span = [true_rel[0], true_rel[1]]
                    if pred_span == true_span:
                        correct += 1
                        break
            
            true_count += len(true_filtered)
            pred_count += len(pred_filtered)
        
        p = correct / pred_count if pred_count != 0 else 0
        r = correct / true_count if true_count != 0 else 0
        f1 = 2 * p * r / (p + r) if p != 0 and r != 0 else 0
        
        return f1
    
    def _calculate_per_label_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Dict]:
        """
        Calculate per-label precision, recall, and F1 to identify problematic labels.
        This helps diagnose which fine-grain labels cause exact_match failures.
        """
        per_label = {}
        
        for label_id in range(len(self.label_set)):
            label_str = self.label_set.id2label(label_id)
            
            # True positives, false positives, false negatives for this label
            tp = np.sum((predictions == label_id) & (labels == label_id))
            fp = np.sum((predictions == label_id) & (labels != label_id))
            fn = np.sum((predictions != label_id) & (labels == label_id))
            
            # Skip labels with no occurrences
            if tp + fp + fn == 0:
                continue
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_label[label_str] = {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'total_occurrences': int(tp + fn)
            }
        
        return per_label
    
    def _calculate_lss(self, true_relations: List[List], pred_relations: List[List]) -> Dict[str, float]:
        """Calculate Labeled Span Score (LSS) and Exact Match - matching Eval_USS_LSS.py logic"""
        correct = 0
        predict_count = 0
        true_count = 0
        em = 0
        tot_comps = 0
        
        for true_rels, pred_rels in zip(true_relations, pred_relations):
            # Filter: only relations with label != 'No_rel' and != 'root'
            true_rels_filtered = [r for r in true_rels if r[2] != 'No_rel' and r[2] != 'root']
            pred_rels_filtered = [r for r in pred_rels if r[2] != 'No_rel' and r[2] != 'root']
            
            # Convert to string format for compound extraction: "token_idx,head_idx,label"
            tr_copy = [','.join(map(str, lst)) for lst in true_rels_filtered]
            pr_copy = [','.join(map(str, lst)) for lst in pred_rels_filtered]
            
            # Extract compounds (groups ending with Comp_root)
            tr_comps = self._comps_from_relations(tr_copy)
            pr_comps = self._comps_from_relations(pr_copy)
            
            # Exact match count
            for comp in pr_comps:
                if comp in tr_comps:
                    em += 1
            tot_comps += len(tr_comps)
            
            # Count matching relations (LSS)
            for rel in pred_rels_filtered:
                if rel in true_rels_filtered:
                    correct += 1
            
            predict_count += len(pred_rels_filtered)
            true_count += len(true_rels_filtered)
        
        # Calculate precision, recall, F1
        if correct == 0:
            p = 0
            r = 0
        else:
            p = correct / predict_count if predict_count != 0 else 0
            r = correct / true_count if true_count != 0 else 0
        
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        
        em_per = em / tot_comps if tot_comps != 0 else 0
        
        return {
            'precision': p,
            'recall': r,
            'f1': f1,
            'exact_match': em_per
        }
    
    def _comps_from_relations(self, relations: List[str]) -> List[List[str]]:
        """Extract compounds from relations list"""
        lst = []
        nested_comp = []
        for rel in relations:
            if 'Comp_root' in rel:
                lst.append(rel)
                nested_comp.append(lst)
                lst = []
            else:
                lst.append(rel)
        return nested_comp

    # Add helper to merge subword tokens into words and align labels
    def _merge_subword_tokens(self, tokens: List[str], labels: List[str]):
        """
        Merge tokenizer subword pieces (e.g., '##ing') into full words.
        Returns (merged_tokens, merged_labels) where each merged token is aligned
        with the label from its first piece.
        """
        merged_tokens = []
        merged_labels = []
        for tok, lab in zip(tokens, labels):
            if tok.startswith("##"):
                # attach to previous token
                if merged_tokens:
                    merged_tokens[-1] = merged_tokens[-1] + tok[2:]
                else:
                    # fallback if first token starts with ##
                    merged_tokens.append(tok[2:])
                    merged_labels.append(lab)
            else:
                merged_tokens.append(tok)
                merged_labels.append(lab)
        return merged_tokens, merged_labels

    def run_inference(self, splits=['test'], batch_size=8, save_predictions=False, output_dir=None,
                  use_mbr=False, num_mbr_samples=5, mbr_temperature=0.8, mbr_metric='uss'):
        """
        Run inference on specified splits
        
        Args:
            splits: List of splits to evaluate on ('test', 'dev', 'ood')
            batch_size: Batch size for inference
            save_predictions: Whether to save predictions to file
            output_dir: Directory to save predictions (if save_predictions=True)
            use_mbr: Whether to use MBR decoding
            num_mbr_samples: Number of samples for MBR
            mbr_temperature: Temperature for MBR sampling
            mbr_metric: Metric for MBR selection ('uss', 'lss', or 'voting')
        
        Returns:
            Dictionary mapping split names to results
        """
        results = {}
        
        for split in splits:
            try:
                dataloader = self._get_dataloader(split, batch_size)
                split_results = self.evaluate(
                    dataloader, split, save_predictions,
                    use_mbr=use_mbr,
                    num_mbr_samples=num_mbr_samples,
                    mbr_temperature=mbr_temperature,
                    mbr_metric=mbr_metric
                )
                results[split] = split_results
                
                # Save predictions if requested
                if save_predictions and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f'{split}_predictions.json')
                    
                    # Remove detailed predictions before saving metrics
                    detailed_preds = split_results.pop('detailed_predictions', [])
                    
                    # Save metrics
                    with open(output_file.replace('_predictions.json', '_metrics.json'), 'w') as f:
                        json.dump(split_results, f, indent=2)
                    
                    # Save detailed predictions
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(detailed_preds, f, indent=2, ensure_ascii=False)
                    
                    print(f"\nPredictions saved to: {output_file}")
                    print(f"Metrics saved to: {output_file.replace('_predictions.json', '_metrics.json')}")
                    
            except FileNotFoundError:
                print(f"\n{split.upper()} dataset not found, skipping...")
                continue
        
        return results
    
    def _mbr_decode(self, input_ids, attention_mask, seq_labels, words2pieces,
                    num_samples=5, temperature=1.0, metric='uss', max_parallel_samples=10):
        """
        Minimum Bayes Risk decoding with structure-aware scoring (parallelized with chunking)
        """
        start_time = time.time()
        
        batch_size = input_ids.size(0)
        seq_len = seq_labels.size(1)
        
        # If num_samples is too large, chunk it
        generation_start = time.time()
        if num_samples > max_parallel_samples:
            chunks = [max_parallel_samples] * (num_samples // max_parallel_samples)
            if num_samples % max_parallel_samples:
                chunks.append(num_samples % max_parallel_samples)
            
            all_samples_list = []
            for chunk_size in chunks:
                chunk_samples = self._generate_parallel_samples(
                    input_ids, attention_mask, seq_labels, words2pieces, chunk_size
                )
                all_samples_list.append(chunk_samples)
            
            # Concatenate along sample dimension: [batch_size, num_samples, seq_len]
            all_samples = torch.cat(all_samples_list, dim=1)
        else:
            all_samples = self._generate_parallel_samples(
                input_ids, attention_mask, seq_labels, words2pieces, num_samples
            )
        
        generation_time = time.time() - generation_start
        
        # Select best sample for each instance
        selection_start = time.time()
        best_predictions = []
        for b in range(batch_size):
            valid_mask = (seq_labels[b] != -100)
            instance_samples = [all_samples[b, s, valid_mask].cpu().numpy() 
                              for s in range(num_samples)]
            
            if metric == 'voting':
                best_pred = self._voting_decode(instance_samples)
            else:
                best_pred = self._select_mbr_sample(instance_samples, metric)
            
            full_pred = seq_labels[b].clone()
            full_pred[valid_mask] = torch.tensor(best_pred, device=seq_labels.device)
            best_predictions.append(full_pred)
        
        selection_time = time.time() - selection_start
        total_time = time.time() - start_time
        
        # Print timing info (can be removed or made optional later)
        if hasattr(self, '_mbr_timing_counter'):
            self._mbr_timing_counter += 1
        else:
            self._mbr_timing_counter = 1
        
        # Print every 10 batches to avoid spam
        if self._mbr_timing_counter % 10 == 0:
            print(f"\n[MBR Timing] Batch {self._mbr_timing_counter}:")
            print(f"  Generation: {generation_time:.3f}s ({generation_time/total_time*100:.1f}%)")
            print(f"  Selection:  {selection_time:.3f}s ({selection_time/total_time*100:.1f}%)")
            print(f"  Total:      {total_time:.3f}s")
        
        return torch.stack(best_predictions)
    
    def _generate_parallel_samples(self, input_ids, attention_mask, seq_labels, 
                                   words2pieces, num_samples):
        """Helper to generate samples in parallel"""
        batch_size = input_ids.size(0)
        seq_len = seq_labels.size(1)
        
        # Expand inputs
        expanded_input_ids = input_ids.repeat_interleave(num_samples, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(num_samples, dim=0)
        expanded_seq_labels = seq_labels.repeat_interleave(num_samples, dim=0)
        
        if words2pieces is not None:
            expanded_words2pieces = words2pieces.repeat_interleave(num_samples, dim=0)
        else:
            expanded_words2pieces = None
        
        with torch.no_grad():
            if self.model.add_lstm:
                all_preds, _ = self.model(expanded_input_ids, expanded_attention_mask,
                                         expanded_seq_labels, expanded_words2pieces)
            else:
                all_preds, _ = self.model(expanded_input_ids, expanded_attention_mask,
                                         expanded_seq_labels)
        
        return all_preds.view(batch_size, num_samples, seq_len)

    def _voting_decode(self, samples):
        """Majority voting across samples"""
        samples_array = np.array(samples)  # [num_samples, seq_len]
        # Get mode (most frequent) for each position
        from scipy import stats
        voted, _ = stats.mode(samples_array, axis=0, keepdims=False)
        return voted
    
    def _select_mbr_sample(self, samples, metric='uss'):
        """
        Select sample with minimum Bayes risk based on structure
        
        Args:
            samples: List of predicted sequences [num_samples, seq_len]
            metric: 'uss' for unlabeled span score, 'lss' for labeled span score
        
        Returns:
            Best sample (numpy array)
        """
        start_time = time.time()
        num_samples = len(samples)
        
        # Convert samples to relations format
        conversion_start = time.time()
        all_relations = []
        for sample in samples:
            relations = []
            for j, label_id in enumerate(sample):
                label_str = self.label_set.id2label(int(label_id))
                relations.append([j + 1, 0, label_str])
            all_relations.append(relations)
        conversion_time = time.time() - conversion_start
        
        # Calculate pairwise scores
        scoring_start = time.time()
        risk_scores = np.zeros(num_samples)
        
        for i in range(num_samples):
            total_score = 0.0
            for j in range(num_samples):
                if i == j:
                    continue
                
                # Calculate agreement score
                if metric == 'uss':
                    score = self._calculate_uss([all_relations[i]], [all_relations[j]])
                else:  # lss
                    score = self._calculate_lss([all_relations[i]], [all_relations[j]])['f1']
                
                # Risk is inverse of agreement (we want high agreement)
                total_score += (1.0 - score)
            
            risk_scores[i] = total_score / (num_samples - 1)
        
        scoring_time = time.time() - scoring_start
        
        # Select sample with minimum risk
        best_idx = np.argmin(risk_scores)
        
        total_time = time.time() - start_time
        
        # Store timing info for reporting
        if not hasattr(self, '_selection_timings'):
            self._selection_timings = {
                'conversion': [],
                'scoring': [],
                'total': []
            }
        
        self._selection_timings['conversion'].append(conversion_time)
        self._selection_timings['scoring'].append(scoring_time)
        self._selection_timings['total'].append(total_time)
        
        # Print aggregate timing every 50 samples
        if len(self._selection_timings['total']) % 50 == 0:
            avg_conversion = np.mean(self._selection_timings['conversion'][-50:])
            avg_scoring = np.mean(self._selection_timings['scoring'][-50:])
            avg_total = np.mean(self._selection_timings['total'][-50:])
            print(f"\n[MBR Selection Timing - Last 50 samples]:")
            print(f"  Avg Conversion: {avg_conversion:.4f}s ({avg_conversion/avg_total*100:.1f}%)")
            print(f"  Avg Scoring:    {avg_scoring:.4f}s ({avg_scoring/avg_total*100:.1f}%)")
            print(f"  Avg Total:      {avg_total:.4f}s")
        
        return samples[best_idx]


def main():
    parser = argparse.ArgumentParser(description='Inference for NeCTI compound identification')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--data_path', type=str, 
                        default='/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data',
                        help='Path to DepNeCTI data directory')
    parser.add_argument('--granularity', type=str, default='Coarse',
                        choices=['Coarse', 'Finegrain'],
                        help='Granularity level')
    parser.add_argument('--use_context', action='store_true',
                        help='Use "With Context" data instead of "Without Context" data')
    parser.add_argument('--splits', type=str, nargs='+', default=['test'],
                        choices=['train', 'dev', 'test', 'ood'],
                        help='Splits to evaluate on')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save predictions')
    parser.add_argument('--use_mbr', action='store_true',
                        help='Use MBR decoding during inference')
    parser.add_argument('--num_mbr_samples', type=int, default=5,
                        help='Number of samples for MBR decoding')
    parser.add_argument('--mbr_temperature', type=float, default=0.8,
                        help='Temperature for MBR decoding')
    parser.add_argument('--mbr_metric', type=str, default='uss',
                        choices=['uss', 'lss', 'voting'],
                        help='Metric for MBR selection (uss=unlabeled span, lss=labeled span, voting=majority vote)')

    args = parser.parse_args()
    
    # Initialize inference
    inference = NeCTIInference(
        model_path=args.model_path,
        data_path=args.data_path,
        granularity=args.granularity,
        use_context=args.use_context,
        device=args.device
    )
    
    # Run inference with MBR support
    results = inference.run_inference(
        splits=args.splits,
        batch_size=args.batch_size,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
        use_mbr=args.use_mbr,
        num_mbr_samples=args.num_mbr_samples,
        mbr_temperature=args.mbr_temperature,
        mbr_metric=args.mbr_metric
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print("INFERENCE SUMMARY")
    print(f"{'='*50}\n")
    
    for split, metrics in results.items():
        print(f"{split.upper()}:")
        print(f"  F1:          {metrics['f1']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  USS (F1):    {metrics['uss']:.4f}")
        print(f"  LSS (F1):    {metrics['lss']:.4f}")
        print(f"  Exact Match: {metrics['exact_match']:.4f}")
        print()


if __name__ == '__main__':
    main()