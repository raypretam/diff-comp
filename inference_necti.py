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


class NeCTIInference:
    """Inference for nested compound identification"""
    
    def __init__(self, model_path: str, data_path: str, granularity: str, device: str = 'cuda'):
        """
        Initialize inference
        
        Args:
            model_path: Path to saved model checkpoint
            data_path: Path to DepNeCTI data directory
            granularity: 'Coarse' or 'Finegrain'
            device: Device to run inference on
        """
        self.model_path = model_path
        self.data_path = data_path
        self.granularity = granularity
        
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
            granularity=self.granularity
        )
        print(f"\nNumber of classes: {len(self.label_set)}")
        
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
        dataset = NeCTIDataset(self.data_path, mode, self.label_set)
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
    def evaluate(self, dataloader, split_name="test", save_predictions=False):
        """
        Evaluate model on given dataloader
        
        Args:
            dataloader: DataLoader for evaluation
            split_name: Name of the split (for logging)
            save_predictions: Whether to save predictions to file
        
        Returns:
            Dictionary with metrics and predictions
        """
        print(f"\n{'='*50}")
        print(f"Evaluating on {split_name} set...")
        print(f"{'='*50}\n")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_compounds = []
        detailed_predictions = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels = batch['seq_labels'].to(self.device)
            compounds = batch['compounds']
            word_ids = batch['word_ids']
            
            # Get predictions
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
                
                # Store detailed predictions if requested
                if save_predictions:
                    sample_compounds = compounds[i] if i < len(compounds) else []
                    
                    # Decode predictions
                    pred_labels = [self.label_set.id2label(p) for p in pred_seq]
                    true_labels = [self.label_set.id2label(l) for l in label_seq]
                    
                    detailed_predictions.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'predictions': pred_labels,
                        'true_labels': true_labels,
                        'compounds': sample_compounds
                    })
        
        # Calculate metrics
        results = self._calculate_metrics(all_predictions, all_labels)
        
        # Print results
        print(f"\n{split_name.upper()} Results:")
        print("-" * 50)
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print("-" * 50)
        
        # Add detailed predictions to results
        if save_predictions:
            results['detailed_predictions'] = detailed_predictions
        
        return results
    
    def _calculate_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate precision, recall, F1, and accuracy"""
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
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    def run_inference(self, splits=['test'], batch_size=8, save_predictions=False, output_dir=None):
        """
        Run inference on specified splits
        
        Args:
            splits: List of splits to evaluate on ('test', 'dev', 'ood')
            batch_size: Batch size for inference
            save_predictions: Whether to save predictions to file
            output_dir: Directory to save predictions (if save_predictions=True)
        
        Returns:
            Dictionary mapping split names to results
        """
        results = {}
        
        for split in splits:
            try:
                dataloader = self._get_dataloader(split, batch_size)
                split_results = self.evaluate(dataloader, split, save_predictions)
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


def main():
    parser = argparse.ArgumentParser(description='Inference for NeCTI compound identification')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--data_path', type=str, 
                        default='/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/With Context',
                        help='Path to DepNeCTI data directory')
    parser.add_argument('--granularity', type=str, default='Coarse',
                        choices=['Coarse', 'Finegrain'],
                        help='Granularity level')
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
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = NeCTIInference(
        model_path=args.model_path,
        data_path=args.data_path,
        granularity=args.granularity,
        device=args.device
    )
    
    # Run inference
    results = inference.run_inference(
        splits=args.splits,
        batch_size=args.batch_size,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print("INFERENCE SUMMARY")
    print(f"{'='*50}\n")
    
    for split, metrics in results.items():
        print(f"{split.upper()}:")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print()


if __name__ == '__main__':
    main()
