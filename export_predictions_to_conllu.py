"""
Export DiffusionSL predictions to CoNLL-U format for DepNeCTI evaluation
Compatible with Eval_USS_LSS.py from DepNeCTI

This script:
1. Loads a trained DiffusionSL model
2. Generates predictions on a dataset split
3. Exports to CoNLL-U format with (token_id, head_id, dependency_label) relations
4. Can be evaluated with: python Eval_USS_LSS.py <true_file> <pred_file>
"""

import argparse
import torch
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

from models.ddim_bitdit import BitDit
from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class ConLLUExporter:
    """Export predictions to CoNLL-U format"""
    
    def __init__(self, model_path: str, data_path: str, granularity: str = 'Coarse', 
                 use_context: bool = False, device: str = 'cuda'):
        """
        Args:
            model_path: Path to saved model checkpoint
            data_path: Path to DepNeCTI data directory
            granularity: 'Coarse' or 'Finegrain'
            use_context: Use "With Context" data
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.data_path = data_path
        self.granularity = granularity
        self.use_context = use_context
        
        # Load checkpoint
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.args = checkpoint['args']
        
        # Load label set
        self.label_set = NeCTILabelSet(
            data_path=self.data_path,
            granularity=self.granularity,
            use_context=self.use_context
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        
        # Initialize model with args from checkpoint
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
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Label set: {len(self.label_set)} classes")
        print(f"✓ Granularity: {granularity}, Use Context: {use_context}")
    
    def _get_dataloader(self, split: str, batch_size: int = 8):
        """Load dataset"""
        dataset = NeCTIDataset(
            data_path=self.data_path,
            mode=split,
            label_set=self.label_set,
            use_context=self.use_context
        )
        collate_fn = NeCTICollator(self.tokenizer, max_length=self.args.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )
        return dataloader, dataset
    
    @torch.no_grad()
    def generate_predictions(self, split: str = 'test', batch_size: int = 8) -> List[Dict]:
        """
        Generate predictions for a dataset split
        
        Returns:
            List of predictions with format:
            {
                'tokens': [word1, word2, ...],
                'token_ids': [1, 2, 3, ...],
                'predictions': [label_id1, label_id2, ...],
                'pred_labels': [label_str1, label_str2, ...],
                'true_labels': [label_str1, label_str2, ...],
            }
        """
        dataloader, dataset = self._get_dataloader(split, batch_size)
        
        all_predictions = []
        
        for batch in tqdm(dataloader, desc=f"Predicting on {split}"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels = batch['seq_labels'].to(self.device)
            words2pieces = batch.get('words2pieces', None)
            if words2pieces is not None:
                words2pieces = words2pieces.to(self.device)
            
            # Model inference - same logic as inference_necti.py
            if self.model.add_lstm:
                predictions, _ = self.model(input_ids, attention_mask, seq_labels, words2pieces)
            else:
                predictions, _ = self.model(input_ids, attention_mask, seq_labels)
            
            # Process batch
            batch_size_actual = input_ids.size(0)
            for i in range(batch_size_actual):
                # Get valid positions (non-padding)
                valid_mask = (seq_labels[i] != -100)
                
                pred_seq = predictions[i][valid_mask].cpu().numpy()
                label_seq = seq_labels[i][valid_mask].cpu().numpy()
                
                # Convert to labels
                pred_labels = [self.label_set.id2label(int(pid)) for pid in pred_seq]
                true_labels = [self.label_set.id2label(int(tid)) for tid in label_seq]
                
                # Get token ids and tokens corresponding to valid positions
                token_ids = input_ids[i][valid_mask].cpu().tolist()
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                
                all_predictions.append({
                    'tokens': tokens,
                    'token_ids': list(range(1, len(tokens) + 1)),
                    'predictions': pred_seq,
                    'pred_labels': pred_labels,
                    'true_labels': true_labels
                })
        
        return all_predictions
    
    def _predict_labels_to_relations(self, tokens: List[str], labels: List[str], 
                                     token_ids: List[int]) -> List[List]:
        """
        Convert token-level labels to dependency relations
        Expected label format: "Comp3_Start", "Comp3_Middle", "Comp_root", etc.
        
        Returns list of relations [token_id, head_id, dependency_label]
        """
        relations = []
        compound_stack = {}  # compound_level -> (start_token, compound_type)
        
        for token_id, label in zip(token_ids, labels):
            if label in ['No_rel', 'root', 'O']:
                continue
            
            # Extract compound level and position marker
            if 'Comp' in label:
                parts = label.split('_')
                comp_level = parts[0]  # "Comp3", "Comp2", etc.
                
                if len(parts) > 1:
                    position = parts[1]  # "Start", "Middle", "End"
                else:
                    position = "root"
                
                if position == "Start":
                    # Mark start of compound
                    compound_stack[comp_level] = token_id
                
                elif position in ["End", "root"] and "Comp_root" not in label:
                    # End of compound (but not the root marker itself)
                    if comp_level in compound_stack:
                        start_token = compound_stack.pop(comp_level)
                        relations.append([str(start_token), str(token_id), comp_level])
                
                elif "Comp_root" in label:
                    # This is the root marker - create relations
                    relations.append([str(token_id), str(token_id), 'Comp_root'])
                    compound_stack.clear()
        
        return relations
    
    def export_to_conllu(self, predictions: List[Dict], output_file: str):
        """
        Export predictions to CoNLL-U format
        
        CoNLL-U format columns (tab-separated):
        1. ID (token_id)
        2. FORM (word)
        3. LEMMA (set to _)
        4. UPOS (set to _)
        5. XPOS (set to _)
        6. FEATS (set to _)
        7. HEAD (head_id)
        8. DEPREL (dependency_label: token_id or Comp_root)
        9. DEPS (set to _)
        10. MISC (set to _)
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                tokens = pred['tokens']
                token_ids = pred['token_ids']
                pred_labels = pred['pred_labels']
                
                # Convert labels to relations
                relations = self._predict_labels_to_relations(tokens, pred_labels, token_ids)
                
                # Create relation lookup: token_id -> (head_id, deprel)
                relation_map = {}
                for token_id, head_id, deprel in relations:
                    relation_map[token_id] = (head_id, deprel)
                
                # Write sentence in CoNLL-U format
                for idx, (token_id, token) in enumerate(zip(token_ids, tokens)):
                    token_id_str = str(token_id)
                    
                    if token_id_str in relation_map:
                        head_id, deprel = relation_map[token_id_str]
                    else:
                        head_id = '0'  # ROOT
                        deprel = 'No_rel'
                    
                    # CoNLL-U format: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                    line = f"{token_id_str}\t{token}\t_\t_\t_\t_\t{head_id}\t{deprel}\t_\t_"
                    f.write(line + '\n')
                
                # Empty line between sentences
                f.write('\n')
        
        print(f"✓ Exported predictions to {output_file}")
    
    def export_predictions(self, split: str = 'test', batch_size: int = 8, 
                          output_file: str = None):
        """Main export function"""
        if output_file is None:
            output_file = f"predictions_{self.granularity}_{split}.conllu"
        
        print(f"\n{'='*70}")
        print(f"Exporting {split.upper()} predictions to CoNLL-U format")
        print(f"{'='*70}")
        
        # Generate predictions
        predictions = self.generate_predictions(split, batch_size)
        
        # Export to CoNLL-U
        self.export_to_conllu(predictions, output_file)
        
        print(f"\n{'='*70}")
        print(f"✓ Export complete!")
        print(f"✓ Output file: {output_file}")
        print(f"✓ Total sentences: {len(predictions)}")
        print(f"{'='*70}\n")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Export DiffusionSL predictions to CoNLL-U format for DepNeCTI evaluation'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--data_path', type=str,
                        default='/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data',
                        help='Path to DepNeCTI data directory')
    parser.add_argument('--granularity', type=str, default='Coarse',
                        choices=['Coarse', 'Finegrain'],
                        help='Granularity level')
    parser.add_argument('--use_context', action='store_true',
                        help='Use "With Context" data')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'dev', 'test', 'ood'],
                        help='Dataset split to export')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CoNLL-U file path')
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = ConLLUExporter(
        model_path=args.model_path,
        data_path=args.data_path,
        granularity=args.granularity,
        use_context=args.use_context,
        device=args.device
    )
    
    # Export predictions
    output_file = exporter.export_predictions(
        split=args.split,
        batch_size=args.batch_size,
        output_file=args.output_file
    )
    
    print(f"\n📊 Next step: Evaluate with DepNeCTI's script:")
    print(f"   python /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py \\")
    print(f"       <true_file.conllu> {output_file}\n")


if __name__ == '__main__':
    main()
