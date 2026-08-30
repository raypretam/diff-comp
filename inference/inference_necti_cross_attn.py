"""
Inference script that supports BitDit and BitDitCrossAttn models (cross-attention).

Usage examples:
  python inference_necti_cross_attn.py --model_path path/to/checkpoint.pt --cross_attn --splits test --batch_size 8 --save_predictions --output_dir ./inference_results

The script loads the checkpoint args and initializes the appropriate model class.
"""
import argparse
import json
import os
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from models.ddim_bitdit import BitDit
from models.ddim_bitdit_cross_attn import BitDitCrossAttn
from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return ckpt


def build_model_from_args(args_ckpt, label_set, device, cross_attn: bool):
    # Use checkpoint args to initialize model similarly to training
    model_args = args_ckpt
    num_labels = len(label_set)
    
    # Debug: print key model parameters
    print(f"\nBuilding model with configuration:")
    print(f"  dim_model: {model_args.dim_model}")
    print(f"  dim_time: {model_args.dim_time}")
    print(f"  depth: {getattr(model_args, 'depth', 12)}")
    print(f"  backbone: {model_args.backbone}")
    print(f"  num_labels: {num_labels}")
    print(f"  cross_attn: {cross_attn}")
    
    if cross_attn:
        model = BitDitCrossAttn(
            device=device,
            num_classes=num_labels,
            backbone=model_args.backbone,
            time_steps=model_args.time_steps,
            sampling_steps=getattr(model_args, 'sampling_steps', model_args.time_steps),
            noise_schedule=model_args.noise_schedule,
            ddim_sampling_eta=model_args.ddim_sampling_eta,
            self_condition=model_args.self_condition,
            snr_scale=model_args.snr_scale,
            dataset=f"necti_{model_args.granularity}",
            dim_model=model_args.dim_model,
            dim_time=model_args.dim_time,
            objective=model_args.objective,
            loss_type=model_args.loss_type,
            add_lstm=getattr(model_args, 'add_lstm', False),
            freeze_bert=getattr(model_args, 'freeze_bert', False),
            max_length=model_args.max_length,
            depth=getattr(model_args, 'depth', 12),
            num_labels=num_labels
        ).to(device)
    else:
        model = BitDit(
            device=device,
            num_classes=num_labels,
            backbone=model_args.backbone,
            time_steps=model_args.time_steps,
            sampling_steps=getattr(model_args, 'sampling_steps', model_args.time_steps),
            noise_schedule=model_args.noise_schedule,
            ddim_sampling_eta=model_args.ddim_sampling_eta,
            self_condition=model_args.self_condition,
            snr_scale=model_args.snr_scale,
            dataset=f"necti_{model_args.granularity}",
            dim_model=model_args.dim_model,
            dim_time=model_args.dim_time,
            objective=model_args.objective,
            loss_type=model_args.loss_type,
            add_lstm=getattr(model_args, 'add_lstm', False),
            freeze_bert=getattr(model_args, 'freeze_bert', False),
            max_length=model_args.max_length,
            depth=getattr(model_args, 'depth', 12),
            num_labels=num_labels
        ).to(device)

    return model


def calculate_metrics(all_predictions, all_labels, all_pred_relations, all_true_relations, label_set):
    """Calculate precision, recall, F1, accuracy, USS, LSS, and Exact Match"""
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = np.mean(predictions == labels)
    
    # Get special label IDs
    no_rel_id = label_set.label2id('No_rel')
    root_id = label_set.label2id('Comp_root')
    
    # Binary classification: compound element vs non-compound
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
    uss = calculate_uss(all_true_relations, all_pred_relations)
    lss_metrics = calculate_lss(all_true_relations, all_pred_relations)
    
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
        'exact_match': lss_metrics['exact_match']
    }


def calculate_uss(true_relations, pred_relations):
    """Calculate Unlabeled Span Score (USS) - matches Eval_USS_LSS.py implementation"""
    correct = 0
    true_count = 0
    pred_count = 0
    
    for true_rels, pred_rels in zip(true_relations, pred_relations):
        # Filter: only consider relations where label != 'No_rel'
        true_filtered = [r for r in true_rels if r[2] != 'No_rel']
        pred_filtered = [r for r in pred_rels if r[2] != 'No_rel']
        
        # Compare spans: [token_idx, head_idx]
        # Exact match logic from Eval_USS_LSS.py
        for pred_rel in pred_filtered:
            pred_span = [str(pred_rel[0]), str(pred_rel[1])]
            for true_rel in true_filtered:
                true_span = [str(true_rel[0]), str(true_rel[1])]
                if pred_span == true_span:
                    correct += 1
                    break
        
        true_count += len(true_filtered)
        pred_count += len(pred_filtered)
    
    p = correct / pred_count if pred_count != 0 else 0
    r = correct / true_count if true_count != 0 else 0
    f1 = 2 * p * r / (p + r) if p != 0 and r != 0 else 0
    
    return f1


def calculate_lss(true_relations, pred_relations):
    """Calculate Labeled Span Score (LSS) and Exact Match - matches Eval_USS_LSS.py implementation"""
    correct = 0
    predict_count = 0
    true_count = 0
    em = 0
    tot_comps = 0
    
    for true_rels, pred_rels in zip(true_relations, pred_relations):
        # Filter: only relations with label != 'No_rel' and != 'root'
        # Note: Eval_USS_LSS.py filters for 'No_rel' and 'root'
        true_rels_filtered = [r for r in true_rels if r[2] != 'No_rel' and r[2] != 'root']
        pred_rels_filtered = [r for r in pred_rels if r[2] != 'No_rel' and r[2] != 'root']
        
        # Convert to string format exactly like Eval_USS_LSS.py: [','.join(lst) for lst in relations]
        tr_copy = [','.join(map(str, lst)) for lst in true_rels_filtered]
        pr_copy = [','.join(map(str, lst)) for lst in pred_rels_filtered]
        
        # Extract compounds using the same function
        tr_comps = comps_from_relations(tr_copy)
        pr_comps = comps_from_relations(pr_copy)
        
        # Exact match count at compound level (not relation level)
        for comp in pr_comps:
            if comp in tr_comps:
                em += 1
        tot_comps += len(tr_comps)
        
        # Count matching relations for LSS (relation-level comparison)
        for rel in pred_rels_filtered:
            if rel in true_rels_filtered:
                correct += 1
        
        predict_count += len(pred_rels_filtered)
        true_count += len(true_rels_filtered)
    
    # Calculate metrics exactly like Eval_USS_LSS.py
    if correct == 0:
        p = 0
        r = 0
    else:
        p = correct / predict_count
        r = correct / true_count
    
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


def comps_from_relations(relations):
    """Extract compounds from relations list - exact implementation from Eval_USS_LSS.py"""
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


def run_inference(model, dataloader, label_set, device, save_predictions=False, output_path=None, batch_start_idx=0):
    model.eval()
    results = []
    all_predictions = []
    all_labels = []
    all_pred_relations = []
    all_true_relations = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference"), start=batch_start_idx):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_labels = batch['seq_labels'].to(device)
            words2pieces = batch.get('words2pieces', None)
            if words2pieces is not None:
                words2pieces = words2pieces.to(device)

            # model returns (predictions, path) in inference mode
            preds, path = model(input_ids, attention_mask, seq_labels, words2pieces) if words2pieces is not None else model(input_ids, attention_mask, seq_labels)

            # preds: tensor [bsz, seq_len]
            bsz = preds.size(0)
            for i in range(bsz):
                mask = (seq_labels[i] != -100)
                pred_seq = preds[i][mask].cpu().tolist()
                true_seq = seq_labels[i][mask].cpu().tolist()
                
                # Get original tokens from batch - try different keys
                original_tokens = batch.get('sentences', batch.get('original_tokens', batch.get('tokens', None)))
                if original_tokens is None:
                    sample_tokens = [f"tok_{j}" for j in range(len(pred_seq))]
                else:
                    sample_tokens = original_tokens[i] if isinstance(original_tokens, list) else []
                
                # Get original text
                original_text = ' '.join(sample_tokens) if sample_tokens else ''

                pred_labels = [label_set.id2label(int(p)) for p in pred_seq]
                true_labels = [label_set.id2label(int(t)) for t in true_seq]

                annotated_pred = ' '.join([f"{t}/{l}" for t, l in zip(sample_tokens, pred_labels)]) if sample_tokens else ''
                annotated_true = ' '.join([f"{t}/{l}" for t, l in zip(sample_tokens, true_labels)]) if sample_tokens else ''

                # Create detailed prediction entry matching inference_necti.py format
                if save_predictions:
                    results.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'original_text': original_text,
                        'original_tokens': sample_tokens,
                        'predictions': pred_labels,
                        'true_labels': true_labels,
                        'annotated_pred': annotated_pred,
                        'annotated_true': annotated_true,
                        'comparison': {
                            'prediction': pred_labels,
                            'ground_truth': true_labels,
                            'tokens': sample_tokens,
                            'matches': [pred == true for pred, true in zip(pred_labels, true_labels)]
                        }
                    })
                
                # Collect for metrics
                all_predictions.extend(pred_seq)
                all_labels.extend(true_seq)
                
                # Create relations format for USS/LSS: [token_idx, head_idx, relation_type]
                pred_relations = []
                true_relations = []
                for j, (pred_label, true_label) in enumerate(zip(pred_labels, true_labels)):
                    token_idx = j + 1  # 1-indexed
                    pred_relations.append([token_idx, 0, pred_label])
                    true_relations.append([token_idx, 0, true_label])
                
                all_pred_relations.append(pred_relations)
                all_true_relations.append(true_relations)
    
    # Calculate metrics with USS/LSS/EM
    metrics = calculate_metrics(all_predictions, all_labels, all_pred_relations, all_true_relations, label_set)

    if save_predictions and output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} predictions to {output_path}")

    return results, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', default='/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data')
    parser.add_argument('--granularity', default='Finegrain', choices=['Coarse', 'Finegrain'])
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--cross_attn', action='store_true', help='Use BitDitCrossAttn model')
    parser.add_argument('--splits', nargs='+', default=['test'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--output_dir', default='./inference_results')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    print(f"Loading checkpoint from: {args.model_path}")
    ckpt = load_checkpoint(args.model_path, device)
    ckpt_args = ckpt.get('args', argparse.Namespace())
    
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint F1 score: {ckpt.get('f1_score', 'unknown')}")

    label_set = NeCTILabelSet(data_path=args.data_path, granularity=args.granularity, use_context=args.use_context)
    print(f"Number of labels: {len(label_set)}")

    model = build_model_from_args(ckpt_args, label_set, device, cross_attn=args.cross_attn)
    
    print(f"\nLoading model state dict...")
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Model loaded successfully!")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_args.backbone if hasattr(ckpt_args, 'backbone') else 'xlm-roberta-base')
    collate = NeCTICollator(tokenizer, max_length=getattr(ckpt_args, 'max_length', 128))

    for split in args.splits:
        dataset = NeCTIDataset(args.data_path, split, label_set, use_context=args.use_context)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        out_path = os.path.join(args.output_dir, f"{os.path.basename(args.model_path).replace('.pt','')}_{split}_predictions.json") if args.save_predictions else None

        print(f"\nRunning inference on split={split} (cross_attn={args.cross_attn})")
        results, metrics = run_inference(model, dataloader, label_set, device, save_predictions=args.save_predictions, output_path=out_path)
        
        print(f"\n{'='*60}")
        print(f"Results for {split} split:")
        print(f"{'='*60}")
        print(f"Samples:      {len(results)}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1 Score:     {metrics['f1']:.4f}")
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"USS (F1):     {metrics['uss']:.4f}")
        print(f"LSS (F1):     {metrics['lss']:.4f}")
        print(f"LSS Prec:     {metrics['lss_precision']:.4f}")
        print(f"LSS Recall:   {metrics['lss_recall']:.4f}")
        print(f"Exact Match:  {metrics['exact_match']:.4f}")
        print(f"TP/FP/FN/TN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")
        if out_path:
            print(f"Saved to:     {out_path}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
