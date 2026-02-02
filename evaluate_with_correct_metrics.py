"""
Re-evaluate DiffusionSL predictions using CORRECT span-based metrics
matching the DepNeCTI paper specification.

This script will:
1. Load your model predictions
2. Extract spans from predictions
3. Calculate CORRECT USS, LSS, EM
4. Compare with current (incorrect) relation-based metrics
"""

import json
import os
from typing import List, Tuple, Dict
from collections import defaultdict
from span_based_evaluator import SpanBasedEvaluator


def load_predictions_from_json(pred_file: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load predictions from DiffusionSL JSON output.
    
    Returns:
        (predictions_per_sentence, true_labels_per_sentence)
    """
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_predictions = []
    all_true_labels = []
    
    for sample in data:
        predictions = sample.get('predictions', [])
        true_labels = sample.get('true_labels', [])
        
        all_predictions.append(predictions)
        all_true_labels.append(true_labels)
    
    return all_predictions, all_true_labels


def labels_to_spans_from_strings(labels: List[str], token_ids: List[int] = None) -> List[Tuple[int, int, str]]:
    """
    Convert string labels to spans with CORRECT semantic labels.
    
    Args:
        labels: List of label strings like ["K1", "Di", "Comp_root", "T6", ...]
        token_ids: Optional token IDs (defaults to 1-indexed)
    
    Returns:
        List of (start, end, semantic_label) spans
    """
    if token_ids is None:
        token_ids = list(range(1, len(labels) + 1))
    
    spans = []
    current_compound_tokens = []
    current_compound_label = None
    
    for token_id, label in zip(token_ids, labels):
        if label in ['No_rel', 'root']:
            continue
        
        # Skip non-compound labels
        if not any(x in label for x in ['Comp', 'Bs', 'Di', 'K', 'T', 'U', 'D', 'B']):
            continue
        
        # Accumulate tokens in current compound
        current_compound_tokens.append(token_id)
        
        # Capture semantic label from first token in compound
        if current_compound_label is None and 'Comp_root' not in label:
            current_compound_label = label
        
        # Check if this ends the compound
        if 'Comp_root' in label:
            if current_compound_tokens:
                start = min(current_compound_tokens)
                end = max(current_compound_tokens)
                span_label = current_compound_label if current_compound_label else 'Compound'
                spans.append((start, end, span_label))
                current_compound_tokens = []
                current_compound_label = None
    
    # Close any remaining open compounds
    if current_compound_tokens:
        start = min(current_compound_tokens)
        end = max(current_compound_tokens)
        span_label = current_compound_label if current_compound_label else 'Compound'
        spans.append((start, end, span_label))
    
    return sorted(list(set(spans)), key=lambda x: (x[0], x[1]))


def extract_compounds_from_predictions(all_predictions: List[List[str]]) -> List[List[Tuple]]:
    """
    Extract compound spans from prediction sequences with CORRECT labels.
    
    Strategy: 
    - Group tokens between Comp_root markers
    - Extract the actual compound label from the tokens
    - Create (start, end, actual_label) spans for LSS
    """
    all_spans = []
    
    for sentence_predictions in all_predictions:
        sentence_spans = []
        current_compound_tokens = []
        current_compound_label = None
        
        for idx, label in enumerate(sentence_predictions):
            token_id = idx + 1  # 1-indexed
            
            if label == 'No_rel' or label == 'root':
                continue
            
            # Accumulate tokens in current compound
            current_compound_tokens.append(token_id)
            
            # Extract the compound type from the label
            # Labels are like "K1", "K2", "T1", "T6", "T7", "Bs5", "Di", "Comp_root", etc.
            # For Comp_root, try to get the actual semantic label from previous tokens
            
            if 'Comp_root' not in label:
                # Use this label as the compound identifier
                # Keep only the non-Comp portion (semantic role like K1, T6, etc.)
                if current_compound_label is None:
                    current_compound_label = label
            
            # Check if this ends the compound
            if 'Comp_root' in label or label == 'Comp_root':
                if current_compound_tokens:
                    start = min(current_compound_tokens)
                    end = max(current_compound_tokens)
                    
                    # Use compound label if found, otherwise use semantic role
                    span_label = current_compound_label if current_compound_label else 'Compound'
                    sentence_spans.append((start, end, span_label))
                    
                    current_compound_tokens = []
                    current_compound_label = None
        
        # Handle any remaining tokens (shouldn't happen if well-formed)
        if current_compound_tokens:
            start = min(current_compound_tokens)
            end = max(current_compound_tokens)
            span_label = current_compound_label if current_compound_label else 'Compound'
            sentence_spans.append((start, end, span_label))
        
        all_spans.append(sorted(list(set(sentence_spans)), key=lambda x: (x[0], x[1])))
    
    return all_spans


def generate_corrected_evaluation_report(pred_file: str, metrics_file: str, output_file: str = None):
    """
    Generate evaluation report with CORRECT span-based metrics.
    """
    print("Loading predictions from:", pred_file)
    all_predictions, all_true_labels = load_predictions_from_json(pred_file)
    
    print(f"Loaded {len(all_predictions)} samples")
    
    # Extract spans
    print("\nExtracting spans from predictions...")
    pred_spans_list = extract_compounds_from_predictions(all_predictions)
    true_spans_list = extract_compounds_from_predictions(all_true_labels)
    
    # Calculate metrics
    print("Calculating span-based metrics...")
    evaluator = SpanBasedEvaluator()
    metrics = evaluator.evaluate_batch(true_spans_list, pred_spans_list)
    
    # Load current metrics for comparison
    with open(metrics_file, 'r') as f:
        current_metrics = json.load(f)
    
    # Generate report
    report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║           NECTI EVALUATION METRICS - SPAN-BASED CORRECTED ANALYSIS          ║
╚════════════════════════════════════════════════════════════════════════════╝

METRICS COMPARISON
═════════════════════════════════════════════════════════════════════════════

CURRENT IMPLEMENTATION (Relation-Based):
  USS (Unlabeled Span Score):     {current_metrics.get('uss', 'N/A'):>8.4f}
  LSS (Labeled Span Score):       {current_metrics.get('lss', 'N/A'):>8.4f}
  Exact Match:                    {current_metrics.get('exact_match', 'N/A'):>8.4f}%
  
  ⚠️  These use (token_id, head_id) relation tuples, NOT span boundaries

CORRECTED IMPLEMENTATION (Span-Based per Paper):
  USS (Unlabeled Span Score):     {metrics['uss']:>8.4f}
  LSS (Labeled Span Score):       {metrics['lss']:>8.4f}
  Exact Match:                    {metrics['exact_match']:>8.4f}
  
  ✅ These use (start, end, label) span tuples as specified in paper

SAMPLE STATISTICS
═════════════════════════════════════════════════════════════════════════════

Total Samples:                      {len(all_predictions):>8d}
Total True Compounds:               {sum(len(s) for s in true_spans_list):>8d}
Total Predicted Compounds:          {sum(len(s) for s in pred_spans_list):>8d}

DETAILED SPAN ANALYSIS
═════════════════════════════════════════════════════════════════════════════

Perfect Spans (both boundaries AND label):   {sum(len(set(true_spans_list[i]) & set(pred_spans_list[i])) for i in range(len(true_spans_list))):>6d} / {sum(len(s) for s in true_spans_list):>6d}

Correct Boundaries (ignoring label):         {sum(len(set((s[0], s[1]) for s in true_spans_list[i]) & set((s[0], s[1]) for s in pred_spans_list[i])) for i in range(len(true_spans_list))):>6d} / {sum(len(s) for s in true_spans_list):>6d}

═════════════════════════════════════════════════════════════════════════════

KEY FINDING:
  The current USS/LSS metrics measure RELATION-BASED accuracy.
  The correct metrics per DepNeCTI paper use SPAN-BASED accuracy.
  
  These are fundamentally different evaluation approaches!
  
  Relation-based: Checks if token->head dependencies are correct
  Span-based:     Checks if compound boundary ranges are correct

RECOMMENDATION:
  1. If comparing to DepNeCTI paper: Use SPAN-BASED metrics above
  2. If reporting your own results: Clearly specify which evaluation method
  3. Consider updating inference_necti.py to use _correct_ evaluation
  
═════════════════════════════════════════════════════════════════════════════

Generated: span-based evaluation
"""
    
    print(report)
    
    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    
    return metrics


if __name__ == '__main__':
    import sys
    
    # Default paths
    pred_file = '/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_no_ctx_new/test_predictions.json'
    metrics_file = '/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_no_ctx_new/test_metrics.json'
    output_file = '/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_no_ctx_new/SPAN_BASED_EVALUATION.txt'
    
    if len(sys.argv) > 1:
        pred_file = sys.argv[1]
    if len(sys.argv) > 2:
        metrics_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    # Generate corrected evaluation
    generate_corrected_evaluation_report(pred_file, metrics_file, output_file)
