"""
Diagnostic script to analyze label confusion patterns in NeCTI predictions.
Helps identify which fine-grain labels are causing low exact_match scores.
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def load_predictions(pred_file: str) -> List[Dict]:
    """Load predictions from JSON file"""
    with open(pred_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_confusion_matrix(predictions: List[Dict]) -> Dict:
    """
    Analyze which label pairs are being confused.
    Returns confusion matrix and error statistics.
    """
    confusion = defaultdict(lambda: defaultdict(int))
    label_errors = defaultdict(int)
    
    for pred in predictions:
        true_labels = pred.get('true_labels', [])
        pred_labels = pred.get('predictions', [])
        
        if len(true_labels) != len(pred_labels):
            print(f"Warning: Length mismatch in sample {pred.get('batch_idx', '?')}")
            continue
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label != pred_label:
                confusion[true_label][pred_label] += 1
                label_errors[true_label] += 1
    
    return dict(confusion), dict(label_errors)


def analyze_compound_mismatches(predictions: List[Dict]) -> Dict:
    """
    Analyze which compounds fail exact_match due to single token errors.
    Groups errors by compound type and position.
    """
    from_relations_func = lambda rels: rels  # Simplified for this analysis
    
    compound_errors = defaultdict(list)
    perfect_compounds = 0
    failed_compounds = 0
    
    for sample_idx, pred in enumerate(predictions):
        true_labels = pred.get('true_labels', [])
        pred_labels = pred.get('predictions', [])
        tokens = pred.get('original_tokens', pred.get('merged_tokens', pred.get('tokens', [])))
        
        if len(true_labels) != len(pred_labels):
            continue
        
        # Find compounds (groups ending with Comp_root)
        current_comp = []
        comp_start_idx = 0
        
        for idx, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
            current_comp.append((true_label, pred_label))
            
            if 'Comp_root' in true_label:
                # End of compound
                errors = sum(1 for t, p in current_comp if t != p)
                compound_str = '_'.join([t for t, _ in current_comp])
                prediction_str = '_'.join([p for _, p in current_comp])
                token_str = ' '.join(tokens[comp_start_idx:idx+1] if idx < len(tokens) else ['?'])
                
                if errors == 0:
                    perfect_compounds += 1
                else:
                    failed_compounds += 1
                    error_pattern = ''.join(['✗' if t != p else '✓' for t, p in current_comp])
                    compound_errors[compound_str].append({
                        'sample_idx': sample_idx,
                        'predicted': prediction_str,
                        'tokens': token_str,
                        'errors': errors,
                        'error_pattern': error_pattern,
                        'details': [(t, p) for t, p in current_comp]
                    })
                
                current_comp = []
                comp_start_idx = idx + 1
    
    return {
        'perfect': perfect_compounds,
        'failed': failed_compounds,
        'error_details': dict(compound_errors)
    }


def print_confusion_report(confusion: Dict, label_errors: Dict):
    """Print human-readable confusion matrix report"""
    print("\n" + "="*80)
    print("LABEL CONFUSION ANALYSIS")
    print("="*80)
    print("\nMost confused labels (True Label → Predicted Label):")
    print("-"*80)
    
    # Get top confusions
    all_confusions = []
    for true_label, pred_dict in confusion.items():
        for pred_label, count in pred_dict.items():
            all_confusions.append((count, true_label, pred_label))
    
    all_confusions.sort(reverse=True)
    
    for count, true_label, pred_label in all_confusions[:20]:
        print(f"{count:4d} errors: {true_label:15s} → {pred_label:15s}")
    
    print("\n" + "-"*80)
    print("Labels with most total errors (by true label):")
    print("-"*80)
    
    sorted_errors = sorted(label_errors.items(), key=lambda x: x[1], reverse=True)
    for label, error_count in sorted_errors[:15]:
        print(f"{error_count:4d} errors: {label}")


def print_compound_report(compound_analysis: Dict):
    """Print compound-level error analysis"""
    print("\n" + "="*80)
    print("COMPOUND-LEVEL ERROR ANALYSIS")
    print("="*80)
    
    perfect = compound_analysis['perfect']
    failed = compound_analysis['failed']
    total = perfect + failed
    accuracy = 100 * perfect / total if total > 0 else 0
    
    print(f"\nPerfect Compounds:  {perfect:5d} ({accuracy:5.2f}%)")
    print(f"Failed Compounds:   {failed:5d} ({100-accuracy:5.2f}%)")
    print(f"Total Compounds:    {total:5d}")
    
    print("\n" + "-"*80)
    print("Most common compound errors (True compound type):")
    print("-"*80)
    
    error_details = compound_analysis['error_details']
    
    # Sort by number of failures
    sorted_comps = sorted(
        error_details.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for true_comp, errors in sorted_comps[:15]:
        failure_count = len(errors)
        avg_errors_per_instance = np.mean([e['errors'] for e in errors])
        
        print(f"\n{true_comp} - {failure_count} failures (avg {avg_errors_per_instance:.2f} errors/instance)")
        
        # Show a few examples
        for i, error in enumerate(errors[:3]):
            tokens = error.get('tokens', '[tokens unavailable]')
            print(f"  Example {i+1}: {tokens}")
            print(f"    True: {true_comp.replace('Comp_root', 'ROOT')}")
            print(f"    Pred: {error['predicted'].replace('Comp_root', 'ROOT')}")
            print(f"    Pattern:  {error['error_pattern']}")
            # Show specific mismatches
            mismatches = [f"{t}→{p}" for t, p in error['details'] if t != p]
            print(f"    Mismatches: {', '.join(mismatches)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze label confusion in NeCTI predictions')
    parser.add_argument('--pred_file', type=str, 
                        default='/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_no_ctx/test_predictions.json',
                        help='Path to predictions JSON file')
    
    args = parser.parse_args()
    
    print(f"Loading predictions from: {args.pred_file}")
    predictions = load_predictions(args.pred_file)
    print(f"Loaded {len(predictions)} samples\n")
    
    # Analyze confusion patterns
    confusion, label_errors = analyze_confusion_matrix(predictions)
    print_confusion_report(confusion, label_errors)
    
    # Analyze compound-level errors
    compound_analysis = analyze_compound_mismatches(predictions)
    print_compound_report(compound_analysis)
    
    # print("\n" + "="*80)
    # print("KEY INSIGHTS:")
    # print("="*80)
    # print("1. Labels with low F1 in per-label metrics are causing exact_match failures")
    # print("2. Most common confusions show which fine-grain distinctions model struggles with")
    # print("3. Compound-level analysis shows if errors cluster in specific compound types")
    # print("\nRECOMMENDATIONS:")
    # print("- Focus training on low-F1 labels with increased weight")
    # print("- Review training data for confused label pairs")
    # print("- Consider: Class imbalance, label similarity, or insufficient training data")


if __name__ == '__main__':
    main()
