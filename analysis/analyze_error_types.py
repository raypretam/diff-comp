"""
Analyze error types in compound identification predictions
Categorizes errors as: correct, label error, span error, or missed
"""

import json
from typing import Dict, List, Tuple, Set


def get_span_tokens(start: int, end: int) -> Set[int]:
    """Get set of token positions in a span"""
    return set(range(start, end + 1))


def spans_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """Check if two spans overlap"""
    tokens1 = get_span_tokens(span1[0], span1[1])
    tokens2 = get_span_tokens(span2[0], span2[1])
    return len(tokens1 & tokens2) > 0


def get_compound_label_sequence(compound: Dict, labels: List[str]) -> Tuple:
    """
    Get the full label sequence for a compound span
    Returns it as a tuple for comparison
    """
    start = compound['start']
    end = compound['end']
    return tuple(labels[start:end+1])


def extract_compounds_from_predictions(pred_labels: List[str], start_offset: int = 0) -> List[Dict]:
    """
    Extract compound spans from prediction labels
    A compound is defined as a sequence ending with Comp_root
    """
    compounds = []
    current_span = []

    for i, label in enumerate(pred_labels):
        if label == 'No_rel' or label == 'root':
            continue

        if label == 'Comp_root':
            if current_span:
                # Found end of compound
                start_idx = current_span[0]
                end_idx = i

                # Get the compound label (first non-root label)
                comp_label = None
                for idx in current_span:
                    if pred_labels[idx] not in ['Comp_root', 'No_rel', 'root']:
                        comp_label = pred_labels[idx]
                        break

                compounds.append({
                    'start': start_idx + start_offset,
                    'end': end_idx + start_offset,
                    'label': comp_label or 'Compound'
                })
            current_span = []
        else:
            current_span.append(i)

    return compounds


def classify_error_type(true_compound: Dict, pred_compounds: List[Dict],
                       true_labels: List[str], pred_labels: List[str]) -> str:
    """
    Classify the error type for a gold compound

    Returns:
        - 'correct': exact span and label sequence match
        - 'label_error': span boundaries correct, label sequence wrong
        - 'span_error': boundaries incorrect but overlapping
        - 'missed': no overlapping prediction
    """
    true_start = true_compound['start']
    true_end = true_compound['end']
    true_label_seq = get_compound_label_sequence(true_compound, true_labels)

    # Check for exact match
    for pred in pred_compounds:
        if pred['start'] == true_start and pred['end'] == true_end:
            # Span boundaries match
            pred_label_seq = get_compound_label_sequence(pred, pred_labels)
            if pred_label_seq == true_label_seq:
                return 'correct'
            else:
                return 'label_error'

    # Check for overlapping predictions
    has_overlap = False
    for pred in pred_compounds:
        if spans_overlap((true_start, true_end), (pred['start'], pred['end'])):
            has_overlap = True
            break

    if has_overlap:
        return 'span_error'
    else:
        return 'missed'


def analyze_error_types(predictions_file: str) -> Dict[str, int]:
    """
    Analyze error types in predictions

    Returns:
        Dictionary with counts for each error type
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    error_counts = {
        'correct': 0,
        'label_error': 0,
        'span_error': 0,
        'missed': 0
    }

    for sample in predictions:
        # Handle both naming conventions
        pred_labels = sample.get('predictions') or sample.get('fine_predictions')
        true_labels = sample.get('true_labels') or sample.get('fine_true_labels')
        true_compounds = sample.get('compounds') or sample.get('true_compounds', [])

        # Check if predicted compounds are pre-extracted
        if 'pred_compounds' in sample:
            # Use pre-extracted predicted compounds (Local Refinement format)
            pred_compounds = sample['pred_compounds']
        else:
            # Extract predicted compounds from labels (Enc-Diffusion format)
            pred_compounds = extract_compounds_from_predictions(pred_labels)

        # Classify each true compound
        for true_comp in true_compounds:
            error_type = classify_error_type(true_comp, pred_compounds,
                                            true_labels, pred_labels)
            error_counts[error_type] += 1

    return error_counts


def calculate_percentages(error_counts: Dict[str, int]) -> Dict[str, float]:
    """Calculate percentages for each error type"""
    total = sum(error_counts.values())

    if total == 0:
        return {k: 0.0 for k in error_counts}

    return {k: (v / total * 100) for k, v in error_counts.items()}


def format_results_text(enc_diff_pct: Dict[str, float], local_ref_pct: Dict[str, float]) -> str:
    """Format results as the paragraph text"""

    # Calculate improvements
    label_improvement = enc_diff_pct['label_error'] - local_ref_pct['label_error']
    span_improvement = enc_diff_pct['span_error'] - local_ref_pct['span_error']

    # Determine the relationship between the two
    if label_improvement > 0 and span_improvement > 0:
        # Both improved
        text = f"""Enc-Diffusion produces {enc_diff_pct['label_error']:.1f}% label errors and {enc_diff_pct['span_error']:.1f}% span errors, while Local Refinement reduces label errors to {local_ref_pct['label_error']:.1f}% and span errors to {local_ref_pct['span_error']:.1f}%. This confirms that the local refinement mechanism successfully corrects both fine-grained type confusions and span boundary errors, leading to improved overall accuracy."""
    elif label_improvement > 0 and span_improvement < 0:
        # Label improved but span degraded
        text = f"""Enc-Diffusion produces {enc_diff_pct['label_error']:.1f}% label errors and {enc_diff_pct['span_error']:.1f}% span errors, while Local Refinement reduces label errors to {local_ref_pct['label_error']:.1f}% at the cost of increasing span errors to {local_ref_pct['span_error']:.1f}%. This confirms that the EM improvement stems specifically from correcting fine-grained type confusions (label errors), while the USS degradation arises from occasional span boundary shifts induced by the restricted receptive field."""
    else:
        # Other cases
        text = f"""Enc-Diffusion produces {enc_diff_pct['label_error']:.1f}% label errors and {enc_diff_pct['span_error']:.1f}% span errors, compared to {local_ref_pct['label_error']:.1f}% label errors and {local_ref_pct['span_error']:.1f}% span errors for Local Refinement."""

    return text


def main():
    # Paths to prediction files
    enc_diff_file = "/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_with_ctx/test_predictions.json"
    local_ref_file = "/home/pretam-pg/DiffusionSL/inference_results/hierarchial_window_7/test_predictions.json"

    print("Analyzing Enc-Diffusion error types...")
    enc_diff_counts = analyze_error_types(enc_diff_file)
    enc_diff_pct = calculate_percentages(enc_diff_counts)

    print("Analyzing Local Refinement error types...")
    local_ref_counts = analyze_error_types(local_ref_file)
    local_ref_pct = calculate_percentages(local_ref_counts)

    # Print detailed results
    print("\n" + "="*80)
    print("ERROR TYPE ANALYSIS")
    print("="*80)

    print("\n--- Enc-Diffusion ---")
    total_ed = sum(enc_diff_counts.values())
    print(f"Total gold spans: {total_ed}")
    for error_type, count in enc_diff_counts.items():
        pct = enc_diff_pct[error_type]
        print(f"  {error_type:>15s}: {count:>5d} ({pct:>5.2f}%)")

    print("\n--- + Local Refinement ---")
    total_lr = sum(local_ref_counts.values())
    print(f"Total gold spans: {total_lr}")
    for error_type, count in local_ref_counts.items():
        pct = local_ref_pct[error_type]
        print(f"  {error_type:>15s}: {count:>5d} ({pct:>5.2f}%)")

    print("\n--- Comparison ---")
    print(f"{'Error Type':<20s} {'Enc-Diff':>10s} {'Local-Ref':>10s} {'Change':>10s}")
    print("-" * 55)
    for error_type in ['correct', 'label_error', 'span_error', 'missed']:
        ed_pct = enc_diff_pct[error_type]
        lr_pct = local_ref_pct[error_type]
        change = lr_pct - ed_pct
        print(f"{error_type:<20s} {ed_pct:>9.2f}% {lr_pct:>9.2f}% {change:>+9.2f}%")

    # Generate formatted paragraph
    print("\n" + "="*80)
    print("FORMATTED PARAGRAPH")
    print("="*80)
    print()
    paragraph = format_results_text(enc_diff_pct, local_ref_pct)
    print(paragraph)

    # Save results
    results = {
        'enc_diffusion': {
            'counts': enc_diff_counts,
            'percentages': enc_diff_pct
        },
        'local_refinement': {
            'counts': local_ref_counts,
            'percentages': local_ref_pct
        }
    }

    output_file = "/home/pretam-pg/DiffusionSL/error_type_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Save paragraph to file
    paragraph_file = "/home/pretam-pg/DiffusionSL/error_types_paragraph.txt"
    with open(paragraph_file, 'w') as f:
        f.write(paragraph)

    print(f"Paragraph saved to: {paragraph_file}")


if __name__ == "__main__":
    main()
