"""
Calculate correlation between compound length (N) and ΔEM (improvement in exact match)
Shows whether Local Refinement helps more on longer vs shorter compounds
"""

import json
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def get_compound_label_sequence(compound: Dict, labels: List[str]) -> Tuple:
    """Get the full label sequence for a compound span"""
    start = compound['start']
    end = compound['end']
    return tuple(labels[start:end+1])


def get_compound_length(compound: Dict) -> int:
    """Get the number of tokens in a compound"""
    if 'tokens' in compound:
        return len(compound['tokens'])
    else:
        return compound['end'] - compound['start'] + 1


def extract_compounds_from_predictions(pred_labels: List[str]) -> List[Dict]:
    """Extract compound spans from prediction labels"""
    compounds = []
    current_span = []

    for i, label in enumerate(pred_labels):
        if label == 'No_rel' or label == 'root':
            continue

        if label == 'Comp_root':
            if current_span:
                compounds.append({
                    'start': current_span[0],
                    'end': i,
                })
            current_span = []
        else:
            current_span.append(i)

    return compounds


def is_exact_match(true_compound: Dict, pred_compounds: List[Dict],
                   true_labels: List[str], pred_labels: List[str]) -> bool:
    """
    Check if a true compound has an exact match in predictions
    (both span boundaries and label sequence must match)
    """
    true_start = true_compound['start']
    true_end = true_compound['end']
    true_seq = get_compound_label_sequence(true_compound, true_labels)

    for pred_comp in pred_compounds:
        if pred_comp['start'] == true_start and pred_comp['end'] == true_end:
            pred_seq = get_compound_label_sequence(pred_comp, pred_labels)
            if pred_seq == true_seq:
                return True

    return False


def process_predictions(predictions_file: str) -> Dict[str, Dict]:
    """
    Process predictions and return compound-level results

    Returns:
        Dict mapping compound_id to {N: length, exact_match: bool}
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    compound_results = {}

    for sample in predictions:
        # Handle both naming conventions
        pred_labels = sample.get('predictions') or sample.get('fine_predictions')
        true_labels = sample.get('true_labels') or sample.get('fine_true_labels')
        true_compounds = sample.get('compounds') or sample.get('true_compounds', [])

        # Get predicted compounds
        if 'pred_compounds' in sample:
            pred_compounds = sample['pred_compounds']
        else:
            pred_compounds = extract_compounds_from_predictions(pred_labels)

        batch_idx = sample['batch_idx']
        sample_idx = sample['sample_idx']

        # Process each true compound
        for comp_idx, true_comp in enumerate(true_compounds):
            # Create unique compound ID
            compound_id = f"{batch_idx}_{sample_idx}_{comp_idx}"

            # Get compound properties
            N = get_compound_length(true_comp)
            em = is_exact_match(true_comp, pred_compounds, true_labels, pred_labels)

            compound_results[compound_id] = {
                'N': N,
                'exact_match': em
            }

    return compound_results


def main():
    print("Processing Enc-Diffusion predictions...")
    base_preds = process_predictions(
        '/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_with_ctx/test_predictions.json'
    )

    print("Processing Local Refinement predictions...")
    lr_preds = process_predictions(
        '/home/pretam-pg/DiffusionSL/inference_results/hierarchial_window_7/test_predictions.json'
    )

    # Calculate ΔEM for each compound
    Ns = []
    delta_ems = []

    for cid in base_preds:
        if cid not in lr_preds:
            continue

        N = base_preds[cid]['N']
        base_em = int(base_preds[cid]['exact_match'])
        lr_em = int(lr_preds[cid]['exact_match'])
        delta_em = lr_em - base_em  # in {-1, 0, +1}

        Ns.append(N)
        delta_ems.append(delta_em)

    # Convert to numpy arrays
    Ns = np.array(Ns)
    delta_ems = np.array(delta_ems)

    # Calculate correlation statistics
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS: Compound Length (N) vs. ΔEM")
    print("="*70)

    print(f"\nTotal compounds: {len(Ns)}")
    print(f"Compounds improved (ΔEM=+1): {np.sum(delta_ems == 1)}")
    print(f"Compounds degraded (ΔEM=-1): {np.sum(delta_ems == -1)}")
    print(f"Compounds unchanged (ΔEM=0): {np.sum(delta_ems == 0)}")

    print(f"\nMean compound length: {np.mean(Ns):.2f}")
    print(f"Std compound length: {np.std(Ns):.2f}")
    print(f"Length range: [{np.min(Ns)}, {np.max(Ns)}]")

    # Pearson correlation
    r, p_value = stats.pearsonr(Ns, delta_ems)
    print(f"\n--- Pearson Correlation ---")
    print(f"r = {r:.4f}")
    print(f"p-value = {p_value:.4f}")
    print(f"n = {len(Ns)}")

    # Determine significance
    if p_value < 0.001:
        sig = "*** (p < 0.001)"
    elif p_value < 0.01:
        sig = "** (p < 0.01)"
    elif p_value < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "ns (not significant)"

    # Effect size interpretation
    if abs(r) > 0.5:
        effect = "large effect"
    elif abs(r) > 0.3:
        effect = "medium effect"
    elif abs(r) > 0.1:
        effect = "small effect"
    else:
        effect = "negligible effect"

    print(f"Significance: {sig}")
    print(f"Effect size: {effect}")

    # Spearman correlation (more robust to outliers)
    rho, p_spearman = stats.spearmanr(Ns, delta_ems)
    print(f"\n--- Spearman Correlation ---")
    print(f"ρ = {rho:.4f}")
    print(f"p-value = {p_spearman:.4f}")

    if p_spearman < 0.001:
        sig_spear = "*** (p < 0.001)"
    elif p_spearman < 0.01:
        sig_spear = "** (p < 0.01)"
    elif p_spearman < 0.05:
        sig_spear = "* (p < 0.05)"
    else:
        sig_spear = "ns (not significant)"

    print(f"Significance: {sig_spear}")

    # Breakdown by length bins
    print("\n--- ΔEM by Length Bins ---")
    bins = [(2, 3), (4, 6), (7, 9), (10, 12), (13, 100)]

    for min_len, max_len in bins:
        mask = (Ns >= min_len) & (Ns <= max_len)
        if np.sum(mask) > 0:
            bin_delta = delta_ems[mask]
            improved = int(np.sum(bin_delta == 1))
            degraded = int(np.sum(bin_delta == -1))
            unchanged = int(np.sum(bin_delta == 0))
            total = int(np.sum(mask))

            net_improvement = (improved - degraded) / total * 100

            if max_len >= 100:
                bin_label = f"{min_len}+"
            else:
                bin_label = f"{min_len}-{max_len}"

            print(f"{bin_label:>8s} tokens (n={total:4d}): "
                  f"+{improved:4d}  -{degraded:4d}  ={unchanged:4d}  "
                  f"(net: {net_improvement:+.1f}%)")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    n = len(Ns)
    if r > 0 and p_value < 0.05:
        print(f"✓ Positive correlation (r={r:.3f}, p={p_value:.4f})")
        print(f"  Local Refinement provides greater benefit for longer compounds.")
        print(f"  This is statistically significant with {effect}.")
    elif r < 0 and p_value < 0.05:
        print(f"✓ Negative correlation (r={r:.3f}, p={p_value:.4f})")
        print(f"  Local Refinement provides greater benefit for shorter compounds.")
        print(f"  This is statistically significant with {effect}.")
    else:
        print(f"✗ No significant correlation (r={r:.3f}, p={p_value:.4f})")
        print(f"  Local Refinement's benefit is independent of compound length.")

    # Save results
    results = {
        'n': int(len(Ns)),
        'pearson_r': float(r),
        'pearson_p': float(p_value),
        'spearman_rho': float(rho),
        'spearman_p': float(p_spearman),
        'improved': int(np.sum(delta_ems == 1)),
        'degraded': int(np.sum(delta_ems == -1)),
        'unchanged': int(np.sum(delta_ems == 0)),
        'mean_N': float(np.mean(Ns)),
        'std_N': float(np.std(Ns))
    }

    output_file = '/home/pretam-pg/DiffusionSL/correlation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
