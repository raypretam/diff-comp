"""
Plot Error Breakdown by Compound Length — Dependency Label Format
====================================================================
Generates visualization of USS, LSS, EM stratified by individual compound length.

Usage:
    python plot_eval_by_length.py --data predictions.json [--output plot.png] [--max-length 15]
"""
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Labels that indicate non-compound tokens
IGNORE_LABELS = {'No_rel', 'Global_rel', 'no_rel', 'global_rel'}


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def is_compound_label(label):
    return label not in IGNORE_LABELS


def extract_compound_length(compound):
    """Extract compound length from compound structure."""
    if 'level' in compound:
        # Extract number from 'Comp2', 'Comp3', etc.
        level = compound['level']
        if level.startswith('Comp'):
            return int(level[4:])
    # Fallback: count tokens
    if 'tokens' in compound:
        return len(compound['tokens'])
    # Another fallback: use start and end indices
    if 'start' in compound and 'end' in compound:
        return compound['end'] - compound['start'] + 1
    return None


def compute_metrics_for_compound(true_labels, pred_labels, start, end):
    """
    Compute metrics for a single compound spanning [start, end].
    
    Returns:
        uss_tp, uss_fp, uss_fn: unlabeled span counts
        lss_tp, lss_fp, lss_fn: labeled span counts
        em: 1 if exact match for this compound, 0 otherwise
    """
    uss_tp = uss_fp = uss_fn = 0
    lss_tp = lss_fp = lss_fn = 0

    compound_match = True
    for i in range(start, end + 1):
        t = true_labels[i]
        p = pred_labels[i]
        
        t_is_comp = is_compound_label(t)
        p_is_comp = is_compound_label(p)

        # USS: structural prediction (is this a compound token?)
        if t_is_comp and p_is_comp:
            uss_tp += 1
        elif t_is_comp and not p_is_comp:
            uss_fn += 1
        elif not t_is_comp and p_is_comp:
            uss_fp += 1

        # LSS: exact label match at compound positions
        if t_is_comp and p_is_comp:
            if t == p:
                lss_tp += 1
            else:
                lss_fp += 1
                lss_fn += 1
                compound_match = False
        elif t_is_comp:
            lss_fn += 1
            compound_match = False
        elif p_is_comp:
            lss_fp += 1
            compound_match = False
        
        # Check for exact match
        if t != p:
            compound_match = False

    em = 1 if compound_match else 0

    return uss_tp, uss_fp, uss_fn, lss_tp, lss_fp, lss_fn, em


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to predictions JSON')
    parser.add_argument('--output', default='compound_length_metrics.png',
                        help='Output plot filename')
    parser.add_argument('--max-length', type=int, default=15,
                        help='Maximum compound length to plot (rest grouped as N+)')
    parser.add_argument('--title', default=None,
                        help='Plot title (default: derived from filename)')
    args = parser.parse_args()

    data = load_data(args.data)
    
    # Determine title
    if args.title:
        plot_title = args.title
    else:
        # Extract meaningful name from path
        import os
        filename = os.path.basename(args.data)
        plot_title = f"Metrics by Compound Length - {filename.replace('.json', '')}"

    # Accumulate per exact length
    stats = defaultdict(lambda: {
        'uss_tp': 0, 'uss_fp': 0, 'uss_fn': 0,
        'lss_tp': 0, 'lss_fp': 0, 'lss_fn': 0,
        'em_correct': 0, 'em_total': 0
    })

    for sample in data:
        true_labels = sample['true_labels']
        pred_labels = sample['predictions']
        
        # Process each compound in this sample
        if 'compounds' not in sample or not sample['compounds']:
            continue
            
        for compound in sample['compounds']:
            # Get compound length
            comp_length = extract_compound_length(compound)
            if comp_length is None:
                continue
            
            # Get start and end indices
            start = compound['start']
            end = compound['end']
            
            # Group lengths beyond max_length
            if comp_length > args.max_length:
                comp_length = args.max_length + 1  # N+ bucket
            
            # Compute metrics for this individual compound
            uss_tp, uss_fp, uss_fn, lss_tp, lss_fp, lss_fn, em = \
                compute_metrics_for_compound(true_labels, pred_labels, start, end)
            
            stats[comp_length]['uss_tp'] += uss_tp
            stats[comp_length]['uss_fp'] += uss_fp
            stats[comp_length]['uss_fn'] += uss_fn
            stats[comp_length]['lss_tp'] += lss_tp
            stats[comp_length]['lss_fp'] += lss_fp
            stats[comp_length]['lss_fn'] += lss_fn
            stats[comp_length]['em_correct'] += em
            stats[comp_length]['em_total'] += 1

    # Compute metrics for each length
    lengths = sorted(stats.keys())
    uss_scores = []
    lss_scores = []
    em_scores = []
    counts = []
    
    for length in lengths:
        s = stats[length]
        _, _, uss = f1_from_counts(s['uss_tp'], s['uss_fp'], s['uss_fn'])
        _, _, lss = f1_from_counts(s['lss_tp'], s['lss_fp'], s['lss_fn'])
        em = s['em_correct'] / s['em_total'] * 100 if s['em_total'] > 0 else 0
        
        uss_scores.append(uss * 100)  # Convert to percentage
        lss_scores.append(lss * 100)
        em_scores.append(em)
        counts.append(s['em_total'])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # X-axis labels
    x_labels = [str(l) if l <= args.max_length else f"{args.max_length}+" for l in lengths]
    x_pos = np.arange(len(lengths))
    
    # Plot 1: Metrics
    width = 0.25
    ax1.bar(x_pos - width, uss_scores, width, label='USS F1', alpha=0.8, color='skyblue')
    ax1.bar(x_pos, lss_scores, width, label='LSS F1', alpha=0.8, color='lightcoral')
    ax1.bar(x_pos + width, em_scores, width, label='Exact Match', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title(plot_title, fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 105])
    
    # Add value labels on bars
    for i, (uss, lss, em) in enumerate(zip(uss_scores, lss_scores, em_scores)):
        if uss > 5:
            ax1.text(i - width, uss + 2, f'{uss:.1f}', ha='center', va='bottom', fontsize=8)
        if lss > 5:
            ax1.text(i, lss + 2, f'{lss:.1f}', ha='center', va='bottom', fontsize=8)
        if em > 5:
            ax1.text(i + width, em + 2, f'{em:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Sample counts
    ax2.bar(x_pos, counts, alpha=0.7, color='mediumpurple')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Compounds', fontsize=12, fontweight='bold')
    ax2.set_title('Compound Distribution by Length', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        ax2.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {args.output}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"{'Length':<10} {'#Compounds':<12} {'USS F1':<10} {'LSS F1':<10} {'EM':<10}")
    print('=' * 55)
    for i, length in enumerate(lengths):
        label = x_labels[i]
        print(f"{label:<10} {counts[i]:<12} {uss_scores[i]:<10.2f} {lss_scores[i]:<10.2f} {em_scores[i]:<10.2f}")


if __name__ == '__main__':
    main()
