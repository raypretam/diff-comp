"""
Per-Class Analysis + Error Type Breakdown — Dependency Label Format
====================================================================
Computes:
  1. Fine-grained LSS per coarse compound type
  2. Error type classification (correct / label-error / span-error / missed)

Based on DepNeCTI-XLMR Trankit_Data format with labels:
- Coarse: Tatpurusha, Bahuvrihi, Dvandva, Avyayibhava, Comp_root, No_rel
- Fine-grained: T1-T7, K1-K7, U, Bs2-Bs7, Bv*, Di, Ds, A1-A7, etc.

Your format:
[
  {
    "batch_idx": 0,
    "sample_idx": 0,
    "predictions": ["Bs5", "Di", "Di", "Comp_root", "No_rel", ...],
    "true_labels":  ["Bs5", "Di", "Di", "Comp_root", "No_rel", ...]
  },
  ...
]

Usage:
    python eval_per_class.py --data predictions.json

    Optionally provide a custom label map:
    python eval_per_class.py --data predictions.json --label-map coarse_map.json

    coarse_map.json format: {"Bs5": "Bahuvrihi", "Di": "Dvandva", "T6": "Tatpurusha", ...}
"""

import json
import argparse
from collections import defaultdict

# Labels that indicate non-compound tokens (skip these)
IGNORE_LABELS = {'No_rel', 'Global_rel', 'no_rel', 'global_rel', 'Comp_root', 'comp_root', 'root'}

# Default fine-grained -> coarse mapping
# Based on DepNeCTI-XLMR Trankit_Data format
DEFAULT_COARSE_MAP = {
    # Tatpurusha subtypes - vibhakti-based (case-based) compounds
    'T': 'Tatpurusha',
    'T1': 'Tatpurusha', 'T2': 'Tatpurusha', 'T3': 'Tatpurusha',
    'T4': 'Tatpurusha', 'T5': 'Tatpurusha', 'T6': 'Tatpurusha',
    'T7': 'Tatpurusha',
    # Karmadharaya variants (determinative compounds)
    'K': 'Tatpurusha',
    'K1': 'Tatpurusha', 'K2': 'Tatpurusha', 'K3': 'Tatpurusha',
    'K4': 'Tatpurusha', 'K5': 'Tatpurusha', 'K6': 'Tatpurusha',
    'K7': 'Tatpurusha',
    'Km': 'Tatpurusha',      # Karmadharaya
    # Special Tatpurusha types
    'Tds': 'Tatpurusha',     # Tatpurusha with specific semantics
    'Tdt': 'Tatpurusha',
    'Tdu': 'Tatpurusha',
    'Tg': 'Tatpurusha',
    'Tk': 'Tatpurusha',
    'Tm': 'Tatpurusha',
    'Tn': 'Tatpurusha',
    'Tp': 'Tatpurusha',
    'U': 'Tatpurusha',       # Upapada
    # Bahuvrihi subtypes - possessive/exocentric compounds
    'Bs': 'Bahuvrihi',
    'Bs2': 'Bahuvrihi', 'Bs3': 'Bahuvrihi',
    'Bs4': 'Bahuvrihi', 'Bs5': 'Bahuvrihi', 'Bs6': 'Bahuvrihi',
    'Bs7': 'Bahuvrihi',
    'Bb': 'Bahuvrihi',
    'Bsmn': 'Bahuvrihi',
    'Bsp': 'Bahuvrihi',
    'Bss': 'Bahuvrihi',
    'Bsu': 'Bahuvrihi',
    'Bv': 'Bahuvrihi',
    'Bvp': 'Bahuvrihi',
    'Bvs': 'Bahuvrihi',
    'BvS': 'Bahuvrihi',
    'BVS': 'Bahuvrihi',
    'BvU': 'Bahuvrihi',
    # Dvandva subtypes - copulative compounds
    'd': 'Dvandva',
    'D': 'Dvandva',
    'Di': 'Dvandva',         # Itaretara-dvandva
    'Ds': 'Dvandva',         # Samahara-dvandva
    # Avyayibhava - adverbial/indeclinable compounds
    'A1': 'Avyayibhava',
    'A2': 'Avyayibhava',
    'A4': 'Avyayibhava',
    'A7': 'Avyayibhava',
    # Comp_root gets its own category for tracking
    'Comp_root': 'Comp_root',
}


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_label_map(filepath):
    if filepath:
        with open(filepath, 'r') as f:
            return json.load(f)
    return DEFAULT_COARSE_MAP


def is_compound_label(label):
    """True if this is a meaningful compound-type label (not structural/padding)."""
    return label not in IGNORE_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to predictions JSON')
    parser.add_argument('--label-map', default=None,
                        help='JSON mapping fine labels -> coarse types')
    args = parser.parse_args()

    data = load_data(args.data)
    label_map = load_label_map(args.label_map)

    # === Per-class LSS ===
    # For each coarse class: accumulate TP, FP, FN
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    # === Error type breakdown ===
    # For each compound-type position in gold, classify the error
    error_types = defaultdict(lambda: {
        'correct': 0,          # both structure and label correct
        'label_error': 0,      # predicted as compound but wrong label
        'structure_error': 0,  # predicted as non-compound (missed)
    })

    # Also track: predicted compound where gold is non-compound (false positive)
    false_positives_by_class = defaultdict(int)

    for sample in data:
        true_labels = sample['true_labels']
        pred_labels = sample['predictions']

        for t, p in zip(true_labels, pred_labels):
            t_is_comp = is_compound_label(t)
            p_is_comp = is_compound_label(p)
            t_coarse = label_map.get(t, 'Unknown') if t_is_comp else None
            p_coarse = label_map.get(p, 'Unknown') if p_is_comp else None

            # --- Per-class LSS ---
            if t_is_comp:
                if t == p:
                    class_stats[t_coarse]['tp'] += 1
                else:
                    class_stats[t_coarse]['fn'] += 1

            if p_is_comp and p != t:
                class_stats[p_coarse]['fp'] += 1

            # --- Error type classification (gold compound positions only) ---
            if t_is_comp:
                if p == t:
                    error_types[t_coarse]['correct'] += 1
                elif p_is_comp:
                    error_types[t_coarse]['label_error'] += 1
                else:
                    error_types[t_coarse]['structure_error'] += 1

    # === Print per-class LSS ===
    print("\n=== Fine-grained LSS per Coarse Class ===")
    print(f"{'Class':<16} {'TP':<7} {'FP':<7} {'FN':<7} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print('-' * 62)

    all_tp = all_fp = all_fn = 0
    for coarse in sorted(class_stats.keys()):
        s = class_stats[coarse]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        all_tp += tp; all_fp += fp; all_fn += fn
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{coarse:<16} {tp:<7} {fp:<7} {fn:<7} {prec:<8.2f} {rec:<8.2f} {f1:<8.2f}")

    prec = all_tp / (all_tp + all_fp) * 100 if (all_tp + all_fp) > 0 else 0
    rec = all_tp / (all_tp + all_fn) * 100 if (all_tp + all_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print('-' * 62)
    print(f"{'Micro-avg':<16} {all_tp:<7} {all_fp:<7} {all_fn:<7} {prec:<8.2f} {rec:<8.2f} {f1:<8.2f}")

    # === Print error type breakdown ===
    print("\n=== Error Type Breakdown (gold compound positions only) ===")
    print(f"{'Class':<16} {'Correct':<10} {'Label Err':<12} {'Struct Err':<12} {'Total':<8} {'LblErr%':<10}")
    print('-' * 70)

    grand = defaultdict(int)
    for coarse in sorted(error_types.keys()):
        e = error_types[coarse]
        total = sum(e.values())
        lbl_pct = e['label_error'] / total * 100 if total > 0 else 0
        print(f"{coarse:<16} {e['correct']:<10} {e['label_error']:<12} {e['structure_error']:<12} {total:<8} {lbl_pct:<10.1f}")
        for k, v in e.items():
            grand[k] += v

    grand_total = sum(grand.values())
    print('-' * 70)
    print(f"{'Overall':<16} {grand['correct']:<10} {grand['label_error']:<12} {grand['structure_error']:<12} {grand_total:<8} "
          f"{grand['label_error']/grand_total*100 if grand_total > 0 else 0:<10.1f}")

    # === Summary for paper ===
    print("\n=== Paper-ready summary ===")
    print(f"Total compound positions: {grand_total}")
    print(f"  Correct:         {grand['correct']:>6} ({grand['correct']/grand_total*100:.1f}%)")
    print(f"  Label errors:    {grand['label_error']:>6} ({grand['label_error']/grand_total*100:.1f}%)")
    print(f"  Structure errors:{grand['structure_error']:>6} ({grand['structure_error']/grand_total*100:.1f}%)")


if __name__ == '__main__':
    main()