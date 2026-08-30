"""
Error Breakdown by Compound Length — Dependency Label Format
=============================================================
Computes USS, LSS, EM stratified by number of components (N).

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

Logic:
- Compound tokens = positions where true_label NOT in {No_rel, Global_rel}
- N (num components) = count of compound tokens
- USS: did we predict a compound label (any non-No_rel/Global_rel) at the right position?
- LSS: did we predict the exact correct compound label at the right position?
- EM: do ALL positions in the sample match exactly?

Usage:
    python eval_by_length.py --data predictions.json [--buckets "2-3,4-6,7+"]
"""

import json
import argparse
from collections import defaultdict

# Labels that indicate non-compound tokens
IGNORE_LABELS = {'No_rel', 'Global_rel', 'no_rel', 'global_rel'}


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def is_compound_label(label):
    return label not in IGNORE_LABELS


def get_num_components(true_labels):
    """N = number of compound-role tokens (non-No_rel, non-Global_rel)."""
    return sum(1 for l in true_labels if is_compound_label(l))


def compute_metrics_for_sample(true_labels, pred_labels):
    """
    Returns:
        uss_tp, uss_fp, uss_fn: unlabeled span counts
        lss_tp, lss_fp, lss_fn: labeled span counts
        em: 1 if exact match, 0 otherwise
    """
    assert len(true_labels) == len(pred_labels), \
        f"Length mismatch: {len(true_labels)} vs {len(pred_labels)}"

    uss_tp = uss_fp = uss_fn = 0
    lss_tp = lss_fp = lss_fn = 0

    for t, p in zip(true_labels, pred_labels):
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
        if t_is_comp and p == t:
            lss_tp += 1
        elif t_is_comp and p != t:
            lss_fn += 1
        if p_is_comp and p != t:
            lss_fp += 1

    # EM: all positions match
    em = 1 if true_labels == pred_labels else 0

    return uss_tp, uss_fp, uss_fn, lss_tp, lss_fp, lss_fn, em


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p * 100, r * 100, f1 * 100


def parse_buckets(bucket_str):
    buckets = []
    for b in bucket_str.split(','):
        b = b.strip()
        if b.endswith('+'):
            lo = int(b[:-1])
            buckets.append((lo, 9999, f'{lo}+'))
        elif '-' in b:
            lo, hi = b.split('-')
            buckets.append((int(lo), int(hi), b))
        else:
            v = int(b)
            buckets.append((v, v, str(v)))
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to predictions JSON')
    parser.add_argument('--buckets', default='2-3,4-6,7+',
                        help='Comma-separated buckets')
    args = parser.parse_args()

    data = load_data(args.data)
    buckets = parse_buckets(args.buckets)

    # Accumulate per bucket
    stats = defaultdict(lambda: {
        'uss_tp': 0, 'uss_fp': 0, 'uss_fn': 0,
        'lss_tp': 0, 'lss_fp': 0, 'lss_fn': 0,
        'em_correct': 0, 'em_total': 0
    })

    for sample in data:
        true_labels = sample['true_labels']
        pred_labels = sample['predictions']
        n = get_num_components(true_labels)

        # Find bucket
        bucket_name = 'other'
        for lo, hi, name in buckets:
            if lo <= n <= hi:
                bucket_name = name
                break

        uss_tp, uss_fp, uss_fn, lss_tp, lss_fp, lss_fn, em = \
            compute_metrics_for_sample(true_labels, pred_labels)

        s = stats[bucket_name]
        s['uss_tp'] += uss_tp
        s['uss_fp'] += uss_fp
        s['uss_fn'] += uss_fn
        s['lss_tp'] += lss_tp
        s['lss_fp'] += lss_fp
        s['lss_fn'] += lss_fn
        s['em_correct'] += em
        s['em_total'] += 1

    # Print
    print(f"\n{'Bucket':<10} {'#Cpds':<8} {'USS':<8} {'LSS':<8} {'EM':<8}")
    print('=' * 45)

    total = defaultdict(int)
    for lo, hi, name in buckets:
        s = stats[name]
        if s['em_total'] == 0:
            continue
        _, _, uss = f1_from_counts(s['uss_tp'], s['uss_fp'], s['uss_fn'])
        _, _, lss = f1_from_counts(s['lss_tp'], s['lss_fp'], s['lss_fn'])
        em = s['em_correct'] / s['em_total'] * 100
        print(f"{name:<10} {s['em_total']:<8} {uss:<8.2f} {lss:<8.2f} {em:<8.2f}")
        for k, v in s.items():
            total[k] += v

    # Overall
    _, _, uss = f1_from_counts(total['uss_tp'], total['uss_fp'], total['uss_fn'])
    _, _, lss = f1_from_counts(total['lss_tp'], total['lss_fp'], total['lss_fn'])
    em = total['em_correct'] / total['em_total'] * 100
    print('-' * 45)
    print(f"{'All':<10} {total['em_total']:<8} {uss:<8.2f} {lss:<8.2f} {em:<8.2f}")


if __name__ == '__main__':
    main()