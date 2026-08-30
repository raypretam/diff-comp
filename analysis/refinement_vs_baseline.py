"""
Does Local Refinement (Stage 2) help or hurt depending on the baseline
performance (Stage-1 / enc-diffusion, i.e. BEFORE refinement)?

Hypothesis under test (user):
    - If the model was already good before refinement -> refinement improves.
    - If it was below some threshold -> refinement makes it worse.

We measure performance per-sentence on the *discriminative* tokens, i.e. the
compound-internal fine-grained type labels (excluding No_rel / Comp_root / root,
which are not what Stage 2 is meant to refine). This mirrors the LSS metric.

For each sentence:
    base_acc = fraction of discriminative tokens correct  (BEFORE refinement)
    lr_acc   = fraction of discriminative tokens correct  (WITH refinement)
    delta    = lr_acc - base_acc

Then we look at delta as a function of base_acc.
"""

import os
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_with_ctx/test_predictions.json"
LR = "/home/pretam-pg/DiffusionSL/inference_results/hierarchial_window_7/test_predictions.json"

NON_DISCRIMINATIVE = {"No_rel", "Comp_root", "root", "ROOT", "O"}


def load(path):
    with open(path) as f:
        return json.load(f)


def index_by_key(data, pred_key, true_key):
    out = {}
    for s in data:
        key = (s["batch_idx"], s["sample_idx"])
        preds = s.get(pred_key) or s.get("predictions") or s.get("fine_predictions")
        trues = s.get(true_key) or s.get("true_labels") or s.get("fine_true_labels")
        out[key] = (preds, trues)
    return out


def sentence_acc(preds, trues):
    """Accuracy over discriminative (compound-internal type) tokens only."""
    correct = 0
    total = 0
    for p, t in zip(preds, trues):
        if t in NON_DISCRIMINATIVE:
            continue
        total += 1
        if p == t:
            correct += 1
    if total == 0:
        return None, 0
    return correct / total, total


def main():
    base = index_by_key(load(BASE), "predictions", "true_labels")
    lr = index_by_key(load(LR), "fine_predictions", "fine_true_labels")

    base_accs, deltas, weights = [], [], []
    for key, (bp, bt) in base.items():
        if key not in lr:
            continue
        lp, lt = lr[key]
        ba, n = sentence_acc(bp, bt)
        la, n2 = sentence_acc(lp, lt)
        if ba is None or la is None:
            continue
        base_accs.append(ba)
        deltas.append(la - ba)
        weights.append(n)  # number of discriminative tokens in the sentence

    base_accs = np.array(base_accs)
    deltas = np.array(deltas)
    weights = np.array(weights)

    n = len(base_accs)
    print("=" * 72)
    print("LOCAL REFINEMENT IMPACT vs. BASELINE PERFORMANCE (per sentence)")
    print("=" * 72)
    print(f"Sentences analysed (with >=1 compound-internal token): {n}")
    print(f"Mean baseline acc : {base_accs.mean():.4f}")
    print(f"Mean delta (LR-base): {deltas.mean():+.4f}")
    print(f"  improved (delta>0): {int((deltas > 0).sum())}")
    print(f"  degraded (delta<0): {int((deltas < 0).sum())}")
    print(f"  unchanged (delta=0): {int((deltas == 0).sum())}")

    r, p = stats.pearsonr(base_accs, deltas)
    rho, ps = stats.spearmanr(base_accs, deltas)
    print(f"\nPearson  r   = {r:+.4f}  (p = {p:.2e})")
    print(f"Spearman rho = {rho:+.4f}  (p = {ps:.2e})")

    # Binned view
    print("\n--- Mean delta by baseline-accuracy bin ---")
    edges = [-0.001, 0.25, 0.5, 0.75, 0.999, 1.0001]
    labels = ["0-25%", "25-50%", "50-75%", "75-<100%", "=100%"]
    bin_centers, bin_means, bin_ns = [], [], []
    for i in range(len(edges) - 1):
        mask = (base_accs > edges[i]) & (base_accs <= edges[i + 1])
        if mask.sum() == 0:
            continue
        m = deltas[mask].mean()
        bin_centers.append(0.5 * (edges[i] + edges[i + 1]))
        bin_means.append(m)
        bin_ns.append(int(mask.sum()))
        imp = int((deltas[mask] > 0).sum())
        deg = int((deltas[mask] < 0).sum())
        print(f"  {labels[i]:>9s}  n={mask.sum():4d}   mean delta={m:+.4f}   "
              f"(+{imp} / -{deg})")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) scatter with jitter + trend line
    ax = axes[0]
    jitter = (np.random.rand(n) - 0.5) * 0.03
    ax.scatter(base_accs + jitter, deltas, s=8, alpha=0.15, color="#3b6fb6")
    # weighted-ish lowess substitute: bin means
    ax.plot(bin_centers, bin_means, "o-", color="crimson", lw=2,
            label="mean delta per bin")
    z = np.polyfit(base_accs, deltas, 1)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, np.polyval(z, xs), "--", color="black",
            label=f"linear fit (r={r:+.2f})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Baseline accuracy BEFORE refinement (per sentence)")
    ax.set_ylabel("delta accuracy (WITH refinement - baseline)")
    ax.set_title("(a) Refinement impact vs. baseline performance")
    ax.legend(fontsize=8)

    # (b) bar of mean delta per bin
    ax = axes[1]
    colors = ["crimson" if m < 0 else "seagreen" for m in bin_means]
    xpos = np.arange(len(bin_means))
    ax.bar(xpos, bin_means, color=colors, alpha=0.8)
    for x, m, nn in zip(xpos, bin_means, bin_ns):
        ax.text(x, m + (0.002 if m >= 0 else -0.002), f"n={nn}",
                ha="center", va="bottom" if m >= 0 else "top", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([labels[i] for i in range(len(bin_means))], rotation=20)
    ax.set_xlabel("Baseline accuracy bin (before refinement)")
    ax.set_ylabel("mean delta accuracy")
    ax.set_title("(b) Mean refinement gain by baseline bin")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "figures", "refinement_vs_baseline.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot -> {out}")

    # Save numbers
    res = {
        "n_sentences": n,
        "mean_baseline_acc": float(base_accs.mean()),
        "mean_delta": float(deltas.mean()),
        "improved": int((deltas > 0).sum()),
        "degraded": int((deltas < 0).sum()),
        "unchanged": int((deltas == 0).sum()),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(ps),
        "bins": {labels[i]: {"mean_delta": bin_means[i], "n": bin_ns[i]}
                 for i in range(len(bin_means))},
    }
    with open("/home/pretam-pg/DiffusionSL/refinement_vs_baseline.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
