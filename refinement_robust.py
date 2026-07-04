"""
Artifact-robust tests of: does baseline competence predict refinement benefit?

The naive per-sentence (delta vs baseline-on-same-sentence) test is confounded
by regression to the mean (a 100% sentence can only drop; a 0% can only rise).
Here we use two cleaner views:

  1. TOKEN TRANSITION decomposition (the honest mechanism):
        among tokens the baseline got RIGHT  -> how many did refinement BREAK?
        among tokens the baseline got WRONG -> how many did refinement FIX?

  2. PER-CLASS view: treat each fine-grained class' aggregate baseline accuracy
     as a *competence* proxy (averaged over many tokens, so not tied to a single
     outcome), and ask whether refinement's net delta per class tracks it.
"""

import json
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_with_ctx/test_predictions.json"
LR = "/home/pretam-pg/DiffusionSL/inference_results/hierarchial_window_7/test_predictions.json"
NON_DISCRIMINATIVE = {"No_rel", "Comp_root", "root", "ROOT", "O"}


def load(p):
    with open(p) as f:
        return json.load(f)


def idx(data, pk, tk):
    out = {}
    for s in data:
        out[(s["batch_idx"], s["sample_idx"])] = (
            s.get(pk) or s.get("predictions") or s.get("fine_predictions"),
            s.get(tk) or s.get("true_labels") or s.get("fine_true_labels"),
        )
    return out


def main():
    base = idx(load(BASE), "predictions", "true_labels")
    lr = idx(load(LR), "fine_predictions", "fine_true_labels")

    # token-level paired records
    base_correct_total = base_correct_broken = 0
    base_wrong_total = base_wrong_fixed = 0
    per_class = defaultdict(lambda: {"n": 0, "base_ok": 0, "lr_ok": 0})

    for key, (bp, bt) in base.items():
        if key not in lr:
            continue
        lp, lt = lr[key]
        for i, t in enumerate(bt):
            if t in NON_DISCRIMINATIVE:
                continue
            b_ok = bp[i] == t
            l_ok = lp[i] == t  # lt[i] should equal bt[i]; refine same gold
            per_class[t]["n"] += 1
            per_class[t]["base_ok"] += int(b_ok)
            per_class[t]["lr_ok"] += int(l_ok)
            if b_ok:
                base_correct_total += 1
                if not l_ok:
                    base_correct_broken += 1
            else:
                base_wrong_total += 1
                if l_ok:
                    base_wrong_fixed += 1

    print("=" * 72)
    print("1) TOKEN TRANSITION DECOMPOSITION (compound-internal tokens)")
    print("=" * 72)
    harm = base_correct_broken / base_correct_total
    fix = base_wrong_fixed / base_wrong_total
    print(f"Baseline CORRECT tokens : {base_correct_total}")
    print(f"   -> broken by refine  : {base_correct_broken}  "
          f"(harm rate = {harm:.3%})")
    print(f"Baseline WRONG tokens   : {base_wrong_total}")
    print(f"   -> fixed by refine   : {base_wrong_fixed}  "
          f"(fix rate  = {fix:.3%})")
    net = base_wrong_fixed - base_correct_broken
    print(f"\nNet tokens changed: {net:+d}  "
          f"(fixed {base_wrong_fixed} - broken {base_correct_broken})")
    tot = base_correct_total + base_wrong_total
    print(f"Token acc  before refine: {base_correct_total/tot:.4f}")
    print(f"Token acc  with  refine : "
          f"{(base_correct_total - base_correct_broken + base_wrong_fixed)/tot:.4f}")
    print("\nInterpretation: refinement helps net-positive ONLY if fixed > broken.")

    # ---- 2) per-class ----
    rows = []
    for c, d in per_class.items():
        if d["n"] < 10:  # skip ultra-rare classes for stable rates
            continue
        b = d["base_ok"] / d["n"]
        l = d["lr_ok"] / d["n"]
        rows.append((c, d["n"], b, l, l - b))
    rows.sort(key=lambda r: r[2])

    print("\n" + "=" * 72)
    print("2) PER-CLASS: baseline competence vs refinement delta (n>=10)")
    print("=" * 72)
    print(f"{'class':>10s} {'n':>5s} {'base_acc':>9s} {'lr_acc':>8s} {'delta':>8s}")
    for c, n, b, l, dlt in rows:
        print(f"{c:>10s} {n:>5d} {b:>9.3f} {l:>8.3f} {dlt:>+8.3f}")

    bacc = np.array([r[2] for r in rows])
    dlt = np.array([r[4] for r in rows])
    ww = np.array([r[1] for r in rows])
    r, p = stats.pearsonr(bacc, dlt)
    rho, ps = stats.spearmanr(bacc, dlt)
    print(f"\nAcross {len(rows)} classes:")
    print(f"  Pearson  r   = {r:+.4f} (p={p:.2e})")
    print(f"  Spearman rho = {rho:+.4f} (p={ps:.2e})")
    # weighted mean delta among 'good' vs 'bad' classes
    good = bacc >= np.median(bacc)
    print(f"  Mean delta, classes with HIGH baseline: "
          f"{np.average(dlt[good], weights=ww[good]):+.4f}")
    print(f"  Mean delta, classes with LOW  baseline: "
          f"{np.average(dlt[~good], weights=ww[~good]):+.4f}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    cats = ["broken\n(was right)", "fixed\n(was wrong)"]
    ax.bar([0, 1], [base_correct_broken, base_wrong_fixed],
           color=["crimson", "seagreen"], alpha=0.85)
    ax.set_xticks([0, 1]); ax.set_xticklabels(cats)
    ax.set_ylabel("# tokens")
    ax.set_title(f"(a) Token transitions\nharm={harm:.1%}  fix={fix:.1%}  net={net:+d}")
    for x, v in zip([0, 1], [base_correct_broken, base_wrong_fixed]):
        ax.text(x, v, str(v), ha="center", va="bottom")

    ax = axes[1]
    sizes = 20 + ww / ww.max() * 300
    ax.scatter(bacc, dlt, s=sizes, alpha=0.6, color="#3b6fb6")
    z = np.polyfit(bacc, dlt, 1)
    xs = np.linspace(bacc.min(), bacc.max(), 50)
    ax.plot(xs, np.polyval(z, xs), "--k", label=f"fit r={r:+.2f}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Per-class baseline accuracy (competence)")
    ax.set_ylabel("delta accuracy with refinement")
    ax.set_title("(b) Per-class: does refinement track competence?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = "/home/pretam-pg/DiffusionSL/plots/refinement_robust.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    main()
