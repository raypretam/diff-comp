"""
Does Local Refinement (Stage 2) help or hurt depending on baseline (Stage-1 /
enc-diffusion) performance BEFORE refinement?

IMPORTANT alignment note: the two prediction files are batched in different
orders, so (batch_idx, sample_idx) does NOT identify the same sentence across
files. We align by the gold fine-label sequence and keep only sequences that are
UNIQUE in both files (clean 1-to-1 match). 2524 / 2940 sentences qualify.

Discriminative tokens = compound-internal fine-type labels (exclude
No_rel/Comp_root/root), which is what Stage 2 is meant to refine (mirrors LSS).

Three views:
  (A) per-sentence delta vs same-sentence baseline  -- CONFOUNDED by regression
      to the mean (a 100% sentence can only drop), shown only for context.
  (B) token-transition decomposition -- honest mechanism: of tokens the baseline
      got RIGHT, how many does refinement BREAK; of WRONG ones, how many FIXED.
  (C) per-class competence vs delta -- baseline accuracy of each fine class
      (aggregate competence proxy) vs refinement's net delta for that class.
"""

import json
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_with_ctx/test_predictions.json"
LR = "/home/pretam-pg/DiffusionSL/inference_results/hierarchial_window_7/test_predictions.json"
ND = {"No_rel", "Comp_root", "root", "ROOT", "O"}


def aligned_pairs():
    base = json.load(open(BASE))
    lr = json.load(open(LR))
    bc = Counter(tuple(s["true_labels"]) for s in base)
    lc = Counter(tuple(s["fine_true_labels"]) for s in lr)
    uniq = {k for k in bc if bc[k] == 1 and lc.get(k, 0) == 1}
    bmap = {tuple(s["true_labels"]): s for s in base if tuple(s["true_labels"]) in uniq}
    lmap = {tuple(s["fine_true_labels"]): s for s in lr if tuple(s["fine_true_labels"]) in uniq}
    pairs = []
    for k in uniq:
        bs, ls = bmap[k], lmap[k]
        pairs.append((bs["predictions"], ls["fine_predictions"], list(k)))
    return pairs


def main():
    pairs = aligned_pairs()
    print(f"Cleanly aligned sentences: {len(pairs)}\n")

    # ---- (A) per-sentence (confounded, context only) ----
    base_accs, deltas = [], []
    for bp, lp, gold in pairs:
        idxs = [i for i, t in enumerate(gold) if t not in ND]
        if not idxs:
            continue
        ba = np.mean([bp[i] == gold[i] for i in idxs])
        la = np.mean([lp[i] == gold[i] for i in idxs])
        base_accs.append(ba)
        deltas.append(la - ba)
    base_accs = np.array(base_accs); deltas = np.array(deltas)
    rA, pA = stats.pearsonr(base_accs, deltas)
    print("=" * 72)
    print("(A) PER-SENTENCE delta vs same-sentence baseline  [CONFOUNDED]")
    print("=" * 72)
    print(f"  n={len(base_accs)}  mean delta={deltas.mean():+.4f}  "
          f"Pearson r={rA:+.3f} (p={pA:.1e})")
    print("  NOTE: negative r here is partly mechanical (ceiling/floor).")
    edges = [-1e-9, 0.25, 0.5, 0.75, 0.999, 1.0001]
    labels = ["0-25%", "25-50%", "50-75%", "75-<100%", "=100%"]
    binA = []
    for i in range(len(edges) - 1):
        m = (base_accs > edges[i]) & (base_accs <= edges[i + 1])
        if m.sum():
            binA.append((labels[i], int(m.sum()), float(deltas[m].mean())))
            print(f"    {labels[i]:>9s} n={m.sum():4d}  mean delta={deltas[m].mean():+.4f}")

    # ---- (B) token transition ----
    bc_tot = bc_break = bw_tot = bw_fix = 0
    per_class = defaultdict(lambda: {"n": 0, "b": 0, "l": 0})
    for bp, lp, gold in pairs:
        for i, t in enumerate(gold):
            if t in ND:
                continue
            b_ok = bp[i] == t
            l_ok = lp[i] == t
            pc = per_class[t]
            pc["n"] += 1; pc["b"] += b_ok; pc["l"] += l_ok
            if b_ok:
                bc_tot += 1; bc_break += (not l_ok)
            else:
                bw_tot += 1; bw_fix += l_ok
    harm = bc_break / bc_tot
    fix = bw_fix / bw_tot
    tot = bc_tot + bw_tot
    print("\n" + "=" * 72)
    print("(B) TOKEN TRANSITION DECOMPOSITION  [artifact-free mechanism]")
    print("=" * 72)
    print(f"  baseline-correct tokens : {bc_tot:5d}  -> broken {bc_break:4d}  "
          f"(harm rate {harm:.2%})")
    print(f"  baseline-wrong   tokens : {bw_tot:5d}  -> fixed  {bw_fix:4d}  "
          f"(fix  rate {fix:.2%})")
    print(f"  net tokens: {bw_fix - bc_break:+d}   "
          f"acc {bc_tot/tot:.4f} -> {(bc_tot - bc_break + bw_fix)/tot:.4f}")

    # ---- (C) per-class ----
    rows = []
    for c, d in per_class.items():
        if d["n"] < 10:
            continue
        b = d["b"] / d["n"]; l = d["l"] / d["n"]
        rows.append((c, d["n"], b, l, l - b))
    rows.sort(key=lambda r: r[2])
    bacc = np.array([r[2] for r in rows]); dlt = np.array([r[4] for r in rows])
    ww = np.array([r[1] for r in rows])
    rC, pC = stats.pearsonr(bacc, dlt)
    rhoC, psC = stats.spearmanr(bacc, dlt)
    print("\n" + "=" * 72)
    print("(C) PER-CLASS competence vs refinement delta (classes n>=10)")
    print("=" * 72)
    print(f"{'class':>10s} {'n':>5s} {'base':>7s} {'lr':>7s} {'delta':>8s}")
    for c, n, b, l, d in rows:
        flag = "  <- hurt" if d < -0.02 else ("  <- help" if d > 0.02 else "")
        print(f"{c:>10s} {n:>5d} {b:>7.3f} {l:>7.3f} {d:>+8.3f}{flag}")
    print(f"\n  across {len(rows)} classes: Pearson r={rC:+.3f} (p={pC:.2e})  "
          f"Spearman rho={rhoC:+.3f} (p={psC:.2e})")
    med = np.median(bacc)
    hi = bacc >= med
    print(f"  weighted mean delta  HIGH-competence classes: "
          f"{np.average(dlt[hi], weights=ww[hi]):+.4f}")
    print(f"  weighted mean delta  LOW-competence  classes: "
          f"{np.average(dlt[~hi], weights=ww[~hi]):+.4f}")

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    xs = np.arange(len(binA))
    vals = [b[2] for b in binA]
    ax.bar(xs, vals, color=["crimson" if v < 0 else "seagreen" for v in vals], alpha=.8)
    for x, b in zip(xs, binA):
        ax.text(x, b[2], f"n={b[1]}", ha="center",
                va="bottom" if b[2] >= 0 else "top", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in binA], rotation=20)
    ax.axhline(0, color="gray", lw=.8)
    ax.set_title(f"(A) per-sentence delta by baseline bin\n[CONFOUNDED, r={rA:+.2f}]")
    ax.set_ylabel("mean delta accuracy")

    ax = axes[1]
    ax.bar([0, 1], [bc_break, bw_fix], color=["crimson", "seagreen"], alpha=.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"broken\n(was right)\n{harm:.1%}",
                        f"fixed\n(was wrong)\n{fix:.1%}"])
    for x, v in zip([0, 1], [bc_break, bw_fix]):
        ax.text(x, v, str(v), ha="center", va="bottom")
    ax.set_title(f"(B) token transitions\nnet {bw_fix - bc_break:+d}")
    ax.set_ylabel("# tokens")

    ax = axes[2]
    ax.scatter(bacc, dlt, s=20 + ww / ww.max() * 320, alpha=.6, color="#3b6fb6")
    z = np.polyfit(bacc, dlt, 1)
    xx = np.linspace(bacc.min(), bacc.max(), 50)
    ax.plot(xx, np.polyval(z, xx), "--k", label=f"fit r={rC:+.2f}")
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xlabel("per-class baseline accuracy (competence)")
    ax.set_ylabel("delta accuracy with refinement")
    ax.set_title("(C) per-class: refinement vs competence")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "/home/pretam-pg/DiffusionSL/plots/refinement_analysis.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot -> {out}")

    json.dump({
        "n_sentences": len(pairs),
        "per_sentence_pearson_r": float(rA),
        "token_harm_rate": harm, "token_fix_rate": fix,
        "tokens_broken": bc_break, "tokens_fixed": bw_fix,
        "acc_before": bc_tot / tot,
        "acc_after": (bc_tot - bc_break + bw_fix) / tot,
        "per_class_pearson_r": float(rC), "per_class_spearman_rho": float(rhoC),
        "per_class": [{"class": c, "n": n, "base": b, "lr": l, "delta": d}
                      for c, n, b, l, d in rows],
    }, open("/home/pretam-pg/DiffusionSL/refinement_analysis.json", "w"), indent=2)


if __name__ == "__main__":
    main()
