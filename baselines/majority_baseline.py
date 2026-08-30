"""
Majority-class baseline for NeCTIS.

Reads CoNLL-U-style files where:
  col 0: token_id
  col 1: token
  col 3: coarse compound class (Comp2, Comp4, CompNo, ...)
  col 6: head index
  col 7: fine-grained label (Bs5, Di, Tp, T6, Comp_root, No_rel, ...)
  col 9: BIO position tag (Comp{N}_Start | Comp{N}_Middle | Comp{N}_End | CompNo)

Computes USS / LSS / EM at coarse and fine-grained levels for a baseline
that predicts the most-frequent training label for every span.

Usage:
    python majority_baseline.py --train train.conllu --test test.conllu
    python majority_baseline.py --train train.conllu --test ood.conllu --label fine
"""

import argparse
from collections import Counter
from pathlib import Path


# ---------- I/O ----------

def read_conllu(path):
    """Yields lists of token-rows. Each token-row is a list of column strings."""
    sent = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                if sent:
                    yield sent
                    sent = []
                continue
            cols = line.split("\t")
            if len(cols) < 10:
                continue
            sent.append(cols)
    if sent:
        yield sent


# ---------- Span extraction ----------

def extract_spans(sent):
    """
    Returns a list of spans for one compound-bearing sentence.
    Each span is a tuple (head_idx, tail_idx, fine_label, coarse_label).
    A span is a contiguous run of tokens whose col 9 matches Comp{N}_(Start|Middle|End).
    Indices are 1-based token IDs (matching col 0).
    """
    spans = []
    cur_start, cur_label_fine, cur_label_coarse = None, None, None

    for tok in sent:
        tok_id = int(tok[0])
        coarse = tok[3]                  # e.g., Comp4 or CompNo
        fine = tok[7]                    # e.g., Bs5, Di, Comp_root, No_rel
        bio = tok[9]                     # Comp4_Start, Comp4_Middle, Comp4_End, CompNo

        if bio.endswith("_Start"):
            cur_start = tok_id
            cur_label_fine = fine        # label of the span = label at the START token
            cur_label_coarse = coarse
        elif bio.endswith("_End") and cur_start is not None:
            spans.append((cur_start, tok_id, cur_label_fine, cur_label_coarse))
            cur_start, cur_label_fine, cur_label_coarse = None, None, None
        elif bio.endswith("_Middle"):
            continue
        else:
            # CompNo or anything else — not part of a span
            cur_start, cur_label_fine, cur_label_coarse = None, None, None

    return spans


# ---------- Metrics ----------

def evaluate(gold_per_compound, pred_per_compound):
    """
    gold_per_compound, pred_per_compound: list of list-of-spans, one list per compound.
    Each span is (head, tail, label).
    Computes USS (span match), LSS (span+label match), EM (full compound match).
    """
    uss_correct = uss_total_pred = uss_total_gold = 0
    lss_correct = lss_total_pred = lss_total_gold = 0
    em_correct = em_total = 0

    for gold, pred in zip(gold_per_compound, pred_per_compound):
        gold_unl = {(h, t) for (h, t, _) in gold}
        pred_unl = {(h, t) for (h, t, _) in pred}
        gold_lab = {(h, t, l) for (h, t, l) in gold}
        pred_lab = {(h, t, l) for (h, t, l) in pred}

        uss_correct += len(gold_unl & pred_unl)
        uss_total_pred += len(pred_unl)
        uss_total_gold += len(gold_unl)

        lss_correct += len(gold_lab & pred_lab)
        lss_total_pred += len(pred_lab)
        lss_total_gold += len(gold_lab)

        em_total += 1
        if gold_lab == pred_lab and len(gold_lab) > 0:
            em_correct += 1

    def f1(c, p, g):
        if p == 0 or g == 0:
            return 0.0
        prec = c / p
        rec = c / g
        if prec + rec == 0:
            return 0.0
        return 200.0 * prec * rec / (prec + rec)

    return {
        "USS": f1(uss_correct, uss_total_pred, uss_total_gold),
        "LSS": f1(lss_correct, lss_total_pred, lss_total_gold),
        "EM":  100.0 * em_correct / em_total if em_total else 0.0,
        "n_compounds": em_total,
        "n_spans_gold": uss_total_gold,
    }


# ---------- Pipeline ----------

def collect_spans(path):
    """Returns one list-of-spans per sentence (compound-level granularity)."""
    out = []
    for sent in read_conllu(path):
        spans = extract_spans(sent)
        if spans:
            out.append(spans)
    return out


def find_majority(train_spans):
    """Returns (majority_fine, majority_coarse) computed over all training spans."""
    fine_counter = Counter()
    coarse_counter = Counter()
    for spans in train_spans:
        for (_, _, fine, coarse) in spans:
            fine_counter[fine] += 1
            coarse_counter[coarse] += 1

    print("\nTop-10 fine labels in train:")
    for lab, c in fine_counter.most_common(10):
        print(f"  {lab:20s} {c:6d}")
    print("\nTop coarse labels in train:")
    for lab, c in coarse_counter.most_common():
        print(f"  {lab:20s} {c:6d}")

    return fine_counter.most_common(1)[0][0], coarse_counter.most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test",  required=True, type=Path)
    args = ap.parse_args()

    print(f"Reading train: {args.train}")
    train_spans = collect_spans(args.train)
    print(f"  {len(train_spans)} compounds, "
          f"{sum(len(s) for s in train_spans)} spans")

    print(f"Reading test: {args.test}")
    test_spans = collect_spans(args.test)
    print(f"  {len(test_spans)} compounds, "
          f"{sum(len(s) for s in test_spans)} spans")

    maj_fine, maj_coarse = find_majority(train_spans)
    print(f"\nMajority fine label:   {maj_fine}")
    print(f"Majority coarse label: {maj_coarse}")

    # Build gold and predictions in two granularities
    gold_fine   = [[(h, t, fine)   for (h, t, fine, _)   in s] for s in test_spans]
    gold_coarse = [[(h, t, coarse) for (h, t, _,    coarse) in s] for s in test_spans]

    # Oracle structure + majority label
    pred_fine   = [[(h, t, maj_fine)   for (h, t, _) in g] for g in gold_fine]
    pred_coarse = [[(h, t, maj_coarse) for (h, t, _) in g] for g in gold_coarse]

    print("\n" + "=" * 60)
    print("Majority-class baseline (oracle structure + majority label)")
    print("=" * 60)

    coarse = evaluate(gold_coarse, pred_coarse)
    fine   = evaluate(gold_fine,   pred_fine)

    print(f"\nCoarse:  USS={coarse['USS']:5.2f}  "
          f"LSS={coarse['LSS']:5.2f}  EM={coarse['EM']:5.2f}")
    print(f"Fine:    USS={fine['USS']:5.2f}  "
          f"LSS={fine['LSS']:5.2f}  EM={fine['EM']:5.2f}")
    print(f"\n({coarse['n_compounds']} compounds, "
          f"{coarse['n_spans_gold']} gold spans)")


if __name__ == "__main__":
    main()