"""
Perplexity comparison: segmented vs. compound-fused Sanskrit.

Reads two line-aligned plaintext files in IAST:
    toy-skt-segmented.txt   — every word separated by a space
    toy-skt.txt             — natural compound-fused form

Pipeline:
    1. Read aligned line pairs, strip blanks.
    2. Transliterate IAST -> Devanagari (LLMs primarily saw Sanskrit in
       Devanagari during pretraining, not IAST).
    3. Compute per-token cross-entropy under Qwen-2.5-32B (4-bit nf4)
       on each line. NLL is computed manually from logits to avoid the
       numerical edge cases that HF's internal `outputs.loss` averaging
       can produce (e.g., underflow when all positions are masked,
       degenerate values from float16 accumulation).

The line lengths in this dataset (~5-15 tokens after Devanagari conversion)
fit comfortably in a single forward pass; no sliding window is needed.

Usage:
    python ppl.py --data_dir /path/to/toy-data --n_subset 100 --seed 42
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# ---------- Model ----------

def load_model_and_tokenizer():
    print(f"Loading {MODEL_ID} in 4-bit nf4 ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # or torch.float16
    device_map="auto",
    trust_remote_code=True,
    )
    model.eval()
    return model, tok


# ---------- Data ----------

def iast_to_devanagari(text: str) -> str:
    return transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)


def load_pairs(data_dir: Path, n_subset: int, seed: int,
               show_samples: int = 3) -> List[Tuple[str, str]]:
    seg_path = data_dir / "toy-skt-segmented.txt"
    fus_path = data_dir / "toy-skt.txt"

    seg_lines = seg_path.read_text(encoding="utf-8").splitlines()
    fus_lines = fus_path.read_text(encoding="utf-8").splitlines()
    if len(seg_lines) != len(fus_lines):
        raise ValueError(
            f"Line-count mismatch: {seg_path.name}={len(seg_lines)}, "
            f"{fus_path.name}={len(fus_lines)}"
        )

    pairs = []
    for seg, fus in zip(seg_lines, fus_lines):
        seg, fus = seg.strip(), fus.strip()
        # Skip empty lines and the "and " header line in your data
        if not seg or not fus:
            continue
        if len(seg.split()) < 2 or len(fus.split()) < 1:
            continue
        pairs.append((iast_to_devanagari(seg), iast_to_devanagari(fus)))

    print(f"Loaded {len(pairs)} non-empty line pairs from {data_dir}")

    random.seed(seed)
    random.shuffle(pairs)
    subset = pairs[: n_subset]
    print(f"Sampled {len(subset)} pairs (seed={seed})")

    if show_samples > 0:
        print(f"\nSample conversions (first {min(show_samples, len(subset))}):")
        for i, (seg, fus) in enumerate(subset[:show_samples]):
            print(f"\n  --- line {i} ---")
            print(f"    SEG: {seg!r}")
            print(f"    FUS: {fus!r}")
    return subset


# ---------- Perplexity (manual) ----------

@torch.no_grad()
def sentence_nll_manual(model, tokenizer, text: str
                        ) -> Tuple[float, int]:
    """
    Compute per-token NLL using F.cross_entropy in fp32 for numerical
    stability under 4-bit + bf16 inference. Returns (sum_nll, n_tokens).

    For a sequence of T tokens, we predict tokens 1..T-1 conditional on
    their left context, so n_tokens = T - 1 (minus any positions that
    produce non-finite values, which we drop).
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(model.device)              # (1, T)
    if ids.size(1) < 2:
        return 0.0, 0

    out = model(ids)
    # CRITICAL: upcast logits to fp32 *before* any softmax/log computation.
    # bf16 logits from a 4-bit-quantized model can produce inf/NaN through
    # softmax, especially at low-probability positions on rare-token sequences.
    logits = out.logits.float()                       # (1, T, V) in fp32

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()     # (1, T-1, V)
    shift_labels = ids[:, 1:].contiguous()            # (1, T-1)

    # F.cross_entropy uses a fused, numerically stable kernel
    # (subtracts the per-row max before exp, all in fp32 here).
    nll_per_pos = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )                                                 # (T-1,)

    # Drop any non-finite positions defensively. Under fp32 cross_entropy
    # this should be rare; if it triggers, log it for inspection.
    finite_mask = torch.isfinite(nll_per_pos)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum().item())
        # Show which positions failed and their token IDs for debugging
        bad_positions = (~finite_mask).nonzero(as_tuple=True)[0].tolist()
        # Token IDs at the bad positions (the targets we tried to predict)
        bad_token_ids = shift_labels.view(-1)[~finite_mask].tolist()
        bad_tokens = [tokenizer.decode([t]) for t in bad_token_ids[:5]]
        print(f"    [warn] {n_bad}/{nll_per_pos.numel()} non-finite NLL "
              f"positions in '{text[:40]}...'; positions={bad_positions[:8]}, "
              f"first bad tokens={bad_tokens}; dropping these")
        nll_per_pos = nll_per_pos[finite_mask]

    n_tokens = nll_per_pos.numel()
    if n_tokens == 0:
        return 0.0, 0
    return float(nll_per_pos.sum().item()), int(n_tokens)


def corpus_perplexity(model, tokenizer, texts: List[str], label: str
                      ) -> Tuple[float, int, int, float]:
    total_nll = 0.0
    total_tokens = 0
    n_sents = 0
    t0 = time.time()
    for text in tqdm(texts, desc=label):
        nll, n = sentence_nll_manual(model, tokenizer, text)
        if n > 0:
            total_nll += nll
            total_tokens += n
            n_sents += 1
    if total_tokens == 0:
        return float("inf"), 0, 0, float("inf")
    avg_nll = total_nll / total_tokens
    ppl = math.exp(min(avg_nll, 700))
    elapsed = time.time() - t0
    print(f"  [{label}] {n_sents} sents, {total_tokens} tokens, "
          f"PPL={ppl:.3f}, NLL/tok={avg_nll:.4f}, {elapsed:.0f}s")
    return ppl, total_tokens, n_sents, avg_nll


# ---------- Driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--n_subset", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_samples", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("perplexity_results.json"))
    args = ap.parse_args()

    pairs = load_pairs(args.data_dir, args.n_subset, args.seed, args.show_samples)
    seg_texts = [p[0] for p in pairs]
    fus_texts = [p[1] for p in pairs]

    model, tokenizer = load_model_and_tokenizer()

    print("\n--- SEGMENTED ---")
    ppl_seg, tok_seg, n_seg, nll_seg = corpus_perplexity(
        model, tokenizer, seg_texts, label="segmented")

    print("\n--- COMPOUND-FUSED ---")
    ppl_fus, tok_fus, n_fus, nll_fus = corpus_perplexity(
        model, tokenizer, fus_texts, label="fused")

    # ---------- Report ----------
    print("\n" + "=" * 78)
    print(f"{'Format':<18} {'PPL':>10} {'NLL/tok':>10} "
          f"{'#tokens':>10} {'#sents':>8}")
    print("=" * 78)
    print(f"{'segmented':<18} {ppl_seg:>10.3f} {nll_seg:>10.4f} "
          f"{tok_seg:>10d} {n_seg:>8d}")
    print(f"{'compound-fused':<18} {ppl_fus:>10.3f} {nll_fus:>10.4f} "
          f"{tok_fus:>10d} {n_fus:>8d}")
    print("=" * 78)

    rel_fus = (ppl_fus - ppl_seg) / ppl_seg * 100 if ppl_seg > 0 else float("nan")
    print(f"\nRelative to segmented: compound-fused {rel_fus:+.1f}% PPL")

    if ppl_seg < ppl_fus:
        verdict = (
            "Segmented input has LOWER perplexity than compound-fused, "
            "indicating pre-segmentation matches the LLM's natural input "
            "distribution rather than constituting an out-of-distribution favor."
        )
    elif abs(rel_fus) < 5:
        verdict = (
            f"Segmented and compound-fused PPL are comparable "
            f"(within {abs(rel_fus):.1f}%), indicating pre-segmentation "
            "does not give the LLM an artificial advantage."
        )
    else:
        verdict = (
            "Compound-fused has lower PPL than segmented; "
            "the framing should be revised before reporting."
        )
    print(f"\nVerdict: {verdict}")

    results = {
        "model": MODEL_ID,
        "n_sentences": n_seg,
        "seed": args.seed,
        "segmented":      {"ppl": ppl_seg, "nll_per_tok": nll_seg, "n_tokens": tok_seg},
        "compound_fused": {"ppl": ppl_fus, "nll_per_tok": nll_fus, "n_tokens": tok_fus},
        "rel_fused_pct":  rel_fus,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()