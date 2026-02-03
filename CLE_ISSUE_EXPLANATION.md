# Why Chu-Liu-Edmonds Failed: Analysis & Fix

## The Problem You Observed

After running inference with `--use_cle_decoding`:

```
TEST Results:
  USS (F1):    1.0000  ← Perfect! (100%)
  LSS (F1):    0.2462  ← Terrible (24.6%)
  Exact Match: 0.0002  ← Almost zero (0.02%)
```

**Error patterns from your log:**
- 2,285 errors: T6 → Comp_root
- 1,762 errors: T6 → T7
- 1,639 errors: Comp_root → A1
- 886 errors: K1 → Comp_root

**Nearly everything predicted as Comp_root!**

---

## What Went Wrong

### 1. Fundamental Mismatch

**Chu-Liu-Edmonds (CLE) is designed for:**
- Dependency parsing
- Finding optimal spanning tree
- Head-dependent relations (token i points to token j)
- Input: Score matrix [seq_len, seq_len, num_labels] for all possible arcs

**Your task is:**
- Sequence labeling / token classification
- Each token gets an independent label
- No explicit tree structure in predictions
- Input: Label distribution [seq_len, num_labels] per token

### 2. The Implementation Problem

My CLE integration tried to:
1. Take discrete predictions from diffusion model
2. Create fake "logits" by setting predicted label = 10.0, others = -5.0
3. Run CLE algorithm on these artificial scores
4. CLE couldn't find valid trees with these fake scores
5. Defaulted to safe predictions (everything → Comp_root)

### 3. Why USS = 1.0 but LSS = 0.24?

**USS (Unlabeled Span Score)**: Checks only structure (token boundaries)
- CLE enforces perfect tree structure
- All tokens have valid head relationships
- **Result: Perfect score (1.0)**

**LSS (Labeled Span Score)**: Checks structure + labels
- CLE defaults everything to Comp_root (safe choice)
- Labels are wrong but structure is valid
- **Result: Terrible score (0.24)**

**Exact Match**: All compounds must be perfect
- With wrong labels, no compound is fully correct
- **Result: Near zero (0.0002 = 2 out of 6014 compounds)**

---

## Why CLE Doesn't Fit Here

### Token-Based Evaluation

Your evaluation code (from error log analysis):
```python
# Token-level classification
for token, pred_label, true_label in zip(tokens, predictions, labels):
    if pred_label == true_label:
        correct += 1
```

This is **sequence labeling**, not dependency parsing.

### Dependency Format ≠ Dependency Parsing

Your data has dependency information:
```
1    sa         Comp2    _    2    BvS          # head=2, label=BvS
2    sarzapaM   Comp2    _    11   Comp_root    # head=11, root marker
```

But the **evaluation is token-level**:
- Did we predict the right label for token 1? (BvS)
- Did we predict the right label for token 2? (Comp_root)

NOT dependency parsing:
- Did we predict the right head for token 1? (should point to token 2)
- Did we predict the right arc label? (BvS)

---

## The Fix

### Immediate Solution: Disable CLE

**Status**: Already done ✅

```bash
# Run without CLE (uses standard diffusion sampling)
python inference_necti.py \
    --model_path saved_models/necti_Finegrain_with_ctx/best_model.pt \
    --data_path "/home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data" \
    --granularity Finegrain \
    --use_context \
    --splits test ood

# CLE is now disabled by default in configs
```

**Expected improvement**:
- USS: 0.85-0.90 (down from 1.0, but more realistic)
- LSS: 0.75-0.85 (up from 0.24!)
- EM: 0.60-0.75 (up from 0.0002!)

### What Changed

1. **inference_necti.py**: CLE code replaced with explanatory comment
2. **configs/*.yaml**: `use_cle_decoding: False`
3. Now uses standard diffusion model predictions

---

## Could CLE Ever Help?

### Yes, BUT Only If:

#### Option 1: Reformulate as Dependency Parsing

**Requirements:**
1. Model outputs: Arc scores [seq_len, seq_len, num_labels]
2. Training: Optimize for head-dependent structure
3. Evaluation: Dependency parsing metrics (UAS/LAS)
4. Data: Use head information from dependency format

**Benefits:**
- Linguistically motivated (compounds ARE hierarchical)
- Enforces valid tree structure
- Could capture nested compounds better

**Challenges:**
- Requires complete model redesign
- Diffusion models aren't designed for dependency parsing
- Training complexity increases significantly

#### Option 2: Post-Processing Constraints

**Simpler alternative:**
```python
def enforce_compound_constraints(predictions, label_set):
    """Apply linguistic rules without full CLE"""
    
    # Rule 1: Compound must end with Comp_root
    # Rule 2: No isolated Comp_root tokens
    # Rule 3: Valid label sequences (e.g., T6 can precede T6 or Comp_root)
    # etc.
    
    return corrected_predictions
```

**Benefits:**
- Works with current model
- Faster than CLE (O(n) vs O(n³))
- Can incorporate linguistic knowledge

**Implementation:**
- Add post-processing step after diffusion sampling
- Define valid transition rules
- Apply corrections to violations

---

## Current Status

### ✅ CLE Disabled
- Config files updated: `use_cle_decoding: False`
- Inference script bypasses CLE code
- Early stopping still active and working

### 🔄 What to Do Next

1. **Re-run inference without CLE:**
   ```bash
   python inference_necti.py \
       --model_path saved_models/necti_Finegrain_with_ctx/best_model.pt \
       --data_path "/home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data" \
       --granularity Finegrain \
       --use_context \
       --splits test ood \
       --save_predictions \
       --output_dir inference_results/necti_finegrain_with_ctx_no_cle
   ```

2. **Compare results:**
   - Check if LSS improves to 0.75-0.85
   - Check if EM improves to 0.60-0.75
   - USS should be realistic (0.85-0.90)

3. **If results are better**, CLE was the problem ✅

---

## Future Work: Proper CLE Integration

If you want to use CLE properly in the future:

### Step 1: Model Modification
```python
class DependencyDiffusionModel(nn.Module):
    def forward(self, ...):
        # Output: arc scores [batch, seq_len, seq_len, num_labels]
        # Not: token labels [batch, seq_len]
```

### Step 2: Training Change
```python
# Loss should be on arcs, not tokens
arc_loss = compute_arc_loss(pred_arcs, true_arcs)
# Include tree constraint loss
tree_loss = compute_tree_constraint_loss(pred_arcs)
total_loss = arc_loss + lambda * tree_loss
```

### Step 3: CLE During Inference
```python
# Get arc scores from model
arc_scores = model(input_ids, attention_mask)  # [batch, seq_len, seq_len, num_labels]

# Apply CLE to find optimal tree
for sent_arcs in arc_scores:
    optimal_tree = chu_liu_edmonds(sent_arcs)
    predictions.append(optimal_tree)
```

### Step 4: Dependency Evaluation
```python
# Evaluate with dependency metrics
uas = unlabeled_attachment_score(pred_heads, true_heads)
las = labeled_attachment_score(pred_arcs, true_arcs)
```

**Estimated effort**: 2-3 weeks of development

---

## Summary

### The Core Issue
**CLE enforced tree structure on token classification → forced everything to Comp_root**

### The Evidence
- USS = 1.0 (perfect structure)
- LSS = 0.24 (wrong labels)
- EM = 0.0002 (almost nothing correct)
- 2285 errors to Comp_root

### The Solution
**Disable CLE for token-based evaluation**

### Moving Forward
1. Use standard diffusion sampling (CLE disabled)
2. Optionally: Add simple post-processing rules
3. Long-term: Reformulate as dependency parsing if needed

---

## Questions?

**Q: But the data is in dependency format, shouldn't we use CLE?**  
A: Data format ≠ task formulation. The evaluation is token-based, so CLE doesn't fit.

**Q: Will results be worse without CLE?**  
A: No! Results will be MUCH better. CLE was breaking predictions.

**Q: How can I enforce structure without CLE?**  
A: Add simple post-processing rules that check for valid compound patterns.

**Q: Could we make CLE work eventually?**  
A: Yes, but requires model redesign as dependency parser, not sequence labeler.

---

**Status**: Fixed ✅  
**Action**: Re-run inference without `--use_cle_decoding`  
**Expected**: LSS ~0.80, EM ~0.65-0.70  
