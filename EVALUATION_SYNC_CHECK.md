# DiffusionSL Evaluation Analysis: Current vs Paper Specification

## 🔴 CRITICAL FINDING: MISMATCH IDENTIFIED

Your DiffusionSL implementation uses **RELATION-BASED** evaluation, but the DepNeCTI paper describes **SPAN-BASED** evaluation.

---

## Comparison Table

| Aspect | DepNeCTI Paper | DiffusionSL Current | Status |
|--------|---|---|---|
| **USS** | F1 on (start, end) boundaries | F1 on (token_idx, head_idx) pairs | ❌ WRONG |
| **LSS** | F1 on (start, end, label) tuples | F1 on (token_idx, head_idx, label) tuples | ❌ WRONG |
| **EM** | % of compounds with all correct spans | % of compounds with all correct relations | ⚠️ PARTIAL |
| **Methodology** | Span-based NER | Dependency-relation based | ❌ DIFFERENT |

---

## Current Implementation Analysis

### What DiffusionSL Currently Does (Lines 350-378)

```python
def _calculate_uss(self, true_relations, pred_relations):
    # Line 360-361
    true_spans = [(r[0], r[1]) for r in true_rels if r[2] != 'No_rel']
    pred_spans = [(r[0], r[1]) for r in pred_rels if r[2] != 'No_rel']
    
    # These are (token_idx, head_idx) pairs, NOT (start, end) positions!
```

**Problem**: `r[0]` is token index, `r[1]` is head index - not a span boundary

### Current USS Interpretation

```
Example: Compound with tokens [1, 2, 3, 4] where 1,2,3 are members, 4 is root

Your code extracts:
- (1, 4) - token 1's head is 4
- (2, 4) - token 2's head is 4  
- (3, 4) - token 3's head is 4
- (4, 0) - token 4 is root

Checks USS as: Did we get these (token, head) pairs right?

Paper expects:
- Compound span: (1, 3) representing tokens 1-3
- Check: Did we identify span (1, 3)?
```

---

## What the Paper Actually Requires

### Span-Based Evaluation (Correct)

**USS (Unlabeled Span Score)**:
```python
# Extract compound spans as (start_token, end_token)
# E.g., compound spanning tokens 1-3 → (1, 3)

# USS = F1 on span boundaries only
# Precision: of all spans we predicted, how many have correct boundaries?
# Recall: of all true spans, how many did we correctly identify boundaries for?
```

**LSS (Labeled Span Score)**:
```python
# Extract compound spans as (start_token, end_token, compound_type)
# E.g., compound spanning tokens 1-3 with type "Comp3" → (1, 3, "Comp3")

# LSS = F1 on (span, label) tuples
# Same precision/recall but requiring both boundaries AND label correct
```

**EM (Exact Match)**:
```python
# For each compound, check if ALL spans within it are correctly identified
# EM = % of compounds where all nested spans and labels are correct
```

---

## The Problem with Current Implementation

### Current Code Issue #1: Span Extraction

```python
# Current (WRONG)
true_spans = [(r[0], r[1]) for r in true_rels if r[2] != 'No_rel']

# This extracts dependency relations, not span boundaries!
# r[0] = token_idx (which token)
# r[1] = head_idx (which token is the head)
# These form a dependency relation, NOT a span
```

### Current Code Issue #2: What Should Be Done

```python
# CORRECT (Span-based)
# Need to extract contiguous token ranges that form compounds

# From conll-u format:
# Token 1: label="Comp3_Start" → marks start of compound
# Token 2: label="Comp3_Middle" → continues compound
# Token 3: label="Comp3_Middle" → continues compound
# Token 4: label="Comp3_End" or contains "Comp_root" → marks end

# Extract: (start=1, end=4, type="Comp3")
```

---

## Current Results Are Misleading

```
Your current metrics:
- USS: 100%
- LSS: 83.30%
- EM: 61.04%

These numbers are NOT comparable to papers using true span-based evaluation
because they're measuring something different (dependency relations, not spans).
```

---

## Implementation Comparison

### Current DiffusionSL (Relation-based) ❌

```python
# USS checks: (token_idx, head_idx) pairs
# Counts: 11,669 TP, 149 FP, 127 FN out of 11,945 relations
# This is NOT span-based!
```

### What Paper Describes (Span-based) ✅

```python
# USS should check: (start_token, end_token) boundaries
# Should count: how many compound spans did we get right?
# This requires extracting SPAN RANGES, not token-head pairs
```

---

## To Summarize

| Metric | DiffusionSL Current | DepNeCTI Paper | Match |
|--------|---|---|---|
| USS | F1 on 11,945 token-head relations | F1 on compound span boundaries | ❌ NO |
| LSS | F1 on 11,945 labeled relations | F1 on (span, label) tuples | ❌ NO |
| EM | 6,014 compound structures | % of perfect compounds | ⚠️ Partial |

---

## Recommendation

**Your current evaluation is NOT wrong, but it IS DIFFERENT from the paper.**

### Option 1: Document the Difference ✅
- Keep current evaluation
- Clearly label it as "Relation-based USS/LSS"
- Note: NOT comparable to span-based methods

### Option 2: Implement Correct Span-Based Evaluation ✅
- Extract compound spans from CoNLL-U format
- Implement true span-based USS/LSS
- Makes results comparable to paper

### Option 3: Use DepNeCTI's Script ✅
- Use /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py
- It has the same issue (relation-based), but at least consistent with original

---

## What Would Need to Change

To implement **CORRECT** span-based evaluation:

1. **Extract spans** from CoNLL-U format:
   - Look for tokens with compound markers (Comp2_Start, Comp3_Middle, etc.)
   - Track start/end positions
   - Build span tuples: (start, end, compound_type)

2. **Calculate USS**:
   - Only compare (start, end) boundaries
   - Ignore labels

3. **Calculate LSS**:
   - Compare full (start, end, label) tuples
   - Include compound type/level

4. **Calculate EM**:
   - For each sentence's compound set
   - Check if ALL spans match perfectly

---

## Your Specific Situation

### Current Metrics (from last run):
```
Token F1:     98.89%  ← Token-level classification
USS:          100.00% ← Relation-based (not true span-based)
LSS:          83.30%  ← Relation-based labeled spans
EM:           61.04%  ← Compound-level exact match
```

### If You Switch to Span-Based:
```
USS:          ? (likely lower than 100%)
LSS:          ? (likely different from 83.30%)
EM:           ~ 61.04% (similar - compound structure stays same)
```

The EM probably stays similar because the compound structure judgment
(whether ALL members are correctly labeled) is independent of whether
you measure via dependencies or spans.

---

**Question for you: Would you like me to implement the CORRECT span-based evaluation?**

If yes, I can:
1. Create proper span extraction from your CoNLL-U format
2. Implement true span-based USS/LSS/EM
3. Re-evaluate your model with correct metrics
4. Show you the actual values

The work would be ~1-2 hours and would make your results directly comparable to the paper.
