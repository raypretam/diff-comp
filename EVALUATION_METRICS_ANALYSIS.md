# DepNeCTI Evaluation Metrics Analysis: Paper vs Implementation

## Executive Summary

**NO - The implementation does NOT match the paper's description.**

The paper claims **SPAN-BASED** evaluation, but the code implements **DEPENDENCY-RELATION-BASED** evaluation. These are fundamentally different metrics.

---

## What the Paper Claims

From DepNeCTI paper:
```
"We evaluate the performance using the Labeled/Unlabeled Span Score (LSS/USS) 
in terms of micro-averaged F1-score.

- LSS: micro-averaged F1-score on tuples of (predicted spans + labels)
- USS: F1-score excluding labels of the spans
- EM:  percentage of compounds where ALL spans and relations are correctly identified"
```

This describes traditional **SPAN-BASED NER evaluation**:
- **Span**: A continuous range (start_token, end_token)
- **USS**: F1 on (start, end) tuples
- **LSS**: F1 on (start, end, label) tuples

---

## What the Code Actually Does

### 1. USS Implementation (WRONG)

```python
# Line 42-43
true_span = [true_lst[0], true_lst[4]]  # [token_id, head_id]
pred_span = [pred_lst[0], pred_lst[4]]  # [token_id, head_id]
```

**What it checks**: (token_id, head_id) PAIRS, not spans

**Example from data**:
```
Token 1 (वीत): head=4 → Checks (1, 4)
Token 2 (राग): head=4 → Checks (2, 4)
Token 3 (भय):  head=4 → Checks (3, 4)
Token 4 (क्रोधः): head=9 → Checks (4, 9)
```

**Problem**: 
- This is **DEPENDENCY-RELATION evaluation**, not span evaluation
- True span should be (1, 4) representing "tokens 1 through 4 form a compound"
- Current code checks 4 separate (token, head) pairs
- **Never verifies that tokens 1-3 are contiguous** or that they bound a span

### 2. LSS Implementation (WRONG)

```python
# Line 13
relation = [lst[0], lst[4], lst[5]]  # [token_id, head_id, label]

# Line 80-81
for rel in pred_relation_oneline:
    if rel in true_relation_oneline:
        correct += 1
```

**What it checks**: (token_id, head_id, label) TUPLES

**Problem**: Same as USS
- Checks dependency relations, not span tuples
- Should check (start_token, end_token, label) not (token, head, label)

### 3. EM Implementation (PARTIALLY CORRECT)

```python
# Line 26-31
def comps_from_relations(relations):
    lst = []
    nested_comp = []
    for rel in relations:
        if 'Comp_root' in rel:
            lst.append(rel)
            nested_comp.append(lst)
            lst = []
        else:
            lst.append(rel)
    return nested_comp
```

**What it does**: 
- Groups relations by compound (ends with Comp_root marker)
- Compares entire compound structures

**Status**: Closer to correct but still using relations, not span boundaries

---

## Concrete Example: Why This Matters

### Test Case: Two-compound sentence

```
Sentence:
Tokens:     [A(1), B(2), C(3), D(4), E(5), F(6)]
Structure:
  Compound 1: tokens 1-3, root=4, label="Comp3"
  Compound 2: tokens 5-6, root=6, label="Comp2"

Data format:
1    A    _    _    _    _    4    Bs    _    Comp3_Start
2    B    _    _    _    _    4    Di    _    Comp3_Middle
3    C    _    _    _    _    4    Di    _    Comp3_Middle
4    D    _    _    _    _    0    Comp_root    _    Comp3_End
5    E    _    _    _    _    6    Bs    _    Comp2_Start
6    F    _    _    _    _    0    Comp_root    _    Comp2_End
```

### Paper's Expected Evaluation (Span-based)

**True spans**: [(1,3,"Comp3"), (5,6,"Comp2")]

Prediction 1 (Correct):
- Predicted spans: [(1,3,"Comp3"), (5,6,"Comp2")]
- USS: 2/2 = 100% (both span boundaries correct)
- LSS: 2/2 = 100% (both spans with labels correct)
- EM: 100% (all compounds correct)

Prediction 2 (One token wrong):
- Predicted spans: [(1,3,"Comp3"), (5,6,"Comp3")] ← Wrong label for span 2
- USS: 2/2 = 100% (span boundaries still correct!)
- LSS: 1/2 = 50% (one label wrong)
- EM: 0% (compound 2 wrong)

### Code's Actual Evaluation (Relation-based)

**True relations**: 
- (1,4,Bs), (2,4,Di), (3,4,Di), (5,6,Bs), (6,0,Comp_root)

Prediction 1 (Correct):
- Predicted relations same as true
- USS F1: 100%
- LSS F1: 100%
- EM: 100%

Prediction 2 (One token wrong):
- Predicted relations: (1,4,Bs), (2,4,Di), (3,4,Di), (5,6,Bs), (6,0,Comp2) ← Wrong
- This causes EM to fail (entire compound 2 fails)
- But USS would still count (5,6) as "correct" if we're just checking (token, head) pairs

**Wait** - actually looking at the code again...

```python
# USS only counts where true_lst[5]!='No_rel'
if true_lst[5]!='No_rel':
    true_span = [true_lst[0], true_lst[4]]
    if true_span==pred_span:
        correct += 1
```

This iterates line-by-line (token-by-token), not compound-by-compound. So:
- Token 1: checks (1,4) ✓
- Token 2: checks (2,4) ✓
- Token 3: checks (3,4) ✓
- Token 5: checks (5,6) ✓
- Token 6: skipped (it's root marker, might be in No_rel range)

So USS is counting **4 out of 5 relations correct = 80% F1**, not checking spans at all.

---

## Summary of Issues

| Metric | Paper Describes | Code Implements | Match? |
|--------|-----------------|-----------------|--------|
| **USS** | F1 on span boundaries (start, end) | F1 on (token, head) pairs | ❌ NO |
| **LSS** | F1 on (start, end, label) tuples | F1 on (token, head, label) tuples | ❌ NO |
| **EM** | Exact compounds with correct spans | Exact compounds as relation sequences | ⚠️ PARTIAL |

---

## The Core Problem

The paper describes **span-based evaluation** where you:
1. Extract compound boundaries as (start, end)
2. Evaluate: Did we predict the right boundaries?
3. For LSS: Did we assign the right label?

The code does **relation-based evaluation** where you:
1. Extract token-to-head relations
2. Evaluate: Did we predict the right relations?
3. This conflates TWO different things:
   - Span boundaries (which tokens form a compound)
   - Dependency structure (token pointing to head)

For nested compounds, this is especially problematic because:
- A compound might have tokens [1,2,3] pointing to root 4
- But there could be SUB-compounds within it (e.g., tokens [1,2] at Comp2 level, tokens [1,2,3] at Comp3 level)
- Relation-based evaluation loses the hierarchical structure
- Span-based evaluation would capture it

---

## Correct Span-Based Implementation (What Should Have Been Done)

```python
def extract_compound_spans(lines):
    """Extract compound spans as (start_token, end_token, label) tuples"""
    spans = []
    compound_stack = []
    
    for line in lines:
        if line == '\n':
            # Add all open compounds to spans list
            while compound_stack:
                comp_level, start_token, label = compound_stack.pop()
                end_token = current_token - 1
                spans.append((start_token, end_token, comp_level))
        else:
            lst = line.strip().split('\t')
            current_token = int(lst[0])
            comp_level = lst[9]  # Get compound level
            
            # Open new compound if marked as start
            if 'Start' in lst[9]:
                compound_stack.append((comp_level, current_token, lst[9]))
    
    return spans

def uss_correct(true_lines, pred_lines):
    """USS: F1 on span boundaries ONLY"""
    true_spans = extract_compound_spans(true_lines)
    pred_spans = extract_compound_spans(pred_lines)
    
    true_boundaries = [(s[0], s[1]) for s in true_spans]  # (start, end) only
    pred_boundaries = [(s[0], s[1]) for s in pred_spans]
    
    correct = len(set(true_boundaries) & set(pred_boundaries))
    p = correct / len(set(pred_boundaries))
    r = correct / len(set(true_boundaries))
    return 2*p*r/(p+r) if (p+r)>0 else 0

def lss_correct(true_lines, pred_lines):
    """LSS: F1 on (start, end, label) tuples"""
    true_spans = extract_compound_spans(true_lines)
    pred_spans = extract_compound_spans(pred_lines)
    
    correct = len(set(true_spans) & set(pred_spans))
    p = correct / len(set(pred_spans))
    r = correct / len(set(true_spans))
    return 2*p*r/(p+r) if (p+r)>0 else 0
```

---

## Conclusion

**The DepNeCTI paper and codebase have a fundamental MISMATCH in evaluation methodology:**

1. ✅ **EM** is relatively correctly implemented (checks compound structure)
2. ❌ **USS** is INCORRECTLY implemented (uses relations, not spans)
3. ❌ **LSS** is INCORRECTLY implemented (uses relations, not spans)

The paper describes **span-based evaluation** (standard for NER/compound extraction) but the code implements **relation-based evaluation** (closer to dependency parsing).

### Impact

This means:
- USS/LSS numbers reported in papers may not be comparable to truly span-based methods
- The metrics are actually measuring something different than described
- USS and LSS conflate span boundary accuracy with dependency structure accuracy
- For proper reproduction, you'd need to clarify which evaluation was actually used

### Recommendation

If you're using this evaluation code, you should:
1. ✅ Trust the EM metric (it's closer to correct)
2. ⚠️ Be cautious with USS/LSS (they're actually relation-based, not span-based)
3. 🔧 Consider fixing the USS/LSS implementation to match the paper's description
4. 📝 Document which evaluation method you actually used when reporting results
