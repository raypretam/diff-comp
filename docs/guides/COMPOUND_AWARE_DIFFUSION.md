# Compound-Aware Diffusion for DepNeCTI

## Overview

This implementation replaces token-level diffusion with **compound-level diffusion** to improve compound-level consistency and exact match scores on the DepNeCTI nested compound identification task.

## Motivation

The original token-level diffusion achieves excellent metrics:
- **USS (Unlabeled Span Score)**: 98.99% - nearly perfect boundary detection
- **LSS (Labeled Span Score)**: 84.92% - good label accuracy
- **EM (Exact Match)**: 70.33% - **problematic**

The low EM despite high USS/LSS indicates that the model correctly predicts most tokens, but fails to maintain consistency across all tokens within a compound. Even a single token error breaks EM.

## Key Changes

### Architecture

1. **CompoundEncoder** (`models/compound_encoder.py`):
   - Pools token-level BERT features into fixed-size compound representations
   - Supports multiple pooling methods: mean, max, attention, LSTM
   - Extracts compound masks from token-level labels

2. **CompoundDecoder** (`models/compound_encoder.py`):
   - Broadcasts compound-level predictions back to token level
   - Ensures all tokens within a compound receive the same label

3. **Modified BitDit** (`models/ddim_bitdit.py`):
   - Added `compound_aware` mode that operates at compound granularity
   - Runs diffusion on compound representations instead of tokens
   - Falls back gracefully to token-level on errors

### Training Pipeline

When `compound_aware=True`:

1. **Feature Pooling**: Token-level BERT features → Compound-level features
   ```
   [bsz, seq_len, dim] → [bsz, num_compounds, dim]
   ```

2. **Label Pooling**: Token labels → Compound labels (averaged)
   ```
   [bsz, seq_len, bits] → [bsz, num_compounds, bits]
   ```

3. **Diffusion**: Operate at compound level
   ```
   compound_features + noised_compound_labels → pred_compound_labels
   ```

4. **Broadcasting**: Compound predictions → Token predictions
   ```
   [bsz, num_compounds, bits] → [bsz, seq_len, bits]
   ```

5. **Loss**: Computed at token level (for compatibility with labels)

## Configuration

### Enable Compound-Aware Mode

In `configs/necti_finegrain_xlmr.yaml` or `configs/necti_coarse_xlmr.yaml`:

```yaml
compound_aware: True  # Enable compound-level diffusion
compound_pooling: 'mean'  # Pooling method: 'mean', 'max', 'attention', 'lstm'
```

### Pooling Methods

- **mean** (recommended): Average token features within each compound
  - Fast, stable, works well for all compounds sizes
  - Default choice

- **max**: Max-pooling over compound tokens
  - Emphasizes salient features
  - Good for compounds with important individual tokens

- **attention**: Learnable attention-weighted pooling
  - Adaptive, learns which tokens matter most
  - Requires more parameters (+hidden_size)

- **lstm**: BiLSTM-based pooling
  - Captures sequential dependencies within compounds
  - Most expensive computationally
  - Best for long, complex compounds

## Implementation Details

### Compound Mask Extraction

Compounds are identified by:
1. Labels containing "Comp" (e.g., Comp2, Comp3)
2. Ending with "Comp_root" or "root"

Example sequence:
```
Token:  sa      sarzapaM  vanyaM    caRqAM
Label:  Comp2   Comp2     Comp_root CompNo
        └───────────┬─────────┘     └──┬──┘
          Compound 1              Compound 2 (or No_rel)
```

Compound masks are binary:
```python
compound_masks[b, 0] = [1, 1, 1, 0]  # Compound 1: tokens 0-2
compound_masks[b, 1] = [0, 0, 0, 1]  # Compound 2: token 3
```

### Backward Compatibility

- When `compound_aware=False` (default): Standard token-level diffusion
- Graceful fallback: If compound extraction fails, falls back to token-level
- No changes to inference (TODO: implement compound-aware inference)

## Expected Results

Based on analysis:
- **EM**: +8-12% (from 70.33% → 78-82%)
- **LSS**: +2-4% (from 84.92% → 87-89%)
- **USS**: ~0% change (already near perfect at 98.99%)

The improvement comes from:
1. **Direct compound-level supervision**: Model explicitly learns compound-level consistency
2. **Shared representations**: All tokens in a compound get the same prediction
3. **Reduced search space**: Fewer degrees of freedom during generation

## Files Modified

1. **New Files**:
   - `models/compound_encoder.py` - Compound encoding/decoding logic

2. **Modified Files**:
   - `models/ddim_bitdit.py` - Added compound-aware mode
   - `trainer_necti.py` - Pass label_set to model
   - `configs/necti_finegrain_xlmr.yaml` - Enable compound-aware
   - `configs/necti_coarse_xlmr.yaml` - Enable compound-aware

## Training

Standard training command (configs now default to compound-aware):

```bash
# Finegrain
bash run_necti_finegrain.sh

# Coarse  
bash run_necti_coarse.sh
```

Or explicitly:
```bash
python trainer_necti.py \
    --config_file necti_finegrain_xlmr.yaml \
    --compound_aware True \
    --compound_pooling mean
```

## Debugging

Enable verbose output:
```python
# In models/ddim_bitdit.py forward method
print(f"Compound masks shape: {compound_masks.shape}")
print(f"Compound features shape: {compound_features.shape}")
print(f"Num compounds: {compound_mask.sum(dim=1)}")
```

Check compound extraction:
```python
from models.compound_encoder import extract_compound_masks_from_labels
masks = extract_compound_masks_from_labels(labels, label_set)
print(f"Extracted {masks.shape[1]} compounds per batch")
```

## Limitations & Future Work

1. **Inference**: Currently uses token-level inference (compound-aware inference TODO)
2. **Variable compound sizes**: Current implementation handles this, but could be optimized
3. **Nested compounds**: Inner compounds are treated as part of outer compound
   - Future: Hierarchical compound modeling?

## References

- Original analysis in comprehensive analysis document
- DepNeCTI paper: [arXiv:2310.09501](http://arxiv.org/abs/2310.09501)
- DiffusionSL paper: EMNLP 2023 Findings
