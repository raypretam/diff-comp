 # Context Mode Implementation Summary

## Quick Reference

### New Argument
- `--use_context`: Use "With Context" data instead of "Without Context" (default)

### Training
```bash
# Without context (default)
./run_necti_coarse.sh

# With context (edit script: USE_CONTEXT="--use_context")
./run_necti_coarse.sh
```

### Inference
```bash
# Without context
./run_inference_necti.sh coarse no_ctx

# With context
./run_inference_necti.sh coarse with_ctx
```

### Model Directories
- Without context: `saved_models/necti_{Granularity}_no_ctx/`
- With context: `saved_models/necti_{Granularity}_with_ctx/`

### Inference Results
- Without context: `inference_results/necti_{Granularity}_no_ctx/`
- With context: `inference_results/necti_{Granularity}_with_ctx/`

## Changes Made

### 1. options.py
Added argument:
```python
parser.add_argument("--use_context", action='store_true',
                   help="Use 'With Context' data instead of 'Without Context' data")
```

### 2. data/ner/necti_dataset.py
- `NeCTILabelSet`: Added `use_context` parameter
- `NeCTIDataset`: Added `use_context` parameter
- Both classes now select the correct data directory based on context mode

### 3. trainer_necti.py
- Passes `use_context` to label set and datasets
- Includes context mode in model save paths
- Includes context mode in WandB run names

### 4. inference_necti.py
- Accepts `use_context` parameter
- Passes it to label set and datasets
- Displays context mode in output

### 5. Shell Scripts
- `run_necti_coarse.sh`: Added `USE_CONTEXT` variable
- `run_necti_finegrain.sh`: Added `USE_CONTEXT` variable
- `run_inference_necti.sh`: Added context mode as second argument

## File Structure

```
DiffusionSL/
├── options.py                          # ✓ Modified
├── trainer_necti.py                    # ✓ Modified
├── inference_necti.py                  # ✓ Modified
├── run_necti_coarse.sh                 # ✓ Modified
├── run_necti_finegrain.sh              # ✓ Modified
├── run_inference_necti.sh              # ✓ Modified
├── data/ner/necti_dataset.py           # ✓ Modified
├── CONTEXT_MODE_GUIDE.md               # ✓ New
├── CONTEXT_MODE_SUMMARY.md             # ✓ New (this file)
└── saved_models/
    ├── necti_Coarse_no_ctx/            # Models without context
    ├── necti_Coarse_with_ctx/          # Models with context
    ├── necti_Finegrain_no_ctx/         # Models without context
    └── necti_Finegrain_with_ctx/       # Models with context
```

## Key Features

1. **Automatic Directory Selection**: Based on `use_context` flag
2. **Separate Model Storage**: Models for each mode stored separately
3. **Separate Results Storage**: Inference results stored separately
4. **WandB Integration**: Run names include context mode
5. **Backward Compatible**: Default behavior unchanged

## Testing Checklist

- [ ] Train model without context: `./run_necti_coarse.sh`
- [ ] Verify model saved to: `saved_models/necti_Coarse_no_ctx/`
- [ ] Run inference without context: `./run_inference_necti.sh coarse no_ctx`
- [ ] Verify results in: `inference_results/necti_Coarse_no_ctx/`
- [ ] Train model with context (edit script first)
- [ ] Verify model saved to: `saved_models/necti_Coarse_with_ctx/`
- [ ] Run inference with context: `./run_inference_necti.sh coarse with_ctx`
- [ ] Verify results in: `inference_results/necti_Coarse_with_ctx/`
- [ ] Compare metrics from both modes

## Notes

- Context mode must match between training and inference
- Each context mode creates independent models and results
- Can train and evaluate both modes simultaneously
- All changes are backward compatible
