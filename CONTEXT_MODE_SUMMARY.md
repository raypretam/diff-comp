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

## Notes

- Context mode must match between training and inference
- Each context mode creates independent models and results
- Can train and evaluate both modes simultaneously
- All changes are backward compatible
