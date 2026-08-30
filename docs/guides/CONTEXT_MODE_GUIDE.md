# Context Mode Feature Guide

## Overview

This guide explains the new `--use_context` argument that allows you to choose between "With Context" and "Without Context" data for training and inference in the NeCTI compound identification system.

## What's New

### Command-Line Argument

A new `--use_context` flag has been added to control which data directory to use:
- **Without flag** (default): Uses data from `Without Context/` directory
- **With flag**: Uses data from `With Context/` directory

## Training

### Training Scripts

Both training scripts now support the context mode argument:

#### Coarse-grained Training
```bash
# Edit the script to choose mode
vim run_necti_coarse.sh

# In the script, comment/uncomment these lines:
# USE_CONTEXT=""              # Without Context (default)
USE_CONTEXT="--use_context"   # With Context

# Run training
./run_necti_coarse.sh
```

#### Fine-grained Training
```bash
# Edit the script to choose mode
vim run_necti_finegrain.sh

# In the script, comment/uncomment these lines:
# USE_CONTEXT=""              # Without Context (default)
USE_CONTEXT="--use_context"   # With Context

# Run training
./run_necti_finegrain.sh
```

#### Direct Python Command
```bash
# Without context (default)
python trainer_necti.py \
    --config_file necti_coarse_xlmr.yaml \
    --granularity Coarse \
    --backbone xlm-roberta-base \
    --data_path "/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data" \
    --batch_size 16 \
    --max_epochs 20

# With context
python trainer_necti.py \
    --config_file necti_coarse_xlmr.yaml \
    --granularity Coarse \
    --backbone xlm-roberta-base \
    --data_path "/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data" \
    --use_context \
    --batch_size 16 \
    --max_epochs 20
```

### Model Saving

Models are now saved with context mode in the directory name:

- **Without Context**: `saved_models/necti_Coarse_no_ctx/`
- **With Context**: `saved_models/necti_Coarse_with_ctx/`
- **Without Context (Fine-grained)**: `saved_models/necti_Finegrain_no_ctx/`
- **With Context (Fine-grained)**: `saved_models/necti_Finegrain_with_ctx/`

Each directory contains:
- `best_model.pt` - Latest best model
- `best_model_epoch{N}_f1{score}.pt` - Checkpoint files

## Inference

### Inference Script

The inference script now accepts a second argument for context mode:

```bash
# Usage: ./run_inference_necti.sh [coarse|finegrain] [no_ctx|with_ctx]

# Examples:
./run_inference_necti.sh coarse no_ctx      # Coarse without context
./run_inference_necti.sh coarse with_ctx    # Coarse with context
./run_inference_necti.sh finegrain no_ctx   # Fine-grained without context
./run_inference_necti.sh finegrain with_ctx # Fine-grained with context
```

### Direct Python Command

```bash
# Without context
python inference_necti.py \
    --model_path "./saved_models/necti_Coarse_no_ctx/best_model.pt" \
    --data_path "/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data" \
    --granularity Coarse \
    --splits test dev ood \
    --device cuda \
    --save_predictions \
    --output_dir "./inference_results/necti_Coarse_no_ctx"

# With context
python inference_necti.py \
    --model_path "./saved_models/necti_Coarse_with_ctx/best_model.pt" \
    --data_path "/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data" \
    --granularity Coarse \
    --use_context \
    --splits test dev ood \
    --device cuda \
    --save_predictions \
    --output_dir "./inference_results/necti_Coarse_with_ctx"
```

### Inference Results

Results are saved with context mode in the directory name:

- **Without Context**: `inference_results/necti_Coarse_no_ctx/`
- **With Context**: `inference_results/necti_Coarse_with_ctx/`

Each directory contains:
- `{split}_predictions.json` - Detailed predictions
- `{split}_metrics.json` - Evaluation metrics

## Data Structure

The system expects the following directory structure:

```
/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/
├── Without Context/
│   ├── Coarse/
│   │   ├── Coarse_train_san
│   │   ├── Coarse_dev_san
│   │   ├── Coarse_test_san
│   │   └── Coarse_ood_san
│   └── Finegrain/
│       ├── Finegrain_train_san
│       ├── Finegrain_dev_san
│       ├── Finegrain_test_san
│       └── Finegrain_ood_san
└── With Context/
    ├── Coarse/
    │   ├── Coarse_train_san
    │   ├── Coarse_dev_san
    │   ├── Coarse_test_san
    │   └── Coarse_ood_san
    └── Finegrain/
        ├── Finegrain_train_san
        ├── Finegrain_dev_san
        ├── Finegrain_test_san
        └── Finegrain_ood_san
```

## Technical Details

### Modified Files

1. **options.py**: Added `--use_context` argument
2. **data/ner/necti_dataset.py**: 
   - Updated `NeCTILabelSet` to accept `use_context` parameter
   - Updated `NeCTIDataset` to accept `use_context` parameter
3. **trainer_necti.py**:
   - Pass `use_context` to datasets
   - Include context mode in model save paths
   - Include context mode in WandB run names
4. **inference_necti.py**:
   - Accept `use_context` parameter
   - Pass it to datasets
   - Display context mode in output
5. **run_necti_coarse.sh**: Added `USE_CONTEXT` variable
6. **run_necti_finegrain.sh**: Added `USE_CONTEXT` variable
7. **run_inference_necti.sh**: Added context mode as second argument

### Backward Compatibility

The changes are backward compatible:
- By default (without the flag), the system uses "Without Context" data
- Existing scripts and commands will continue to work
- The `use_context` parameter defaults to `False` in all functions

## Examples

### Complete Training Workflow

```bash
# 1. Train without context
./run_necti_coarse.sh
# Model saved to: saved_models/necti_Coarse_no_ctx/

# 2. Train with context (edit script first)
# Change: USE_CONTEXT="--use_context"
./run_necti_coarse.sh
# Model saved to: saved_models/necti_Coarse_with_ctx/
```

### Complete Inference Workflow

```bash
# 1. Inference on model trained without context
./run_inference_necti.sh coarse no_ctx
# Results in: inference_results/necti_Coarse_no_ctx/

# 2. Inference on model trained with context
./run_inference_necti.sh coarse with_ctx
# Results in: inference_results/necti_Coarse_with_ctx/
```

### Comparing Results

```bash
# Train both models
./run_necti_coarse.sh  # Edit script to switch between modes

# Run inference on both
./run_inference_necti.sh coarse no_ctx
./run_inference_necti.sh coarse with_ctx

# Compare metrics
cat inference_results/necti_Coarse_no_ctx/test_metrics.json
cat inference_results/necti_Coarse_with_ctx/test_metrics.json
```

## WandB Integration

WandB run names now include the context mode:
- Without context: `necti-Coarse-no_ctx--lr_bert_2e-5--lr_other_5e-4--epochs_20`
- With context: `necti-Coarse-with_ctx--lr_bert_2e-5--lr_other_5e-4--epochs_20`

This makes it easy to compare experiments with different context modes in the WandB dashboard.

## Troubleshooting

### Model Not Found Error

If you get a "Model not found" error during inference:
1. Check that you used the correct context mode during training
2. Verify the model path matches the context mode:
   - `saved_models/necti_{Granularity}_no_ctx/` for without context
   - `saved_models/necti_{Granularity}_with_ctx/` for with context

### Data File Not Found Error

If you get a "Data file not found" error:
1. Verify the data directory structure matches the expected layout
2. Check that both "With Context" and "Without Context" directories exist
3. Ensure the data files are named correctly (e.g., `Coarse_train_san`)

## Notes

- The context mode must match between training and inference
- Different context modes produce different models stored in separate directories
- You can train and evaluate models with both context modes simultaneously
- All metrics and predictions are saved separately for each context mode
