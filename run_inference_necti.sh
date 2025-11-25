#!/bin/bash

# Inference script for NeCTI compound identification
# Usage: ./run_inference_necti.sh [coarse|finegrain] [no_ctx|with_ctx]
# Examples:
#   ./run_inference_necti.sh coarse no_ctx    # Coarse-grained without context
#   ./run_inference_necti.sh coarse with_ctx  # Coarse-grained with context
#   ./run_inference_necti.sh finegrain with_ctx  # Fine-grained with context

GRANULARITY=${1:-Coarse}  # Default to Coarse if not specified
USE_CONTEXT_FLAG=${2:-no_ctx}  # Default to no_ctx if not specified (use 'with_ctx' for With Context)

# Convert to proper case
if [ "$GRANULARITY" = "coarse" ]; then
    GRANULARITY="Coarse"
elif [ "$GRANULARITY" = "finegrain" ]; then
    GRANULARITY="Finegrain"
fi

# Set context mode
if [ "$USE_CONTEXT_FLAG" = "with_ctx" ]; then
    CONTEXT_MODE="with_ctx"
    USE_CONTEXT_ARG="--use_context"
else
    CONTEXT_MODE="no_ctx"
    USE_CONTEXT_ARG=""
fi

# Paths
DATA_PATH="/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data"
MODEL_PATH="./saved_models/necti_${GRANULARITY}_${CONTEXT_MODE}/best_model.pt"
OUTPUT_DIR="./inference_results/necti_${GRANULARITY}_${CONTEXT_MODE}"

echo "=================================================="
echo "NeCTI Inference - ${GRANULARITY}"
echo "Context Mode: ${CONTEXT_MODE}"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    echo "Please train the model first using trainer_necti.py"
    exit 1
fi

# Run inference
python inference_necti.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --granularity "$GRANULARITY" \
    ${USE_CONTEXT_ARG} \
    --splits test dev ood \
    --batch_size 16 \
    --device cuda \
    --save_predictions \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=================================================="
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================================="
