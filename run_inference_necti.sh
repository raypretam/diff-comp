#!/bin/bash

# Inference script for NeCTI compound identification
# Usage: ./run_inference_necti.sh [coarse|finegrain]

GRANULARITY=${1:-Coarse}  # Default to Coarse if not specified

# Convert to proper case
if [ "$GRANULARITY" = "coarse" ]; then
    GRANULARITY="Coarse"
elif [ "$GRANULARITY" = "finegrain" ]; then
    GRANULARITY="Finegrain"
fi

# Paths
DATA_PATH="/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/With Context"
MODEL_PATH="./saved_models/necti_${GRANULARITY}/best_model.pt"
OUTPUT_DIR="./inference_results/necti_${GRANULARITY}"

echo "=================================================="
echo "NeCTI Inference - ${GRANULARITY}"
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
    --splits test dev \
    --batch_size 16 \
    --device cuda \
    --save_predictions \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=================================================="
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================================="
