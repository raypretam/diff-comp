#!/bin/bash

# Training script for NeCTI Finegrain nested compound identification
# Using DiffusionSL with XLM-RoBERTa encoder

echo "========================================"
echo "NeCTI Finegrain Training"
echo "Using XLM-RoBERTa encoder"
echo "========================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Training configuration
CONFIG_FILE="necti_finegrain_xlmr.yaml"
GRANULARITY="Finegrain"
BACKBONE="FacebookAI/xlm-roberta-large"  # Changed from base to large (1024 dims)
DATA_PATH="/home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data"

# Context mode: uncomment one of the following
USE_CONTEXT=""  # Without Context (default)
# USE_CONTEXT="--use_context"  # With Context

# Run training
CUDA_VISIBLE_DEVICES=0 python trainer_necti.py \
    --config_file ${CONFIG_FILE} \
    --granularity ${GRANULARITY} \
    --backbone ${BACKBONE} \
    --data_path ${DATA_PATH} \
    ${USE_CONTEXT} \
    --logger none \
    --batch_size 16 \
    --max_epochs 200 \
    --lr_bert 2e-5 \
    --lr_other 5e-4 \
    --num_workers 4 \
    --max_length 256 \
    --time_steps 1000 \
    --sampling_steps 10 \
    --depth 12

echo "Training completed!"
