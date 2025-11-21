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
BACKBONE="xlm-roberta-base"
DATA_PATH="/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data"

# Run training
python trainer_necti.py \
    --config_file ${CONFIG_FILE} \
    --granularity ${GRANULARITY} \
    --backbone ${BACKBONE} \
    --data_path ${DATA_PATH} \
    --logger wandb \
    --batch_size 16 \
    --max_epochs 20 \
    --lr_bert 2e-5 \
    --lr_other 5e-4 \
    --num_workers 4 \
    --max_length 256 \
    --time_steps 1000 \
    --sampling_steps 10 \
    --depth 6

echo "Training completed!"
