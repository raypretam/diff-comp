#!/bin/bash

# Run SaCTI training with coarse-grained labels
# Usage: bash run_sacti_coarse.sh

echo "=========================================="
echo "SaCTI Coarse-grained Training"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python3 trainer_sacti.py \
    --config_file sacti_coarse.yaml \
    --data_path "/home/pretam-pg/SaCTI/data" \
    --output_dir "saved_models/sacti_coarse" \
    --max_epochs 50 \
    --batch_size 32 \
    --lr_bert 2e-5 \
    --lr_other 1e-4 \
    --patience 10 \
    --granularity coarse

echo ""
echo "Training completed! Check saved_models/sacti_coarse/ for results."
