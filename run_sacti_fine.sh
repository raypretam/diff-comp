#!/bin/bash

# Run SaCTI training with fine-grained labels
# Usage: bash run_sacti_fine.sh

echo "=========================================="
echo "SaCTI Fine-grained Training"
echo "=========================================="

CUDA VISIBLE_DEVICES=1 python3 trainer_sacti.py \
    --config_file configs/sacti_fine.yaml \
    --data_path "/home/pretam-pg/SaCTI/data" \
    --output_dir "saved_models/sacti_fine" \
    --max_epochs 70 \
    --batch_size 16 \
    --lr_bert 2e-5 \
    --lr_other 1e-4 \
    --patience 15 \
    --granularity fine

echo ""
echo "Training completed! Check saved_models/sacti_fine/ for results."
