#!/bin/bash
# Bracket-augmented NeCTI training — Coarse granularity
# Each token gets a JOINT label: bracket_string × compound_type
# An auxiliary bracket head provides structural supervision on top of XLM-R.

echo "========================================"
echo "NeCTI Bracket Training (Coarse)"
echo "Bracket k=2 planar encoding"
echo "========================================"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

DATA_PATH="/home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data"
BACKBONE="FacebookAI/xlm-roberta-large"

# Toggle context mode:
# USE_CONTEXT=""          # Without Context
USE_CONTEXT="--use_context"  # With Context

python trainer_necti_bracket.py \
    --config_file necti_bracket_coarse.yaml \
    --granularity Coarse \
    --backbone ${BACKBONE} \
    --data_path ${DATA_PATH} \
    ${USE_CONTEXT} \
    --bracket_k 2 \
    --bracket_aux_weight 0.3 \
    --logger None \
    --batch_size 4 \
    --max_epochs 50 \
    --lr_bert 5e-6 \
    --lr_other 3e-4 \
    --num_workers 4 \
    --max_length 256 \
    --time_steps 1000 \
    --sampling_steps 10 \
    --depth 6

echo "Training completed!"
