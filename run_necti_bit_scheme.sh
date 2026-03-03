#!/bin/bash
# Run Bit4 / Bit7 / HexaTagging NeCTI experiment.
#
# Usage:
#   bash run_necti_bit_scheme.sh <scheme> <granularity> [extra_flags...]
#
# Examples:
#   bash run_necti_bit_scheme.sh bit4 Coarse --use_context
#   bash run_necti_bit_scheme.sh bit7 Finegrain --use_context
#   bash run_necti_bit_scheme.sh hexa Coarse
#   bash run_necti_bit_scheme.sh bit4 Finegrain --use_context --logger None

SCHEME=${1:-bit4}        # bit4 | bit7 | hexa
GRANULARITY=${2:-Coarse} # Coarse | Finegrain
shift 2 2>/dev/null || true

echo "========================================"
echo "NeCTI Bit-Scheme Training"
echo "  Scheme      : ${SCHEME}"
echo "  Granularity : ${GRANULARITY}"
echo "  Extra args  : $@"
echo "========================================"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

BACKBONE="FacebookAI/xlm-roberta-large"
DATA_PATH="/home/pretam-pg/DepNeCTI/DepNeCTI-XLMR/Trankit_Data"

python trainer_necti_bit_schemes.py \
    --config_file necti_bit_scheme.yaml \
    --scheme         "${SCHEME}" \
    --granularity    "${GRANULARITY}" \
    --backbone       "${BACKBONE}" \
    --data_path      "${DATA_PATH}" \
    --struct_aux_weight 0.3 \
    --logger         wandb \
    --batch_size     4 \
    --max_epochs     50 \
    --lr_bert        5e-6 \
    --lr_other       3e-4 \
    --num_workers    4 \
    --max_length     256 \
    --time_steps     1000 \
    --sampling_steps 10 \
    --depth          6 \
    "$@"

echo "Done: ${SCHEME} / ${GRANULARITY}"
