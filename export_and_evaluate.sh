#!/bin/bash
# Export predictions to CoNLL-U format and evaluate against DepNeCTI

set -e

# Configuration
MODEL_PATH="${1:?Error: MODEL_PATH required}"
DATA_PATH="${2:-/home/pretam-pg/DepNeCTI/data/NeCTIS\ Model\ Data}"
GRANULARITY="${3:-Coarse}"
SPLIT="${4:-test}"
USE_CONTEXT="${5:-false}"

echo "=========================================="
echo "DiffusionSL → CoNLL-U Export & Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Granularity: $GRANULARITY"
echo "Split: $SPLIT"
echo "Context: $USE_CONTEXT"
echo "=========================================="

# Export predictions
CONTEXT_FLAG=""
if [ "$USE_CONTEXT" = "true" ]; then
    CONTEXT_FLAG="--use_context"
fi

OUTPUT_FILE="predictions_${GRANULARITY}_${SPLIT}.conllu"
OUTPUT_DIR="./inference_results"
mkdir -p "$OUTPUT_DIR"
FULL_OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

echo ""
echo "📤 Exporting predictions to CoNLL-U format..."
python export_predictions_to_conllu.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --granularity "$GRANULARITY" \
    --split "$SPLIT" \
    --output_file "$FULL_OUTPUT_PATH" \
    $CONTEXT_FLAG

echo ""
echo "✓ Predictions exported to: $FULL_OUTPUT_PATH"

# Locate DepNeCTI true file
if [ "$USE_CONTEXT" = "true" ]; then
    CONTEXT_SUFFIX="With Context"
else
    CONTEXT_SUFFIX="Without Context"
fi

TRUE_FILE_DIR="$DATA_PATH/$CONTEXT_SUFFIX/$GRANULARITY/formatted"
TRUE_FILE="$TRUE_FILE_DIR/${SPLIT}.txt"

if [ ! -f "$TRUE_FILE" ]; then
    echo "⚠️  Warning: True file not found at $TRUE_FILE"
    echo "   Please specify the correct path manually"
    echo ""
    echo "📊 To evaluate, run:"
    echo "   python /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py \\"
    echo "       <true_file.txt> $FULL_OUTPUT_PATH"
else
    echo ""
    echo "📊 Evaluating predictions..."
    echo "   True file: $TRUE_FILE"
    echo "   Pred file: $FULL_OUTPUT_PATH"
    echo ""
    python /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py "$TRUE_FILE" "$FULL_OUTPUT_PATH"
fi

echo ""
echo "=========================================="
echo "✓ Export & Evaluation Complete!"
echo "=========================================="
