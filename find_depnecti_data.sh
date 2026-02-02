#!/bin/bash
# Helper script to locate DepNeCTI true data files

DATA_BASE="/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data"

echo "=========================================="
echo "DepNeCTI Data File Locations"
echo "=========================================="
echo ""

for context in "Without Context" "With Context"; do
    echo "📁 $context:"
    context_path="$DATA_BASE/$context"
    if [ -d "$context_path" ]; then
        for granularity in "Coarse" "Finegrain"; do
            gran_path="$context_path/$granularity/formatted"
            echo "   └─ $granularity:"
            if [ -d "$gran_path" ]; then
                for file in train.txt dev.txt test.txt ood.txt; do
                    if [ -f "$gran_path/$file" ]; then
                        lines=$(wc -l < "$gran_path/$file")
                        printf "      ✓ %-10s (%6d lines) %s\n" "$file" "$lines" "$gran_path/$file"
                    fi
                done
            else
                echo "      ✗ Not found: $gran_path"
            fi
        done
    else
        echo "   ✗ Not found: $context_path"
    fi
    echo ""
done

echo "=========================================="
echo ""
echo "Usage Example:"
echo "  python export_predictions_to_conllu.py \\"
echo "      --model_path ./saved_models/necti_Coarse_best.pt \\"
echo "      --granularity Coarse \\"
echo "      --split test"
echo ""
echo "Then evaluate:"
echo "  python /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py \\"
echo "      '$DATA_BASE/Without Context/Coarse/formatted/test.txt' \\"
echo "      predictions_Coarse_test.conllu"
