#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1   # run from repo root

# Quick Start Guide for NeCTI Training with DiffusionSL
# This script helps you get started with nested compound identification

echo "=========================================="
echo "NeCTI + DiffusionSL Setup & Quick Start"
echo "=========================================="

# Navigate to DiffusionSL directory

echo -e "\n1. Testing dataset loading..."
python tests/test_necti_dataset.py

if [ $? -eq 0 ]; then
    echo -e "\n✓ Dataset loading test passed!"
else
    echo -e "\n✗ Dataset loading test failed. Please check the error messages."
    exit 1
fi

echo -e "\n=========================================="
echo "Setup Complete! You can now:"
echo "=========================================="
echo ""
echo "1. Train Coarse-grained model:"
echo "   bash run_necti_coarse.sh"
echo ""
echo "2. Train Finegrain model:"
echo "   bash run_necti_finegrain.sh"
echo ""
echo "3. Custom training:"
echo "   python -m trainers.trainer_necti --config_file necti_coarse_xlmr.yaml --batch_size 8"
echo ""
echo "4. Monitor training:"
echo "   - With wandb: Login at https://wandb.ai"
echo "   - Without logging: Add --logger None to command"
echo ""
echo "5. Check results:"
echo "   ls saved_models/necti_coarse/"
echo "   ls saved_models/necti_finegrain/"
echo ""
echo "For more details, see docs/tasks/NECTI_README.md"
echo "=========================================="
