# DiffusionSL for Nested Compound Identification (DepNeCTI)

This directory contains the integration of DiffusionSL framework with DepNeCTI data for nested compound identification in Sanskrit texts using XLM-RoBERTa encoder.

## Overview

The implementation adapts the DiffusionSL sequence labeling framework for the task of nested compound identification. The model uses:
- **Encoder**: XLM-RoBERTa (xlm-roberta-base) - a multilingual pretrained transformer
- **Decoder**: Diffusion-based sequence labeling model (DiT - Diffusion Transformer)
- **Dataset**: DepNeCTI with context (from `/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/With Context`)

## Architecture

### Model Components
1. **XLM-R Encoder**: Extracts contextual representations for Sanskrit tokens
2. **Diffusion Process**: Models compound label sequences through iterative denoising
3. **DiT Decoder**: Transformer-based decoder for predicting compound labels at each diffusion step

### Key Features
- Handles nested compound structures (e.g., <<A-B>-C> type compounds)
- Supports both Coarse and Finegrain granularity levels
- Uses DDIM sampling for efficient inference
- Implements self-conditioning for improved performance

## File Structure

```
DiffusionSL/
├── data/ner/
│   ├── necti_dataset.py          # Dataset loader for DepNeCTI data
│   └── ner_dataset.py             # Original NER dataset (for reference)
├── models/
│   ├── ddim_bitdit.py             # Main diffusion model (BitDit)
│   └── dit_discrete.py            # DiT architecture
├── configs/
│   ├── necti_coarse_xlmr.yaml     # Config for coarse-grained training
│   └── necti_finegrain_xlmr.yaml  # Config for finegrain training
├── trainer_necti.py               # Training script for NeCTI
├── run_necti_coarse.sh           # Shell script for coarse training
├── run_necti_finegrain.sh        # Shell script for finegrain training
└── NECTI_README.md               # This file
```

## Data Format

The DepNeCTI data is in CoNLL-U format:

```
1    sa         Comp2    _    2    BvS          # Coarse label: BvS (Bahuvrihi)
2    sarzapaM   Comp2    _    11   Comp_root
3    tumburu    Comp3    _    5    Ds           # Coarse label: Ds (Dvandva)
4    DAnya      Comp3    _    5    Ds
5    vanyaM     Comp3    _    11   Comp_root
6    caRqAM     CompNo   _    11   No_rel       # Not a compound
```

### Label Categories

**Coarse Granularity:**
- CompNo: Not part of a compound
- Comp2, Comp3, etc.: Compound members
- Relation types: Tatpurusha, Dvandva, Bahuvrihi, Avyayibhava, etc.

**Finegrain Granularity:**
- Uses abbreviated codes: T6, T7, Ds, BvS, U, K1, etc.
- Provides more detailed compound type information

## Installation & Requirements

### Dependencies
```bash
pip install torch transformers wandb pyyaml prettytable tqdm
```

### Pretrained Model
The XLM-RoBERTa model will be automatically downloaded from HuggingFace:
- Model: `xlm-roberta-base`
- Size: ~278M parameters (encoder only)

## Usage

### Training

#### 1. Coarse-grained Compound Identification
```bash
cd /home/pretam-pg/DiffusionSL
bash run_necti_coarse.sh
```

Or run directly with Python:
```bash
python trainer_necti.py \
    --config_file necti_coarse_xlmr.yaml \
    --granularity Coarse \
    --backbone xlm-roberta-base \
    --data_path /home/pretam-pg/DepNeCTI/data/NeCTIS\ Model\ Data \
    --logger wandb \
    --batch_size 16 \
    --max_epochs 20
```

#### 2. Finegrain Compound Identification
```bash
bash run_necti_finegrain.sh
```

### Training Configuration

Key hyperparameters (can be modified in config files or command line):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | xlm-roberta-base | Encoder model |
| `batch_size` | 16 | Training batch size |
| `max_epochs` | 20 | Number of training epochs |
| `lr_bert` | 2e-5 | Learning rate for XLM-R encoder |
| `lr_other` | 5e-4 | Learning rate for diffusion decoder |
| `time_steps` | 1000 | Number of diffusion steps |
| `sampling_steps` | 10 | Inference sampling steps (DDIM) |
| `max_length` | 256 | Maximum sequence length |
| `depth` | 6 | DiT decoder depth |

### Monitoring Training

If using wandb logger:
```bash
wandb login
# Training metrics will be logged to wandb dashboard
```

To disable logging:
```bash
python trainer_necti.py --config_file necti_coarse_xlmr.yaml --logger None
```

### Evaluation

The training script automatically evaluates on:
- **Dev set**: After each epoch (used for model selection)
- **Test set**: After training completes (final evaluation)
- **OOD set**: If available (out-of-domain generalization)

Metrics reported:
- **Precision**: Ratio of correctly identified compounds
- **Recall**: Ratio of actual compounds detected
- **F1 Score**: Harmonic mean of precision and recall

## Model Outputs

Trained models are saved in:
```
saved_models/necti_{granularity}/
├── best_model.pt                    # Best model based on dev F1
└── best_model_epoch{N}_f1{score}.pt # Checkpoints with epoch info
```

## Customization

### Using Different Encoder

To use a different multilingual model:
```bash
python trainer_necti.py \
    --config_file necti_coarse_xlmr.yaml \
    --backbone ai4bharat/IndicBERT \  # Or any HuggingFace model
    --dim_model 768  # Adjust based on model hidden size
```

### Adjusting Diffusion Parameters

For faster inference (fewer sampling steps):
```bash
python trainer_necti.py \
    --config_file necti_coarse_xlmr.yaml \
    --sampling_steps 5  # Reduce from 10
```

For better quality (more diffusion steps):
```bash
python trainer_necti.py \
    --config_file necti_coarse_xlmr.yaml \
    --time_steps 2000  # Increase from 1000
```

## Dataset Path Configuration

The dataset path points to:
```
/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/With Context/
├── Coarse/
│   ├── Coarse_train_san
│   ├── Coarse_dev_san
│   ├── Coarse_test_san
│   └── Coarse_ood_san
└── Finegrain/
    ├── Finegrain_train_san
    ├── Finegrain_dev_san
    ├── Finegrain_test_san
    └── Finegrain_ood_san
```

To use a different data location, modify `--data_path` argument.

## Technical Details

### Data Processing
1. **Tokenization**: XLM-R subword tokenization with word-level alignment
2. **Label Alignment**: First subword of each word gets the label, others are masked (-100)
3. **Padding**: Dynamic padding to longest sequence in batch

### Training Process
1. **Forward Diffusion**: Add noise to ground-truth labels over T timesteps
2. **Denoising**: Train model to predict clean labels from noisy versions
3. **Sampling**: Use DDIM for fast generation during inference

### Loss Function
- MSE loss (L2) on predicted noise/labels in latent space
- Masked loss (ignores padding tokens)
- Gradient clipping for stability

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or sequence length:
```bash
python trainer_necti.py --config_file necti_coarse_xlmr.yaml --batch_size 8 --max_length 128
```

### Slow Training
- Reduce `num_workers` if CPU is bottleneck
- Use fewer `time_steps` for faster training
- Enable mixed precision training (requires code modification)

### Poor Performance
- Increase `max_epochs` (20 → 30)
- Adjust learning rates (`lr_bert`, `lr_other`)
- Try different `snr_scale` values (1.0 - 3.0)
- Enable `self_condition` for better quality

## Citation

If you use this code, please cite:

**DiffusionSL:**
```bibtex
@inproceedings{diffusionsl2023,
    title={DiffusionSL: Sequence Labeling via Tag Diffusion Process},
    booktitle={EMNLP 2023 Findings},
    year={2023}
}
```
**DepNeCTI**
```bibtex
@inproceedings{sandhan-etal-2023-depnecti,
    title = "{D}ep{N}e{CTI}: Dependency-based Nested Compound Type Identification for {S}anskrit",
    author = "Sandhan, Jivnesh  and
      Narsupalli, Yaswanth  and
      Muppirala, Sreevatsa  and
      Krishnan, Sriram  and
      Satuluri, Pavankumar  and
      Kulkarni, Amba  and
      Goyal, Pawan",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.914/",
    doi = "10.18653/v1/2023.findings-emnlp.914",
    pages = "13679--13692",
    abstract = "Multi-component compounding is a prevalent phenomenon in Sanskrit, and understanding the implicit structure of a compound{'}s components is crucial for deciphering its meaning. Earlier approaches in Sanskrit have focused on binary compounds and neglected the multi-component compound setting. This work introduces the novel task of nested compound type identification (NeCTI), which aims to identify nested spans of a multi-component compound and decode the implicit semantic relations between them. To the best of our knowledge, this is the first attempt in the field of lexical semantics to propose this task. We present 2 newly annotated datasets including an out-of-domain dataset for this task. We also benchmark these datasets by exploring the efficacy of the standard problem formulations such as nested named entity recognition, constituency parsing and seq2seq, etc. We present a novel framework named DepNeCTI: Dependency-based Nested Compound Type Identifier that surpasses the performance of the best baseline with an average absolute improvement of 13.1 points F1-score in terms of Labeled Span Score (LSS) and a 5-fold enhancement in inference efficiency. In line with the previous findings in the binary Sanskrit compound identification task, context provides benefits for the NeCTI task. The codebase and datasets are publicly available at: https://github.com/yaswanth-iitkgp/DepNeCTI"
}
```

## License

This integration follows the licenses of both DiffusionSL and DepNeCTI projects.
