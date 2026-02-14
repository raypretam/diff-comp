# SaCTI Integration with DiffusionSL

This integration adapts DiffusionSL for **Sanskrit Compound Type Identification (SaCTI)**, a token-level classification task.

## Task Description

**SaCTI** (Sanskrit Compound Type Identifier) classifies each token in a sentence with its compound type:

- **Coarse-grained**: Main compound categories (Tatpurusha, Avyayibhava, Bahuvrihi, etc.)
- **Fine-grained**: Specific subtypes (T6, Km, Bs6, U, K1, etc.)

Unlike DepNeCTI which predicts dependency relations and extracts spans, SaCTI is a simpler **token-level classification task**.

## Dataset Format

SaCTI uses CoNLL format:
```
ID  WORD  _  _  _  _  HEAD  COMPOUND_TYPE  GRAM_ROLE  HEAD_G  _  _
1   योगः  _  _  _  _  15    T6            karwa      2       _  _
2   विहितः _  _  _  _  15    T6            root       0       _  _
```

- **Column 7**: Compound type (our target label)
- **Column 8**: Grammatical role (auxiliary information)

## Differences from DepNeCTI

| Aspect | DepNeCTI | SaCTI |
|--------|----------|-------|
| **Task** | Nested compound span extraction | Token classification |
| **Labels** | Dependency relations (distance:type) | Compound types only |
| **Output** | Spans with types | Label per token |
| **Complexity** | High (nested structures) | Moderate (flat classification) |
| **Evaluation** | USS, LSS, EM (span-based) | Accuracy, Precision, Recall, F1 |

## Files Created

```
data/ner/sacti_dataset.py      - Dataset and collator for SaCTI
trainer_sacti.py               - Training loop for SaCTI
configs/sacti_coarse.yaml      - Config for coarse-grained task
configs/sacti_fine.yaml        - Config for fine-grained task
run_sacti_coarse.sh            - Script to run coarse training
run_sacti_fine.sh              - Script to run fine training
```

## Usage

### 1. Coarse-grained Classification
```bash
bash run_sacti_coarse.sh
# OR
python3 trainer_sacti.py --config_file configs/sacti_coarse.yaml
```

### 2. Fine-grained Classification
```bash
bash run_sacti_fine.sh
# OR
python3 trainer_sacti.py --config_file configs/sacti_fine.yaml
```

## Configuration

Key parameters in YAML configs:

```yaml
data_path: "/home/pretam-pg/SaCTI/data"  # Path to SaCTI data
granularity: 'coarse'  # or 'fine'
backbone: 'google/muril-base-cased'  # Multilingual Indic BERT
batch_size: 16
max_epochs: 50
lr_bert: 2e-5
snr_scale: 1.0  # Important: prevents mode collapse
```

## Expected Performance

Based on the SaCTI paper (COLING 2022):

- **Coarse-grained**: ~94% accuracy
- **Fine-grained**: ~88% accuracy

DiffusionSL may achieve comparable or better results due to:
- Better handling of label dependencies through diffusion
- Self-conditioning for iterative refinement
- Pretrained MURIL encoder

## Troubleshooting

### Issue: Model predicts same label for all tokens
**Solution**: Increase `snr_scale` in config (try 1.5 or 2.0)

### Issue: Loss not decreasing
**Solution**: 
- Check data_path is correct
- Verify labels are being loaded (check console output)
- Reduce learning rate

### Issue: Out of memory
**Solution**: Reduce `batch_size` or `max_length` in config

## Comparison with Original SaCTI

| Model | Method | Coarse F1 | Fine F1 |
|-------|--------|-----------|---------|
| Original SaCTI | Multi-task BERT | ~94% | ~88% |
| DiffusionSL | Bit diffusion | **TBD** | **TBD** |

## Citation

If you use this integration, please cite both papers:

**SaCTI**:
```bibtex
@inproceedings{sandhan-etal-2022-novel,
    title = "A Novel Multi-Task Learning Approach for Context-Sensitive Compound Type Identification in {S}anskrit",
    author = "Sandhan, Jivnesh and Gupta, Ashish and Terdalkar, Hrishikesh and ...",
    booktitle = "Proceedings of COLING",
    year = "2022",
}
```

**DiffusionSL**:
```bibtex
@inproceedings{diffusionsl2023,
    title = "Diffusion-Based Sequence Labeling",
    booktitle = "Proceedings of EMNLP",
    year = "2023",
}
```
