# Export DiffusionSL Predictions to CoNLL-U Format

This script exports predictions from your trained DiffusionSL model to CoNLL-U format, compatible with DepNeCTI's official evaluation script (`Eval_USS_LSS.py`).

## Quick Start

### 1. Export Predictions Only

```bash
python export_predictions_to_conllu.py \
    --model_path ./saved_models/necti_Coarse_best.pt \
    --data_path /home/pretam-pg/DepNeCTI/data/NeCTIS\ Model\ Data \
    --granularity Coarse \
    --split test \
    --output_file ./predictions_test.conllu
```

### 2. Export + Automatic Evaluation

```bash
bash export_and_evaluate.sh \
    ./saved_models/necti_Coarse_best.pt \
    /home/pretam-pg/DepNeCTI/data/NeCTIS\ Model\ Data \
    Coarse \
    test \
    false
```

### 3. Manual Evaluation (if needed)

```bash
python /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py \
    <true_file.txt> predictions_test.conllu
```

---

## Output Format

The exported CoNLL-U file follows DepNeCTI's format with 10 tab-separated columns:

```
ID      FORM    LEMMA   UPOS    XPOS    FEATS   HEAD     DEPREL      DEPS    MISC
1       वीत     _       _       _       _       4        Comp3       _       _
2       राग     _       _       _       _       4        Comp3       _       _
3       भय      _       _       _       _       4        Comp3       _       _
4       क्रोधः   _       _       _       _       4        Comp_root   _       _

```

Where:
- **ID**: Token ID (1-indexed)
- **FORM**: Word/token text
- **HEAD**: Head token ID (target of the dependency)
- **DEPREL**: Dependency relation label (e.g., "Comp3", "Comp2", "Comp_root", "No_rel")

---

## Arguments

### export_predictions_to_conllu.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | **required** | Path to saved model checkpoint |
| `--data_path` | str | DepNeCTI data | Path to DepNeCTI data directory |
| `--granularity` | str | Coarse | 'Coarse' or 'Finegrain' |
| `--use_context` | flag | false | Use "With Context" data |
| `--split` | str | test | 'train', 'dev', 'test', or 'ood' |
| `--batch_size` | int | 8 | Batch size for inference |
| `--device` | str | cuda | 'cuda' or 'cpu' |
| `--output_file` | str | auto | Output CoNLL-U file path |

### export_and_evaluate.sh

```bash
bash export_and_evaluate.sh \
    <MODEL_PATH> \
    [DATA_PATH] \
    [GRANULARITY] \
    [SPLIT] \
    [USE_CONTEXT]
```

---

## Examples

### Example 1: Export Coarse Granularity (Without Context)

```bash
python export_predictions_to_conllu.py \
    --model_path ./saved_models/necti_Coarse_best.pt \
    --granularity Coarse \
    --split test
```

Output: `predictions_Coarse_test.conllu`

### Example 2: Export Fine-Grain with Context

```bash
python export_predictions_to_conllu.py \
    --model_path ./saved_models/necti_Finegrain_with_ctx_best.pt \
    --granularity Finegrain \
    --use_context \
    --split test \
    --output_file ./fine_grain_predictions.conllu
```

### Example 3: Export + Evaluate All Splits

```bash
for split in train dev test; do
    python export_predictions_to_conllu.py \
        --model_path ./saved_models/necti_Coarse_best.pt \
        --split $split \
        --output_file ./predictions_${split}.conllu
    
    python /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py \
        /home/pretam-pg/DepNeCTI/data/NeCTIS\ Model\ Data/Without\ Context/Coarse/formatted/${split}.txt \
        ./predictions_${split}.conllu
done
```

---

## Expected Output

When evaluation runs, you'll see:

```
Results for /home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/Without Context/Coarse/formatted/test.txt are:

USS: XX.XX
LSS: XX.XX
Exact match: XX.XX
```

These are the metrics you compare against DepNeCTI's reported results.

---

## Troubleshooting

### Model Loading Error
Ensure your checkpoint has `model_state_dict` or is directly the state dict:
```python
checkpoint = torch.load(model_path)
# Should have either: checkpoint['model_state_dict'] or be the dict directly
```

### CoNLL-U Format Issues
The script handles label conversion:
- Labels like "Comp3_Start", "Comp3_Middle" → relations
- "Comp_root" marker → compound root
- "No_rel" → filtered out (not included in relations)

### Data Path Issues
Find the correct path:
```bash
ls -la /home/pretam-pg/DepNeCTI/data/NeCTIS\ Model\ Data/Without\ Context/Coarse/formatted/
```

---

## Integration with Your Pipeline

This script fits into your workflow:

```
DiffusionSL Model
    ↓
[export_predictions_to_conllu.py]
    ↓
CoNLL-U Format Predictions
    ↓
[Eval_USS_LSS.py]
    ↓
USS / LSS / EM Scores
    ↓
Compare with DepNeCTI Results
```

---

## Notes

1. **Dependency-Based Evaluation**: USS/LSS here refer to dependency-based metrics (not span-based), matching DepNeCTI's original implementation.

2. **Evaluation Script Location**: 
   ```bash
   /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py
   ```

3. **True Data Files**: Located in:
   ```
   /home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/
   └── [With Context | Without Context]
       └── [Coarse | Finegrain]
           └── formatted
               ├── train.txt
               ├── dev.txt
               ├── test.txt
               └── ood.txt
   ```

---

## Contact

For questions about the evaluation format or metrics, refer to:
- [DepNeCTI Repository](https://github.com/yaswanth-iitkgp/DepNeCTI)
- [DepNeCTI Paper](https://arxiv.org/abs/2310.09501)
