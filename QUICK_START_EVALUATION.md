# 🚀 Quick Start: Export & Evaluate DiffusionSL

## Option 1: Quickest Way (One Command)

```bash
python run_full_pipeline.py \
    --model_path ./saved_models/necti_Coarse_best.pt \
    --granularity Coarse \
    --output_dir ./eval_results
```

**Output**: USS, LSS, EM scores + comparison guide

---

## Option 2: Step-by-Step

### Step 1️⃣: Find Your Data Files
```bash
bash find_depnecti_data.sh
```

### Step 2️⃣: Export Predictions
```bash
python export_predictions_to_conllu.py \
    --model_path ./saved_models/necti_Coarse_best.pt \
    --granularity Coarse \
    --split test
```

**Output**: `predictions_Coarse_test.conllu`

### Step 3️⃣: Evaluate with DepNeCTI Script
```bash
python /home/pretam-pg/DepNeCTI/Evaluation/Eval_USS_LSS.py \
    "/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/Without Context/Coarse/formatted/test.txt" \
    predictions_Coarse_test.conllu
```

**Output**:
```
USS: XX.XX
LSS: XX.XX
Exact match: XX.XX
```

---

## Option 3: One-Liner Bash Script

```bash
bash export_and_evaluate.sh ./saved_models/necti_Coarse_best.pt
```

---

## 📊 Expected Output

```
========================================
USS (Unlabeled Span F1): 91.23
LSS (Labeled Span F1):   85.67
EM (Exact Match):        72.34
========================================
```

Compare with DepNeCTI baseline (Table 2 in paper):
- **Coarse**: USS ~93.5%, LSS ~88.2%, EM ~75.4%
- **Fine-grain**: USS ~89.3%, LSS ~82.1%, EM ~62.8%

---

## 🔧 Common Variations

### Fine-Grain with Context
```bash
python export_predictions_to_conllu.py \
    --model_path ./saved_models/best.pt \
    --granularity Finegrain \
    --use_context \
    --split test
```

### All Splits at Once
```bash
for split in train dev test; do
    python export_predictions_to_conllu.py \
        --model_path ./saved_models/best.pt \
        --split $split
done
```

### Evaluate on OOD Data
```bash
python export_predictions_to_conllu.py \
    --model_path ./saved_models/best.pt \
    --split ood \
    --output_file ood_predictions.conllu
```

---

## 📁 Files Generated

```
./evaluation_results/
├── predictions_Coarse_test.conllu    # CoNLL-U format predictions
├── predictions_Coarse_dev.conllu
├── evaluation_summary.txt            # Results summary
└── [evaluation logs]
```

---

## ❓ Troubleshooting

### Q: Model not found?
```bash
ls -la ./saved_models/
```

### Q: Data path wrong?
```bash
bash find_depnecti_data.sh
```

### Q: GPU memory error?
```bash
python export_predictions_to_conllu.py ... --device cpu
```

### Q: Missing dependencies?
```bash
pip install torch transformers tqdm
```

---

## 📚 Format Reference

### Input (CoNLL-U)
```
1	वीत	_	_	_	_	4	Comp3	_	_
2	राग	_	_	_	_	4	Comp3	_	_
3	भय	_	_	_	_	4	Comp3	_	_
4	क्रोधः	_	_	_	_	4	Comp_root	_	_

```

Columns:
- ID: token position
- FORM: word
- HEAD: target token (dependency head)
- DEPREL: relation type (Comp3, Comp_root, No_rel)

---

## 🎯 Your Goal

**Beat DepNeCTI Results:**
- [ ] Coarse: USS > 93.5%, LSS > 88.2%, EM > 75.4%
- [ ] Fine-grain: USS > 89.3%, LSS > 82.1%, EM > 62.8%

Use these scripts to quickly benchmark your model! 🚀
