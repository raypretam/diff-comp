# DiffusionSL for DepNeCTI: Quick Reference

## 🚀 Quick Start

```bash
cd /home/pretam-pg/DiffusionSL

# 1. Test setup
bash setup_necti.sh

# 2. Start training
bash run_necti_coarse.sh      # For coarse-grained
# OR
bash run_necti_finegrain.sh   # For finegrain
```

## 📁 Key Files Created

| File | Purpose |
|------|---------|
| `data/ner/necti_dataset.py` | Dataset loader for DepNeCTI |
| `trainer_necti.py` | Training script |
| `configs/necti_coarse_xlmr.yaml` | Coarse config |
| `configs/necti_finegrain_xlmr.yaml` | Finegrain config |
| `run_necti_coarse.sh` | Run script (coarse) |
| `run_necti_finegrain.sh` | Run script (finegrain) |
| `test_necti_dataset.py` | Dataset verification |
| `NECTI_README.md` | Full documentation |

## 🔧 Architecture

- **Encoder**: XLM-RoBERTa (768-dim, multilingual)
- **Decoder**: DiT (Diffusion Transformer, 6 layers)
- **Process**: Diffusion-based sequence labeling (1000 steps → 10 inference steps)

## 📊 Dataset

**Path**: `/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data/With Context`

**Splits**: train, dev, test, ood  
**Formats**: Coarse (broad categories) | Finegrain (detailed subcategories)

## ⚙️ Key Parameters

```yaml
backbone: xlm-roberta-base
batch_size: 16
max_epochs: 20
lr_bert: 2e-5      # Encoder learning rate
lr_other: 5e-4     # Decoder learning rate
max_length: 256
time_steps: 1000
sampling_steps: 10
```

## 📈 Training Commands

### Basic
```bash
python trainer_necti.py --config_file necti_coarse_xlmr.yaml
```

### Custom
```bash
python trainer_necti.py \
    --config_file necti_coarse_xlmr.yaml \
    --batch_size 8 \
    --max_epochs 30 \
    --logger wandb
```

### Without Logging
```bash
python trainer_necti.py \
    --config_file necti_coarse_xlmr.yaml \
    --logger None
```

## 🎯 Expected Results

- **Coarse F1**: 85-92%
- **Finegrain F1**: 75-85%
- **Training**: ~2-6 hours (20 epochs)

## 💾 Outputs

Models saved in:
```
saved_models/necti_coarse/best_model.pt
saved_models/necti_finegrain/best_model.pt
```

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| CUDA OOM | `--batch_size 8 --max_length 128` |
| Dataset not found | Check `--data_path` in config |
| Slow training | Reduce `--num_workers` |

## 📚 Documentation

- **Full Guide**: `NECTI_README.md`
- **Integration Details**: `INTEGRATION_SUMMARY.md`
- **This File**: `QUICK_REFERENCE.md`

## ✅ Verification

```bash
# Test dataset loading
python test_necti_dataset.py

# Check GPU
nvidia-smi

# Verify files
ls configs/necti_*.yaml
ls data/ner/necti_dataset.py
```

---

**Ready to train!** Run `bash setup_necti.sh` to begin.
