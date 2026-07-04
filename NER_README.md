# Sanskrit NER with DiffusionSL (mahanama-NER)

Named Entity Recognition on the **mahanama** Sanskrit corpus, built on the
DiffusionSL sequence-labeling framework. Every method here shares one backbone —
a **MuRIL-large** encoder feeding a **BitDiT** bit-diffusion decoder
([`models/ddim_bitdit.py`](models/ddim_bitdit.py)) — and differs only in how the
label hierarchy is exploited during training and decoding.

> **Run everything from the repo root** (`/home/pretam-pg/DiffusionSL`).
> Dataset paths are resolved relative to the current working directory.

---

## 1. Table of contents

1. [The dataset](#2-the-dataset)
2. [Label hierarchy](#3-label-hierarchy)
3. [Methods overview](#4-methods-overview)
4. [Method 1 — Base diffusion NER](#5-method-1--base-diffusion-ner)
5. [Method 2 — Hyperbolic hierarchical contrastive NER](#6-method-2--hyperbolic-hierarchical-contrastive-ner)
6. [Method 3 — Two-stage hierarchical diffusion NER](#7-method-3--two-stage-hierarchical-diffusion-ner)
7. [Inference & evaluation](#8-inference--evaluation)
8. [Serving a trained model](#9-serving-a-trained-model)
9. [Baseline / comparison methods](#10-baseline--comparison-methods)
10. [Configuration reference](#11-configuration-reference)
11. [Outputs & checkpoints](#12-outputs--checkpoints)
12. [Installation](#13-installation)
13. [File map](#14-file-map)
14. [Troubleshooting](#15-troubleshooting)

---

## 2. The dataset

The NER data lives in [`data/ner/`](data/ner/) as JSON. Each record is a
sentence with parallel BMES tag labels:

```json
{"sentence": ["rāmaḥ", "ayodhyām", "agacchat"],
 "label":    ["S-manuṣyaḥ", "S-janapadaḥ", "O"]}
```

Two granularities are shipped:

| Dataset directory | Granularity | `num_classes` | Config |
|---|---|---|---|
| [`data/ner/mahanama_ner_coarse/`](data/ner/mahanama_ner_coarse/) | Coarse (6 entity types) | 25 | [`configs/mahanama_ner_coarse.yaml`](configs/mahanama_ner_coarse.yaml) |
| [`data/ner/mahanama_ner_fine/`](data/ner/mahanama_ner_fine/) | Fine (per-entity) | 77 | [`configs/mahanama_ner_fine.yaml`](configs/mahanama_ner_fine.yaml) |

Each directory holds `train.json`, `dev.json`, `test.json`. The loader
([`data/ner/ner_dataset.py`](data/ner/ner_dataset.py)) reads
`data/ner/<dataset>/<split>.json`, so the `dataset:` field in a config must
match the directory name.

### Rebuilding the dataset

To regenerate the JSON from the raw mahanama corpus (CoNLL-U-style train +
Label Studio test):

```bash
python mahanama_ner/build_dataset.py
```

This writes both the coarse and fine `{train,dev,test}.json`. The dev split is
5% of train (seeded). Source files live in [`mahanama_ner/`](mahanama_ner/).

---

## 3. Label hierarchy

The 77 fine labels collapse cleanly into **6 coarse categories**, defined in
[`data/ner/mahanama_hierarchy.py`](data/ner/mahanama_hierarchy.py):

```
Person   Location   NORP   Misc   Time   O
```

Every fine label maps to exactly one coarse parent
(`build_fine_to_coarse(label_set)` returns the `fine_id → coarse_id` list).
Methods 2 and 3 exploit this prior explicitly; Method 1 treats all fine labels
as flat classes.

---

## 4. Methods overview

| # | Method | Trainer | Entrypoint | `base:` |
|---|---|---|---|---|
| 1 | Base diffusion NER | [`trainer_ner.py`](trainer_ner.py) | `run.py` | `ner` |
| 2 | Hyperbolic hierarchical contrastive | [`trainer_ner_hyp.py`](trainer_ner_hyp.py) | `run.py` | `ner_hyp` |
| 3 | Two-stage hierarchical diffusion | [`train_hierarchial_ner.py`](train_hierarchial_ner.py) | `train_hierarchial_ner.py` | `hierarchical_ner` |
| — | Inference / eval | — | [`infer_ner.py`](infer_ner.py) | (any) |
| — | Baselines | — | `majority_baseline.py`, `gemini_fine.py`, `ppl.py` | — |

[`run.py`](run.py) is the dispatcher for Methods 1 & 2: it reads the `base:`
field from the config and instantiates the matching trainer, then runs
`train() → save() → eval_epoch('test')`. Method 3 has its own entrypoint
(different CLI: `--config`, not `--config_file`).

---

## 5. Method 1 — Base diffusion NER

Plain MuRIL + BitDiT diffusion. No hierarchy signal. This is the baseline
diffusion model.

```bash
cd /home/pretam-pg/DiffusionSL

# Fine-grained (77 classes)
python run.py --config_file mahanama_ner_fine.yaml --base ner

# Coarse-grained (6 types / 25 BMES)
python run.py --config_file mahanama_ner_coarse.yaml --base ner
```

- `--config_file` takes **just the filename**; [`options.py`](options.py) joins
  it with `configs/`.
- Any config value can be overridden on the CLI, e.g.
  `--batch_size 8 --max_epochs 30 --logger wandb`.
- Defaults in the shipped configs: `logger: None`, `gpus: 0`, `use_gpu: True`.

---

## 6. Method 2 — Hyperbolic hierarchical contrastive NER

Same backbone as Method 1 **plus an auxiliary hyperbolic contrastive loss** on
the encoder's token features. Tokens are embedded on the Poincaré ball and the
known fine→coarse hierarchy organizes them: same-fine tokens pulled together,
same-coarse siblings kept close (cheap to confuse), cross-coarse cousins pushed
apart (expensive to confuse). Full write-up: [`hyp_README.md`](hyp_README.md).

```bash
cd /home/pretam-pg/DiffusionSL
python run.py --config_file mahanama_ner_fine_hyp.yaml --base ner_hyp
```

Key knobs in [`configs/mahanama_ner_fine_hyp.yaml`](configs/mahanama_ner_fine_hyp.yaml):

| Key | Default | Meaning |
|---|---|---|
| `use_hyp_contrastive` | `True` | enable the auxiliary loss |
| `hyp_contrastive_weight` | `0.001` | keep small — it's a *nudge*, not the objective |
| `hyp_proj_dim` | `128` | contrastive projection dim |
| `hyp_temperature` | `0.5` | InfoNCE temperature |
| `hyp_w_sibling` / `hyp_w_cross` | `1.0` / `2.0` | repulsion weights (cross > sibling) |
| `use_hyp_encoder` | `False` | optional hHTM-style hyperbolic Hawkes encoder |

> `batch_size` is `2` in this config because the pairwise `[M, M]` distance
> matrix over tokens is memory-heavy. Raise it only if the GPU allows.

Inference is **identical to Method 1** — the auxiliary loss only shapes training;
decoding uses the same diffusion head ([`infer_ner.py`](infer_ner.py) auto-detects
`base: ner_hyp`).

---

## 7. Method 3 — Two-stage hierarchical diffusion NER

Coarse-then-fine cascade ([`models/hierarchial_diffusion.py`](models/hierarchial_diffusion.py)):

- **Stage 1** — global attention over the 6 coarse entity types.
- **Stage 2** — local/windowed attention over the 77 fine-grained BMES tags.

This trainer is **not** routed through `run.py`. It has its own entrypoint and
uses `--config` (a full path) rather than `--config_file`:

```bash
cd /home/pretam-pg/DiffusionSL

# Both stages jointly (default)
python train_hierarchial_ner.py --config configs/hierarchial_mahanama_ner.yaml

# Stage 1 only
python train_hierarchial_ner.py --config configs/hierarchial_mahanama_ner.yaml \
    --stage stage1

# Stage 2 on top of a trained stage-1 checkpoint
python train_hierarchial_ner.py --config configs/hierarchial_mahanama_ner.yaml \
    --stage stage2 --stage1_checkpoint path/to/stage1.pt
```

CLI flags: `--stage {stage1,stage2,both,sequential}`, `--stage1_checkpoint`,
`--resume`, `--seed`.

Training strategy and stage sizes are set in
[`configs/hierarchial_mahanama_ner.yaml`](configs/hierarchial_mahanama_ner.yaml)
under `training.strategy` (`joint | sequential | stage1_only | stage2_only`),
`stage1:` / `stage2:` blocks, etc. This config also enables the same hyperbolic
hierarchical contrastive aux loss on the encoder side (`hyp_contrastive.enabled:
True`).

> **Precision:** use `precision: 32` (the default here). The fp16/AMP path is
> documented as unstable for this model — early batches can produce inf grads
> and stall the run. fp32 is ~30% slower but avoids the failure mode.

Outputs go to `saved_models/hierarchical_mahanama_ner/`.

---

## 8. Inference & evaluation

[`infer_ner.py`](infer_ner.py) evaluates a trained checkpoint for **Method 1 or
Method 2** (it rebuilds the model through the same Trainer the config's `base:`
selects, so the architecture always matches the checkpoint). It reports
entity-level P/R/F1 (BMES decoding), token accuracy, and a per-entity-type
breakdown, and dumps per-sentence predictions.

```bash
# base model
python infer_ner.py --config_file mahanama_ner_fine.yaml \
    --checkpoint output/mahanama_ner_fine/best_f1_0.XXXX

# hyperbolic variant
python infer_ner.py --config_file mahanama_ner_fine_hyp.yaml \
    --checkpoint output/mahanama_ner_fine_hyp/best_f1_0.XXXX

# evaluate the dev split; write predictions to a chosen path
python infer_ner.py --config_file mahanama_ner_fine.yaml \
    --checkpoint <path> --split dev --output preds.json
```

- `--split {test,dev}` (default `test`).
- If `--checkpoint` is omitted, it falls back to
  `<output_dir>/<config-stem>/<model_path>`.
- Predictions default to `predictions_<split>.json` next to the checkpoint.

### Standalone metric scripts

Additional analysis/evaluation utilities (operate on gold CoNLL-U + a
predictions JSON):

```bash
python evaluate_fine.py   gold.conllu predictions.json -o eval_fine.json
python evaluate_coarse.py gold.conllu predictions.json -o eval_coarse.json
```

Other analysis scripts: `analyze_per_class.py`, `analyze_error_types.py`,
`analyze_by_length.py`, `analyze_label_confusion.py`, `span_based_evaluator.py`.

---

## 9. Serving a trained model

[`muril_diffusion_ner_server.py`](muril_diffusion_ner_server.py) exposes a
trained model as an inference server (FastAPI). Configure the backbone /
checkpoint via the `args = Namespace(...)` block near the bottom of the file,
then launch:

```bash
python muril_diffusion_ner_server.py > ner_server_log.txt 2>&1 &
```

---

## 10. Baseline / comparison methods

Non-diffusion reference points for the labeling task. (These operate on the
compound/NeCTI CoNLL-U data, not the mahanama JSON, but are the standard
comparison baselines.)

```bash
# Majority-class baseline — predicts the most frequent training label per span
python majority_baseline.py --train train.conllu --test test.conllu --label fine

# LLM baseline via Gemini API (requires a Google API key)
python gemini_fine.py ...        # fine-grained;  gemini.py for coarse

# Perplexity study: segmented vs compound-fused Sanskrit under an LLM
python ppl.py
```

---

## 11. Configuration reference

Common knobs (set in the YAML config or overridden on the CLI via
[`options.py`](options.py)):

| Parameter | Typical | Description |
|---|---|---|
| `backbone` | `google/muril-large-cased` | HuggingFace encoder |
| `dim_model` | `1024` | encoder hidden size (match the backbone) |
| `num_classes` | `25` / `77` | number of BMES labels (must match dataset) |
| `max_length` | `128` / `256` | max sequence length |
| `time_steps` | `1000` | forward diffusion steps |
| `sampling_steps` | `10` | DDIM inference steps |
| `noise_schedule` | `linear` / `cosine` | noise schedule |
| `objective` | `pred_x0` | diffusion prediction target |
| `loss_type` | `l2` | diffusion loss |
| `self_condition` | `True` | self-conditioning |
| `snr_scale` | `0.1` | signal-to-noise scaling |
| `depth` | `6` | DiT decoder depth |
| `batch_size` | `16` (`2` for hyp) | training batch size |
| `max_epochs` | `10`–`30` | epochs |
| `lr_bert` / `lr_other` | `1e-5` | encoder / decoder learning rates |
| `use_gpu`, `gpus` | `True`, `0` | GPU on, device index |
| `logger` | `None` | `wandb` \| `tensorboard` \| `None` |
| `save_limit` | `3` | max non-best checkpoints kept |

Available configs in [`configs/`](configs/): `mahanama_ner_coarse.yaml`,
`mahanama_ner_fine.yaml`, `mahanama_ner_fine_hyp.yaml`,
`hierarchial_mahanama_ner.yaml`, `sanskrit_ner.yaml`.

---

## 12. Outputs & checkpoints

Methods 1 & 2 save under `<output_dir>/<config-stem>/`, where the config-stem is
the config filename with its extension dropped (e.g. `mahanama_ner_fine.yaml` →
`output/mahanama_ner_fine/`). Files written during a run:

```
output/mahanama_ner_fine/
├── best_f1_0.XXXX               # best dev-F1 checkpoint (state_dict)
├── epoch_{N}_f1_{score}         # per-epoch checkpoints (pruned to save_limit)
├── last_{f1}_{E}epoch.pt        # final checkpoint
└── model.pt                     # default model_path
```

Checkpoints are plain `state_dict`s (`torch.save(model.state_dict(), ...)`),
loaded back with `model.load_state_dict(...)`. Method 3 writes to
`saved_models/hierarchical_mahanama_ner/` (see the config's `output:` block).

To log metrics to Weights & Biases:

```bash
wandb login
python run.py --config_file mahanama_ner_fine.yaml --base ner --logger wandb
```

---

## 13. Installation

```bash
pip install torch transformers wandb pyyaml prettytable tqdm \
            indic-transliteration
# Gemini baseline additionally needs: google-generativeai
# Full pinned environment: requirements.txt
```

The MuRIL backbone downloads automatically from HuggingFace on first run.

---

## 14. File map

```
DiffusionSL/
├── run.py                          # dispatcher for Methods 1 & 2 (--base)
├── options.py                      # CLI / config argument parser
│
├── trainer_ner.py                  # Method 1: base diffusion NER
├── trainer_ner_hyp.py              # Method 2: hyperbolic contrastive NER
├── train_hierarchial_ner.py        # Method 3: two-stage hierarchical NER
├── infer_ner.py                    # inference / evaluation (Methods 1 & 2)
├── muril_diffusion_ner_server.py   # inference server
│
├── configs/
│   ├── mahanama_ner_coarse.yaml    # Method 1, coarse
│   ├── mahanama_ner_fine.yaml      # Method 1, fine
│   ├── mahanama_ner_fine_hyp.yaml  # Method 2
│   ├── hierarchial_mahanama_ner.yaml   # Method 3
│   └── sanskrit_ner.yaml
│
├── data/ner/
│   ├── ner_dataset.py              # JSON dataset loader (LabelSet1D, NERDataset1D)
│   ├── mahanama_hierarchy.py       # fine → coarse label hierarchy
│   ├── mahanama_ner_coarse/        # {train,dev,test}.json
│   └── mahanama_ner_fine/          # {train,dev,test}.json
│
├── mahanama_ner/
│   └── build_dataset.py            # regenerate the JSON from raw corpus
│
├── models/
│   ├── ddim_bitdit.py              # BitDiT diffusion backbone
│   └── hierarchial_diffusion.py    # two-stage hierarchical model
│
├── hyp_README.md                   # deep dive on Method 2
├── evaluate_fine.py / evaluate_coarse.py   # metric scripts
├── majority_baseline.py / gemini_fine.py / ppl.py   # baselines
└── analyze_*.py, span_based_evaluator.py   # analysis utilities
```

---

## 15. Troubleshooting

- **`FileNotFoundError` on `data/ner/...`** — you're not in the repo root. `cd`
  to `/home/pretam-pg/DiffusionSL` first; paths are CWD-relative.
- **`num_classes` mismatch** — the trainer asserts the config's `num_classes`
  equals the label count found in the dataset. Use `25` for coarse, `77` for
  fine, or match your rebuilt data.
- **CUDA OOM** — lower `--batch_size` and/or `--max_length`. Method 2 already
  uses `batch_size: 2` for the pairwise distance matrix.
- **Training stalls / inf grads (Method 3)** — keep `precision: 32`; the fp16
  path is unstable for this model.
- **Wrong trainer picked** — Methods 1/2 select the trainer from the config's
  `base:` field (overridable with `--base`); Method 3 is a separate script.
- **Hyp aux loss dominates** — keep `hyp_contrastive_weight` small (`~0.001`);
  the raw hyperbolic loss (~7–8) dwarfs the diffusion loss (~0.01) if unscaled.

---

### Quick reference

| You want… | Command |
|---|---|
| Standard diffusion NER (fine) | `python run.py --config_file mahanama_ner_fine.yaml --base ner` |
| Coarse NER | `python run.py --config_file mahanama_ner_coarse.yaml --base ner` |
| Hierarchy-aware (hyperbolic) NER | `python run.py --config_file mahanama_ner_fine_hyp.yaml --base ner_hyp` |
| Two-stage coarse→fine NER | `python train_hierarchial_ner.py --config configs/hierarchial_mahanama_ner.yaml` |
| Evaluate a checkpoint | `python infer_ner.py --config_file <cfg> --checkpoint <ckpt>` |
| Rebuild the dataset | `python mahanama_ner/build_dataset.py` |
| Non-neural baseline | `python majority_baseline.py --train <t> --test <t> --label fine` |
