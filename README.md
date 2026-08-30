# Diffusion-based Nested Compound Type Identification for Sanskrit

Code and configurations accompanying our work on **nested compound type identification
(NeCTI)** in Sanskrit, built on a diffusion-based sequence labeling framework
(DiffusionSL) with multilingual transformer encoders.

The model pairs a pretrained encoder (XLM-RoBERTa / MuRIL) with a **Diffusion
Transformer (DiT)** decoder that denoises tag sequences, and supports a
**hierarchical two-stage** variant that first predicts coarse compound categories
and then refines them into fine-grained relation labels.

---

## Repository layout

```
.
├── configs/            YAML experiment configurations (one per setting)
├── models/             Model definitions (DiT decoders, diffusion heads, encoders)
├── data/               Dataset readers and label sets  (ner/, pos/, cws/)
├── trainers/           Training entry points  (trainer_*.py, train_*.py)
├── inference/          Inference and checkpoint evaluation entry points
├── analysis/           Result analysis, per-class/length breakdowns, plotting
├── baselines/          LLM and majority-class baselines
├── tools/              Debugging, pipeline verification, prediction export
├── tests/              Unit and integration tests
├── scripts/            Shell launchers for the main experiments
├── docs/               Extended documentation (see below)
└── run.py              Generic entry point for the NER / POS / CWS tasks
```

`docs/` is organised as:

| Directory           | Contents                                                        |
| ------------------- | --------------------------------------------------------------- |
| `docs/tasks/`       | Per-task guides (NeCTI, NER, SACTI, hyperbolic variant)          |
| `docs/guides/`      | Method and feature documentation (diffusion mechanics, MST decoding, cross-attention, data balancing, …) |
| `docs/development/` | Development notes and design logs, kept for transparency         |

### Important: run everything from the repository root

Configuration files are resolved relative to the current working directory
(`os.getcwd()/configs/...`, see [options.py:118](options.py#L118)), and the entry
points live inside packages. Launch them as **modules** from the repo root:

```bash
python -m trainers.trainer_necti --config_file necti_coarse_xlmr.yaml
python -m inference.inference_necti --config_file necti_coarse_xlmr.yaml
python -m analysis.evaluate_coarse
```

The shell launchers in `scripts/` `cd` to the repository root themselves, so they
can be invoked from anywhere.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12 and PyTorch 2.6 (CUDA 12.4). Encoder weights
(`FacebookAI/xlm-roberta-large`, `google/muril-large-cased`) download
automatically from the HuggingFace Hub on first run.

## Data

Experiments use the **DepNeCTI** datasets (Sandhan et al., 2023), which are
distributed separately and are *not* included in this repository. Obtain them
from <https://github.com/yaswanth-iitkgp/DepNeCTI> and point the code at your
copy with `--data_path`:

```
<DATA_ROOT>/{With Context,Without Context}/
├── Coarse/      Coarse_{train,dev,test,ood}_san
└── Finegrain/   Finegrain_{train,dev,test,ood}_san
```

> **Note.** The YAML files in `configs/` and some launchers in `scripts/` still
> contain absolute paths from the development machine
> (`/home/pretam-pg/DepNeCTI/...`). Override `data_path` / `test_path` for your
> environment, either by editing the config or by passing the flag on the
> command line.

Data is read in CoNLL-U style, one token per line:

```
1    sa         Comp2    _    2    BvS          # Bahuvrihi
2    sarzapaM   Comp2    _    11   Comp_root
3    tumburu    Comp3    _    5    Ds           # Dvandva
4    DAnya      Comp3    _    5    Ds
5    vanyaM     Comp3    _    11   Comp_root
6    caRqAM     CompNo   _    11   No_rel       # not part of a compound
```

Two label granularities are supported: **Coarse** (Tatpurusha, Dvandva,
Bahuvrihi, Avyayibhava, …) and **Finegrain** (57 labels: `T6`, `T7`, `Ds`,
`BvS`, `U`, `K1`, …).

## Training

```bash
bash scripts/run_necti_coarse.sh        # coarse-grained NeCTI
bash scripts/run_necti_finegrain.sh     # fine-grained NeCTI
bash scripts/run_necti_bracket_coarse.sh
bash scripts/run_necti_bit_scheme.sh
bash scripts/run_sacti_coarse.sh
```

Equivalent direct invocation:

```bash
python -m trainers.trainer_necti \
    --config_file necti_coarse_xlmr.yaml \
    --granularity Coarse \
    --backbone FacebookAI/xlm-roberta-large \
    --data_path <DATA_ROOT> \
    --use_context \
    --batch_size 16 \
    --max_epochs 20 \
    --logger None
```

Hierarchical (two-stage) model:

```bash
python -m trainers.train_hierarchial --config configs/hierarchial_necti.yaml
```

### Key hyperparameters

| Parameter        | Default             | Description                          |
| ---------------- | ------------------- | ------------------------------------ |
| `backbone`       | xlm-roberta-large   | Pretrained encoder                   |
| `batch_size`     | 16                  | Training batch size                  |
| `max_epochs`     | 20                  | Training epochs                      |
| `lr_bert`        | 2e-5                | Encoder learning rate                |
| `lr_other`       | 5e-4                | Diffusion decoder learning rate      |
| `time_steps`     | 1000                | Forward diffusion steps              |
| `sampling_steps` | 10                  | DDIM sampling steps at inference     |
| `max_length`     | 256                 | Maximum sequence length              |
| `depth`          | 6                   | DiT decoder depth                    |
| `self_condition` | True                | Self-conditioning during denoising   |

Pass `--logger wandb` to log to Weights & Biases (`wandb login` first), or
`--logger None` to disable logging.

## Inference and evaluation

```bash
bash scripts/run_inference_necti.sh

python -m inference.inference_necti --config_file necti_coarse_xlmr.yaml
python -m inference.evaluate_checkpoint --config_file necti_coarse_xlmr.yaml
```

Training reports Precision / Recall / F1 (Labeled Span Score) on the dev set
after every epoch, and on the test and out-of-domain sets after training.

Analysis and figures:

```bash
python -m analysis.evaluate_coarse
python -m analysis.analyze_per_class
python -m analysis.plot_eval_by_length
python -m analysis.refinement_analysis     # figures are written to assets/figures/
```

Baselines:

```bash
python -m baselines.majority_baseline
python -m baselines.gemini                 # requires an API key in the environment
```

## Tests

```bash
pytest tests/
bash tests/test_integration.sh
```

`tests/conftest.py` puts the repository root on `sys.path`, so the tests can be
run from any working directory.

> `tests/test_focal_loss.py` currently fails: it imports `focal_mse_loss` from
> `models.ddim_bitdit`, which does not exist in this codebase.

## Outputs

This repository tracks source code only. Checkpoints, predictions, figures and
paper sources are written to directories that are excluded from version control
(`saved_models/`, `output/`, `inference_results/`, `logs/`, `wandb/`, `plots/`,
`assets/`, `paper/`):

```
saved_models/necti_{granularity}/
├── best_model.pt
└── best_model_epoch{N}_f1{score}.pt
```

## Citation

If you use this code, please cite DiffusionSL and DepNeCTI:

```bibtex
@inproceedings{diffusionsl2023,
    title     = {DiffusionSL: Sequence Labeling via Tag Diffusion Process},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
    year      = {2023}
}
```

```bibtex
@inproceedings{sandhan-etal-2023-depnecti,
    title     = "{D}ep{N}e{CTI}: Dependency-based Nested Compound Type Identification for {S}anskrit",
    author    = "Sandhan, Jivnesh and Narsupalli, Yaswanth and Muppirala, Sreevatsa and
                 Krishnan, Sriram and Satuluri, Pavankumar and Kulkarni, Amba and Goyal, Pawan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month     = dec,
    year      = "2023",
    address   = "Singapore",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2023.findings-emnlp.914/",
    doi       = "10.18653/v1/2023.findings-emnlp.914",
    pages     = "13679--13692"
}
```

## License

This work builds on the DiffusionSL and DepNeCTI projects and follows their
respective licenses.
