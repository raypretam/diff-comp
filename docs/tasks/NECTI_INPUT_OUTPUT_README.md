# NeCTI Dataset: Input Processing and Output Generation Guide

## Overview

This document explains how input data is processed, passed to the model, and how the model generates outputs for the **NeCTI (Nested Compound Type Identification)** task in the DiffusionSL framework. The NeCTI task identifies compound types and their nested structures in Sanskrit text using a diffusion-based sequence labeling approach.

---

## Table of Contents

1. [Data Format](#data-format)
2. [Input Processing Pipeline](#input-processing-pipeline)
3. [Model Architecture](#model-architecture)
4. [Output Generation](#output-generation)
5. [Span and Label Extraction](#span-and-label-extraction)
6. [Evaluation Metrics](#evaluation-metrics)

---

## Data Format

### Source Data Structure

The NeCTI dataset is stored in **CoNLL-U format** with the following structure:

```
<data_path>/
├── With Context/
│   ├── Coarse/
│   │   ├── Coarse_train_san
│   │   ├── Coarse_dev_san
│   │   ├── Coarse_test_san
│   │   └── Coarse_ood_san
│   └── Finegrain/
│       ├── Finegrain_train_san
│       ├── Finegrain_dev_san
│       ├── Finegrain_test_san
│       └── Finegrain_ood_san
└── Without Context/
    ├── Coarse/
    └── Finegrain/
```

### CoNLL-U Format

Each line in the data file contains:
```
<idx>  <token>  <comp_label>  <placeholder>  <head_idx>  <relation>
```

**Example:**
```
1    naraH       CompNo      _    0    No_rel
2    rAjA        Comp2       _    3    Karmadharaya
3    hasti       Comp2       _    0    Comp_root
4    DUMMY       CompNo      _    0    No_rel
```

**Field Descriptions:**
- **idx**: Token position (1-indexed)
- **token**: Sanskrit token in SLP1 transliteration
- **comp_label**: Compound level indicator (`CompNo`, `Comp2`, `Comp3`, etc.)
- **head_idx**: Index of the head token in dependency structure
- **relation**: Compound type label (e.g., `Karmadharaya`, `Tatpurusha`, `No_rel`, `Comp_root`)

### Label Types

**Granularity Levels:**
- **Coarse**: Broad compound categories (e.g., `Karmadharaya`, `Tatpurusha`)
- **Finegrain**: Fine-grained compound subtypes (e.g., `Tatpurusha_Pancami`, `Bahuvrihi_Samanadhikarana`)

**Special Labels:**
- `No_rel`: Token is not part of any compound
- `Comp_root`: Token is the root of a compound

---

## Input Processing Pipeline

### 1. Data Loading (`NeCTIDataset`)

The dataset class parses CoNLL-U files and extracts:

```python
{
    'tokens': ['naraH', 'rAjA', 'hasti'],         # List of tokens
    'relation_labels': ['No_rel', 'Karmadharaya', 'Comp_root'],  # Relation per token
    'compounds': [                                 # Extracted compound spans
        {
            'start': 1,                           # Start index (0-indexed)
            'end': 2,                             # End index (0-indexed)
            'type': 'Comp_root',                  # Compound type
            'level': 'Comp2',                     # Nesting level
            'internal_types': ['Karmadharaya'],   # Component relations
            'tokens': ['rAjA', 'hasti'],          # Tokens in SLP1
            'tokens_devanagari': ['राजा', 'हस्ति'] # Tokens in Devanagari
        }
    ]
}
```

**Key Processing Steps:**
1. **Parse CoNLL-U**: Read tokens, compound labels, head indices, and relations
2. **Extract Compounds**: Identify compound boundaries using dependency structure
3. **Build Spans**: Create span information with start/end positions and types
4. **Transliteration**: Convert SLP1 to Devanagari for readability

### 2. Tokenization and Collation (`NeCTICollator`)

The collator prepares batches for the model using **XLM-RoBERTa tokenizer**:

```python
# Input: List of word-level tokens
sentences = [['naraH', 'rAjA', 'hasti'], ...]

# Output: Subword-tokenized batch
{
    'input_ids': tensor([[101, 2345, 6789, ...], ...]),      # Subword token IDs
    'attention_mask': tensor([[1, 1, 1, ...], ...]),         # Valid token mask
    'seq_labels': tensor([[0, 2, 1, ...], ...]),             # Aligned labels
    'compounds': [...],                                       # Original compound info
    'word_ids': [[None, 0, 0, 1, 2, None, ...], ...]        # Word-to-subword mapping
}
```

**Label Alignment:**
- XLM-R tokenizer splits words into subwords
- Only the **first subword** of each word receives the label
- Subsequent subwords are marked with `-100` (ignored in loss)
- Special tokens (`[CLS]`, `[SEP]`) are also marked `-100`

**Example:**
```
Word:       ['naraH',  'rAjA',      'hasti']
Subwords:   ['nara', 'H', 'rA', 'jA', 'hasti']
Labels:     [0,      -100, 2,    -100,  1    ]
```

---

## Model Architecture

### BitDiT (Bit Diffusion Transformer)

The model uses a **diffusion-based approach** for sequence labeling:

```
Input → XLM-RoBERTa Encoder → [Optional LSTM] → Diffusion Denoising → Output Labels
```

**Components:**

1. **Backbone Encoder (XLM-RoBERTa)**
   - Encodes input tokens into contextual embeddings
   - Output shape: `[batch_size, seq_len, hidden_dim]`

2. **Optional LSTM Layer**
   - Additional sequence modeling (if `add_lstm=True`)
   - Aggregates subword features to word-level

3. **Diffusion Process**
   - **Training**: Forward diffusion adds noise to label representations
   - **Inference**: Reverse diffusion denoises random vectors to predictions

4. **DiT (Diffusion Transformer)**
   - Predicts noise or clean labels at each diffusion timestep
   - Uses time embeddings and cross-attention to BERT features

### Label Representation

Labels are converted to **bit representations** for diffusion:

```python
# Convert label IDs to binary representation
label_id = 5  # Karmadharaya
bits = [1, 0, 1]  # 5 in binary (for 3-bit encoding)
scaled_bits = [1.0, -1.0, 1.0]  # Scaled to [-1, 1]
```

---

## Output Generation

### Training Phase

1. **Convert labels to bits**: `decimal_to_bits(labels, num_bits)`
2. **Add noise**: Forward diffusion process
   ```python
   noised_labels = sqrt(alpha_t) * clean_labels + sqrt(1 - alpha_t) * noise
   ```
3. **Predict**: Model predicts noise or clean labels
4. **Compute loss**: MSE/L1 loss between prediction and target

### Inference Phase (DDIM Sampling)

1. **Initialize**: Start with random noise
   ```python
   x_t = torch.randn([batch_size, seq_len, num_bits])
   ```

2. **Iterative Denoising**: For each timestep `t` from `T` to `0`:
   ```python
   # Predict clean labels (x_0) from noisy labels (x_t)
   x_0_pred = model(x_t, t, bert_features, attention_mask)
   
   # Compute x_{t-1} using DDIM update rule
   x_{t-1} = sqrt(alpha_{t-1}) * x_0_pred + sqrt(1 - alpha_{t-1}) * predicted_noise
   ```

3. **Convert to labels**: Transform final bits to label IDs
   ```python
   label_ids = bits_to_decimal(x_0, num_bits)
   ```

**Denoising Path:**
```
Random Noise → ... → Noisy Labels → ... → Clean Labels
    x_T            x_{T/2}              x_0
```

### Output Format

```python
# Per-token predictions
predictions = [0, 2, 1, 0, 0, ...]  # Label IDs for each token

# Decoded labels
decoded_labels = ['No_rel', 'Karmadharaya', 'Comp_root', 'No_rel', ...]
```

---

## Span and Label Extraction

### Token-Level Predictions

The model outputs a label for each token:

```
Tokens:     ['naraH', 'rAjA', 'hasti', 'gajaH']
Predictions: [No_rel, Karmadharaya, Comp_root, No_rel]
```

### Compound Span Extraction

Spans are extracted during data parsing based on **dependency structure**:

1. **Find Compound Roots**: Identify tokens with `Comp_root` relation
2. **Extract Members**: Follow dependency links to find all compound members
3. **Determine Boundaries**: Sort members by index to get span `[start, end]`
4. **Assign Types**: Collect relation types from compound members

**Example Extraction:**
```python
# Dependency structure
Token 1: naraH   -> head=0, relation=No_rel
Token 2: rAjA    -> head=3, relation=Karmadharaya
Token 3: hasti   -> head=0, relation=Comp_root

# Extracted compound
{
    'start': 1,                        # rAjA (index 1)
    'end': 2,                          # hasti (index 2)
    'type': 'Comp_root',
    'internal_types': ['Karmadharaya'],
    'tokens': ['rAjA', 'hasti']
}
```

### Nested Compounds

Compounds can be nested (compound within compound):

```
Level 1: [rAja-puruSa] (Tatpurusha)
Level 2: [[rAja-puruSa]-yukta] (Karmadharaya)

Compound 1: start=0, end=1, level='Comp2', type='Tatpurusha'
Compound 2: start=0, end=2, level='Comp3', type='Karmadharaya'
```

---

## Evaluation Metrics

### Standard Classification Metrics

- **Accuracy**: Proportion of correctly predicted tokens
- **Precision**: TP / (TP + FP) for compound vs non-compound
- **Recall**: TP / (TP + FN) for compound vs non-compound
- **F1 Score**: Harmonic mean of precision and recall

### Compound-Specific Metrics

#### 1. USS (Unlabeled Span Score)

Measures span boundary accuracy **without** considering types:

```python
# Count matching spans (start, end) regardless of label
true_span = (1, 2)
pred_span = (1, 2)  # Match! (even if types differ)

USS F1 = 2 * (USS_P * USS_R) / (USS_P + USS_R)
```

#### 2. LSS (Labeled Span Score)

Measures span accuracy **with** correct types:

```python
# Count matching (token_idx, head_idx, relation_type) tuples
true_relation = [2, 3, 'Karmadharaya']
pred_relation = [2, 3, 'Karmadharaya']  # Match!

LSS F1 = 2 * (LSS_P * LSS_R) / (LSS_P + LSS_R)
```

#### 3. Exact Match (EM)

Percentage of compounds with **all relations predicted correctly**:

```python
# Compound is correct only if ALL internal relations match
compound_correct = (all relations in compound match exactly)
EM = correct_compounds / total_compounds
```

### Evaluation Output Example

```
TEST Results:
--------------------------------------------------
Precision:    0.9234
Recall:       0.9156
F1 Score:     0.9195
Accuracy:     0.9567
USS (F1):     0.8923  # Span boundaries only
LSS (F1):     0.8745  # Span boundaries + types
LSS Prec:     0.8812
LSS Recall:   0.8679
Exact Match:  0.7834  # Fully correct compounds
--------------------------------------------------
```

---

## Summary Workflow

### Complete Pipeline

```
1. Load CoNLL-U File
   ↓
2. Parse Tokens & Dependencies
   ↓
3. Extract Compound Spans
   ↓
4. Tokenize with XLM-RoBERTa
   ↓
5. Align Labels to Subwords
   ↓
6. Encode with BERT + [LSTM]
   ↓
7. Diffusion Denoising (Inference)
   ↓
8. Convert Bits to Label IDs
   ↓
9. Decode to Relation Names
   ↓
10. Evaluate USS/LSS/EM Metrics
```

### Key Files

- **Dataset**: `data/ner/necti_dataset.py`
  - `NeCTILabelSet`: Manages label vocabulary
  - `NeCTIDataset`: Loads and parses data
  - `NeCTICollator`: Prepares batches

- **Model**: `models/ddim_bitdit.py`
  - `BitDit`: Main diffusion model
  - Handles training (forward diffusion) and inference (reverse diffusion)

- **Training**: `trainer_necti.py`
  - Training loop with USS/LSS evaluation

- **Inference**: `inference_necti.py`
  - Loads trained model and evaluates on test/OOD sets

---

## Example End-to-End

### Input Sentence
```
Tokens: ['nara', 'siMha', 'iva', 'yuddhe']
Translation: "Like a man-lion in battle"
```

### Ground Truth
```
Relations: ['Karmadharaya', 'Comp_root', 'No_rel', 'No_rel']
Compound: [nara-siMha] (Karmadharaya, indices 0-1)
```

### Model Processing
```
1. Tokenize: ['▁nara', '▁si', 'M', 'ha', '▁iva', '▁yuddhe']
2. Encode: BERT embeddings [6, 768]
3. Diffusion: Denoise random bits → label bits
4. Decode: [1, 3, 0, 0] → ['Karmadharaya', 'Comp_root', 'No_rel', 'No_rel']
```

### Output
```
Predictions: ['Karmadharaya', 'Comp_root', 'No_rel', 'No_rel']
Extracted Span: {start: 0, end: 1, type: 'Karmadharaya'}
Match: ✓ Correct!
```

---

## Configuration Options

### Context Modes
- **With Context**: Compounds with surrounding sentence context
- **Without Context**: Compounds in isolation

### Granularity
- **Coarse**: 6-8 broad compound categories
- **Finegrain**: 15-20 detailed compound subtypes

### Model Hyperparameters
- `time_steps`: Number of diffusion steps (default: 1000)
- `sampling_steps`: Steps during inference (default: 50)
- `num_bits`: Bits for label encoding (computed from num_classes)
- `max_length`: Max sequence length for tokenizer (default: 256)

---

## References

- **Dataset**: DepNeCTI (Dependency-based Nested Compound Type Identification)
- **Model**: DiffusionSL (Diffusion for Sequence Labeling)
- **Backbone**: XLM-RoBERTa (multilingual transformer)
- **Sampling**: DDIM (Denoising Diffusion Implicit Models)

For more details, see:
- `NECTI_README.md` - Dataset and task description
- `README.md` - General DiffusionSL framework
- `CONTEXT_MODE_GUIDE.md` - Context mode usage

---

**Last Updated**: November 30, 2025
