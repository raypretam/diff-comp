# Cross-Attention DiT Architecture Diagrams

## 1. High-Level System Architecture

### Standard BitDit + DiT
```
┌─────────────────────────────────────────────────────────────┐
│                    NeCTI Trainer                             │
│  Loads data → Forward through BitDit → Compute Loss         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    BitDit Model                              │
│  (models/ddim_bitdit.py)                                    │
│                                                              │
│  1. Extract BERT features                                   │
│  2. Convert labels to bits                                  │
│  3. Add noise (forward diffusion)                           │
│  4. Denoise using DiT                                       │
│  5. Compute diffusion loss                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              DiT (Diffusion Transformer)                     │
│  (models/dit_discrete.py)                                   │
│                                                              │
│  ┌──────────────────────────────────────┐                   │
│  │ Embed noisy bits                     │                   │
│  ├──────────────────────────────────────┤                   │
│  │ ┌──────────────────────────────────┐ │                   │
│  │ │ Self-Attention Block 1           │ │                   │
│  │ │ - Query/Key/Value from same seq  │ │                   │
│  │ ├──────────────────────────────────┤ │                   │
│  │ │ Self-Attention Block 2           │ │                   │
│  │ │ - Pattern: Self-Attn → MLP       │ │                   │
│  │ ├──────────────────────────────────┤ │                   │
│  │ │          ...                      │ │                   │
│  │ ├──────────────────────────────────┤ │                   │
│  │ │ Self-Attention Block N           │ │                   │
│  │ └──────────────────────────────────┘ │                   │
│  ├──────────────────────────────────────┤                   │
│  │ Linear output projection             │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
│ Context: bert_features + time_embedding (added early)       │
└─────────────────────────────────────────────────────────────┘
```

### BitDitCrossAttn + DiT (Cross-Attention)
```
┌─────────────────────────────────────────────────────────────┐
│              NeCTITrainerCrossAttn                           │
│  Loads data → Forward through BitDitCrossAttn → Loss        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 BitDitCrossAttn Model                        │
│  (models/ddim_bitdit_cross_attn.py)                         │
│                                                              │
│  1. Extract BERT features [bsz, len, 768]                   │
│  2. Convert labels to bits [bsz, len, num_bits]             │
│  3. Add noise (forward diffusion)                           │
│  4. Denoise using DiT with Cross-Attention                  │
│  5. Compute diffusion loss                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│      DiT with Cross-Attention                               │
│  (models/dit_discrete_cross_attention.py)                   │
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │ Embed noisy bits + Positional Embeddings     │           │
│  ├──────────────────────────────────────────────┤           │
│  │ ┌────────────────────────────────────────┐  │           │
│  │ │ Cross-Attention Block 1                │  │           │
│  │ │ ┌──────────────────────────────────┐   │  │           │
│  │ │ │ 1. Self-Attention                │   │  │           │
│  │ │ │    Q,K,V from current signal     │   │  │           │
│  │ │ ├──────────────────────────────────┤   │  │           │
│  │ │ │ 2. Cross-Attention  ← NEW!       │   │  │           │
│  │ │ │    Q from signal                 │   │  │           │
│  │ │ │    K,V from BERT features        │   │  │           │
│  │ │ ├──────────────────────────────────┤   │  │           │
│  │ │ │ 3. MLP                           │   │  │           │
│  │ │ └──────────────────────────────────┘   │  │           │
│  │ ├────────────────────────────────────────┤  │           │
│  │ │ Cross-Attention Block 2                │  │           │
│  │ │ (Self-Attn → Cross-Attn → MLP)         │  │           │
│  │ ├────────────────────────────────────────┤  │           │
│  │ │          ...                            │  │           │
│  │ ├────────────────────────────────────────┤  │           │
│  │ │ Cross-Attention Block N                │  │           │
│  │ └────────────────────────────────────────┘  │           │
│  ├──────────────────────────────────────────────┤           │
│  │ Linear output projection                     │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
│ Context: BERT features used explicitly in cross-attention   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. DiTBlock Comparison

### Standard DiTBlock
```
┌────────────────────────────────────────────────┐
│ Input: x [bsz, len, dim]                       │
│ Context: c [bsz, len, dim] = BERT + time      │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Residual Connection #1                         │
│  ┌──────────────────────────────────────────┐  │
│  │ Norm: LayerNorm(x)                       │  │
│  ├──────────────────────────────────────────┤  │
│  │ Self-Attention:                          │  │
│  │   Q, K, V all from (x)                   │  │
│  │   Compute attention within x             │  │
│  ├──────────────────────────────────────────┤  │
│  │ Apply to x: x = x + attention_output     │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Residual Connection #2                         │
│  ┌──────────────────────────────────────────┐  │
│  │ Norm: LayerNorm(x)                       │  │
│  ├──────────────────────────────────────────┤  │
│  │ MLP:                                     │  │
│  │   Linear → GELU → Linear                 │  │
│  ├──────────────────────────────────────────┤  │
│  │ Apply to x: x = x + mlp_output           │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Output: x [bsz, len, dim]                      │
│ (context c used only for adaptive normalization)
└────────────────────────────────────────────────┘

Context Fusion: Limited (mainly decorative)
Information Flow: Self-contained within x
```

### Cross-Attention DiTBlock
```
┌────────────────────────────────────────────────┐
│ Input:                                         │
│   x [bsz, len, dim]     - noisy signal         │
│   c [bsz, len, dim]     - context (BERT+time)  │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Residual Connection #1: Self-Attention         │
│  ┌──────────────────────────────────────────┐  │
│  │ Norm: LayerNorm(x)                       │  │
│  ├──────────────────────────────────────────┤  │
│  │ Self-Attention:                          │  │
│  │   Q, K, V all from normalized(x)         │  │
│  │   Compute attention within x             │  │
│  │   (allows temporal coherence)             │  │
│  ├──────────────────────────────────────────┤  │
│  │ Apply to x: x = x + self_attn_output     │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Residual Connection #2: Cross-Attention ← NEW! │
│  ┌──────────────────────────────────────────┐  │
│  │ Norm: LayerNorm(x)                       │  │
│  ├──────────────────────────────────────────┤  │
│  │ Cross-Attention:                         │  │
│  │   Q from normalized(x)  ← denoising sig  │  │
│  │   K,V from c            ← BERT context   │  │
│  │   Compute attention from x TO c          │  │
│  │   (allows contextual conditioning)       │  │
│  ├──────────────────────────────────────────┤  │
│  │ Apply to x: x = x + cross_attn_output    │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Residual Connection #3: MLP                    │
│  ┌──────────────────────────────────────────┐  │
│  │ Norm: LayerNorm(x)                       │  │
│  ├──────────────────────────────────────────┤  │
│  │ MLP:                                     │  │
│  │   Linear → GELU → Linear                 │  │
│  │   (non-linear feature mixing)             │  │
│  ├──────────────────────────────────────────┤  │
│  │ Apply to x: x = x + mlp_output           │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Output: x [bsz, len, dim]                      │
│ (now influenced by both self and context)      │
└────────────────────────────────────────────────┘

Context Fusion: Explicit and strong (cross-attention)
Information Flow: x ← self ← cross ← context
```

---

## 3. Attention Mechanism Details

### Standard Attention (Self-Attention)
```
Input sequence x:  [bsz, len, dim]
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
    Query (Q)            Key (K), Value (V)
    from x              both from x
         ↓                     ↓
    Linear(dim)         Linear(dim)
    [bsz, len, dim]     [bsz, len, dim]
         ↓                     ↓
      Reshape             Reshape
   [bsz, num_heads,   [bsz, num_heads,
    len, head_dim]     len, head_dim]
         ↓                     ↓
         └──────────┬──────────┘
                    ↓
         Attention(Q, K, V)
         = softmax(Q·K^T / √d) · V
                    ↓
         [bsz, num_heads, len, head_dim]
                    ↓
         Concatenate heads & project
                    ↓
         Output: [bsz, len, dim]

Purpose: Allows positions to attend to each other
Effect: Self-coherence within the sequence
```

### Cross-Attention
```
Query source: x          Key/Value source: context
   [noisy signal]              [BERT features]
         ↓                            ↓
    Linear(dim)              Linear(dim)
    [bsz, len, dim]         [bsz, len, dim]
         ↓                            ↓
      Reshape                      Reshape
   [bsz, num_heads,         [bsz, num_heads,
    len, head_dim]           len, head_dim]
         ↓                            ↓
         └──────────┬────────────────┘
                    ↓
         Cross-Attention(Q_x, K_c, V_c)
         = softmax(Q_x·K_c^T / √d) · V_c
                    ↓
         [bsz, num_heads, len, head_dim]
                    ↓
         Concatenate heads & project
                    ↓
         Output: [bsz, len, dim]

Purpose: Allows positions in x to attend to context
Effect: Contextual grounding of denoising
```

---

## 4. Data Flow During Training

### Standard Training
```
BATCH
├─ input_ids [bsz, len]
├─ attention_mask [bsz, len]
└─ seq_labels [bsz, len]
      ↓
BERT Forward Pass
├─ input_ids → BERT encoder
└─ bert_features [bsz, len, 768]
      ↓
Label to Bits Conversion
├─ seq_labels → decimal_to_bits()
└─ bits_labels [bsz, len, num_bits]
      ↓
Forward Diffusion (Add Noise)
├─ x_t = α·x_0 + β·ε
└─ noisy_bits [bsz, len, num_bits]
      ↓
Model Forward (BitDit)
├─ Input:
│  ├─ noisy_bits
│  ├─ timestep
│  └─ bert_features (concatenated early)
└─ Prediction: pred [bsz, len, num_bits]
      ↓
Loss Computation
├─ L = MSE(pred, target)
└─ Loss (scalar)
      ↓
Backward & Optimization
└─ Update parameters
```

### Cross-Attention Training (Similar)
```
BATCH
├─ input_ids [bsz, len]
├─ attention_mask [bsz, len]
└─ seq_labels [bsz, len]
      ↓
BERT Forward Pass
├─ input_ids → BERT encoder
└─ bert_features [bsz, len, 768]
      ↓
Label to Bits Conversion
├─ seq_labels → decimal_to_bits()
└─ bits_labels [bsz, len, num_bits]
      ↓
Forward Diffusion (Add Noise)
├─ x_t = α·x_0 + β·ε
└─ noisy_bits [bsz, len, num_bits]
      ↓
Model Forward (BitDitCrossAttn)
├─ Input:
│  ├─ noisy_bits
│  ├─ timestep
│  └─ bert_features (used in cross-attention blocks)
│
└─ DiT with Cross-Attention:
   ├─ Self-Attention: positions attend to each other
   ├─ Cross-Attention: positions attend to BERT context ← KEY DIFF
   └─ MLP: non-linear processing
      ↓
Prediction: pred [bsz, len, num_bits]
      ↓
Loss Computation
├─ L = MSE(pred, target)
└─ Loss (scalar)
      ↓
Backward & Optimization
└─ Update parameters
```

---

## 5. Inference Flow

### Standard Inference
```
Input
├─ input_ids [bsz, len]
├─ attention_mask [bsz, len]
└─ seq_labels [bsz, len] (for teacher forcing)
      ↓
BERT Features Extraction
└─ bert_features [bsz, len, 768]
      ↓
Sampling Loop (DDIM)
├─ Start with x_T ~ N(0, I)
│
├─ For t = T, T-1, ..., 1:
│  ├─ Model predicts p(x_{t-1} | x_t)
│  ├─ Sample x_{t-1}
│  └─ (using bert_features throughout)
│
└─ Final x_0 [bsz, len, num_bits]
      ↓
Convert Bits to Predictions
├─ bits_to_decimal()
└─ predictions [bsz, len] (class indices)
      ↓
Metrics Computation
├─ F1, Precision, Recall
└─ Results dict
```

### Cross-Attention Inference (Same structure, better context handling)
```
Input
├─ input_ids [bsz, len]
├─ attention_mask [bsz, len]
└─ seq_labels [bsz, len]
      ↓
BERT Features Extraction
└─ bert_features [bsz, len, 768]
      ↓
Sampling Loop (DDIM)
├─ Start with x_T ~ N(0, I)
│
├─ For t = T, T-1, ..., 1:
│  ├─ Model predicts p(x_{t-1} | x_t)
│  │  ├─ Self-Attention: understand current denoising state
│  │  ├─ Cross-Attention: attend to BERT context ← KEY DIFF
│  │  └─ MLP: process features
│  ├─ Sample x_{t-1}
│  └─ (using bert_features in cross-attention)
│
└─ Final x_0 [bsz, len, num_bits]
      ↓
Convert Bits to Predictions
├─ bits_to_decimal()
└─ predictions [bsz, len] (class indices)
      ↓
Metrics Computation
├─ F1, Precision, Recall
└─ Results dict
```

---

## 6. Parameter Flow

### Standard BitDit Parameters
```
┌─ BERT Backbone (frozen or trainable)
│  └─ 110M+ parameters
│
├─ Diffusion Components
│  ├─ Beta schedule (registered buffers)
│  ├─ Alpha schedule (registered buffers)
│  └─ Posterior variance (registered buffers)
│
├─ DiT Model
│  ├─ Input Embedding: in_channels × dim_model
│  ├─ Time Embedding: num_timesteps × dim_time
│  ├─ N × DiT Blocks (depth=12 typical)
│  │  └─ Each block:
│  │     ├─ Self-Attention: O(d²)
│  │     ├─ MLP: O(d × 4d)
│  │     └─ LayerNorms: O(d)
│  └─ Final Layer: dim_model → out_channels
│
└─ Optional LSTM (if add_lstm=True)
   └─ LSTM(dim_model, dim_model)

Total: ~200M-400M parameters depending on config
```

### Cross-Attention BitDit Parameters
```
┌─ BERT Backbone (frozen or trainable)
│  └─ 110M+ parameters
│
├─ Diffusion Components
│  ├─ Beta schedule (registered buffers)
│  ├─ Alpha schedule (registered buffers)
│  └─ Posterior variance (registered buffers)
│
├─ DiT Model with Cross-Attention
│  ├─ Input Embedding: in_channels × dim_model
│  ├─ Time Embedding: num_timesteps × dim_time
│  ├─ N × Cross-Attention DiT Blocks (depth=12)
│  │  └─ Each block: ← EXPANDED
│  │     ├─ Self-Attention: O(d²)
│  │     ├─ LayerNorm
│  │     ├─ Cross-Attention: O(d²) ← NEW
│  │     │  ├─ Q projection from input
│  │     │  ├─ K projection from context
│  │     │  └─ V projection from context
│  │     ├─ LayerNorm
│  │     ├─ MLP: O(d × 4d)
│  │     └─ LayerNorm
│  └─ Final Layer: dim_model → out_channels
│
└─ Optional LSTM (if add_lstm=True)
   └─ LSTM(dim_model, dim_model)

Total: ~200M-400M parameters (~same as standard)
Note: Cross-attention adds projections but minimal parameter increase
```

---

## 7. Memory Usage

### Standard BitDit Forward Pass
```
┌─ Input tensors
│  ├─ input_ids: bsz × len × dtype(int64) ≈ negligible
│  ├─ attention_mask: bsz × len × dtype(float32) ≈ 4 bytes per element
│  └─ seq_labels: bsz × len × dtype(int64) ≈ negligible
│
├─ Intermediate: BERT features
│  └─ [bsz, len, 768] × 4 bytes ≈ 3 MB (for bsz=16, len=512)
│
├─ DiT Forward Pass
│  ├─ Attention: Q,K,V, attention matrix, output
│  │  └─ Per head: [bsz, num_heads, len, dim_head] × multiple ≈ 10-50 MB
│  └─ Activations for all blocks: ≈ 500-1000 MB
│
└─ Gradients: ≈ Same as activations ≈ 500-1000 MB

Total per sample: ≈ 1-2 GB for batch_size=16, len=512
```

### Cross-Attention BitDit (Similar)
```
┌─ Input tensors (same)
│
├─ Intermediate: BERT features (same)
│
├─ DiT with Cross-Attention Forward Pass
│  ├─ Self-Attention: [bsz, num_heads, len, dim_head] × multiple
│  ├─ Cross-Attention: ← ADDITIONAL
│  │  ├─ Q from input [bsz, num_heads, len, dim_head]
│  │  ├─ K from context [bsz, num_heads, len, dim_head]
│  │  ├─ V from context [bsz, num_heads, len, dim_head]
│  │  └─ Attention matrix [bsz, num_heads, len, len]
│  └─ Other layers (same as standard)
│
└─ Total: ≈ 10-20% more memory than standard (usually <100 MB extra)

Total per sample: ≈ 1.1-2.2 GB (similar to standard)
```

---

## Summary

**Standard DiT**: Simple fusion via early concatenation
```
x ← Embed(concat[x, BERT])
```

**Cross-Attention DiT**: Explicit context conditioning
```
x ← Self-Attn(x) + Cross-Attn(x→BERT) + MLP(x)
```

**Result**: Better context utilization, slight training overhead, improved accuracy
