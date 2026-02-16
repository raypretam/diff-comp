# Hierarchical Diffusion with Local Refinement for Sanskrit Compound Type Identification

## 1. Introduction

We propose a **two-stage hierarchical diffusion framework** for Named Entity-based Compound Type Identification (NeCTI) in Sanskrit. The architecture addresses a fundamental limitation identified by Hahn (2020): global self-attention distributes information uniformly across all positions, causing local sensitivity to decay as $O(1/n)$ with sequence length $n$. For fine-grained compound type classification — where distinctions between subtypes (e.g., Tatpurusha-K1 vs. Tatpurusha-K3) depend on local morpho-syntactic cues — this loss of locality is detrimental.

Our approach decomposes the task into:

- **Stage 1 (Global Diffusion):** Full self-attention over the entire sequence to predict **coarse compound categories** (6 classes), capturing long-range structural patterns.
- **Stage 2 (Local Refinement Diffusion):** Windowed local attention to predict **fine-grained compound subtypes** (56 classes), conditioned on the coarse predictions from Stage 1.

Both stages are trained **jointly** in a single pass with a combined loss objective.

---

## 2. Problem Formulation

Given a Sanskrit compound sentence tokenized into words $\mathbf{w} = (w_1, w_2, \ldots, w_n)$, the task is to assign each word a label $y_i$ from a set of 56 fine-grained compound relation types, including `No_rel` (not part of a compound) and `Comp_root` (compound head).

The fine-grained label set $\mathcal{Y}^{\text{fine}}$ can be partitioned into 6 coarse categories $\mathcal{Y}^{\text{coarse}}$:

| Coarse ID | Category    | Fine-grained Subtypes (examples)                     |
|-----------|-------------|-------------------------------------------------------|
| 0         | Tatpurusha  | T1–T7, K1–K7, Km, U, Tm, Tds, Tdt, Tdu             |
| 1         | Bahuvrihi   | Bv, Bs, Bb, Bs2–Bs7, Bvs, BvS, BVS, Bvp, BvU, etc.|
| 2         | Dvandva     | D, d, Di, Ds                                         |
| 3         | Avyayibhava | A1, A2, A4, A7                                       |
| 4         | ROOT        | Comp_root                                             |
| 5         | No_rel      | No_rel                                                |

A deterministic mapping $\phi: \mathcal{Y}^{\text{fine}} \to \mathcal{Y}^{\text{coarse}}$ projects each fine-grained label to its coarse category.

---

## 3. Architecture

### 3.1 Overview

The model consists of three components:

1. **Contextual Encoder** (shared): XLM-RoBERTa-large backbone producing contextualized word representations.
2. **Stage 1 — Global DiT**: A Diffusion Transformer with full self-attention operating on coarse label bit representations.
3. **Stage 2 — Local Refinement DiT**: A Diffusion Transformer with windowed local attention operating on fine-grained label bit representations, conditioned on Stage 1's coarse output.

```
Input Sentence
      │
      ▼
┌─────────────────────┐
│  XLM-RoBERTa-large  │  ← Shared backbone (1024-dim)
│  (Contextual Encoder)│
└─────────┬───────────┘
          │ features f ∈ ℝ^{n×1024}
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌────────────┐
│Stage 1 │  │  Stage 2   │
│Global  │──│  Local     │
│DiT     │  │  Refinement│
│(coarse)│  │  DiT (fine) │
└───┬────┘  └─────┬──────┘
    │             │
    ▼             ▼
 ŷ_coarse     ŷ_fine
 (6 classes)  (56 classes)
```

### 3.2 Bit Encoding for Discrete Diffusion

Discrete labels are encoded into continuous bit representations for the diffusion process. For a label $y \in \{0, 1, \ldots, C-1\}$, we compute:

$$\mathbf{b} = \text{decimal\_to\_bits}(y, B) \in \{-1, +1\}^B$$

where $B = \lceil \log_2(C + 1) \rceil$ is the number of bits required. Each bit is computed via bitwise AND with powers of 2:

$$b_k = \begin{cases} +1 & \text{if } y \mathbin{\&} 2^{B-1-k} \neq 0 \\ -1 & \text{otherwise} \end{cases}$$

For our setup:

- **Coarse labels** (6 classes): $B_{\text{coarse}} = \lceil \log_2 7 \rceil = 3$ bits
- **Fine labels** (56 classes): $B_{\text{fine}} = \lceil \log_2 57 \rceil = 6$ bits

At inference, bits are decoded back to discrete labels:

$$\hat{y} = \sum_{k=0}^{B-1} \mathbb{1}[\hat{b}_k > 0] \cdot 2^{B-1-k}$$

### 3.3 Diffusion Process

Both stages use the same diffusion framework with a **cosine noise schedule** (Nichol & Dhariwal, 2021).

**Forward process.** Given clean bit representation $\mathbf{x}_0$, the noisy version at timestep $t$ is:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ and $\beta_t$ follows the cosine schedule:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

with offset $s = 0.008$ and $T = 1000$ total timesteps.

**Reverse process (inference).** We use deterministic DDIM sampling with $S = 50$ sampling steps. The model predicts $\hat{\mathbf{x}}_0$ directly (`pred_x0` objective), and the DDIM update is:

$$\mathbf{x}_{t'} = \sqrt{\bar{\alpha}_{t'}} \, \hat{\mathbf{x}}_0 + \sqrt{1 - \bar{\alpha}_{t'}} \cdot \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \, \hat{\mathbf{x}}_0}{\sqrt{1 - \bar{\alpha}_t}}$$

where $t' < t$ is the next timestep in the sub-sampled schedule.

---

## 4. Stage 1: Global Diffusion Transformer (Coarse Prediction)

### 4.1 Architecture

Stage 1 uses a standard Diffusion Transformer (DiT) with **full global self-attention**, identical to the architecture of Chen et al. (DiffusionNER). It consists of:

- **Input projection:** Linear mapping from $B_{\text{coarse}}$ bits to hidden dimension $d = 1024$
- **Timestep embedding:** Learned embedding table ($T = 1000$ entries) followed by a 2-layer MLP with SiLU activation
- **$L_1 = 6$ DiT blocks**, each containing:
  - **Adaptive Layer Normalization (AdaLN):** Modulated by the sum of timestep and feature conditioning
  - **Multi-Head Self-Attention:** Full attention over all $n$ positions with $H = 16$ heads (head dimension = 64)
  - **Feed-Forward Network:** 2-layer MLP with GELU activation, expansion ratio $r = 4$
- **Final layer:** AdaLN + linear projection back to $B_{\text{coarse}}$ bits

### 4.2 Conditioning

Each DiT block receives conditioning $\mathbf{c} = \mathbf{t}_{\text{emb}} + \mathbf{f}$, where:

- $\mathbf{t}_{\text{emb}} \in \mathbb{R}^d$: Timestep embedding (broadcast across sequence)
- $\mathbf{f} \in \mathbb{R}^{n \times d}$: XLM-RoBERTa features

The AdaLN mechanism produces 6 modulation parameters $(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2)$ from $\mathbf{c}$:

$$\mathbf{h} = \mathbf{x} + \alpha_1 \odot \text{Attn}\!\left(\gamma_1 \odot \text{LN}(\mathbf{x}) + \beta_1\right)$$
$$\mathbf{x}' = \mathbf{h} + \alpha_2 \odot \text{MLP}\!\left(\gamma_2 \odot \text{LN}(\mathbf{h}) + \beta_2\right)$$

### 4.3 Attention Masking

Padding positions are masked with a large negative value ($-10{,}000$) added to attention logits before softmax, ensuring padded tokens receive zero attention weight without producing NaN gradients (unlike $-\infty$).

### 4.4 Rationale

Global attention at this stage captures **long-range structural relationships** — for example, identifying that a sequence of words forms a Tatpurusha compound vs. a Bahuvrihi compound often requires understanding the overall syntactic structure of the sentence. The coarse distinction (4 major compound types + ROOT + No_rel) is well-suited to global attention because it depends on broad compositional patterns rather than local morphological cues.

---

## 5. Stage 2: Local Refinement Diffusion Transformer (Fine-grained Prediction)

### 5.1 Motivation: The Locality Argument

Hahn (2020) proved that for Transformer self-attention with $H$ heads and sequence length $n$, the mutual information between any single position and the full context is bounded by $O(H \log n / n)$, which vanishes for long sequences. While this is acceptable for coarse category prediction (where global patterns suffice), fine-grained distinctions — such as differentiating a genitive Tatpurusha (T6) from an instrumental Tatpurusha (T3) — depend critically on **local morpho-syntactic markers** (case suffixes, sandhi patterns) in the immediate neighborhood of each word.

Local windowed attention restricts each token to attend only within a fixed window, ensuring $O(1)$ information per position regardless of sequence length, thus preserving local sensitivity.

### 5.2 Local Window Attention

The core innovation of Stage 2 is **Local Window Attention**, which restricts each token's attention field to a symmetric window of size $W$:

$$\text{LocalAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_i = \sum_{j : |j - i| \leq \lfloor W/2 \rfloor} \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h})}{\sum_{j' : |j' - i| \leq \lfloor W/2 \rfloor} \exp(\mathbf{q}_i^\top \mathbf{k}_{j'} / \sqrt{d_h})} \mathbf{v}_j$$

Implementation:

1. Compute full $\mathbf{Q}\mathbf{K}^\top$ attention matrix $\in \mathbb{R}^{n \times n}$
2. Construct a binary local mask $\mathbf{M}$ where $M_{ij} = \mathbb{1}[|i - j| \leq \lfloor W/2 \rfloor]$
3. Set $\text{attn}_{ij} = -10{,}000$ for all $(i, j)$ where $M_{ij} = 0$
4. Apply softmax and compute weighted values

With $W = 5$ (2 left + self + 2 right), each token attends to at most 5 neighbors, capturing local morphological context while ignoring distant irrelevant tokens.

### 5.3 Architecture

Stage 2 uses a **Local Refinement DiT** with:

- **Input projection:** Linear mapping from $B_{\text{fine}} = 6$ bits to hidden dimension $d = 1024$
- **Coarse label embedding:** Learned embedding table ($|\mathcal{Y}^{\text{coarse}}| + 1$ entries, the extra entry for padding) maps the Stage 1 coarse prediction to $\mathbb{R}^d$, providing hierarchical conditioning
- **Timestep embedding:** Same architecture as Stage 1
- **Feature projection:** Linear layer to project XLM-RoBERTa features into conditioning space
- **$L_2 = 4$ Local DiT blocks**, each containing:
  - **AdaLN** modulation (same mechanism as Stage 1)
  - **Local Window Attention** with $W = 5$, $H = 16$ heads
  - **Feed-Forward Network** with GELU, expansion ratio $r = 4$
- **Final layer:** AdaLN + linear projection back to $B_{\text{fine}} = 6$ bits

### 5.4 Conditioning

The combined conditioning signal is:

$$\mathbf{c}^{(2)} = \mathbf{t}_{\text{emb}} + \text{CoarseEmbed}(\hat{y}^{\text{coarse}}) + \text{Proj}(\mathbf{f})$$

where:

- $\hat{y}^{\text{coarse}} \in \{0, \ldots, 5\}^n$: Coarse label predictions from Stage 1 (ground truth during training, sampled during inference)
- $\mathbf{f}$: Shared XLM-RoBERTa features (same features used in both stages)

The coarse embedding provides a **hierarchical prior**: knowing that a token belongs to a Tatpurusha compound narrows the fine-grained search space from 56 to ~21 subtypes, enabling the local attention to focus on discriminating within the correct category.

### 5.5 Coarse-to-Fine Consistency Constraint

At inference time, after sampling fine-grained predictions $\hat{y}^{\text{fine}}$, an optional **consistency constraint** ensures the predicted fine label falls within the predicted coarse category:

$$\hat{y}^{\text{fine}}_i \leftarrow \begin{cases} \hat{y}^{\text{fine}}_i & \text{if } \phi(\hat{y}^{\text{fine}}_i) = \hat{y}^{\text{coarse}}_i \\ \arg\min_{y : \phi(y) = \hat{y}^{\text{coarse}}_i} |y - \hat{y}^{\text{fine}}_i| & \text{otherwise} \end{cases}$$

This is implemented via the `_constrain_to_coarse` method using the pre-registered `fine_to_coarse` mapping buffer.

---

## 6. Joint Training Strategy

### 6.1 Loss Function

Both stages are trained **jointly** in each forward pass. The total loss is a weighted combination:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{coarse}} + \lambda_2 \mathcal{L}_{\text{fine}}$$

where $\lambda_1 = \lambda_2 = 1.0$ (equal weighting). Each stage loss is the MSE between the model's prediction and the clean bit representation, masked to exclude padding positions ($y = -100$):

$$\mathcal{L}_{\text{stage}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \|\hat{\mathbf{x}}_0^{(i)} - \mathbf{x}_0^{(i)}\|_2^2$$

where $\mathcal{V} = \{i : y_i \neq -100\}$ is the set of valid (non-padded) positions. Each stage independently samples its own timestep $t \sim \text{Uniform}\{0, \ldots, T-1\}$.

### 6.2 Why Joint Training?

Joint training offers several advantages over sequential (Stage 1 → freeze → Stage 2):

1. **Shared backbone gradients:** Both stages backpropagate through XLM-RoBERTa, encouraging features useful for both coarse structure and fine-grained distinctions.
2. **Co-adaptation:** The coarse embedding in Stage 2 sees the ground-truth coarse labels during training. This is a form of teacher forcing that stabilizes early training while the Stage 1 diffusion is still noisy.
3. **Efficiency:** A single training loop with one optimizer step per batch for all parameters.

### 6.3 Training Details

| Hyperparameter | Value |
|----------------|-------|
| Backbone | XLM-RoBERTa-large (1024-dim) |
| Stage 1 depth | 6 DiT blocks |
| Stage 2 depth | 4 Local DiT blocks |
| Attention heads (both) | 16 (head dim = 64) |
| Local window size $W$ | 5 |
| Diffusion timesteps $T$ | 1000 |
| Sampling steps (DDIM) $S$ | 50 |
| Noise schedule | Cosine ($s = 0.008$) |
| Objective | `pred_x0` (predict clean signal) |
| Loss | L2 (MSE) |
| Batch size | 16 |
| Backbone LR | $2 \times 10^{-5}$ |
| Stage 1 LR | $1 \times 10^{-4}$ |
| Stage 2 LR | $1 \times 10^{-4}$ |
| Optimizer | AdamW (weight decay $0.01$) |
| LR schedule | Linear warmup (500 steps) → Cosine decay |
| Gradient clipping | Max norm $1.0$ |
| Mixed precision | FP16 via `torch.amp.GradScaler` |
| Max epochs | 50 |
| Early stopping | Patience 10 (monitoring USS) |

### 6.4 Optimizer Configuration

Parameters are grouped with separate learning rates:

- **Group 1 (Backbone):** All XLM-RoBERTa parameters, LR = $2 \times 10^{-5}$
- **Group 2 (Stage 1):** Global DiT parameters, LR = $1 \times 10^{-4}$
- **Group 3 (Stage 2):** Local Refinement DiT parameters, LR = $1 \times 10^{-4}$

The lower backbone LR prevents catastrophic forgetting of pretrained multilingual representations while allowing the diffusion heads to learn faster.

---

## 7. Inference Pipeline

At test time, inference proceeds sequentially through both stages:

```
Step 1: Encode input with XLM-RoBERTa → features f
Step 2: Stage 1 — DDIM sample coarse labels from noise (50 steps)
Step 3: Stage 2 — DDIM sample fine labels from noise (50 steps),
         conditioned on coarse predictions from Step 2
Step 4: (Optional) Apply coarse-to-fine consistency constraint
Step 5: Decode bits → discrete fine-grained labels
```

Both stages use deterministic DDIM sampling ($\eta = 0$, no stochastic noise injection during denoising). The SNR scaling factor ($s = 1.0$) is applied to bit representations before noising and removed after sampling:

$$\mathbf{x}_0^{\text{input}} = s \cdot \text{bits}(y), \quad \hat{y} = \text{decode}(\hat{\mathbf{x}}_0 / s)$$

---

## 8. Evaluation Metrics

Evaluation follows the DepNeCTI benchmark protocol with three metrics:

### 8.1 Compound Span Extraction

From the predicted label sequence, compound spans are extracted as:

1. **Identify compound regions:** Contiguous runs of tokens with labels $\notin \{\text{No\_rel}, \text{root}\}$
2. **Split at Comp_root boundaries:** Each `Comp_root` token marks the end of a compound within a region
3. **Assign compound type:** Majority vote of non-`Comp_root` member labels

For example, given labels `[T6, T6, Comp_root, No_rel, Bs6, K1, Comp_root]`:
- Compound 1: span $(0, 2)$, type = T6 (majority of {T6, T6})
- Compound 2: span $(4, 6)$, type = voted from {Bs6, K1}

### 8.2 USS (Unlabeled Span Score)

F1 over compound **boundary spans** $(s_i, e_i)$, ignoring the type label:

$$\text{USS} = F_1\!\left(\{(s, e) : (s, e, \cdot) \in \mathcal{P}\},\; \{(s, e) : (s, e, \cdot) \in \mathcal{G}\}\right)$$

### 8.3 LSS (Labeled Span Score)

F1 over compound **labeled spans** $(s_i, e_i, \ell_i)$, requiring both correct boundary and correct type:

$$\text{LSS} = F_1\!\left(\{(s, e, \ell) \in \mathcal{P}\},\; \{(s, e, \ell) \in \mathcal{G}\}\right)$$

### 8.4 EM (Exact Match)

Fraction of ground-truth compounds that are exactly matched (correct span + correct label) in the predictions:

$$\text{EM} = \frac{|\mathcal{P} \cap \mathcal{G}|}{|\mathcal{G}|}$$

---

## 9. Theoretical Justification

### 9.1 Why Hierarchical?

Sanskrit compounds exhibit a natural hierarchy:

- **Coarse level:** The four major compound types (Tatpurusha, Bahuvrihi, Dvandva, Avyayibhava) are distinguished by broad semantic and syntactic patterns — e.g., Dvandva compounds list coordinate items, while Bahuvrihi compounds are exocentric.
- **Fine level:** Subtypes within each category are distinguished by specific grammatical relations (case roles) — e.g., T3 (instrumental Tatpurusha) vs. T6 (genitive Tatpurusha) depends on the case suffix of the dependent member.

Global attention excels at coarse distinctions (sentence-level patterns), while local attention excels at fine-grained distinctions (morpheme-level cues).

### 9.2 Why Diffusion?

Diffusion models offer several advantages for structured sequence labeling:

1. **Iterative refinement:** The denoising process naturally implements a coarse-to-fine prediction strategy within each stage.
2. **Global coherence:** Unlike autoregressive models, diffusion generates all positions simultaneously, enabling consistency across the label sequence.
3. **Flexible conditioning:** The AdaLN mechanism elegantly incorporates multiple conditioning signals (timestep, features, coarse labels) without architectural changes.

### 9.3 Information-Theoretic Perspective

For a sequence of length $n$ with global attention:

- **Per-position mutual information with context:** $I(x_i; \mathbf{x}_{-i}) = O(H \log n / n)$ (Hahn, 2020)
- This means fine-grained local cues are **diluted** by distant, irrelevant tokens

For local window attention with window $W$:

- **Per-position mutual information with local context:** $I(x_i; \mathbf{x}_{|j-i| \leq W/2}) = O(H \log W / W)$
- Since $W$ is constant (= 5), this is $O(1)$ — **independent of sequence length**

The hierarchical design gets the best of both worlds: global patterns for coarse prediction, preserved locality for fine-grained refinement.

---

## 10. Alternative Training Strategies

While joint training is the default, the framework supports:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `joint` | Train both stages simultaneously with combined loss | Default; best for end-to-end optimization |
| `sequential` | Train Stage 1 for $E_1$ epochs, then freeze and train Stage 2 for $E_2$ epochs | When Stage 1 needs to converge first |
| `stage1_only` | Train only the global DiT (coarse labels) | Ablation study; coarse-only baseline |
| `stage2_only` | Train only the local DiT (requires pretrained Stage 1 checkpoint) | Fine-tuning Stage 2 with fixed coarse predictions |
