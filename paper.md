## 1. Key Theoretical Result (Existing Work)

**Hahn (2020), TACL** - "Theoretical Limitations of Self-Attention in Neural Sequence Models"

### Theorem (Lemma 5 from Hahn 2020):
> Let a soft attention transformer be given, and let n be the input length. If we exchange one input symbol $x_i$ (i < n), then the change in the resulting activation $y_n^{(L)}$ at the decoder layer is bounded as **O(1/n)** with constants depending on the parameter matrices.

**Implication:** As input length increases, the influence of any single token on the final prediction diminishes to zero.

---

## 2. Application to NeCTI: Why Coarse Works but Fine-Grained Fails

### 2.1 The NeCTI Label Structure

**Coarse labels (4 types):** Tatpuruṣa, Bahuvrīhi, Dvandva, Avyayībhāva

**Fine-grained labels (56 types):** Subdivisions based on **case relationships**
- Example: Tatpuruṣa subdivides into:
  - dvitīyā tatpuruṣa (2nd case - accusative)
  - tṛtīyā tatpuruṣa (3rd case - instrumental)  
  - caturthī tatpuruṣa (4th case - dative)
  - pañcamī tatpuruṣa (5th case - ablative)
  - ṣaṣṭhī tatpuruṣa (6th case - genitive/possessive)
  - saptamī tatpuruṣa (7th case - locative)

### 2.2 Mathematical Formulation

Let compound $C = (c_1, c_2, ..., c_k)$ be a sequence of $k$ components.

**Coarse label prediction** can be formulated as:
$$y_{coarse} = f_{global}(h_1, h_2, ..., h_k)$$

where $h_i$ are contextual embeddings and $f_{global}$ captures the overall semantic relationship.

**Fine-grained label prediction** requires:
$$y_{fine}(c_i, c_j) = g_{local}(h_i, h_j, r_{ij})$$

where $r_{ij}$ is the **specific syntactic relationship** (case) between component $c_i$ and its head $c_j$.

### 2.3 The Critical Difference

**Claim:** Fine-grained labels encode **local pairwise relationships** that cannot be recovered from global attention alone.

**Proof Sketch:**

Consider a 3-component compound: $C = (a, b, c)$

Two possible structures with **same coarse label** (Tatpuruṣa) but **different fine-grained labels**:

1. Structure 1: $\langle\langle a - b \rangle_{T6} - c \rangle_{T6}$
   - $a$ modifies $b$ with ṣaṣṭhī (6th case, possessive)
   - $\langle a-b \rangle$ modifies $c$ with ṣaṣṭhī (6th case, possessive)

2. Structure 2: $\langle\langle a - b \rangle_{T7} - c \rangle_{T6}$
   - $a$ modifies $b$ with saptamī (7th case, locative)
   - $\langle a-b \rangle$ modifies $c$ with ṣaṣṭhī (6th case, possessive)

**Key Observation:** The difference between T6 and T7 at the inner span depends **only** on the relationship between $a$ and $b$, not on $c$ or any global context.

By Hahn's Lemma 5, in a transformer:
$$\frac{\partial y^{(L)}}{\partial x_a} = O(1/n)$$

As compound length $n$ grows, the model's ability to distinguish whether $a$ has a possessive (T6) or locative (T7) relationship with $b$ **diminishes**.

---

## 3. Formal Theorem for NeCTI

### Theorem 1 (Local Dependency Requirement for Fine-Grained Labels):

Let $\mathcal{L}_{coarse}$ be the set of coarse labels and $\mathcal{L}_{fine}$ be the set of fine-grained labels with $|\mathcal{L}_{fine}| >> |\mathcal{L}_{coarse}|$.

Define the **label sensitivity** of position $i$ as:
$$S_i(y) = \max_{x'_i \neq x_i} |P(y|x) - P(y|x')|$$

where $x'$ differs from $x$ only at position $i$.

**For coarse labels:**
$$S_i(y_{coarse}) \leq \epsilon_{coarse}$$

where $\epsilon_{coarse}$ is small, meaning coarse labels can be determined without high sensitivity to individual positions.

**For fine-grained labels:**
$$S_i(y_{fine}) \geq \delta_{fine}$$

where $\delta_{fine}$ is large for positions involved in the local dependency.

**Corollary:** A model with bounded local sensitivity (like transformers, by Hahn's Lemma 5) can approximate $y_{coarse}$ but not $y_{fine}$.

---

## 4. Information-Theoretic Argument

### 4.1 Mutual Information Analysis

Define:
- $X_i$ = embedding of component $i$
- $X_j$ = embedding of component $j$ (head of $i$)
- $Y_{coarse}$ = coarse label
- $Y_{fine}$ = fine-grained label

**For coarse labels:**
$$I(Y_{coarse}; X_1, X_2, ..., X_k) \approx I(Y_{coarse}; \sum_i X_i)$$

The coarse label has high mutual information with the **aggregate** representation.

**For fine-grained labels:**
$$I(Y_{fine}; X_i, X_j | X_{-i,-j}) >> I(Y_{fine}; \sum_k X_k)$$

The fine-grained label has high **conditional** mutual information with the specific pair $(X_i, X_j)$, which cannot be captured by global aggregation.

### 4.2 Why Global Attention Loses Fine-Grained Information

In self-attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

The output for position $i$ is:
$$z_i = \sum_j \alpha_{ij} v_j$$

where $\alpha_{ij} = \frac{\exp(q_i \cdot k_j)}{\sum_l \exp(q_i \cdot k_l)}$

**Problem:** As sequence length $n$ increases:
- Each $\alpha_{ij} \rightarrow \frac{1}{n}$ (softmax dilution)
- The specific pairwise information between $i$ and its head $j$ is averaged away

**Contrast with Bi-affine parsing (DepNeCTI):**
$$s_{arc}(i,j) = h_i^T W h_j + b$$
$$s_{rel}(i,j,r) = h_i^T U_r h_j + c_r$$

This **explicitly models** the pairwise relationship without dilution.

---

## 5. Empirical Prediction

Based on the theoretical analysis:

| Granularity | Requires Local Deps? | Global Attention | With Bi-affine |
|-------------|---------------------|------------------|----------------|
| Coarse (4 labels) | No | ✓ Works | ✓ Works |
| Fine-grained (86 labels) | Yes | ✗ Degrades | ✓ Works |

**Expected behavior:**
- As compound length increases, fine-grained accuracy should degrade faster for global attention models
- The gap between global attention and bi-affine should widen for fine-grained labels

---

## 6. Summary for Paper

### Core Argument:

1. **Hahn (2020) proves** that self-attention's influence of any single token diminishes as O(1/n)

2. **Fine-grained compound types encode case relationships** that depend on specific head-modifier pairs

3. **Case relationships are inherently local** - changing the case between $a$ and $b$ should not depend on distant component $c$

4. **Therefore**, global attention (which averages over all positions) loses the fine-grained local signal needed to distinguish subtypes

5. **Bi-affine/dependency parsing** explicitly models pairwise relationships, preserving this local information
