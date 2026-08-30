# Diffusion Mechanics with Graph-Aware Compound Encoding

## The Core Question: Why Does Graph Encoding Help?

### Scenario: Predicting a Compound's Label

Consider this example:
```
Token 0: "सीता" (Sita)     - Compound A
Token 1: "पुस्तक" (book)    - Compound B  
Token 2: "आलय" (house)      - Compound B

Ground truth:
  Token 0: "+1:Tatpurusha" (possessive: Sita's)
  Token 1: "+1:Avyaya"      (modifier)
  Token 2: "ROOT:Comp_root" (compound root)

Dependency structure:
  Compound A → Compound B (Tatpurusha relation)
  Token 1 → Token 2 (within Compound B, Avyaya relation)
```

### Diffusion Training: What the Model Learns

#### Without Graph Encoding (Independent Compounds)

**Diffusion condition for Compound A:**
```python
condition_A = mean([BERT("सीता")])  # Only local token features
```

**What the model learns:**
- "सीता" typically appears in certain contexts
- The bit pattern for "+1:Tatpurusha" 
- **MISSING**: Any information about what comes after (Compound B)

**Problem during prediction:**
The model must predict "+1:Tatpurusha" based ONLY on "सीता" features.
It has no information about:
- What compound follows (B)
- What relation type makes sense given B's features
- Whether B is a valid target for A's pointer

**Result:** Model sometimes predicts wrong distance or relation type because it's "blind" to the target compound.

#### With Graph Encoding (Dependency-Aware)

**Diffusion condition for Compound A:**
```python
# Step 1: Base features
base_features_A = mean([BERT("सीता")])

# Step 2: Get parent compound features (in this case, A points to B, so B is parent)
features_B = mean([BERT("पुस्तक"), BERT("आलय")])

# Step 3: Get relation embedding
rel_emb = relation_embeddings[id("Tatpurusha")]  # Learned embedding!

# Step 4: Message from parent B
message_from_B = message_network(concat(features_B, rel_emb))

# Step 5: Update A's features
condition_A = base_features_A + aggregated(message_from_B)
```

**What the model learns:**
- "सीता" local features (same as before)
- **NEW**: Features of the compound it points to (B)
- **NEW**: Relation type embedding for Tatpurusha
- **NEW**: Pattern: "When A is possessive of B, A and B have this relationship"

**During prediction:**
The model predicts "+1:Tatpurusha" based on:
1. "सीता" features (possessive marker in Sanskrit)
2. "पुस्तकालय" features (what is being possessed)
3. Learned pattern: "This feature combination → Tatpurusha relation"

**Result:** Much more informed prediction! The model "sees" both the dependent and the head compound.

## Diffusion Process: Step by Step

### Forward Diffusion (Training - Adding Noise)

```
Original label: "+1:Tatpurusha" → bits: [0,1,1,0,1,0,0,1]

Timestep t=0:     [0,1,1,0,1,0,0,1]  (clean)
Timestep t=1000:  [0.1,0.9,0.8,0.2,0.9,0.1,0.1,0.8]  (slight noise)
Timestep t=2500:  [0.3,0.7,0.6,0.4,0.6,0.3,0.2,0.6]  (medium noise)
Timestep t=4000:  [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]  (pure noise)
```

At each timestep t, the model tries to:
- **Predict the noise** that was added, OR
- **Predict the original clean label**

Conditioned on:
- The noisy observation at time t
- The compound features (GRAPH-AWARE!)
- The timestep t (encoded as embedding)

### Reverse Diffusion (Inference - Removing Noise)

```
Start: [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]  (random noise)

Step 1 (t=4000): Model predicts noise, denoise partially
  → [0.4,0.6,0.6,0.4,0.6,0.3,0.3,0.6]

Step 2 (t=3000): Model predicts noise, denoise more
  → [0.3,0.7,0.7,0.3,0.7,0.2,0.2,0.7]

...

Step 100 (t=50): Almost clean
  → [0.05,0.95,0.9,0.1,0.9,0.05,0.05,0.85]

Final: Round to bits
  → [0,1,1,0,1,0,0,1] → Decode to "+1:Tatpurusha"
```

**Critical point**: At EVERY denoising step, the model uses the **graph-aware compound features** as conditioning. This means:
- When denoising Compound A's label, the model "knows" about Compound B
- The model can leverage learned patterns: "If A points to B, and B looks like X, then A's relation is likely Tatpurusha"

## Why Graph Features Lead to Better Exact Match

### Information Flow in the Dependency Graph

```
Example nested structure:
  राम (Rama) → सीता (Sita) → विवाह (wedding) → समारोह (ceremony)
  Comp A     → Comp B       → Comp C          → Comp D (root)
           Dvandva      Tatpurusha       Karmadhara
```

**Without graph encoding:**
- Predict label for A: only uses A's tokens
- Predict label for B: only uses B's tokens  
- Predict label for C: only uses C's tokens
- Predict label for D: only uses D's tokens

**4 independent predictions → High chance of errors**

**With graph encoding (2 GNN layers):**

After GNN Layer 1:
- A's features: base features only (no parents in depth 1)
- B's features: base + message from A (+ Dvandva embedding)
- C's features: base + message from B (+ Tatpurusha embedding)
- D's features: base + message from C (+ Karmadhara embedding)

After GNN Layer 2:
- A's features: still base only
- B's features: base + message from A
- C's features: base + message from B (which includes info about A!)
- D's features: base + message from C (which includes info about B and A!)

**Result**: Information flows hierarchically!
- D's prediction considers C, B, and A (transitively through 2 GNN layers)
- C's prediction considers B and A  
- B's prediction considers A
- All predictions are **correlated** and **consistent**

### Concrete Example: Error Correction

**Scenario**: The model is uncertain about Compound B's label.

**Without graph encoding:**
```
A: 70% Dvandva, 30% Tatpurusha   → Predicts Dvandva ✓
B: 51% Tatpurusha, 49% Bahuvrihi → Predicts Tatpurusha (barely)
C: 80% Karmadhara                → Predicts Karmadhara ✓

Problem: B's prediction is weak and could flip to wrong label
```

**With graph encoding:**
```
After GNN:
  B receives message from A with Dvandva embedding
  B's features now include: "I'm the target of a Dvandva from A"
  
  The model learns: "Compounds that are Dvandva targets are typically nouns of similar status"
  This boosts the probability of compatible relations for B.

Result:
  B: 65% Tatpurusha, 35% Bahuvrihi → Predicts Tatpurusha ✓ (more confident!)
  
Also:
  C receives message from B with updated features
  C's prediction benefits from knowing about both B and A
```

**Key insight**: The dependency structure provides **mutual information** that helps resolve ambiguities.

## Training Dynamics

### What Relation Embeddings Learn

During training, the relation embeddings learn:

```python
relation_embeddings["Tatpurusha"] → learns to capture:
  - Possessive patterns (X's Y)
  - Modifier-head relationships
  - Typical distance (+1, +2 common for local possession)
  
relation_embeddings["Dvandva"] → learns to capture:
  - Coordinative patterns (X and Y)
  - Symmetric relationships
  - Compounds of equal status
  
relation_embeddings["Bahuvrihi"] → learns to capture:
  - Adjectival relationships
  - External reference patterns
  - Descriptive compounds
```

These embeddings act as **relation type classifiers** that guide the diffusion process.

### Loss Landscape

**Without graph encoding:**
```
Loss space is ROUGH and INDEPENDENT
Each compound's prediction explores its own space
No correlation between compound predictions
→ Harder to find globally consistent solutions
```

**With graph encoding:**
```
Loss space is SMOOTHER and CORRELATED  
Compound predictions are coupled through graph structure
Inconsistent predictions have higher loss (because features conflict)
→ Easier to find globally consistent solutions
→ Better generalization
```

## Summary: The Mechanism

1. **Graph Construction**: Parse labels to build dependency graph between compounds

2. **GNN Message Passing**: 
   - Each compound receives messages from parent compounds
   - Messages include parent features + relation type embeddings
   - Information propagates hierarchically through multiple layers

3. **Enhanced Conditioning**: 
   - Diffusion model sees graph-aware features
   - Each prediction is informed by structural context
   - Relation type patterns are learned through embeddings

4. **Consistent Predictions**:
   - Compounds with dependencies make correlated predictions
   - Structural constraints are implicitly enforced
   - Errors are less likely to be isolated and inconsistent

5. **Better Exact Match**:
   - Fewer isolated errors
   - Structurally consistent predictions across the dependency tree
   - Relation-specific patterns improve label accuracy

**Bottom line**: The graph encoding provides the diffusion model with **structural context** that was missing before. Instead of predicting labels in isolation, the model now predicts labels that are consistent with the entire compound dependency structure.
