# Hyperbolic Hierarchical Contrastive Learning for NER

The implementation lives in:

- `models/hyp_hier_contrastive.py` — the auxiliary loss module
- `data/ner/mahanama_hierarchy.py` — the fine→coarse label hierarchy
- `configs/mahanama_ner_fine_hyp.yaml` — training config with the loss enabled
- `trainer_ner_hyp.py` — trainer wiring the auxiliary loss into the main loop

---

## 1. What is the model about?

The module is an **auxiliary loss**, not a standalone classifier. It is added
on top of the existing MuRIL + diffusion NER backbone and shapes the geometry
of the encoder's token representations.

### 1.1 Core idea

Entity labels in mahanama-NER form a natural two-level hierarchy:

```
COARSE  →  FINE
Person     īśvaraḥ, devatā, ṛṣiḥ, manuṣyaḥ, jantuḥ, ...
Location   janapadaḥ, prākṛtikasthānam, mānavanirmitaḥ, ...
NORP       samūhaḥ
Misc       calanirjīvaḥ, śabdaḥ, unknown, ...
Time       kālaḥ
O          (non-entity)
```

A standard cross-entropy classifier treats all 77 fine labels as flat,
unrelated classes. The hyperbolic hierarchical contrastive loss instead
**embeds tokens on the Poincaré ball** and uses the known hierarchy to
organize them:

1. **Graded supervised contrastive** — for every anchor token,
   - positives = tokens with the **same fine label**,
   - *sibling* negatives = same coarse parent, different fine label
     (repelled with low weight `w_sibling`),
   - *cousin* negatives = different coarse parent
     (repelled with higher weight `w_cross`).

2. **Parent pull** — one learnable "parent anchor" point per coarse class
   (initialized near the origin); each token is pulled toward its parent.

3. **Norm-ordering / entailment** — on the Poincaré ball, ball-norm corresponds
   to depth in the hierarchy. A `relu(‖parent‖ − ‖child‖ + margin)` term
   forces parents to sit closer to the origin than their children
   (general → specific = origin → boundary).

4. **Anchor separation** — coarse anchors are pushed apart so the six coarse
   categories occupy distinct sectors of the ball.

The total auxiliary loss is

```
L_aux = λ_supcon·L_supcon + λ_parent·L_parent + λ_norm·L_norm + λ_sep·L_sep
```

and is added to the main diffusion NER loss with a small weight
(`hyp_contrastive_weight: 0.001` in the config) so it nudges the geometry
without dominating training.

### 1.2 Why **hyperbolic** geometry?

Hyperbolic space (here, the Poincaré ball with curvature `c = 1`) has
**exponentially growing volume** with radius. This is exactly the shape of a
tree / hierarchy: the number of descendants grows exponentially with depth.

Consequences used by this module:

- General concepts (coarse parents) can sit near the origin where there is
  little room, and many specific descendants can fit near the boundary
  without crowding — Euclidean space cannot do this without distortion.
- Distance on the ball naturally encodes hierarchical similarity:
  same-fine tokens cluster tight, same-coarse tokens stay in the same sector,
  cross-coarse tokens are far apart.
- Norm = depth gives a free "is-a" signal: simply by pulling parents toward
  the origin, the model learns entailment without any extra structural
  supervision.

Distances are computed with the numerically stable `acosh` form

```
d_c(x, y) = (1/√c) · arccosh(1 + 2c·‖x−y‖² / ((1−c‖x‖²)(1−c‖y‖²)))
```

and tangent vectors are clipped before `expmap0` to keep embeddings away
from the boundary where distances diverge.

---

## 2. Why use this for the NER task?

The mahanama Sanskrit-NER label space has properties that make a flat
classifier suboptimal:

### 2.1 Strong, known label hierarchy
77 fine labels collapse cleanly into 6 coarse buckets (Person / Location /
NORP / Misc / Time / O). This prior is free supervision — a flat
cross-entropy throws it away. The contrastive term injects it directly into
the representation space.

### 2.2 Severe fine-grained class imbalance and confusion
Several fine classes (e.g. `devayoniḥ`, `alaukikaprāṇī`,
`alaukika_acalanirjīvavastu`) are rare and routinely confused with sibling
classes under the **same** coarse parent. Pure CE provides no signal that
"`devatā` vs `ṛṣiḥ` is a smaller mistake than `devatā` vs `kālaḥ`."
The graded contrastive weights (`w_sibling < w_cross`) make sibling
confusions cheap and cross-coarse confusions expensive — which matches what
downstream evaluation actually cares about (coarse F1 is meaningful even
when fine F1 is hard).

### 2.3 Hierarchical metrics matter
The project reports both fine-grained and coarse-grained metrics (see
`evaluate_coarse.py`, `evaluate_fine.py`). A representation that already
respects the hierarchy makes the coarse evaluation much more robust without
hurting fine accuracy.

### 2.4 Geometry beats Euclidean for this label tree
Earlier experiments in this repo with Euclidean supervised-contrastive
(`models/contrastive_learning_*`, see `CONTRASTIVE_LEARNING_GUIDE.md`)
improved separation but could not simultaneously enforce both
*similarity* (same coarse stays close) **and** *entailment* (parent more
general than child). Hyperbolic embeddings encode both at once: closeness
via Poincaré distance, generality via ball-norm.

### 2.5 Complementary to the diffusion backbone
The main NER model uses bit-diffusion over labels (`models/ddim_bitdit.py`).
The diffusion loss optimizes the *output* distribution but does not directly
shape encoder features. The contrastive loss operates on the encoder side,
so the two are orthogonal — adding the auxiliary term improves the features
the diffusion head conditions on without changing the decoding procedure.

### 2.6 Cheap and stable
- Only one extra `Linear` projection head and a small `[num_coarse, proj_dim]`
  anchor parameter (≈ 6 × 128 floats).
- Tokens are subsampled to `max_tokens=512` per batch (entity tokens
  prioritized) so the pairwise [M, M] distance matrix stays small.
- Curvature is fixed (learnable curvature destabilizes the loss in practice).
- The auxiliary weight is tiny (`0.001`) so if the term ever misbehaves it
  cannot derail the main objective.

---

## 3. How it plugs into training

Enable in any config based on `ner_hyp`:

```yaml
use_hyp_contrastive: True
hyp_contrastive_weight: 0.001
hyp_proj_dim: 128
hyp_contrastive_curvature: 1.0
hyp_temperature: 0.5
hyp_w_sibling: 1.0
hyp_w_cross: 2.0
```

`trainer_ner_hyp.py` builds the `fine_to_coarse` list via
`data/ner/mahanama_hierarchy.build_fine_to_coarse(...)`, instantiates
`HyperbolicHierarchicalContrastive`, and adds its scalar output to the main
loss each step.

Example run:

```bash
python run.py --config configs/mahanama_ner_fine_hyp.yaml
```

Inference is unchanged — the auxiliary loss only affects training; at test
time you decode from the same diffusion head as before
(`infer_ner.py`).

---

## 4. References

- Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical
  Representations*, NeurIPS 2017.
- Ganea, Bécigneul & Hofmann, *Hyperbolic Neural Networks*, NeurIPS 2018.
- Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020.
- Rao et al., *hHTM: Hyperbolic Hierarchical Transformer*, LREC-COLING 2024 —the hyperbolic Hawkes encoder this module pairs with.