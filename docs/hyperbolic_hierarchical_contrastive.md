# Hyperbolic Hierarchical Contrastive Learning for NER

This note explains **why** we add a hyperbolic contrastive auxiliary loss to
the mahanama NER model and **how** it works. It assumes you already know
contrastive learning and basic transformer NER; the hyperbolic part is
explained from scratch.

Files:

- Module: [`models/hyp_hier_contrastive.py`](../models/hyp_hier_contrastive.py)
- Hierarchy data: [`data/ner/mahanama_hierarchy.py`](../data/ner/mahanama_hierarchy.py)
- Integration: [`models/ddim_bitdit.py`](../models/ddim_bitdit.py)
- Config flags: [`configs/mahanama_ner_fine_hyp.yaml`](../configs/mahanama_ner_fine_hyp.yaml)


## 1. Why hyperbolic geometry for hierarchical labels?

Our label space is hierarchical:

```
              Person   Location   NORP   Misc   Time   O      ← 6 coarse
                │
        ┌───────┼────────┬─────────┐
   manuṣyaḥ  devatā    ṛṣiḥ    jantuḥ  …            ← 18 Person fine labels
```

Embedding such a tree in **Euclidean** space carries an intrinsic problem.
The number of nodes at depth `d` of a tree grows **exponentially** in `d`,
but the volume of a Euclidean ball grows only **polynomially**. So as the
tree gets bigger you cannot place children without crowding — sibling
distances and cousin distances collapse to similar values. This is a
well-known representation-distortion bound for trees in `ℝⁿ`.

**Hyperbolic space** (we use the Poincaré ball model) has volume that grows
**exponentially** with radius. A tree is embeddable in hyperbolic space with
arbitrarily low distortion (Sarkar, 2011). This gives two properties we
exploit:

1. **Norm encodes depth.** Points near the origin are "general", points
   near the boundary are "specific". A parent (coarse) embedding naturally
   sits closer to the origin than its children (fine).
2. **Geodesics route through the lowest common ancestor (LCA).** The
   shortest path between two leaf embeddings bows *toward the origin*,
   passing near their LCA. So distance between two leaves on the
   Poincaré ball approximates their tree-path length: same-coarse siblings
   stay close, cross-coarse cousins are far.


## 2. When does this help? (and when not)

| Situation | Does this help? |
|---|---|
| Labels form a known hierarchy (our mahanama case) | **Yes** — geometry matches the structure naturally |
| Labels are flat / no hierarchy | No real gain over Euclidean contrastive |
| Hierarchy exists but isn't supplied to the loss | No gain — geometry alone doesn't "discover" siblings |

The geometry is necessary but **not sufficient**. The hierarchy must still
enter the loss explicitly. In our implementation it enters through
(a) graded negative weighting and (b) learnable coarse parent anchors with
norm-ordering — see §4.

This module assumes the hierarchy is **known** (we have a fine→coarse
table). It is therefore distinct from the relationship-matrix + NOTEARS
mechanism in the original hHTM paper, which *learns* the hierarchy.


## 3. The Poincaré ball, in one screen

We use the Poincaré ball `𝔻^d_c = { x ∈ ℝ^d : c‖x‖² < 1 }` of curvature `c`.
Two operations matter:

- **Exponential map at the origin** lifts a tangent (Euclidean) vector `v`
  onto the ball:
  ```
  exp_0^c(v)  =  tanh(√c · ‖v‖) · v / (√c · ‖v‖)
  ```
- **Poincaré distance** (numerically stable form via `arccosh`):
  ```
  d_c(x, y)  =  (1/√c) · arccosh( 1 + 2c · ‖x − y‖² /
                                       ((1 − c‖x‖²)(1 − c‖y‖²)) )
  ```

Implementation:
[`hyp_hier_contrastive.py: _dist_from_terms / poincare_pairwise / poincare_rowwise`](../models/hyp_hier_contrastive.py).

**Numerical caveat.** As points approach the boundary, distances diverge
and gradients explode. We **clip tangent vectors** to a bounded norm
(`tangent_clip=1.0`) before `expmap_0`, so every embedding sits at a safe
radius `≤ tanh(√c · 1.0) ≈ 0.76`. This is essential — without it, training
NaNs after a few hundred steps.


## 4. The loss: four terms, all hierarchy-aware

For each batch we flatten valid (non-`-100`) tokens to `[M, D]`, project
them to `[M, proj_dim]`, clip-tangent, `expmap_0` onto the ball → token
embeddings `z_i`. We also lift the learnable coarse parent anchors
`a_c ∈ ℝ^{proj_dim}` (one per coarse class) onto the ball the same way.

Let `f_i` be token `i`'s fine label and `c_i = parent(f_i)` its coarse
category, looked up from a buffer derived from the
`fine_to_coarse_map`.

### Term 1 — Graded hyperbolic SupCon

A supervised contrastive loss with negatives weighted by tree distance:

```
positives P(i)   = { j : f_j = f_i, j ≠ i }
relation w(i,j)  = 1          if f_j = f_i           (positive)
                 = w_sibling  if c_j = c_i ≠ f_j     (same-coarse sibling)
                 = w_cross    if c_j ≠ c_i           (cross-coarse cousin)

L_supcon  = − mean_{i with |P(i)|>0} mean_{p ∈ P(i)} log[
              exp(−d_c(z_i, z_p) / τ)
            ─────────────────────────────────────────────────
            Σ_{a ≠ i} w(i,a) · exp(−d_c(z_i, z_a) / τ)
           ]
```

`w_sibling < w_cross` means **siblings are repelled less than cousins**.
Same-coarse types therefore stay geometrically close → the coarse structure
emerges in the embedding without ever being a direct label.

### Term 2 — Parent pull

Each token is pulled toward its coarse parent anchor:

```
L_parent  =  mean_i  d_c(z_i, a_{c_i})
```

This materializes the parent–child link explicitly, so a coarse anchor
becomes a concrete attractor for its fine children.

### Term 3 — Norm ordering (entailment)

A parent should be **more general** than its child — translated to the
ball, that means smaller norm. We add a margin hinge:

```
L_norm  =  mean_i  relu( ‖a_{c_i}‖ − ‖z_i‖ + ε_norm )
```

This is what makes the geometry *literally* encode depth = norm.

### Term 4 — Anchor separation

Coarse anchors should occupy distinct regions, otherwise the parent pull
collapses everything to one centroid:

```
L_sep  =  mean over c ≠ c'  relu( ε_sep − d_c(a_c, a_{c'}) )
```

### Total

```
L_hyp  =  λ_supcon · L_supcon
        + λ_parent · L_parent
        + λ_norm   · L_norm
        + λ_sep    · L_sep
```

returned from the module's `forward(features, seq_labels)`; the BitDit
trainer adds `hyp_contrastive_weight · L_hyp` to the main diffusion loss.


## 5. How the hierarchy actually enters

```
yaml flag  ─►  Trainer  ─►  BitDit(label_set=…)
                              │
                              ▼
            build_fine_to_coarse(label_set)        ← ENTITY_TO_COARSE
                              │     returns List[int] (length = 77)
                              ▼
            HyperbolicHierarchicalContrastive(fine_to_coarse=…)
                              │
                              ▼
            self.register_buffer('fine_to_coarse', …)
                              │
                              ▼
            forward:  coarse = self.fine_to_coarse[labels]   ← per batch
```

The mapping is **not in the YAML**. It is built from
[`ENTITY_TO_COARSE`](../data/ner/mahanama_hierarchy.py) by reading every
fine label from `LabelSet1D` and looking up its coarse parent. To change
the hierarchy you edit that dict; everything downstream re-derives itself.



## 6. Practical use

Enable in [`configs/mahanama_ner_fine_hyp.yaml`](../configs/mahanama_ner_fine_hyp.yaml):

```yaml
use_hyp_contrastive: True
hyp_contrastive_weight: 0.001   # the key knob
hyp_proj_dim: 128
hyp_contrastive_curvature: 1.0
hyp_temperature: 0.5
hyp_w_sibling: 1.0
hyp_w_cross: 2.0
```

Run:

```bash
python run.py --config_file mahanama_ner_fine_hyp.yaml
```

At runtime the model logs each batch's loss components as
`model.last_losses`:

```
{'diff_tok': 0.010, 'hyp_contrastive': 7.6}
```

The **scale gap** is important: the diffusion loss is ~`0.01` (MSE on bits
scaled by `snr_scale=0.1`) while the raw contrastive loss is ~`5–8`. With
`hyp_contrastive_weight = 0.001` the contribution becomes ~`0.008`,
comparable to `diff_tok`. **If the contrastive contribution
(`weight × hyp_contrastive`) drowns out `diff_tok`, lower the weight; if
it is negligible, raise it.**

Reasonable tuning knobs, in order of impact:

| Knob | What it does |
|---|---|
| `hyp_contrastive_weight` | Overall strength of the aux loss |
| `hyp_w_cross / hyp_w_sibling` | How strongly hierarchy is encoded (ratio = hierarchy strength) |
| `hyp_temperature` | Softness of the contrastive distribution |
| `hyp_proj_dim` | Capacity of the hyperbolic head |


## 8. References

1. Sarkar, R. (2011). *Low distortion Delaunay embedding of trees in
   hyperbolic plane.* GD.
2. Nickel, M. & Kiela, D. (2017). *Poincaré embeddings for learning
   hierarchical representations.* NeurIPS.
3. Ganea, O. et al. (2018). *Hyperbolic neural networks.* NeurIPS.
4. Khosla, P. et al. (2020). *Supervised contrastive learning.* NeurIPS.
5. Rao et al. (2024). *Hierarchical Topic Modeling via Contrastive
   Learning and Hyperbolic Embedding.* LREC-COLING.
   https://github.com/YRaoGroup/hHTM
6. Zheng, X. et al. (2018). *DAGs with NO TEARS.* NeurIPS. (Used by hHTM
   for the DAG constraint; we do **not** use it, since our hierarchy
   is given.)
