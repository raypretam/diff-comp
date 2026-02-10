# Contrastive Learning for DiffusionSL

## Overview

InfoNCE contrastive learning has been successfully integrated into the DiffusionSL framework for sequence labeling. This enhancement helps the model learn better label representations by pulling embeddings of the same label closer while pushing different labels apart during the diffusion process.

## Implementation Details

### 1. **Contrastive Loss Module** ([models/contrastive_loss.py](models/contrastive_loss.py))

Two types of contrastive losses are available:

#### **InfoNCELoss** (Simple)
- Standard InfoNCE loss with temperature scaling
- Treats all label pairs equally (same label = positive, different = negative)
- Best for tasks without hierarchical label structure

#### **HierarchicalInfoNCELoss**
- Designed for nested/hierarchical labels (like NeCTI compound types)
- Fine positives: exact same label (weight = 1.0)
- Coarse positives: same coarse category (weight = 0.5)
- Hard negatives: different coarse categories

### 2. **Integration with Diffusion Model**

The contrastive loss is computed on the predicted bit embeddings during training:

```python
# In ddim_bitdit.py forward():
diffusion_loss = F.mse_loss(pred, target)  # Standard diffusion loss

if use_contrastive:
    contrastive_loss = contrastive_loss_fn(
        embeddings=pred,      # Predicted bit embeddings [B, L, bits]
        labels=seq_labels,    # Ground truth labels [B, L]
        mask=label_mask       # Attention mask [B, L]
    )
    total_loss = diffusion_loss + contrastive_weight * contrastive_loss
```

### 3. **Configuration Parameters**

Add these parameters to your YAML config:

```yaml
# Contrastive Learning Configuration
use_contrastive: True          # Enable/disable contrastive learning
contrastive_weight: 0.1        # Weight for contrastive loss (0.1 = 10% of total)
contrastive_temp: 0.07         # Temperature (lower = harder negatives)
contrastive_type: 'simple'     # 'simple' or 'hierarchical'
```

### 4. **Training with Contrastive Learning**

Run training normally:

```bash
cd /home/pretam-pg/DiffusionSL
conda activate diff
bash run_necti_finegrain.sh
```

The trainer will automatically:
- Initialize the contrastive loss module
- Compute both diffusion and contrastive losses
- Log separate loss components to WandB:
  - `train/loss` (total)
  - `train/diffusion_loss`
  - `train/contrastive_loss`

## Benefits for NeCTI Task

1. **Better Label Discrimination**: Model learns to distinguish between O, B-Compound, I-Compound, and 54 fine-grained types
2. **Reduced Confusion**: Similar compound types (e.g., Nominative vs. Accusative) get distinct representations
3. **Improved Exact Match**: Better token-level predictions → higher span-level EM scores
4. **Complementary to Diffusion**: Works alongside the diffusion objective without interference

## Hyperparameter Tuning

### Temperature (`contrastive_temp`)
- **Lower (0.01-0.05)**: Harder negatives, stricter separation
- **Default (0.07)**: Balanced, works well for most tasks
- **Higher (0.1-0.2)**: Softer negatives, more forgiving

### Weight (`contrastive_weight`)
- **Lower (0.05)**: Subtle guidance, diffusion dominates
- **Default (0.1)**: Balanced contribution (10% of loss)
- **Higher (0.2-0.5)**: Stronger contrastive signal

Start with defaults and adjust based on validation performance.

## Monitoring Training

Watch for these metrics in WandB:
- **Diffusion loss should decrease steadily**: Model learning to denoise
- **Contrastive loss should decrease**: Label representations improving
- **Total loss = diffusion_loss + contrastive_weight × contrastive_loss**

If contrastive loss stays high:
- Lower the temperature (make task easier)
- Increase the weight (give it more influence)
- Check label distribution (imbalanced labels can hurt contrastive learning)

## Test Results

All tests passed successfully ✅:

```
Test 1: Basic InfoNCE Loss                    ✓ PASS
Test 2: InfoNCE Loss with Masking            ✓ PASS
Test 3: InfoNCE with No Positive Pairs       ✓ PASS
Test 4: Hierarchical InfoNCE Loss            ✓ PASS
Test 5: Gradient Flow                         ✓ PASS
Test 6: Various Batch Sizes                   ✓ PASS
```

Run tests anytime with:
```bash
conda activate diff
python test_contrastive_loss.py
```

## Future Enhancements

Potential improvements:
1. **Hard Negative Mining**: Focus on most confusing label pairs
2. **Momentum Encoder**: Use EMA of embeddings for stable contrastive learning
3. **Multi-scale Contrastive**: Apply at different diffusion timesteps
4. **Supervised Contrastive**: Use label similarities as weights

## References

- **InfoNCE**: Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018)
- **Supervised Contrastive**: Khosla et al., "Supervised Contrastive Learning" (2020)
- **Contrastive for NER**: Zhou et al., "Contrastive Learning for Label Enhancement in Named Entity Recognition" (2022)
