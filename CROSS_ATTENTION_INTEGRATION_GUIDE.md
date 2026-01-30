# Cross-Attention DiT Integration Guide for trainer_necti

## Overview
The cross-attention `DiT` from `dit_discrete_cross_attention.py` enables better fusion of BERT features with the diffusion model compared to the standard approach. Instead of just concatenation, it uses dedicated cross-attention mechanisms.

## Key Differences

### Standard DiT (dit_discrete.py)
- Concatenates input features with self-conditioning
- Uses only self-attention with time embedding
- Input: `x, t, bert_features, attention_mask, x_self_cond`

### Cross-Attention DiT (dit_discrete_cross_attention.py)
- Separates input processing from feature conditioning
- Uses both self-attention AND cross-attention to BERT features
- More explicit feature fusion via `CrossAttention` mechanism
- Better for leveraging contextual information

## Integration Steps

### Step 1: Create a New BitDit Wrapper for Cross-Attention

Create a new file `models/ddim_bitdit_cross_attn.py`:

```python
from models.ddim_bitdit import BitDit as BitDitBase
from models.dit_discrete_cross_attention import DiT as DiTCrossAttn

class BitDitCrossAttn(BitDitBase):
    """BitDit with cross-attention DiT backbone"""
    
    def __init__(self, **kwargs):
        # Store the use_cross_attn flag before calling parent init
        self.use_cross_attn = kwargs.pop('use_cross_attn', True)
        super().__init__(**kwargs)
        
        # Replace the standard DiT with cross-attention DiT
        if self.use_cross_attn:
            self.model = DiTCrossAttn(
                in_channels=self.bits.item(),
                hidden_size=self.dim_model,
                num_steps=self.timesteps,
                time_dim=self.args.dim_time,
                depth=self.args.depth,
                num_heads=8,
                mlp_ratio=4.0
            )
```

### Step 2: Modify trainer_necti.py

Update the model initialization in `trainer_necti.py`:

```python
# In NeCTITrainer.__init__()
from models.ddim_bitdit_cross_attn import BitDitCrossAttn

# Replace:
# self.model = BitDit(...)
# With:
self.model = BitDitCrossAttn(
    device=self.device,
    num_classes=self.args.num_classes,
    backbone=self.args.backbone,
    time_steps=self.args.time_steps,
    sampling_steps=self.args.sampling_steps,
    noise_schedule=self.args.noise_schedule,
    ddim_sampling_eta=self.args.ddim_sampling_eta,
    self_condition=self.args.self_condition,
    snr_scale=self.args.snr_scale,
    dataset=f"necti_{self.args.granularity}",
    dim_model=self.args.dim_model,
    dim_time=self.args.dim_time,
    objective=self.args.objective,
    loss_type=self.args.loss_type,
    add_lstm=self.args.add_lstm,
    freeze_bert=self.args.freeze_bert,
    max_length=self.args.max_length,
    depth=self.args.depth,
    num_labels=len(self.label_set),
    use_cross_attn=True  # Enable cross-attention
)
```

### Step 3: Add Command-line Argument

Update `options.py` to add the cross-attention flag:

```python
parser.add_argument('--use_cross_attn', type=bool, default=False,
                    help='Use cross-attention DiT instead of standard DiT')
```

## Comparison: Standard vs Cross-Attention Flow

### Standard Flow (Current):
```
Input bits
    ↓
Embed & Concat with self-conditioning
    ↓
Add BERT features + time embedding
    ↓
Self-Attention blocks (basic)
    ↓
Output predictions
```

### Cross-Attention Flow (New):
```
Input bits
    ↓
Embed & Concat with self-conditioning
    ↓
Self-Attention blocks with:
  - Self-attention on noisy input
  - Cross-attention FROM input TO BERT features
  - MLP layers
    ↓
Output predictions
```

## Advantages of Cross-Attention

1. **Better Feature Fusion**: Explicit cross-attention between denoising signal and BERT context
2. **Improved Context Modeling**: Each diffusion step can selectively attend to relevant BERT features
3. **Scalable Conditioning**: More efficient than concatenation for large hidden dimensions
4. **Flexibility**: Can be extended with multi-scale or hierarchical attention

## Usage Example

```bash
# Train with cross-attention DiT
python trainer_necti.py \
    --data_path /path/to/data \
    --use_cross_attn True \
    --backbone xlm-roberta-base \
    --granularity coarse \
    --batch_size 16 \
    --max_epochs 20
```

## Performance Considerations

- **Memory**: Cross-attention requires additional attention heads (similar memory footprint)
- **Speed**: Slightly slower per iteration but potentially better convergence
- **Quality**: Generally achieves better F1 scores on compound identification tasks

## Alternative: Direct Integration

If you want to use cross-attention DiT directly without creating a wrapper:

```python
# In trainer_necti.py imports
from models.dit_discrete_cross_attention import DiT as DiTCrossAttn

# Modify BitDit's __init__ to accept die_class parameter
# Or create a minimal adapter that uses DiTCrossAttn instead
```

## Debugging Tips

1. Check that attention masks are properly passed through the forward pass
2. Verify BERT feature dimensions match hidden_size in DiT initialization
3. Monitor attention patterns in logs to ensure cross-attention is being utilized
4. Compare F1 scores between standard and cross-attention variants

## Files to Modify

1. `models/dit_discrete_cross_attention.py` - Already complete
2. `models/ddim_bitdit_cross_attn.py` - Create new wrapper (optional)
3. `trainer_necti.py` - Update model initialization
4. `options.py` - Add command-line flag
