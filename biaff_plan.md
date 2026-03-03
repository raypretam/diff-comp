PROBLEM ANALYSIS
================

Your baseline Enc-Diffusion achieves 86.11 LSS, while DepNeCTI achieves 89.24 LSS.
The 3.13% gap comes from DepNeCTI's explicit bi-affine pairwise scoring:

    s(i,j) = h_i^T W h_j + U h_i + V h_j + b

This directly models the relationship between head (i) and modifier (j) positions.

Your diffusion model uses global self-attention where, per Hahn (2020):
- Each token's influence is O(1/n)
- Pairwise signals get diluted as sequence length increases
- Fine-grained case distinctions require strong pairwise signal

SOLUTION: Inject bi-affine pairwise information into diffusion WITHOUT changing
the core diffusion mechanics (which work well).

===============================================================================
APPROACH 1: BI-AFFINE GUIDED ATTENTION
===============================================================================

Idea: Use bi-affine scores to CREATE ATTENTION BIAS that guides the DiT's
attention toward relevant head-modifier pairs.

    Standard attention: A = softmax(QK^T / sqrt(d))
    Bi-affine guided:   A = softmax(QK^T / sqrt(d) + λ * B)
    
    where B[i,j] = h_i^T W h_j  (bi-affine score)

This "hints" to the attention mechanism which token pairs are grammatically related.

```python
class BiAffineGuidedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        # Bi-affine components
        self.head_mlp = nn.Linear(hidden_size, hidden_size)
        self.dep_mlp = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Parameter(torch.zeros(hidden_size, num_heads, hidden_size))
        self.bias_scale = nn.Parameter(torch.tensor(0.1))  # Learnable scale
        
    def compute_bias(self, features):
        '''Compute bi-affine attention bias.'''
        head_repr = self.head_mlp(features)  # [batch, seq, hidden]
        dep_repr = self.dep_mlp(features)    # [batch, seq, hidden]
        
        # Bi-affine: [batch, num_heads, seq, seq]
        bias = torch.einsum('bih,hnk,bjk->bnij', head_repr, self.W, dep_repr)
        return bias * self.bias_scale
    
    def forward(self, q, k, v, features, mask=None):
        # Standard attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # Add bi-affine bias
        biaffine_bias = self.compute_bias(features)
        attn_scores = attn_scores + biaffine_bias
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v)
```

WHY THIS WORKS:
- Bi-affine bias is ADDITIVE - doesn't break existing attention
- Guides attention toward related pairs
- Learnable scale prevents dominating the attention
- Single-stage diffusion preserved

EXPECTED: +1-2% LSS improvement

===============================================================================
APPROACH 2: BI-AFFINE FEATURE INJECTION
===============================================================================

Idea: For each token, compute a "pairwise context" vector that summarizes
its relationships with nearby tokens, weighted by bi-affine scores.

    pairwise_ctx[i] = Σ_j (softmax(biaffine[i,j]) * h_j)  for j in window(i)

Then inject this into the diffusion conditioning:

    conditioning = time_emb + bert_features + pairwise_ctx

```python
class BiAffinePairwiseContext(nn.Module):
    def __init__(self, hidden_size, window_size=7):
        self.window_size = window_size
        
        # Bi-affine scorer
        self.head_proj = nn.Linear(hidden_size, hidden_size)
        self.dep_proj = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        
        # Projection for output
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        '''Compute pairwise context for each position.'''
        batch, seq_len, hidden = features.shape
        half_w = self.window_size // 2
        
        head_repr = self.head_proj(features)
        dep_repr = self.dep_proj(features)
        
        # Compute bi-affine scores: [batch, seq, seq]
        scores = torch.einsum('bih,hk,bjk->bij', head_repr, self.W, dep_repr)
        
        # For each position, aggregate from local window
        pairwise_ctx = torch.zeros_like(features)
        
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            
            # Local scores and features
            local_scores = scores[:, i, start:end]  # [batch, window]
            local_feats = features[:, start:end, :]  # [batch, window, hidden]
            
            # Attention-weighted aggregation
            weights = F.softmax(local_scores, dim=-1).unsqueeze(-1)
            ctx = (weights * local_feats).sum(dim=1)  # [batch, hidden]
            
            pairwise_ctx[:, i] = ctx
        
        # Gate the pairwise context
        pairwise_ctx = self.out_proj(pairwise_ctx)
        gate_input = torch.cat([features, pairwise_ctx], dim=-1)
        gate = self.gate(gate_input)
        
        return features + gate * pairwise_ctx  # Residual addition
```

WHY THIS WORKS:
- Each token gets explicit information about its relationships
- Local window respects compound structure
- Gated addition prevents destroying original features
- Directly addresses O(1/n) dilution problem

EXPECTED: +1-3% LSS improvement

===============================================================================
APPROACH 3: BI-AFFINE CROSS-ATTENTION IN DiT
===============================================================================

Idea: Add a dedicated cross-attention layer in each DiT block that attends
to bi-affine relationship representations.

Architecture:
    x -> Self-Attn -> Cross-Attn(x, biaffine_repr) -> MLP -> output

```python
class DiTBlockWithBiAffineCrossAttn(nn.Module):
    def __init__(self, hidden_size, num_heads):
        # Standard self-attention
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        
        # Cross-attention to bi-affine representations
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        
        # MLP
        self.mlp = MLP(hidden_size)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
    
    def forward(self, x, conditioning, biaffine_repr, mask=None):
        # Self-attention
        x = x + self.self_attn(self.norm1(x), mask=mask)
        
        # Cross-attention to bi-affine relationships
        x = x + self.cross_attn(
            query=self.norm2(x),
            key=biaffine_repr,
            value=biaffine_repr,
            mask=mask
        )
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x


class BiAffineRepresentationEncoder(nn.Module):
    '''Encode bi-affine scores as sequence of relationship representations.'''
    
    def __init__(self, hidden_size, num_labels):
        self.scorer = BiAffineScorer(hidden_size, num_labels)
        self.label_embed = nn.Embedding(num_labels, hidden_size)
        self.proj = nn.Linear(hidden_size * 2 + num_labels, hidden_size)
    
    def forward(self, features):
        batch, seq_len, hidden = features.shape
        
        # Get bi-affine scores and representations
        scores, head_repr, dep_repr = self.scorer(features)
        # scores: [batch, seq, seq, num_labels]
        
        # For each position, create representation encoding its relationships
        # Use top-k most related positions
        
        k = 3  # Top-3 relationships per position
        biaffine_reprs = []
        
        for i in range(seq_len):
            # Scores from position i to all others
            pos_scores = scores[:, i, :, :]  # [batch, seq, num_labels]
            max_scores = pos_scores.max(dim=-1).values  # [batch, seq]
            
            # Top-k positions
            topk_idx = max_scores.topk(k, dim=-1).indices  # [batch, k]
            
            # Gather representations
            batch_idx = torch.arange(batch).unsqueeze(1).expand(-1, k)
            topk_scores = pos_scores[batch_idx, topk_idx]  # [batch, k, num_labels]
            topk_feats = features[batch_idx, topk_idx]     # [batch, k, hidden]
            
            # Combine: [head_repr, aggregated_dep, label_dist]
            head_repr_i = head_repr[:, i:i+1, :].expand(-1, k, -1)
            combined = torch.cat([head_repr_i, topk_feats, topk_scores], dim=-1)
            
            # Project and pool
            proj = self.proj(combined)  # [batch, k, hidden]
            pooled = proj.mean(dim=1)   # [batch, hidden]
            
            biaffine_reprs.append(pooled)
        
        return torch.stack(biaffine_reprs, dim=1)  # [batch, seq, hidden]
```

WHY THIS WORKS:
- Dedicated pathway for pairwise information
- Cross-attention allows selective use of relationships
- Doesn't modify the self-attention (preserves what works)

EXPECTED: +2-4% LSS improvement (but more complex)

===============================================================================
APPROACH 4: AUXILIARY BI-AFFINE ARC PREDICTION (Simplest, Recommended First)
===============================================================================

Idea: Add bi-affine arc prediction as auxiliary task. This doesn't change
diffusion at all, but encourages BERT features to encode pairwise relationships.

```python
class DiffusionWithBiAffineAux(nn.Module):
    def __init__(self, ...):
        self.backbone = AutoModel.from_pretrained(backbone)
        self.diffusion = StandardDiT(...)  # Unchanged
        
        # Auxiliary bi-affine head (like DepNeCTI)
        self.arc_head = nn.Linear(hidden_size, hidden_size)
        self.arc_dep = nn.Linear(hidden_size, hidden_size)
        self.arc_biaffine = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        
        self.rel_head = nn.Linear(hidden_size, hidden_size)
        self.rel_dep = nn.Linear(hidden_size, hidden_size)
        self.rel_biaffine = nn.Parameter(torch.zeros(hidden_size, num_labels, hidden_size))
    
    def compute_arc_scores(self, features):
        '''Predict arc existence (is there a dependency?).'''
        arc_h = self.arc_head(features)
        arc_d = self.arc_dep(features)
        return torch.einsum('bih,hk,bjk->bij', arc_h, self.arc_biaffine, arc_d)
    
    def compute_rel_scores(self, features):
        '''Predict relationship type given arc.'''
        rel_h = self.rel_head(features)
        rel_d = self.rel_dep(features)
        return torch.einsum('bih,hlk,bjk->bijl', rel_h, self.rel_biaffine, rel_d)
    
    def forward(self, input_ids, attention_mask, fine_labels, head_indices=None):
        features = self.backbone(input_ids, attention_mask).last_hidden_state
        
        # Main diffusion loss (UNCHANGED)
        diffusion_loss = self.diffusion.loss(features, fine_labels, attention_mask)
        
        # Auxiliary bi-affine losses
        if head_indices is not None:
            # Arc prediction loss
            arc_scores = self.compute_arc_scores(features)
            arc_targets = self._create_arc_targets(head_indices, features.shape[1])
            arc_loss = F.binary_cross_entropy_with_logits(arc_scores, arc_targets)
            
            # Relationship prediction loss
            rel_scores = self.compute_rel_scores(features)
            rel_loss = self._compute_rel_loss(rel_scores, head_indices, fine_labels)
            
            aux_loss = arc_loss + rel_loss
        else:
            # No head indices - use diagonal as self-relationship proxy
            rel_scores = self.compute_rel_scores(features)
            diag_scores = rel_scores.diagonal(dim1=1, dim2=2).permute(0, 2, 1)
            aux_loss = F.cross_entropy(
                diag_scores.reshape(-1, self.num_labels),
                fine_labels.reshape(-1),
                ignore_index=-100
            )
        
        return diffusion_loss + 0.2 * aux_loss
```

WHY THIS WORKS:
- Forces BERT to learn pairwise-aware representations
- Diffusion sees better features without any changes
- Simple to implement, minimal risk

EXPECTED: +1-2% LSS improvement

===============================================================================
APPROACH 5: DIFFUSION IN BI-AFFINE SCORE SPACE (Most Novel)
===============================================================================

Idea: Instead of diffusing in label-bit space, diffuse in the bi-affine score space.
The bi-affine scores directly represent relationships.

Standard diffusion: x_0 = bits(labels), denoise bits -> predict labels
Bi-affine diffusion: x_0 = biaffine_scores[i,j,l], denoise scores -> predict relationships

```python
class BiAffineScoreDiffusion(nn.Module):
    '''Diffusion in bi-affine score space.'''
    
    def __init__(self, hidden_size, num_labels, ...):
        self.backbone = AutoModel.from_pretrained(backbone)
        
        # Initial bi-affine scorer (provides x_0)
        self.initial_scorer = BiAffineScorer(hidden_size, num_labels)
        
        # Denoising network operates on score matrices
        self.score_denoiser = ScoreMatrixDiT(
            input_dim=num_labels,  # Score vector dimension
            hidden_size=hidden_size,
            ...
        )
    
    def forward(self, input_ids, attention_mask, fine_labels, head_indices):
        features = self.backbone(input_ids, attention_mask).last_hidden_state
        
        batch, seq_len = features.shape[:2]
        
        # Compute initial bi-affine scores (this is x_0)
        scores, _, _ = self.initial_scorer(features)  # [batch, seq, seq, num_labels]
        
        # Create target score matrix from labels
        # For each (i, j) pair where j is head of i, target[i,j,label] should be high
        target_scores = self._create_target_scores(fine_labels, head_indices, seq_len)
        
        # Diffusion: add noise to scores
        t = torch.randint(0, self.num_timesteps, (batch,), device=self.device)
        noise = torch.randn_like(scores)
        noisy_scores = self.q_sample(scores, t, noise)
        
        # Denoise
        pred_scores = self.score_denoiser(noisy_scores, t, features, attention_mask)
        
        # Loss
        loss = F.mse_loss(pred_scores, scores)  # or target_scores
        
        return loss
    
    def inference(self, input_ids, attention_mask):
        features = self.backbone(input_ids, attention_mask).last_hidden_state
        
        # Sample denoised bi-affine scores
        scores = self.sample(features, attention_mask)  # [batch, seq, seq, num_labels]
        
        # Decode: for each position, find best head and label
        # (Similar to dependency parsing decoding)
        max_scores, labels = scores.max(dim=-1)  # [batch, seq, seq]
        heads = max_scores.argmax(dim=-1)        # [batch, seq]
        
        # Get label for each position based on its predicted head
        fine_preds = labels[torch.arange(batch).unsqueeze(1), torch.arange(seq_len), heads]
        
        return fine_preds
```

WHY THIS WORKS:
- Diffusion directly models pairwise relationships
- Score matrix is the natural representation for dependency-like tasks
- Leverages diffusion's strength in structured prediction

EXPECTED: Potentially +3-5% LSS (but needs careful implementation)

===============================================================================
RECOMMENDED IMPLEMENTATION ORDER
===============================================================================

1. **FIRST (Today)**: Approach 4 - Auxiliary Bi-Affine Loss
   - Zero changes to diffusion
   - Easy to implement
   - Safe baseline improvement

2. **SECOND (1-2 days)**: Approach 2 - Bi-Affine Feature Injection
   - Add pairwise context to conditioning
   - Moderate complexity
   - Good balance of improvement vs effort

3. **THIRD (2-3 days)**: Approach 1 - Bi-Affine Guided Attention
   - Modify attention with bi-affine bias
   - More invasive but principled
   - Strong theoretical motivation

4. **OPTIONAL (Research direction)**: Approach 5 - Score Space Diffusion
   - Novel contribution
   - More complex
   - Good for future work / another paper

===============================================================================
QUICK IMPLEMENTATION: Auxiliary Bi-Affine (Start Here!)
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBiAffineHead(nn.Module):
    """Simple bi-affine head for auxiliary supervision."""
    
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mlp_dep = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        reduced = hidden_size // 2
        self.W = nn.Parameter(torch.zeros(reduced, num_labels, reduced))
        nn.init.xavier_uniform_(self.W.view(reduced, -1))
        
        self.U = nn.Linear(reduced, num_labels)
        self.V = nn.Linear(reduced, num_labels)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, hidden]
        Returns:
            scores: [batch, seq_len, seq_len, num_labels]
        """
        head = self.mlp_head(features)
        dep = self.mlp_dep(features)
        
        bilinear = torch.einsum('bih,hlk,bjk->bijl', head, self.W, dep)
        head_contrib = self.U(head).unsqueeze(2)
        dep_contrib = self.V(dep).unsqueeze(1)
        
        return bilinear + head_contrib + dep_contrib


def add_biaffine_aux_to_existing_model(model, hidden_size, num_labels, aux_weight=0.2):
    """
    Monkey-patch an existing diffusion model to add bi-affine auxiliary loss.
    
    Usage:
        model = YourExistingDiffusionModel(...)
        model = add_biaffine_aux_to_existing_model(model, 768, 56)
    """
    
    # Add bi-affine head
    model.biaffine_head = SimpleBiAffineHead(hidden_size, num_labels).to(next(model.parameters()).device)
    model.aux_weight = aux_weight
    
    # Store original forward
    original_forward = model.forward
    
    def new_forward(input_ids, attention_mask, fine_labels, **kwargs):
        # Call original forward
        result = original_forward(input_ids, attention_mask, fine_labels, **kwargs)
        
        # Get features (assuming model has backbone)
        with torch.no_grad():
            features = model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        features = features.detach().requires_grad_(True)
        
        # Bi-affine auxiliary loss
        biaffine_scores = model.biaffine_head(features)
        
        # Use diagonal scores (token predicting its own label)
        diag_scores = biaffine_scores.diagonal(dim1=1, dim2=2).permute(0, 2, 1)
        biaffine_loss = F.cross_entropy(
            diag_scores.reshape(-1, num_labels),
            fine_labels.reshape(-1),
            ignore_index=-100
        )
        
        # Add to result
        if isinstance(result, dict):
            result['loss'] = result['loss'] + model.aux_weight * biaffine_loss
            result['biaffine_loss'] = biaffine_loss
        else:
            result = result + model.aux_weight * biaffine_loss
        
        return result
    
    model.forward = new_forward
    return model


# Example of how to add bi-affine features to conditioning
class BiAffineConditioningWrapper(nn.Module):
    """
    Wraps any DiT to add bi-affine pairwise features to conditioning.
    """
    
    def __init__(self, dit_model, hidden_size, window_size=7):
        super().__init__()
        self.dit = dit_model
        self.window_size = window_size
        
        # Bi-affine for pairwise features
        self.head_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.dep_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.W = nn.Parameter(torch.randn(hidden_size // 2, hidden_size // 2) * 0.02)
        
        # Gate for combining
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(hidden_size // 2, hidden_size)
    
    def compute_pairwise_features(self, features):
        """Compute local pairwise context for each position."""
        batch, seq_len, hidden = features.shape
        half_w = self.window_size // 2
        
        head = self.head_proj(features)
        dep = self.dep_proj(features)
        
        # Bi-affine scores
        scores = torch.einsum('bih,hk,bjk->bij', head, self.W, dep)
        
        # Aggregate local context
        pairwise = []
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            
            local_scores = scores[:, i, start:end]
            local_feats = dep[:, start:end]
            
            weights = F.softmax(local_scores, dim=-1).unsqueeze(-1)
            ctx = (weights * local_feats).sum(dim=1)
            pairwise.append(ctx)
        
        pairwise = torch.stack(pairwise, dim=1)
        pairwise = self.proj(pairwise)
        
        # Gate
        gate = self.gate(torch.cat([features, pairwise], dim=-1))
        return features + gate * pairwise
    
    def forward(self, x_t, t, features, attention_mask, **kwargs):
        # Enhance features with pairwise context
        enhanced_features = self.compute_pairwise_features(features)
        
        # Call original DiT with enhanced features
        return self.dit(x_t, t, enhanced_features, attention_mask, **kwargs)


print("Bi-Affine Conditioning Strategies for Fine-Grained NeCTI")
print("=" * 60)
print("See docstring for detailed implementation approaches")
print("Recommended: Start with Approach 4 (Auxiliary Bi-Affine Loss)")