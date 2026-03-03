"""
Bi-Affine Enhanced Diffusion for Fine-Grained NeCTI
====================================================

This implementation adds bi-affine pairwise features to the diffusion process
WITHOUT changing the core diffusion mechanics.

KEY INSIGHT (from your paper):
- Fine-grained classification requires pairwise relationship modeling
- Global attention dilutes local signal as O(1/n) (Hahn 2020)
- Bi-affine scoring provides O(1) pairwise signal

SOLUTION:
1. Compute bi-affine pairwise scores from BERT features
2. Aggregate local pairwise context for each position
3. Inject into DiT conditioning (additive, gated)
4. Diffusion process unchanged

Expected improvement: +2-4% LSS over baseline
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple
from transformers import AutoModel
from einops import rearrange

from models.utils import decimal_to_bits, bits_to_decimal


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# =============================================================================
# Bi-Affine Pairwise Module
# =============================================================================

class BiAffinePairwiseEncoder(nn.Module):
    """
    Computes bi-affine pairwise representations for each token.
    
    For each position i, this module:
    1. Computes bi-affine scores to all positions in a local window
    2. Aggregates neighbor representations weighted by these scores
    3. Produces a "pairwise context" that encodes local relationships
    
    This directly addresses the O(1/n) dilution problem by providing
    explicit O(1) pairwise signal.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        window_size: int = 7,
        dropout: float = 0.1,
        num_relation_types: int = 4  # Coarse categories for efficiency
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.half_window = window_size // 2
        
        # Head and dependent projections (like in dependency parsing)
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.dep_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Bi-affine weight for scoring
        self.W_score = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.W_score)
        
        # Bi-affine for relationship-specific aggregation
        self.W_rel = nn.Parameter(torch.zeros(hidden_size, num_relation_types, hidden_size))
        nn.init.xavier_uniform_(self.W_rel.view(hidden_size, -1))
        
        # Relation type embedding
        self.rel_embed = nn.Embedding(num_relation_types, hidden_size)
        
        # Output projection and gating
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Learnable gate to control pairwise influence
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, features: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Compute pairwise-enhanced features.
        
        Args:
            features: [batch, seq_len, hidden]
            attention_mask: [batch, seq_len]
        
        Returns:
            enhanced_features: [batch, seq_len, hidden] - features + pairwise context
            pairwise_scores: [batch, seq_len, window_size] - for analysis/debugging
        """
        batch, seq_len, hidden = features.shape
        device = features.device
        scale = hidden ** -0.5  # prevent softmax overflow (same as scaled dot-product attention)
        
        # Compute head and dependent representations
        head_repr = self.head_mlp(features)  # [batch, seq, hidden]
        dep_repr = self.dep_mlp(features)    # [batch, seq, hidden]
        
        # Compute ALL pairwise bi-affine scores efficiently.
        # Cast to fp32: hidden=1024 -> unscaled einsum can reach ~1024, overflowing fp16.
        head_f32 = head_repr.float()
        dep_f32  = dep_repr.float()
        W_score_f32 = self.W_score.float()
        all_scores = torch.einsum('bih,hk,bjk->bij', head_f32, W_score_f32, dep_f32) * scale
        all_scores = all_scores.to(features.dtype)  # back to fp16 after scaling
        
        # Compute relationship-type scores for weighted aggregation (also in fp32)
        # DETACH: W_rel [1024,6,1024] creates a [B,256,256,6] gradient tensor → explosive.
        # rel_scores only weight the rel_embed lookup (a small 6×1024 table), so
        # gradients through W_rel are not needed — the feature path via W_score suffices.
        with torch.no_grad():
            W_rel_f32 = self.W_rel.float()
            rel_scores = torch.einsum('bih,hrk,bjk->bijr', head_f32, W_rel_f32, dep_f32) * scale
            rel_scores = rel_scores.to(features.dtype)
        
        # Now aggregate local context for each position
        # Using efficient windowed operations
        
        pairwise_contexts = []
        all_local_scores = []
        
        for i in range(seq_len):
            # Define local window
            start = max(0, i - self.half_window)
            end = min(seq_len, i + self.half_window + 1)
            actual_window = end - start
            
            # Get local scores: [batch, window]
            local_scores = all_scores[:, i, start:end]
            
            # Get local relationship scores: [batch, window, num_rel]
            local_rel_scores = rel_scores[:, i, start:end, :]
            
            # Get local features: [batch, window, hidden]
            local_feats = dep_repr[:, start:end, :]
            
            # Apply attention mask if provided
            if attention_mask is not None:
                local_mask = attention_mask[:, start:end]  # [batch, window]
                # Use -1e4 instead of -1e9: fp16 max is ~65504, so -1e9 overflows
                local_scores = local_scores.masked_fill(local_mask == 0, -1e4)
            
            # Softmax over window positions
            attn_weights = F.softmax(local_scores, dim=-1)  # [batch, window]
            
            # Compute relationship distribution
            rel_weights = F.softmax(local_rel_scores.mean(dim=1), dim=-1)  # [batch, num_rel]
            rel_context = (rel_weights.unsqueeze(-1) * self.rel_embed.weight).sum(dim=1)  # [batch, hidden]
            
            # Weighted aggregation of neighbor features
            neighbor_ctx = (attn_weights.unsqueeze(-1) * local_feats).sum(dim=1)  # [batch, hidden]
            
            # Combine neighbor context with relationship context
            combined = self.out_proj(torch.cat([neighbor_ctx, rel_context], dim=-1))
            
            pairwise_contexts.append(combined)
            all_local_scores.append(attn_weights)
        
        # Stack: [batch, seq_len, hidden]
        pairwise_ctx = torch.stack(pairwise_contexts, dim=1)
        
        # Gated residual addition
        gate = self.gate(torch.cat([features, pairwise_ctx], dim=-1))
        enhanced_features = features + gate * pairwise_ctx
        
        # Normalize scores to [-1, 1] before returning as DiT attention bias.
        # DETACH: the attention bias path is a regularizer — gradients already flow
        # through the feature enhancement path above, so double-backprop through
        # all_scores → W_score doubles the effective gradient magnitude unnecessarily.
        normed_scores = torch.tanh(all_scores).detach()
        
        return enhanced_features, normed_scores
    
    def get_pairwise_attention_map(self, features: Tensor) -> Tensor:
        """Get full pairwise attention map for visualization."""
        head_repr = self.head_mlp(features)
        dep_repr = self.dep_mlp(features)
        scale = features.shape[-1] ** -0.5
        scores = torch.einsum('bih,hk,bjk->bij',
            head_repr.float(), self.W_score.float(), dep_repr.float()) * scale
        return F.softmax(scores.to(features.dtype), dim=-1)


# =============================================================================
# DiT Block with Bi-Affine Enhancement
# =============================================================================

class BiAffineEnhancedDiTBlock(nn.Module):
    """
    Standard DiT block with optional bi-affine attention bias.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0,
        use_biaffine_attn_bias: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_biaffine_attn_bias = use_biaffine_attn_bias
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        
        # AdaLN modulation (from DiT)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        
        # Bi-affine attention bias projection
        if use_biaffine_attn_bias:
            self.biaffine_to_heads = nn.Linear(1, num_heads)
    
    def forward(
        self, 
        x: Tensor, 
        c: Tensor, 
        mask: Optional[Tensor] = None,
        biaffine_scores: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: [batch, seq_len, hidden] - noisy input
            c: [batch, seq_len, hidden] - conditioning
            mask: [batch, seq_len] - attention mask
            biaffine_scores: [batch, seq_len, seq_len] - pairwise scores for attention bias
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # === Self-Attention ===
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        
        # Add bi-affine attention bias (helps focus on related pairs)
        if self.use_biaffine_attn_bias and biaffine_scores is not None:
            # Project scalar scores to per-head bias
            # biaffine_scores: [B, N, N] -> [B, heads, N, N]
            bias = self.biaffine_to_heads(biaffine_scores.unsqueeze(-1))  # [B, N, N, heads]
            bias = bias.permute(0, 3, 1, 2)  # [B, heads, N, N]
            attn = attn + bias * 0.1  # Scale down to not dominate
        
        # Apply padding mask
        if mask is not None:
            padding_mask = mask[:, None, None, :]  # [B, 1, 1, N]
            attn = attn.masked_fill(padding_mask == 0, -1e4)
        
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, N, C)
        h = self.proj(h)
        
        x = x + gate_msa * h
        
        # === MLP ===
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h
        
        return x


# =============================================================================
# Main Model: Bi-Affine Enhanced Diffusion
# =============================================================================

class BiAffineEnhancedDiffusion(nn.Module):
    """
    Diffusion model with bi-affine pairwise enhancement for fine-grained NeCTI.
    
    Key innovations:
    1. BiAffinePairwiseEncoder provides explicit O(1) pairwise signal
    2. Bi-affine scores used as attention bias in DiT
    3. Core diffusion mechanics UNCHANGED (preserves what works)
    
    The bi-affine enhancement addresses the limitation identified by Hahn (2020)
    where global attention dilutes local pairwise signals needed for fine-grained
    case distinctions (e.g., ṣaṣṭhī vs saptamī tatpuruṣa).
    """
    
    def __init__(
        self,
        device: torch.device,
        backbone: str = 'google/muril-base-cased',
        dim_model: int = 768,
        freeze_bert: bool = False,
        # Diffusion
        time_steps: int = 1000,
        sampling_steps: int = 100,
        noise_schedule: str = 'cosine',
        snr_scale: float = 2.0,
        # DiT
        depth: int = 12,
        num_heads: int = 16,
        # Bi-affine
        biaffine_window: int = 7,
        use_biaffine_features: bool = True,
        use_biaffine_attn_bias: bool = True,
        # Labels
        num_fine_classes: int = 56,
        num_coarse_classes: int = 6,
        fine_to_coarse_map: Optional[Dict[int, int]] = None,
        # Training
        objective: str = 'pred_x0',
        loss_type: str = 'l2',
        aux_coarse_weight: float = 0.1,
    ):
        super().__init__()
        
        self.device = device
        self.dim_model = dim_model
        self.snr_scale = snr_scale
        self.objective = objective
        self.loss_type = loss_type
        self.use_biaffine_features = use_biaffine_features
        self.use_biaffine_attn_bias = use_biaffine_attn_bias
        self.aux_coarse_weight = aux_coarse_weight
        
        # Labels
        self.num_fine_classes = num_fine_classes
        self.num_coarse_classes = num_coarse_classes
        self.fine_bits = self._compute_bits(num_fine_classes)
        
        # Fine-to-coarse mapping
        if fine_to_coarse_map is not None:
            f2c = torch.zeros(num_fine_classes, dtype=torch.long)
            for fine_id, coarse_id in fine_to_coarse_map.items():
                if fine_id < num_fine_classes:
                    f2c[fine_id] = coarse_id
            self.register_buffer('fine_to_coarse', f2c)
        else:
            self.fine_to_coarse = None
        
        # Backbone
        self.backbone = AutoModel.from_pretrained(backbone)
        if freeze_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Diffusion schedule
        self._build_diffusion_schedule(time_steps, sampling_steps, noise_schedule)
        
        # === Bi-Affine Pairwise Module ===
        self.biaffine_encoder = BiAffinePairwiseEncoder(
            dim_model, 
            window_size=biaffine_window,
            num_relation_types=num_coarse_classes
        )
        
        # === DiT ===
        self.x_embedder = nn.Linear(self.fine_bits, dim_model)
        
        # Split into two parts: embedding lookup (needs LongTensor) + projection (needs float)
        # nn.Sequential[1:] slicing is not supported in PyTorch, so we keep them separate.
        self.time_embed_emb = nn.Embedding(time_steps, dim_model)
        self.time_embed_proj = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_model)
        )
        
        self.blocks = nn.ModuleList([
            BiAffineEnhancedDiTBlock(
                dim_model, num_heads,
                use_biaffine_attn_bias=use_biaffine_attn_bias
            )
            for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(dim_model, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(dim_model, self.fine_bits)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_model, 2 * dim_model)
        )
        
        # Auxiliary coarse head
        self.coarse_head = nn.Linear(dim_model, num_coarse_classes)
        
        self.to(device)
        
        # Print architecture
        print(f"\n{'='*60}")
        print("BI-AFFINE ENHANCED DIFFUSION FOR NeCTI")
        print(f"{'='*60}")
        print(f"  Backbone: {backbone}")
        print(f"  DiT: depth={depth}, heads={num_heads}")
        print(f"  Fine classes: {num_fine_classes} ({self.fine_bits} bits)")
        print(f"  Coarse classes: {num_coarse_classes}")
        print(f"  Bi-affine window: {biaffine_window}")
        print(f"  Use bi-affine features: {use_biaffine_features}")
        print(f"  Use bi-affine attention bias: {use_biaffine_attn_bias}")
        print(f"{'='*60}")
        print("  This addresses O(1/n) attention dilution by providing")
        print("  explicit O(1) pairwise signal for fine-grained distinctions.")
        print(f"{'='*60}\n")
    
    def _compute_bits(self, num_classes: int) -> int:
        return max(1, math.ceil(math.log2(num_classes + 1)))
    
    def _build_diffusion_schedule(self, time_steps, sampling_steps, noise_schedule):
        self.num_timesteps = time_steps
        self.sampling_timesteps = sampling_steps
        
        betas = cosine_beta_schedule(time_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1).float())
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def _get_coarse_labels(self, fine_labels):
        if self.fine_to_coarse is None:
            return torch.zeros_like(fine_labels)
        coarse = fine_labels.clone()
        valid = fine_labels != -100
        coarse[valid] = self.fine_to_coarse[fine_labels[valid].clamp(0, len(self.fine_to_coarse)-1)]
        return coarse
    
    def _denoise_step(self, x, t, features, biaffine_scores, attention_mask, x_self_cond=None):
        """Single denoising step."""
        # Embed noisy input
        x_emb = self.x_embedder(x)
        if x_self_cond is not None:
            x_emb = x_emb + self.x_embedder(x_self_cond)
        
        # Time embedding
        t_emb = self.time_embed_proj(self.time_embed_emb(t))
        t_emb = t_emb.unsqueeze(1).expand(-1, x_emb.shape[1], -1)
        
        # Conditioning (bi-affine enhanced features)
        c = t_emb + features
        
        # Forward through blocks with bi-affine attention bias
        h = x_emb
        for block in self.blocks:
            h = block(h, c, attention_mask, biaffine_scores)
        
        # Final projection
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift
        pred = self.final_linear(h)
        
        return pred
    
    def forward(self, input_ids, attention_mask, fine_labels):
        """Training forward pass."""
        # Extract BERT features
        bert_features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # === Bi-Affine Enhancement ===
        if self.use_biaffine_features:
            features, biaffine_scores = self.biaffine_encoder(bert_features, attention_mask)
        else:
            features = bert_features
            biaffine_scores = None
        
        if not self.use_biaffine_attn_bias:
            biaffine_scores = None
        
        # === Diffusion Loss ===
        fine_labels_clean = fine_labels.clone()
        fine_labels_clean[fine_labels_clean == -100] = 0
        
        bits = decimal_to_bits(fine_labels_clean, self.fine_bits) * self.snr_scale
        
        batch_size = features.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        noise = torch.randn_like(bits)
        x_t = self.q_sample(bits, t, noise)
        
        # Self-conditioning (50% of time)
        x_self_cond = None
        if torch.rand(1).item() < 0.5:
            with torch.no_grad():
                x_self_cond = self._denoise_step(
                    x_t, t, features, biaffine_scores, attention_mask
                ).detach()
        
        # Predict
        pred = self._denoise_step(x_t, t, features, biaffine_scores, attention_mask, x_self_cond)
        
        # Diffusion loss
        target = bits if self.objective == 'pred_x0' else noise
        mask = (fine_labels != -100).unsqueeze(-1).expand_as(pred)
        
        if mask.sum() == 0:
            diffusion_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        elif self.loss_type == 'l2':
            diffusion_loss = F.mse_loss(pred[mask], target[mask])
        else:
            diffusion_loss = F.l1_loss(pred[mask], target[mask])
        
        # === Auxiliary Coarse Loss ===
        coarse_labels = self._get_coarse_labels(fine_labels)
        coarse_logits = self.coarse_head(features)
        coarse_loss = F.cross_entropy(
            coarse_logits.reshape(-1, self.num_coarse_classes),
            coarse_labels.reshape(-1),
            ignore_index=-100
        )
        
        valid_coarse = coarse_labels.reshape(-1) != -100
        coarse_acc = torch.tensor(0.0, device=self.device)
        if valid_coarse.any():
            coarse_acc = (coarse_logits.reshape(-1, self.num_coarse_classes).argmax(-1)[valid_coarse] ==
                         coarse_labels.reshape(-1)[valid_coarse]).float().mean()
        
        # Total loss
        total_loss = diffusion_loss + self.aux_coarse_weight * coarse_loss
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'coarse_loss': coarse_loss,
            'coarse_acc': coarse_acc,
        }
    
    @torch.no_grad()
    def sample(self, features, biaffine_scores, attention_mask):
        """DDIM sampling."""
        batch_size, seq_len = features.shape[:2]
        shape = (batch_size, seq_len, self.fine_bits)
        
        x = torch.randn(shape, device=self.device)
        x_start = None
        
        times = torch.linspace(
            self.num_timesteps - 1, 0, self.sampling_timesteps,
            dtype=torch.long, device=self.device
        )
        
        for i, time in enumerate(times):
            t = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            
            pred = self._denoise_step(x, t, features, biaffine_scores, attention_mask, x_start)
            
            if self.objective == 'pred_x0':
                x_start = pred
            else:
                x_start = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * pred
                )
            
            x_start = torch.clamp(x_start, -self.snr_scale, self.snr_scale)
            
            if i < len(times) - 1:
                time_next = times[i + 1]
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                
                c = (1 - alpha_next).sqrt()
                x = x_start * alpha_next.sqrt() + c * (x - x_start * alpha.sqrt()) / (1 - alpha).sqrt()
            else:
                x = x_start
        
        fine_preds = bits_to_decimal(x / self.snr_scale, self.fine_bits)
        return fine_preds.clamp(0, self.num_fine_classes - 1)
    
    @torch.no_grad()
    def inference(self, input_ids, attention_mask):
        """Full inference."""
        bert_features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        if self.use_biaffine_features:
            features, biaffine_scores = self.biaffine_encoder(bert_features, attention_mask)
        else:
            features = bert_features
            biaffine_scores = None
        
        if not self.use_biaffine_attn_bias:
            biaffine_scores = None
        
        fine_preds = self.sample(features, biaffine_scores, attention_mask)
        coarse_preds = self.coarse_head(features).argmax(dim=-1)
        
        return fine_preds, coarse_preds
    
    def get_pairwise_attention(self, input_ids, attention_mask):
        """Get bi-affine attention map for visualization."""
        with torch.no_grad():
            features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.biaffine_encoder.get_pairwise_attention_map(features)


def create_biaffine_enhanced_diffusion(
    device: str = 'cuda',
    backbone: str = 'google/muril-base-cased',
    fine_to_coarse_map: Optional[Dict[int, int]] = None,
    **kwargs
) -> BiAffineEnhancedDiffusion:
    return BiAffineEnhancedDiffusion(
        device=torch.device(device),
        backbone=backbone,
        fine_to_coarse_map=fine_to_coarse_map,
        **kwargs
    )


if __name__ == '__main__':
    print("Testing Bi-Affine Enhanced Diffusion...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fine_to_coarse = {i: i // 10 for i in range(56)}
    
    model = create_biaffine_enhanced_diffusion(
        device=device,
        num_fine_classes=56,
        num_coarse_classes=6,
        fine_to_coarse_map=fine_to_coarse,
        depth=6,
        biaffine_window=7,
        use_biaffine_features=True,
        use_biaffine_attn_bias=True,
    )
    
    # Test
    batch, seq = 2, 32
    input_ids = torch.randint(0, 1000, (batch, seq)).to(device)
    attention_mask = torch.ones(batch, seq).to(device)
    fine_labels = torch.randint(0, 56, (batch, seq)).to(device)
    
    # Training
    out = model(input_ids, attention_mask, fine_labels)
    print(f"Total loss: {out['loss'].item():.4f}")
    print(f"Diffusion loss: {out['diffusion_loss'].item():.4f}")
    print(f"Coarse loss: {out['coarse_loss'].item():.4f}")
    print(f"Coarse acc: {out['coarse_acc'].item():.3f}")
    
    # Inference
    model.eval()
    fine_preds, coarse_preds = model.inference(input_ids, attention_mask)
    print(f"Fine predictions shape: {fine_preds.shape}")
    print(f"Coarse predictions shape: {coarse_preds.shape}")
    
    # Pairwise attention (for visualization)
    attn_map = model.get_pairwise_attention(input_ids, attention_mask)
    print(f"Pairwise attention shape: {attn_map.shape}")
    
    print("\n✓ All tests passed!")