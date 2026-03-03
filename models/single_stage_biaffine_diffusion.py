"""
Single-Stage Bi-Affine Diffusion for NeCTI (NO CASCADE)
========================================================

THIS IS THE FIX: Remove the two-stage cascade that's killing performance.

Your hierarchical bi-affine model failed because:
1. Stage 2 conditions on coarse predictions from Stage 1
2. Stage 1 has ~6% error rate
3. Those errors cascade into Stage 2, destroying USS/LSS
4. Bi-affine features get overwhelmed by wrong coarse embeddings

Results:
    Baseline:              USS=98.71, LSS=86.11, EM=68.33
    Hierarchical Bi-Affine: USS=92.19, LSS=73.73, EM=74.09  ← WORSE

Solution: Single-stage diffusion with bi-affine features, NO coarse cascade.
    - Bi-affine provides O(1) pairwise signal (vs O(1/n) from global attention)
    - Auxiliary coarse loss for regularization (NOT conditioning)
    - No error propagation

Expected: USS ~98-99%, LSS ~87-89% (maintain baseline + bi-affine improvement)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple
from random import random
from transformers import AutoModel

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


def sinusoidal_position_embedding(batch_size, seq_len, dim):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0).expand(batch_size, -1, -1)


# =============================================================================
# Bi-Affine Pairwise Encoder (Efficient, No Loops)
# =============================================================================

class BiAffinePairwiseEncoder(nn.Module):
    """
    Efficient bi-affine pairwise context encoder.
    
    For each position i:
    1. Compute bi-affine scores to all other positions
    2. Aggregate neighbor representations weighted by scores
    3. Produce pairwise context that encodes local relationships
    
    This provides O(1) pairwise signal vs O(1/n) from global attention.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Head/dependent projections (like dependency parsing)
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
        
        # Bi-affine weight
        self.W = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.W)
        
        # Output projection and gating
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, features: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: [B, N, D]
            attention_mask: [B, N]
        
        Returns:
            enhanced_features: [B, N, D] - features + pairwise context
            biaffine_scores: [B, N, N] - for attention bias (detached)
        """
        B, N, D = features.shape
        scale = D ** -0.5
        
        head = self.head_mlp(features)  # [B, N, D]
        dep = self.dep_mlp(features)    # [B, N, D]
        
        # Bi-affine scores (use fp32 for stability with large hidden dim)
        scores = torch.einsum('bih,hk,bjk->bij', head.float(), self.W.float(), dep.float()) * scale
        scores = scores.to(features.dtype)
        
        # Mask padding
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1)  # [B, 1, N]
            scores = scores.masked_fill(pad_mask == 0, -1e4)
        
        # Pairwise context via attention-weighted aggregation
        attn = F.softmax(scores, dim=-1)        # [B, N, N]
        pairwise_ctx = torch.bmm(attn, dep)     # [B, N, D]
        pairwise_ctx = self.out_proj(pairwise_ctx)
        
        # Gated residual
        gate = self.gate(torch.cat([features, pairwise_ctx], dim=-1))
        enhanced = features + gate * pairwise_ctx
        
        # Detached, normalized scores for attention bias
        normed_scores = torch.tanh(scores).detach()
        
        return enhanced, normed_scores


# =============================================================================
# DiT Block with Bi-Affine Attention Bias
# =============================================================================

class BiAffineAttention(nn.Module):
    """Self-attention with additive bi-affine bias."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Project scalar bi-affine scores to per-head bias
        self.bias_proj = nn.Linear(1, num_heads, bias=False)
        self.bias_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: Tensor, mask: Optional[Tensor], biaffine_scores: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add bi-affine attention bias
        if biaffine_scores is not None:
            bias = self.bias_proj(biaffine_scores.unsqueeze(-1))  # [B, N, N, heads]
            bias = bias.permute(0, 3, 1, 2)  # [B, heads, N, N]
            attn = attn + bias * self.bias_scale
        
        if mask is not None:
            pad = mask[:, None, None, :]
            attn = attn.masked_fill(pad == 0, -1e4)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class BiAffineDiTBlock(nn.Module):
    """DiT block with bi-affine attention bias."""
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = BiAffineAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
    
    def forward(self, x: Tensor, c: Tensor, mask: Optional[Tensor], biaffine_scores: Optional[Tensor] = None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(h, mask, biaffine_scores)
        
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        
        return x


# =============================================================================
# Main Model: Single-Stage Bi-Affine Diffusion
# =============================================================================

class SingleStageBiAffineDiffusion(nn.Module):
    """
    Single-stage diffusion with bi-affine pairwise conditioning.
    
    NO TWO-STAGE CASCADE. NO COARSE CONDITIONING AT INFERENCE.
    
    Architecture:
        BERT → features
             ↓
        BiAffinePairwiseEncoder → enhanced_features, biaffine_scores
             ↓
        BiAffineDiT(enhanced_features, biaffine_scores) → fine_labels
    
    Hierarchy is used ONLY for:
        1. Auxiliary coarse loss (regularization during training)
        2. Optional post-hoc constraint (inference-time, after sampling)
    
    This preserves baseline architecture while adding O(1) pairwise signal.
    """
    
    def __init__(
        self,
        device: torch.device,
        backbone: str = 'FacebookAI/xlm-roberta-large',
        dim_model: int = 1024,
        freeze_bert: bool = False,
        # Diffusion
        time_steps: int = 1000,
        sampling_steps: int = 100,
        snr_scale: float = 2.0,
        # DiT
        depth: int = 12,
        num_heads: int = 16,
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
        self.aux_coarse_weight = aux_coarse_weight
        
        # Labels
        self.num_fine_classes = num_fine_classes
        self.num_coarse_classes = num_coarse_classes
        self.fine_bits = max(1, math.ceil(math.log2(num_fine_classes + 1)))
        
        # Fine-to-coarse mapping (for auxiliary loss and constraint)
        if fine_to_coarse_map is not None:
            f2c = torch.zeros(num_fine_classes, dtype=torch.long)
            for fine_id, coarse_id in fine_to_coarse_map.items():
                if fine_id < num_fine_classes:
                    f2c[fine_id] = coarse_id
            self.register_buffer('fine_to_coarse', f2c)
            
            # Build reverse mapping for constraint
            self.coarse_to_fine_map: Dict[int, list] = {}
            for fine_id, coarse_id in fine_to_coarse_map.items():
                self.coarse_to_fine_map.setdefault(coarse_id, []).append(fine_id)
        else:
            self.fine_to_coarse = None
            self.coarse_to_fine_map = None
        
        # Backbone
        self.backbone = AutoModel.from_pretrained(backbone)
        if freeze_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Diffusion schedule
        self._build_diffusion_schedule(time_steps, sampling_steps)
        
        # Bi-affine pairwise encoder
        self.biaffine_encoder = BiAffinePairwiseEncoder(dim_model)
        
        # Single-stage DiT with bi-affine attention bias
        self.x_embedder = nn.Linear(self.fine_bits * 2, dim_model)  # *2 for self-conditioning
        
        self.time_embed = nn.Sequential(
            nn.Embedding(time_steps, 256),
            nn.Linear(256, dim_model // 2),
        )
        self.feature_proj = nn.Linear(dim_model, dim_model // 2)
        self.cond_fuse = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.LayerNorm(dim_model),
        )
        
        self.blocks = nn.ModuleList([
            BiAffineDiTBlock(dim_model, num_heads)
            for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(dim_model, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(dim_model, self.fine_bits)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_model, 2 * dim_model)
        )
        
        # Auxiliary coarse classifier (regularization only)
        self.coarse_head = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.GELU(),
            nn.Linear(dim_model // 2, num_coarse_classes)
        )
        
        self._initialize_weights()
        self.to(device)
        
        print(f"\n{'='*60}")
        print("SINGLE-STAGE BI-AFFINE DIFFUSION (NO CASCADE)")
        print(f"{'='*60}")
        print(f"  Backbone: {backbone}")
        print(f"  DiT: depth={depth}, heads={num_heads}")
        print(f"  Fine classes: {num_fine_classes} ({self.fine_bits} bits)")
        print(f"  Bi-affine pairwise encoding: YES")
        print(f"  Bi-affine attention bias: YES")
        print(f"  Coarse conditioning at inference: NO (this is the fix!)")
        print(f"  Auxiliary coarse weight: {aux_coarse_weight}")
        print(f"{'='*60}\n")
    
    def _initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)
    
    def _build_diffusion_schedule(self, time_steps, sampling_steps):
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
    
    def _get_coarse_labels(self, fine_labels: Tensor) -> Tensor:
        if self.fine_to_coarse is None:
            return torch.zeros_like(fine_labels)
        coarse = fine_labels.clone()
        valid = fine_labels != -100
        coarse[valid] = self.fine_to_coarse[fine_labels[valid].clamp(0, len(self.fine_to_coarse)-1)]
        return coarse
    
    def _denoise_step(self, x, t, enhanced_features, biaffine_scores, attention_mask, x_self_cond=None):
        """Single denoising step."""
        B, N = x.shape[:2]
        
        # Position embedding
        pos = sinusoidal_position_embedding(B, N, self.dim_model).to(x.device)
        
        # Input embedding with self-conditioning
        x_cond = torch.zeros_like(x) if x_self_cond is None else x_self_cond
        x_emb = self.x_embedder(torch.cat([x, x_cond], dim=-1).float()) + pos
        
        # Conditioning: time + enhanced features
        t_emb = self.time_embed[0](t)
        t_emb = self.time_embed[1](t_emb).unsqueeze(1).expand(-1, N, -1)
        f_emb = self.feature_proj(enhanced_features)
        c = self.cond_fuse(torch.cat([t_emb, f_emb], dim=-1))
        
        # Through bi-affine enhanced blocks
        h = x_emb
        for block in self.blocks:
            h = block(h, c, attention_mask, biaffine_scores)
        
        # Final projection
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift
        return self.final_linear(h)
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor, fine_labels: Tensor) -> Dict[str, Tensor]:
        """Training forward pass."""
        # Extract features
        bert_features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Bi-affine enhancement (NO coarse conditioning!)
        enhanced_features, biaffine_scores = self.biaffine_encoder(bert_features, attention_mask)
        
        # Prepare labels
        fine_labels_clean = fine_labels.clone()
        fine_labels_clean[fine_labels_clean == -100] = 0
        bits = decimal_to_bits(fine_labels_clean, self.fine_bits) * self.snr_scale
        
        # Sample timestep
        B = bert_features.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(bits)
        x_t = self.q_sample(bits, t, noise)
        
        # Self-conditioning (50%)
        x_self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                x_self_cond = self._denoise_step(
                    x_t, t, enhanced_features, biaffine_scores, attention_mask
                ).detach()
        
        # Predict
        pred = self._denoise_step(x_t, t, enhanced_features, biaffine_scores, attention_mask, x_self_cond)
        
        # Diffusion loss
        target = bits if self.objective == 'pred_x0' else noise
        mask = (fine_labels != -100).unsqueeze(-1).expand_as(pred)
        
        if mask.sum() == 0:
            diffusion_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        elif self.loss_type == 'l2':
            diffusion_loss = F.mse_loss(pred[mask], target[mask])
        else:
            diffusion_loss = F.l1_loss(pred[mask], target[mask])
        
        # Auxiliary coarse loss (regularization only, NOT conditioning)
        coarse_labels = self._get_coarse_labels(fine_labels)
        coarse_logits = self.coarse_head(enhanced_features)
        coarse_loss = F.cross_entropy(
            coarse_logits.reshape(-1, self.num_coarse_classes),
            coarse_labels.reshape(-1),
            ignore_index=-100
        )
        
        # Coarse accuracy
        valid = coarse_labels.reshape(-1) != -100
        if valid.any():
            coarse_acc = (coarse_logits.reshape(-1, self.num_coarse_classes).argmax(-1)[valid] ==
                         coarse_labels.reshape(-1)[valid]).float().mean()
        else:
            coarse_acc = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = diffusion_loss + self.aux_coarse_weight * coarse_loss
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'coarse_loss': coarse_loss,
            'coarse_acc': coarse_acc,
        }
    
    @torch.no_grad()
    def sample(self, enhanced_features: Tensor, biaffine_scores: Tensor, attention_mask: Tensor) -> Tensor:
        """DDIM sampling - NO coarse conditioning!"""
        B, N = enhanced_features.shape[:2]
        
        x = torch.randn(B, N, self.fine_bits, device=self.device)
        x_start = None
        
        times = torch.linspace(
            self.num_timesteps - 1, 0, self.sampling_timesteps,
            dtype=torch.long, device=self.device
        )
        
        for i, time in enumerate(times):
            t = torch.full((B,), time, device=self.device, dtype=torch.long)
            
            pred = self._denoise_step(x, t, enhanced_features, biaffine_scores, attention_mask, x_start)
            
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
    def inference(self, input_ids: Tensor, attention_mask: Tensor, apply_constraint: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Full inference - NO coarse conditioning during sampling!
        
        Coarse predictions are derived AFTER sampling for metrics only.
        Optional post-hoc constraint can be applied.
        """
        # Extract and enhance features
        bert_features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        enhanced_features, biaffine_scores = self.biaffine_encoder(bert_features, attention_mask)
        
        # Sample fine labels directly (NO coarse conditioning!)
        fine_preds = self.sample(enhanced_features, biaffine_scores, attention_mask)
        
        # Derive coarse predictions (for metrics only)
        coarse_preds = self.coarse_head(enhanced_features).argmax(dim=-1)
        
        # Optional: post-hoc constraint (inference-time only)
        if apply_constraint and self.coarse_to_fine_map is not None:
            fine_preds = self._apply_constraint(fine_preds, coarse_preds)
        
        return coarse_preds, fine_preds
    
    def _apply_constraint(self, fine_preds: Tensor, coarse_preds: Tensor) -> Tensor:
        """Post-hoc constraint: ensure fine pred is valid for predicted coarse."""
        if self.coarse_to_fine_map is None:
            return fine_preds
        
        B, N = fine_preds.shape
        for b in range(B):
            for s in range(N):
                coarse = coarse_preds[b, s].item()
                fine = fine_preds[b, s].item()
                if coarse in self.coarse_to_fine_map:
                    valid = self.coarse_to_fine_map[coarse]
                    if fine not in valid and len(valid) > 0:
                        fine_preds[b, s] = valid[0]
        return fine_preds


def create_single_stage_biaffine_model(
    device: str = 'cuda',
    backbone: str = 'FacebookAI/xlm-roberta-large',
    fine_to_coarse_map: Optional[Dict[int, int]] = None,
    **kwargs
) -> SingleStageBiAffineDiffusion:
    return SingleStageBiAffineDiffusion(
        device=torch.device(device),
        backbone=backbone,
        fine_to_coarse_map=fine_to_coarse_map,
        **kwargs
    )


if __name__ == '__main__':
    print("Testing Single-Stage Bi-Affine Diffusion...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fine_to_coarse = {i: i // 10 for i in range(56)}
    
    model = create_single_stage_biaffine_model(
        device=device,
        backbone='google/muril-base-cased',  # Smaller for testing
        dim_model=768,
        num_fine_classes=56,
        num_coarse_classes=6,
        fine_to_coarse_map=fine_to_coarse,
        depth=6,
    )
    
    # Test
    batch, seq = 2, 32
    input_ids = torch.randint(0, 1000, (batch, seq)).to(device)
    attention_mask = torch.ones(batch, seq).to(device)
    fine_labels = torch.randint(0, 56, (batch, seq)).to(device)
    
    out = model(input_ids, attention_mask, fine_labels)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Diffusion: {out['diffusion_loss'].item():.4f}")
    print(f"Coarse: {out['coarse_loss'].item():.4f}")
    print(f"Coarse acc: {out['coarse_acc'].item():.3f}")
    
    model.eval()
    coarse_preds, fine_preds = model.inference(input_ids, attention_mask)
    print(f"Fine preds: {fine_preds.shape}")
    print(f"Coarse preds: {coarse_preds.shape}")
    
    print("\n✓ Single-stage bi-affine diffusion ready!")