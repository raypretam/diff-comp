"""
Hierarchical Diffusion with Biaffine Conditioning for NeCTI
============================================================

Architecture:
  Stage 1: Global DiT → coarse labels (standard, unchanged)
  Stage 2: Biaffine-Conditioned DiT → fine-grained labels
    - Efficient biaffine pairwise scoring on BERT features
    - Biaffine attention bias injected into DiT blocks
    - Coarse label conditioning from Stage 1
    - Scheduled sampling for train-test gap bridging

Why biaffine conditioning fixes the USS→LSS gap:
  1. Pairwise signal is O(1) — no dilution from sequence length
  2. Directly models component relationships (like dep parsing)
  3. Attention bias guides DiT to focus on related token pairs
  4. Richer conditioning: [BERT] + [biaffine ctx] + [coarse embed]

Data flow:
  backbone(input_ids) → features [B, N, D]
                          ↓
  BiaffineConditioner(features) → enhanced_features, biaffine_scores
                          ↓
  Stage 1: GlobalDiT(features)     → coarse preds  (no biaffine needed)
  Stage 2: BiaffineDiT(enhanced + coarse_embed, biaffine_scores) → fine preds
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List
from collections import namedtuple
from random import random
from transformers import AutoModel
from timm.models.vision_transformer import Mlp

from models.utils import decimal_to_bits, bits_to_decimal
from models.dit_discrete import DiT as GlobalDiT, sinusoidal_position_embedding


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def extract(a, t, x_shape):
    """Extract the appropriate t index for a batch of indices."""
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


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# =============================================================================
# Efficient Biaffine Conditioner
# =============================================================================

class BiaffineConditioner(nn.Module):
    """
    Efficient biaffine pairwise feature encoder — fully vectorized, no loops.

    Computes biaffine scores between all token pairs:
        score(i,j) = head(h_i)^T W dep(h_j) / sqrt(d)

    Then aggregates pairwise context via attention:
        ctx(i) = sum_j softmax(score(i,:))_j * dep(h_j)

    Returns:
        enhanced_features: features + gate * pairwise_ctx  [B, N, D]
        biaffine_scores:   tanh-normalized scores           [B, N, N]
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Head/dep projections (like dependency parsing)
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

        # Biaffine weight matrix
        self.W = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.W)

        # Output projection + gating
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

    def forward(
        self, features: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: [B, N, D] BERT features
            attention_mask: [B, N]
        Returns:
            enhanced_features: [B, N, D]
            biaffine_scores:   [B, N, N] (tanh-normalized, detached for attn bias)
        """
        B, N, D = features.shape
        scale = D ** -0.5

        head = self.head_mlp(features)   # [B, N, D]
        dep  = self.dep_mlp(features)    # [B, N, D]

        # Full biaffine scores — use fp32 for stability (hidden=1024 can overflow fp16)
        scores = torch.einsum(
            'bih,hk,bjk->bij', head.float(), self.W.float(), dep.float()
        ) * scale
        scores = scores.to(features.dtype)

        # Mask padding
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1)  # [B, 1, N]
            scores = scores.masked_fill(pad_mask == 0, -1e4)

        # Pairwise context via attention-weighted aggregation (fully vectorized)
        attn = F.softmax(scores, dim=-1)        # [B, N, N]
        pairwise_ctx = torch.bmm(attn, dep)     # [B, N, D]
        pairwise_ctx = self.out_proj(pairwise_ctx)

        # Gated residual
        gate = self.gate(torch.cat([features, pairwise_ctx], dim=-1))
        enhanced = features + gate * pairwise_ctx

        # Detached, normalized scores for attention bias
        # (gradients flow through feature path; bias is a soft structural prior)
        normed_scores = torch.tanh(scores).detach()

        return enhanced, normed_scores


# =============================================================================
# Biaffine-Enhanced DiT Components
# =============================================================================

class BiaffineAttention(nn.Module):
    """Self-attention with additive biaffine attention bias."""

    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Project scalar biaffine scores to per-head bias
        self.bias_proj = nn.Linear(1, num_heads, bias=False)
        # Learnable scale — starts small so biaffine doesn't dominate early
        self.bias_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, mask, biaffine_scores=None):
        """
        Args:
            x: [B, N, D]
            mask: [B, N]
            biaffine_scores: [B, N, N] — tanh-normalized pairwise scores
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, heads, N, N]

        # Biaffine attention bias
        if biaffine_scores is not None:
            # [B, N, N] -> [B, N, N, 1] -> project -> [B, N, N, heads]
            bias = self.bias_proj(biaffine_scores.unsqueeze(-1))   # [B, N, N, heads]
            bias = bias.permute(0, 3, 1, 2)                       # [B, heads, N, N]
            attn = attn + bias * self.bias_scale

        # Padding mask
        if mask is not None:
            pad = mask[:, None, None, :]     # [B, 1, 1, N]
            attn = attn.masked_fill(pad == 0, -1e4)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class BiaffineDiTBlock(nn.Module):
    """DiT block with biaffine attention bias (adaLN-gated)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = BiaffineAttention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden,
            act_layer=lambda: nn.GELU(),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, mask, biaffine_scores=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h, mask, biaffine_scores)

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)
        return x


class BiaffineDiT(nn.Module):
    """
    DiT with biaffine attention bias — used for Stage 2 (fine-grained).

    Same structure as the standard DiT from dit_discrete.py but each block
    receives biaffine_scores as an additive attention bias.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        num_steps: int,
        time_dim: int = 256,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Input: bits + self-cond → hidden
        self.x_embedder = nn.Linear(2 * in_channels, hidden_size)
        self.time_embed = nn.Embedding(num_steps, time_dim)
        self.time_mlp = nn.Linear(time_dim, hidden_size // 2)
        self.bert_mlp = nn.Linear(hidden_size, hidden_size // 2)
        self.fuse_mlp = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        # Biaffine-enhanced blocks
        self.blocks = nn.ModuleList([
            BiaffineDiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer (same as standard DiT)
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, in_channels, bias=True)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.weight, std=0.02)
        nn.init.normal_(self.time_mlp.weight, std=0.02)
        nn.init.normal_(self.bert_mlp.weight, std=0.02)

        # Zero-out adaLN modulation (standard DiT practice)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        bert_features: Tensor,
        attention_mask: Tensor,
        biaffine_scores: Optional[Tensor] = None,
        x_self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:               [B, N, bits]  noisy label bits
            t:               [B]           diffusion timestep
            bert_features:   [B, N, D]     conditioning features
            attention_mask:  [B, N]
            biaffine_scores: [B, N, N]     pairwise bias (from BiaffineConditioner)
            x_self_cond:     [B, N, bits]  self-conditioning
        Returns:
            pred:            [B, N, bits]  predicted x_0 or noise
        """
        B, N = x.shape[:2]

        # Positional embedding
        pos = sinusoidal_position_embedding(B, N, self.hidden_size).to(x.device)

        # Input embedding
        x_cond = torch.zeros_like(x) if x_self_cond is None else x_self_cond
        x = self.x_embedder(torch.cat([x, x_cond], dim=-1).float()) + pos

        # Conditioning: fuse time + features
        t_emb = self.time_mlp(self.time_embed(t)).unsqueeze(1).expand(-1, N, -1)
        b_emb = self.bert_mlp(bert_features)
        c = self.norm(self.fuse_mlp(torch.cat([t_emb, b_emb], dim=-1)))

        # Through biaffine-enhanced blocks
        for block in self.blocks:
            x = block(x, c, attention_mask, biaffine_scores)

        # Final projection
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.final_linear(x)
        return x


# =============================================================================
# Main Model: Hierarchical Biaffine Diffusion
# =============================================================================

class HierarchicalBiaffineDiffusion(nn.Module):
    """
    Hierarchical Diffusion with Biaffine Conditioning for NeCTI.

    Stage 1: Global DiT  → coarse labels  (3 bits, 6 classes)
    Stage 2: BiaffineDiT → fine labels    (6 bits, 56 classes)
             conditioned on: biaffine pairwise context + coarse embeddings
             with biaffine attention bias in every DiT block

    Scheduled sampling bridges the train-test gap for coarse conditioning.
    """

    def __init__(
        self,
        device: torch.device,
        # Backbone
        backbone: str = 'FacebookAI/xlm-roberta-large',
        dim_model: int = 1024,
        freeze_bert: bool = False,
        # Diffusion
        time_steps: int = 1000,
        sampling_steps: int = 100,
        noise_schedule: str = 'cosine',
        snr_scale: float = 2.0,
        # Stage 1 (Global — coarse)
        global_depth: int = 6,
        global_num_heads: int = 16,
        num_coarse_classes: int = 6,
        # Stage 2 (Biaffine — fine, coarse-conditioned)
        fine_depth: int = 12,
        fine_num_heads: int = 16,
        num_fine_classes: int = 56,
        # Biaffine
        biaffine_dropout: float = 0.1,
        # Label mapping
        fine_to_coarse_map: Optional[Dict[int, int]] = None,
        # Training
        objective: str = 'pred_x0',
        loss_type: str = 'l2',
    ):
        super().__init__()

        self.device = device
        self.dim_model = dim_model
        self.snr_scale = snr_scale
        self.objective = objective
        self.loss_type = loss_type

        # Label configuration
        self.num_coarse_classes = num_coarse_classes
        self.num_fine_classes = num_fine_classes
        self.coarse_bits = self._compute_bits(num_coarse_classes)
        self.fine_bits = self._compute_bits(num_fine_classes)

        # Fine-to-coarse mapping
        if fine_to_coarse_map is not None:
            self.register_buffer(
                'fine_to_coarse',
                torch.tensor([fine_to_coarse_map.get(i, 0) for i in range(num_fine_classes)])
            )
        else:
            self.fine_to_coarse = None

        self._build_coarse_to_fine_map(fine_to_coarse_map)

        # ------ Backbone ------
        self.backbone = AutoModel.from_pretrained(backbone)
        if freeze_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ------ Diffusion schedule ------
        self._build_diffusion_schedule(time_steps, sampling_steps, noise_schedule)

        # ------ Stage 1: Global DiT for coarse labels ------
        self.global_dit = GlobalDiT(
            in_channels=self.coarse_bits,
            hidden_size=dim_model,
            depth=global_depth,
            num_heads=global_num_heads,
            num_steps=time_steps,
            time_dim=256,
        )

        # ------ Biaffine Conditioner (for Stage 2) ------
        self.biaffine_conditioner = BiaffineConditioner(
            hidden_size=dim_model,
            dropout=biaffine_dropout,
        )

        # ------ Stage 2: Biaffine DiT for fine-grained labels ------
        self.fine_dit = BiaffineDiT(
            in_channels=self.fine_bits,
            hidden_size=dim_model,
            depth=fine_depth,
            num_heads=fine_num_heads,
            num_steps=time_steps,
            time_dim=256,
        )

        # ------ Coarse conditioning embedding ------
        self.coarse_embed = nn.Embedding(num_coarse_classes + 1, dim_model)  # +1 for padding
        nn.init.normal_(self.coarse_embed.weight, std=0.02)

        # ------ Scheduled sampling config ------
        self.ss_max_ratio = 0.5
        self.ss_warmup_epochs = 5

        self.to(device)

        print(f"\n{'='*60}")
        print("HIERARCHICAL BIAFFINE DIFFUSION")
        print(f"{'='*60}")
        print(f"  Stage 1: Global DiT, depth={global_depth}, heads={global_num_heads}")
        print(f"  Stage 2: Biaffine DiT, depth={fine_depth}, heads={fine_num_heads}")
        print(f"           + biaffine pairwise conditioning")
        print(f"           + biaffine attention bias in every block")
        print(f"           + coarse label embedding conditioning")
        print(f"  Coarse classes: {num_coarse_classes} ({self.coarse_bits} bits)")
        print(f"  Fine classes: {num_fine_classes} ({self.fine_bits} bits)")
        print(f"  SNR scale: {snr_scale}")
        print(f"  Scheduled sampling: warmup={self.ss_warmup_epochs}, max_ratio={self.ss_max_ratio}")
        print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_bits(self, num_classes: int) -> int:
        return max(1, math.ceil(math.log2(num_classes + 1)))

    def _build_coarse_to_fine_map(self, fine_to_coarse_map: Optional[Dict[int, int]]):
        if fine_to_coarse_map is None:
            self.coarse_to_fine_map = None
            return
        self.coarse_to_fine_map: Dict[int, List[int]] = {}
        for fine_id, coarse_id in fine_to_coarse_map.items():
            self.coarse_to_fine_map.setdefault(coarse_id, []).append(fine_id)

    def _build_diffusion_schedule(self, time_steps, sampling_steps, noise_schedule):
        self.num_timesteps = time_steps
        self.sampling_timesteps = sampling_steps

        betas = cosine_beta_schedule(time_steps) if noise_schedule == 'cosine' \
            else linear_beta_schedule(time_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1).float())

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # -------------------------------------------------------------------------
    # Coarse conditioning helpers
    # -------------------------------------------------------------------------

    def _condition_features(self, features: Tensor, coarse_labels: Tensor) -> Tensor:
        """Add coarse label embeddings to (biaffine-enhanced) features."""
        cl = coarse_labels.clone()
        cl[cl == -100] = self.num_coarse_classes   # padding → last embed slot
        cl = cl.clamp(0, self.num_coarse_classes)
        return features + self.coarse_embed(cl)

    def _get_scheduled_sampling_ratio(self, epoch: int) -> float:
        if epoch < self.ss_warmup_epochs:
            return 0.0
        ramp = (epoch - self.ss_warmup_epochs) / max(1, self.ss_warmup_epochs)
        return min(self.ss_max_ratio, ramp)

    def _get_noisy_coarse_labels(self, coarse_labels: Tensor) -> Tensor:
        """Simulate Stage 1 errors (~10% token-level corruption)."""
        noisy = coarse_labels.clone()
        valid = coarse_labels != -100
        corrupt = (torch.rand_like(coarse_labels.float()) < 0.10) & valid
        noisy[corrupt] = torch.randint(
            0, self.num_coarse_classes, coarse_labels.shape, device=coarse_labels.device
        )[corrupt]
        return noisy

    # -------------------------------------------------------------------------
    # Stage 1: Global Coarse Prediction (unchanged)
    # -------------------------------------------------------------------------

    def forward_stage1(self, features, coarse_labels, attention_mask):
        coarse_clean = coarse_labels.clone()
        coarse_clean[coarse_clean == -100] = 0

        bits = decimal_to_bits(coarse_clean, self.coarse_bits) * self.snr_scale
        B = features.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)

        noise = torch.randn_like(bits)
        x_t = self.q_sample(bits, t, noise)

        pred = self.global_dit(x_t, t, features, attention_mask)

        target = bits if self.objective == 'pred_x0' else noise
        mask = (coarse_labels != -100).unsqueeze(-1).expand_as(pred)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if self.loss_type == 'l2':
            return F.mse_loss(pred[mask], target[mask])
        return F.l1_loss(pred[mask], target[mask])

    @torch.no_grad()
    def sample_stage1(self, features, attention_mask):
        B, N = features.shape[:2]
        x = torch.randn(B, N, self.coarse_bits, device=self.device)
        x_start = None

        times = torch.linspace(
            self.num_timesteps - 1, 0, self.sampling_timesteps,
            dtype=torch.long, device=self.device,
        )
        for i, time in enumerate(times):
            t = torch.full((B,), time, device=self.device, dtype=torch.long)

            pred = self.global_dit(x, t, features, attention_mask, x_self_cond=x_start)
            x_start = pred if self.objective == 'pred_x0' else self.predict_start_from_noise(x, t, pred)
            x_start = x_start.clamp(-self.snr_scale, self.snr_scale)

            if i < len(times) - 1:
                time_next = times[i + 1]
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                c = (1 - alpha_next).sqrt()
                x = x_start * alpha_next.sqrt() + c * (x - x_start * alpha.sqrt()) / (1 - alpha).sqrt()
            else:
                x = x_start

        preds = bits_to_decimal(x / self.snr_scale, self.coarse_bits)
        return preds.clamp(0, self.num_coarse_classes - 1)

    # -------------------------------------------------------------------------
    # Stage 2: Biaffine-Conditioned Fine-grained Prediction
    # -------------------------------------------------------------------------

    def forward_stage2(
        self,
        enhanced_features: Tensor,
        biaffine_scores: Tensor,
        coarse_labels: Tensor,
        fine_labels: Tensor,
        attention_mask: Tensor,
        epoch: int = 0,
    ) -> Tensor:
        """
        Stage 2 training with biaffine conditioning.

        enhanced_features: BERT + biaffine pairwise context  [B, N, D]
        biaffine_scores:   pairwise attention bias            [B, N, N]
        """
        fine_clean = fine_labels.clone()
        fine_clean[fine_clean == -100] = 0

        bits = decimal_to_bits(fine_clean, self.fine_bits) * self.snr_scale
        B = enhanced_features.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)

        noise = torch.randn_like(bits)
        x_t = self.q_sample(bits, t, noise)

        # Scheduled sampling: sometimes use noisy coarse labels
        ss_ratio = self._get_scheduled_sampling_ratio(epoch)
        cond_coarse = self._get_noisy_coarse_labels(coarse_labels) \
            if (ss_ratio > 0 and random() < ss_ratio) else coarse_labels

        # Combine biaffine-enhanced features + coarse embedding
        cond_features = self._condition_features(enhanced_features, cond_coarse)

        # Self-conditioning (50% during training)
        x_self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.fine_dit(
                    x_t, t, cond_features, attention_mask, biaffine_scores
                ).detach()

        pred = self.fine_dit(x_t, t, cond_features, attention_mask, biaffine_scores, x_self_cond)

        target = bits if self.objective == 'pred_x0' else noise
        mask = (fine_labels != -100).unsqueeze(-1).expand_as(pred)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if self.loss_type == 'l2':
            return F.mse_loss(pred[mask], target[mask])
        return F.l1_loss(pred[mask], target[mask])

    @torch.no_grad()
    def sample_stage2(
        self,
        enhanced_features: Tensor,
        biaffine_scores: Tensor,
        coarse_labels: Tensor,
        attention_mask: Tensor,
        apply_constraint: bool = True,
    ) -> Tensor:
        """Stage 2 inference: biaffine + coarse conditioning → fine preds."""
        B, N = enhanced_features.shape[:2]
        cond_features = self._condition_features(enhanced_features, coarse_labels)

        x = torch.randn(B, N, self.fine_bits, device=self.device)
        x_start = None

        times = torch.linspace(
            self.num_timesteps - 1, 0, self.sampling_timesteps,
            dtype=torch.long, device=self.device,
        )
        for i, time in enumerate(times):
            t = torch.full((B,), time, device=self.device, dtype=torch.long)

            pred = self.fine_dit(x, t, cond_features, attention_mask, biaffine_scores, x_start)
            x_start = pred if self.objective == 'pred_x0' else self.predict_start_from_noise(x, t, pred)
            x_start = x_start.clamp(-self.snr_scale, self.snr_scale)

            if i < len(times) - 1:
                time_next = times[i + 1]
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                c = (1 - alpha_next).sqrt()
                x = x_start * alpha_next.sqrt() + c * (x - x_start * alpha.sqrt()) / (1 - alpha).sqrt()
            else:
                x = x_start

        fine_preds = bits_to_decimal(x / self.snr_scale, self.fine_bits)
        fine_preds = fine_preds.clamp(0, self.num_fine_classes - 1)

        if apply_constraint and self.coarse_to_fine_map is not None:
            fine_preds = self._constrain_to_coarse(fine_preds, coarse_labels)
        return fine_preds

    def _constrain_to_coarse(self, fine_preds, coarse_labels):
        if self.coarse_to_fine_map is None:
            return fine_preds
        B, N = fine_preds.shape
        for b in range(B):
            for s in range(N):
                coarse = coarse_labels[b, s].item()
                fine = fine_preds[b, s].item()
                if coarse in self.coarse_to_fine_map:
                    valid = self.coarse_to_fine_map[coarse]
                    if fine not in valid and len(valid) > 0:
                        fine_preds[b, s] = valid[0]
        return fine_preds

    # -------------------------------------------------------------------------
    # Combined Forward / Inference (same interface as HierarchicalDiffusionNeCTI)
    # -------------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        coarse_labels: Optional[Tensor] = None,
        fine_labels: Optional[Tensor] = None,
        stage: str = 'both',
        epoch: int = 0,
    ):
        """Combined forward — same interface as HierarchicalDiffusionNeCTI."""
        features = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # Biaffine conditioning (computed once, reused)
        enhanced_features, biaffine_scores = self.biaffine_conditioner(
            features, attention_mask
        )

        if stage == 'stage1':
            assert coarse_labels is not None
            return self.forward_stage1(features, coarse_labels, attention_mask)

        elif stage == 'stage2':
            assert coarse_labels is not None and fine_labels is not None
            return self.forward_stage2(
                enhanced_features, biaffine_scores,
                coarse_labels, fine_labels, attention_mask, epoch=epoch
            )

        elif stage in ('both', 'joint'):
            assert coarse_labels is not None and fine_labels is not None
            loss1 = self.forward_stage1(features, coarse_labels, attention_mask)
            loss2 = self.forward_stage2(
                enhanced_features, biaffine_scores,
                coarse_labels, fine_labels, attention_mask, epoch=epoch
            )
            return {
                'loss': loss1 + loss2,
                'stage1_loss': loss1,
                'stage2_loss': loss2,
            }

        raise ValueError(f"Unknown stage: {stage}")

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        apply_constraint: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Full inference: Stage 1 → biaffine + coarse → Stage 2 → fine."""
        features = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # Biaffine conditioning
        enhanced_features, biaffine_scores = self.biaffine_conditioner(
            features, attention_mask
        )

        # Stage 1: coarse prediction (uses raw BERT features)
        coarse_preds = self.sample_stage1(features, attention_mask)

        # Stage 2: fine prediction (uses biaffine-enhanced features + coarse preds)
        fine_preds = self.sample_stage2(
            enhanced_features, biaffine_scores,
            coarse_preds, attention_mask,
            apply_constraint=apply_constraint,
        )
        return coarse_preds, fine_preds

    def get_coarse_from_fine(self, fine_preds: Tensor) -> Tensor:
        if self.fine_to_coarse is None:
            return torch.zeros_like(fine_preds)
        return self.fine_to_coarse[fine_preds.clamp(0, len(self.fine_to_coarse) - 1)]


# =============================================================================
# Factory Function
# =============================================================================

def create_hierarchical_biaffine_model(
    device: str = 'cuda',
    backbone: str = 'FacebookAI/xlm-roberta-large',
    fine_to_coarse_map: Optional[Dict[int, int]] = None,
    **kwargs,
) -> HierarchicalBiaffineDiffusion:
    return HierarchicalBiaffineDiffusion(
        device=torch.device(device),
        backbone=backbone,
        fine_to_coarse_map=fine_to_coarse_map,
        **kwargs,
    )
