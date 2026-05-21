"""
Hyperbolic Hawkes Encoder
=========================

Encoder-side module inspired by hHTM (Rao et al., LREC-COLING 2024
"Hierarchical Topic Modeling via Contrastive Learning and Hyperbolic Embedding",
https://github.com/YRaoGroup/hHTM).

The original hHTM uses a hyperbolic Hawkes attention (HypHawkes) that:
    1. Maps Euclidean inputs onto the Poincaré ball via expmap_0
    2. Applies a linear "general" attention transform
    3. Computes attention scores and softmax in tangent space
    4. Maps the result back into hyperbolic space

We adapt this idea for token-level sequence encoders: stack of self-attention
blocks operating in the tangent space of the Poincaré ball (with curvature c
learnable), producing enriched contextual features in the same dim as the
BERT backbone. We implement the Poincaré-ball exp/log maps directly to avoid
a `geoopt` dependency.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Poincaré ball helpers (no external dependency)
# -----------------------------------------------------------------------------

_EPS = 1e-7
_MAX_NORM = 1.0 - 1e-5  # to keep points strictly inside the ball


def _safe_clamp_norm(x: Tensor, c: Tensor) -> Tensor:
    """Project x to lie strictly inside the Poincaré ball of curvature c."""
    sqrt_c = c.clamp(min=_EPS).sqrt()
    max_radius = (_MAX_NORM / sqrt_c).expand_as(torch.norm(x, dim=-1, keepdim=True))
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    cond = norm > max_radius
    projected = x / norm * max_radius
    return torch.where(cond, projected, x)


def expmap0(v: Tensor, c: Tensor) -> Tensor:
    """Exponential map at origin onto the Poincaré ball of curvature c."""
    sqrt_c = c.clamp(min=_EPS).sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    coef = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    return _safe_clamp_norm(coef * v, c)


def logmap0(y: Tensor, c: Tensor) -> Tensor:
    """Logarithmic map at origin from the Poincaré ball back to tangent space."""
    sqrt_c = c.clamp(min=_EPS).sqrt()
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=_EPS, max=_MAX_NORM)
    coef = torch.atanh((sqrt_c * y_norm).clamp(max=1.0 - _EPS)) / (sqrt_c * y_norm)
    return coef * y


# -----------------------------------------------------------------------------
# Hyperbolic Hawkes Attention Block
# -----------------------------------------------------------------------------

class HypHawkesBlock(nn.Module):
    """
    One block of hyperbolic Hawkes-style self-attention.

    Inputs/outputs live in Euclidean space (BERT features). Internally we
    map to/from the Poincaré ball so attention is computed in the tangent
    space of a learnable-curvature hyperbolic space.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        init_curvature: float = 1.0,
        learnable_curvature: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.c = nn.Parameter(
            torch.tensor([init_curvature], dtype=torch.float32),
            requires_grad=learnable_curvature,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def _attn(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        B, N, D = x.shape

        # Map into Poincaré ball, then back to tangent for linear ops
        c = self.c.to(x.device)
        x_h = expmap0(x, c)
        x_tan = logmap0(x_h, c)

        qkv = self.qkv(x_tan).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, N, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        if mask is not None:
            attn_mask = mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        # Map result back to hyperbolic space, then re-project to Euclidean
        out_h = expmap0(out, c)
        return logmap0(out_h, c)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self._attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Stack
# -----------------------------------------------------------------------------

class HypHawkesEncoder(nn.Module):
    """
    Stack of hyperbolic Hawkes attention blocks, applied on top of BERT
    features. Same input/output shape: [B, N, dim].
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        init_curvature: float = 1.0,
        learnable_curvature: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            HypHawkesBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                init_curvature=init_curvature,
                learnable_curvature=learnable_curvature,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, features: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        x = features
        for blk in self.blocks:
            x = blk(x, attention_mask)
        return self.norm(x)
