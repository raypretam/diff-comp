"""
Hyperbolic Hierarchical Contrastive Loss
========================================

An auxiliary loss that organizes token representations on the Poincaré ball
according to a *known* label hierarchy (fine entity type -> coarse category).

Unlike the `HypHawkesEncoder` (which only borrowed hyperbolic attention and,
on its own, carries no hierarchy), this module makes the geometry genuinely
active: every term below is computed with the Poincaré distance, and the
label hierarchy enters explicitly through

  1. graded contrastive negatives
       - positive            : same fine label
       - sibling (soft neg)  : same coarse parent, different fine label
       - cousin (hard neg)   : different coarse parent
     siblings are repelled less than cousins (weight w_sibling < w_cross),
     so same-coarse types stay geometrically close;

  2. coarse "parent" anchors
     one learnable point per coarse category, initialised near the origin
     (origin = most general). Each token is pulled toward its parent anchor;

  3. norm-ordering / entailment
     a parent anchor must have a *smaller* ball-norm than its children
     (norm on the Poincaré ball ~ depth in the hierarchy);

  4. anchor separation
     coarse anchors are pushed apart so categories occupy distinct sectors.

The total is returned as a single scalar to be added to the main loss.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.hyp_hawkes_encoder import expmap0, _safe_clamp_norm


_EPS = 1e-7


# -----------------------------------------------------------------------------
# Poincaré-ball distance (curvature c), via the numerically stable acosh form:
#   d_c(x, y) = (1/√c) · arccosh(1 + 2c·‖x−y‖² / ((1−c‖x‖²)(1−c‖y‖²)))
# This avoids building an [M, N, d] Möbius-addition tensor.
# -----------------------------------------------------------------------------

def _dist_from_terms(sq_dist: Tensor, x2: Tensor, y2: Tensor, c: Tensor) -> Tensor:
    denom = ((1.0 - c * x2) * (1.0 - c * y2)).clamp(min=_EPS)
    arg = (1.0 + 2.0 * c * sq_dist / denom).clamp(min=1.0 + _EPS)
    return torch.acosh(arg) / c.clamp(min=_EPS).sqrt()


def poincare_pairwise(x: Tensor, y: Tensor, c: Tensor) -> Tensor:
    """Pairwise Poincaré distance. x:[M,d], y:[N,d] -> [M,N]."""
    x2 = (x * x).sum(-1)                       # [M]
    y2 = (y * y).sum(-1)                       # [N]
    sq = (x2[:, None] + y2[None, :] - 2.0 * (x @ y.t())).clamp(min=0.0)
    return _dist_from_terms(sq, x2[:, None], y2[None, :], c)


def poincare_rowwise(x: Tensor, y: Tensor, c: Tensor) -> Tensor:
    """Element-wise Poincaré distance. x,y:[M,d] -> [M]."""
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    sq = ((x - y) ** 2).sum(-1).clamp(min=0.0)
    return _dist_from_terms(sq, x2, y2, c)


def clip_tangent(v: Tensor, max_norm: float) -> Tensor:
    """Scale down tangent vectors whose norm exceeds max_norm.

    Without this, large projection outputs map (via expmap0) to points right
    at the ball boundary, where Poincaré distances diverge -> exploding loss
    and eventual NaNs. Clipping keeps every embedding at a controlled radius.
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    factor = (max_norm / norm).clamp(max=1.0)
    return v * factor


# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------

class HyperbolicHierarchicalContrastive(nn.Module):
    """
    Hierarchy-aware hyperbolic contrastive auxiliary loss.

    Call:  loss = module(features, seq_labels)
        features   : [B, N, in_dim]  encoder/BERT token features
        seq_labels : [B, N]          fine label ids (-100 = ignore)
    Returns a scalar tensor (0.0 if a batch has too few usable tokens).
    """

    def __init__(
        self,
        in_dim: int,
        fine_to_coarse: List[int],
        num_coarse: int,
        o_label_id: int = -1,
        proj_dim: int = 128,
        curvature: float = 1.0,
        temperature: float = 0.5,
        tangent_clip: float = 1.0,
        w_sibling: float = 1.0,
        w_cross: float = 2.0,
        lambda_supcon: float = 1.0,
        lambda_parent: float = 0.5,
        lambda_norm: float = 0.3,
        lambda_sep: float = 0.3,
        norm_margin: float = 0.1,
        sep_margin: float = 1.0,
        max_tokens: int = 512,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.tangent_clip = tangent_clip
        self.w_sibling = w_sibling
        self.w_cross = w_cross
        self.lambda_supcon = lambda_supcon
        self.lambda_parent = lambda_parent
        self.lambda_norm = lambda_norm
        self.lambda_sep = lambda_sep
        self.norm_margin = norm_margin
        self.sep_margin = sep_margin
        self.max_tokens = max_tokens
        self.ignore_index = ignore_index
        self.o_label_id = o_label_id

        # fixed curvature (learnable curvature tends to destabilize this loss)
        self.register_buffer('c', torch.tensor([float(curvature)]))
        # fine id -> coarse id lookup
        self.register_buffer('fine_to_coarse',
                             torch.as_tensor(fine_to_coarse, dtype=torch.long))

        # projection head: encoder dim -> contrastive dim
        self.proj = nn.Linear(in_dim, proj_dim)
        # one learnable coarse "parent" anchor (tangent space, near origin)
        self.parent_anchors = nn.Parameter(torch.randn(num_coarse, proj_dim) * 0.01)

    # ------------------------------------------------------------------
    def _select_tokens(self, feats: Tensor, labels: Tensor):
        """Flatten valid tokens and subsample, prioritising entity tokens."""
        valid = labels != self.ignore_index
        feats = feats[valid]
        labels = labels[valid]
        if feats.shape[0] <= self.max_tokens:
            return feats, labels

        if self.o_label_id >= 0:
            ent_idx = (labels != self.o_label_id).nonzero(as_tuple=True)[0]
            o_idx = (labels == self.o_label_id).nonzero(as_tuple=True)[0]
            if ent_idx.numel() >= self.max_tokens:
                keep = ent_idx[torch.randperm(ent_idx.numel(), device=feats.device)[:self.max_tokens]]
            else:
                n_o = self.max_tokens - ent_idx.numel()
                o_keep = o_idx[torch.randperm(o_idx.numel(), device=feats.device)[:n_o]]
                keep = torch.cat([ent_idx, o_keep])
        else:
            keep = torch.randperm(feats.shape[0], device=feats.device)[:self.max_tokens]
        return feats[keep], labels[keep]

    # ------------------------------------------------------------------
    def forward(self, features: Tensor, seq_labels: Tensor) -> Tensor:
        device = features.device
        zero = torch.zeros((), device=device)

        feats, labels = self._select_tokens(
            features.reshape(-1, features.shape[-1]), seq_labels.reshape(-1))
        m = feats.shape[0]
        if m < 4:
            return zero

        c = self.c.to(device)
        # project tokens onto the Poincaré ball (tangent vectors clipped so
        # embeddings stay at a controlled radius, away from the boundary)
        z_tan = clip_tangent(self.proj(feats), self.tangent_clip)
        a_tan = clip_tangent(self.parent_anchors, self.tangent_clip)
        z = _safe_clamp_norm(expmap0(z_tan, c), c)          # [M, d]
        anchors = _safe_clamp_norm(expmap0(a_tan, c), c)    # [C, d]

        coarse = self.fine_to_coarse.to(device)[labels]                # [M]

        # ---- relation masks ----
        same_fine = labels[:, None] == labels[None, :]
        same_coarse = coarse[:, None] == coarse[None, :]
        self_mask = torch.eye(m, dtype=torch.bool, device=device)
        pos_mask = same_fine & ~self_mask

        # ---- (1) graded hyperbolic supervised contrastive ----
        dist = poincare_pairwise(z, z, c)                              # [M, M]
        logits = -dist / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # negative weights: sibling (same coarse) < cousin (cross coarse)
        weight = torch.full((m, m), self.w_cross, device=device)
        weight[same_coarse] = self.w_sibling
        weight[same_fine] = 1.0                                        # positives

        exp_logits = torch.exp(logits) * weight * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + _EPS)

        pos_count = pos_mask.sum(dim=1)
        has_pos = pos_count > 0
        if has_pos.any():
            mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp(min=1)
            loss_supcon = -mean_log_prob_pos[has_pos].mean()
        else:
            loss_supcon = zero

        # ---- (2) parent pull: token -> its coarse anchor ----
        anchor_per_token = anchors[coarse]                             # [M, d]
        d_parent = poincare_rowwise(z, anchor_per_token, c)            # [M]
        loss_parent = d_parent.mean()

        # ---- (3) norm-ordering / entailment: parent norm < child norm ----
        norm_token = z.norm(dim=-1)                                    # [M]
        norm_anchor = anchor_per_token.norm(dim=-1)                    # [M]
        loss_norm = F.relu(norm_anchor - norm_token + self.norm_margin).mean()

        # ---- (4) separate coarse anchors ----
        d_anchors = poincare_pairwise(anchors, anchors, c)             # [C, C]
        cc = anchors.shape[0]
        off_diag = ~torch.eye(cc, dtype=torch.bool, device=device)
        if off_diag.any():
            loss_sep = F.relu(self.sep_margin - d_anchors[off_diag]).mean()
        else:
            loss_sep = zero

        return (self.lambda_supcon * loss_supcon
                + self.lambda_parent * loss_parent
                + self.lambda_norm * loss_norm
                + self.lambda_sep * loss_sep)
