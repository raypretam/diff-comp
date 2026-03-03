"""
Lightweight Biaffine Conditioner for feature enhancement.

This module enhances BERT features with pairwise biaffine context,
designed to be inserted BETWEEN the backbone and the DiT without
changing anything else about the diffusion pipeline.

Key design choices:
    - Low-rank biaffine factorization: O(n * rank) instead of O(n^2)
    - Gated addition with near-zero init: model starts ≈ identical to baseline
    - No architectural changes to DiT — only feature quality changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiaffineConditioner(nn.Module):
    """
    Enhances sequence features with pairwise biaffine context.
    
    Given features h_i, computes low-rank biaffine scores:
        score(i,j) = h_i^T W_1^T W_2 h_j + bias
    Then uses attention over scores to create pairwise context:
        ctx_i = sum_j softmax(score(i,:))_j * V(h_j)
    Finally gates the enhancement:
        enhanced_i = h_i + gate * ctx_i
    
    The gate is initialized to near-zero (sigmoid(gate_bias) ≈ 0.05)
    so the model starts nearly identical to baseline.
    
    Args:
        hidden_size: Feature dimension (must match BERT output / DiT input)
        rank: Low-rank factorization dimension for biaffine scores
        num_heads: Number of attention heads for pairwise context
        dropout: Dropout rate
        gate_init_bias: Initial bias for the gate sigmoid.
                        -3.0 → sigmoid ≈ 0.05 (near-zero enhancement initially)
                        0.0  → sigmoid = 0.5 (equal mix)
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        rank: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init_bias: float = -3.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        
        # Low-rank biaffine scoring: score(i,j) = (W1 h_i)^T (W2 h_j)
        # Factorized as two projections to rank-dim
        self.W1 = nn.Linear(hidden_size, rank, bias=False)
        self.W2 = nn.Linear(hidden_size, rank, bias=False)
        
        # Value projection for pairwise context aggregation
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Learnable gate — initialized to produce near-zero output
        self.gate_bias = nn.Parameter(torch.tensor(gate_init_bias))
        
        # Layer norm for the context (stabilizes the gated addition)
        self.ctx_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights so initial contribution is minimal
        self._init_weights()
    
    def _init_weights(self):
        """Small initialization to minimize initial perturbation."""
        for module in [self.W1, self.W2, self.V, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, features, attention_mask):
        """
        Enhance features with pairwise biaffine context.
        
        Args:
            features: [B, N, H] — BERT output features
            attention_mask: [B, N] — 1 for valid tokens, 0 for padding
            
        Returns:
            enhanced: [B, N, H] — features + gated biaffine context
        """
        B, N, H = features.shape
        
        # Compute low-rank biaffine scores: [B, N, N]
        q = self.W1(features)  # [B, N, rank]
        k = self.W2(features)  # [B, N, rank]
        scores = torch.bmm(q, k.transpose(1, 2))  # [B, N, N]
        scores = scores / (self.rank ** 0.5)  # Scale
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)  # [B, 1, N]
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # Softmax → attention weights
        attn = F.softmax(scores, dim=-1)  # [B, N, N]
        attn = self.dropout(attn)
        
        # Compute pairwise context via attention over values
        v = self.V(features)  # [B, N, H]
        ctx = torch.bmm(attn, v)  # [B, N, H]
        ctx = self.out_proj(ctx)   # [B, N, H]
        ctx = self.ctx_norm(ctx)
        
        # Gated addition: gate starts near 0
        gate = torch.sigmoid(self.gate_bias)  # scalar
        enhanced = features + gate * ctx
        
        return enhanced
    
    def get_gate_value(self):
        """Return current gate value for logging."""
        return torch.sigmoid(self.gate_bias).item()
