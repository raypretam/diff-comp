"""
Hierarchical Diffusion with Local Refinement for NeCTI
======================================================

Two-stage diffusion where:
- Stage 1: Global attention for coarse labels / structure
- Stage 2: Local attention window for fine-grained refinement

This addresses the theoretical limitation (Hahn 2020) that global attention
loses local sensitivity as O(1/n), which is needed for fine-grained case distinctions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List
from collections import namedtuple
from functools import partial
from transformers import AutoModel

# Assuming these exist in your codebase
from models.utils import decimal_to_bits, bits_to_decimal


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    """Extract the appropriate t index for a batch of indices"""
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


# =============================================================================
# Local Attention Module
# =============================================================================

class LocalWindowAttention(nn.Module):
    """
    Attention restricted to a local window around each token.
    This preserves local sensitivity that global attention loses.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, window_size: int = 3, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size  # Total window (e.g., 3 = 1 left + self + 1 right)
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            mask: [batch, seq_len] - 1 for valid, 0 for padding
        
        Returns:
            [batch, seq_len, hidden_size]
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Create local attention mask
        # Each position can only attend to positions within window_size
        local_mask = self._create_local_mask(N, self.window_size, x.device)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        
        # Apply local window mask (positions outside window get large negative)
        # Use -1e4 instead of -inf to avoid NaN from softmax when all positions are masked
        # (This matches the working global attention in DiT)
        attn = attn.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), -1e4)
        
        # Apply padding mask if provided
        if mask is not None:
            padding_mask = mask[:, None, None, :]  # [B, 1, 1, N]
            attn = attn.masked_fill(padding_mask == 0, -1e4)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out
    
    def _create_local_mask(self, seq_len: int, window_size: int, device: torch.device) -> Tensor:
        """Create a mask where each position can only attend to nearby positions"""
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        # Distance matrix
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # Mask: True if within window
        half_window = window_size // 2
        mask = distance <= half_window
        return mask


# =============================================================================
# Local DiT Block (for Stage 2)
# =============================================================================

class LocalDiTBlock(nn.Module):
    """
    DiT block with LOCAL attention instead of global.
    Used in Stage 2 for fine-grained refinement.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, window_size: int = 3, mlp_ratio: float = 4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.local_attn = LocalWindowAttention(hidden_size, num_heads, window_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        
        # AdaLN modulation (conditioned on timestep)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
    
    def forward(self, x: Tensor, c: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size] - noisy input
            c: [batch, seq_len, hidden_size] - conditioning (time + features)
            mask: [batch, seq_len] - attention mask
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Local attention with AdaLN
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        h = self.local_attn(h, mask)
        x = x + gate_msa * h
        
        # MLP with AdaLN
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h
        
        return x


# =============================================================================
# Local Refinement DiT (Stage 2 Model)
# =============================================================================

class LocalRefinementDiT(nn.Module):
    """
    Stage 2: Local attention DiT for fine-grained refinement.
    
    Key difference from global DiT:
    - Uses LocalWindowAttention instead of full attention
    - Takes coarse prediction as additional conditioning
    - Operates on compound spans with limited context
    """
    
    def __init__(
        self,
        in_channels: int = 7,  # bits for fine-grained labels
        hidden_size: int = 512,
        depth: int = 4,  # Fewer layers than global model
        num_heads: int = 8,
        window_size: int = 5,  # Local attention window
        num_coarse_classes: int = 4,
        mlp_ratio: float = 4.0,
        num_timesteps: int = 1000
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.window_size = window_size
        
        # Input projection
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        
        # Coarse label embedding (conditioning)
        # +1 for padding (-100 will be mapped to num_coarse_classes index)
        self.coarse_embed = nn.Embedding(num_coarse_classes + 1, hidden_size)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Embedding(num_timesteps, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Feature projection (from BERT)
        self.feature_proj = nn.Linear(hidden_size, hidden_size)
        
        # Local attention blocks
        self.blocks = nn.ModuleList([
            LocalDiTBlock(hidden_size, num_heads, window_size, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Output projection
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, in_channels)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        bert_features: Tensor,
        coarse_labels: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: [batch, seq_len, bits] - noisy fine-grained bits
            t: [batch] - timesteps
            bert_features: [batch, seq_len, hidden] - encoder features
            coarse_labels: [batch, seq_len] - predicted coarse labels (from Stage 1)
            attention_mask: [batch, seq_len]
        
        Returns:
            [batch, seq_len, bits] - predicted clean bits
        """
        # Embed inputs
        x = self.x_embedder(x)
        
        # Time conditioning
        t_emb = self.time_embed[0](t)  # [batch, hidden]
        t_emb = self.time_embed[1:](t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [batch, seq_len, hidden]
        
        # Coarse label conditioning
        # Map -100 (padding) to the last embedding index to avoid negative indexing.
        # During inference coarse_labels are sampled predictions (bits_to_decimal),
        # which can decode to values outside [0, num_coarse_classes-1]; clamp so the
        # embedding lookup never goes out of bounds (avoids CUDA device-side assert).
        coarse_labels_clamped = coarse_labels.clone()
        coarse_labels_clamped[coarse_labels == -100] = self.coarse_embed.num_embeddings - 1
        coarse_labels_clamped = coarse_labels_clamped.clamp_(0, self.coarse_embed.num_embeddings - 1)
        coarse_emb = self.coarse_embed(coarse_labels_clamped)  # [batch, seq_len, hidden]
        
        # Feature conditioning
        feat_emb = self.feature_proj(bert_features)
        
        # Combined conditioning
        c = t_emb + coarse_emb + feat_emb
        
        # Apply local attention blocks
        for block in self.blocks:
            x = block(x, c, attention_mask)
        
        # Final projection
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.final_linear(x)
        
        return x


# =============================================================================
# Hierarchical Diffusion Model (Main Class)
# =============================================================================

class HierarchicalDiffusionNeCTI(nn.Module):
    """
    Hierarchical Diffusion for NeCTI with Local Refinement.
    
    Stage 1: Global attention diffusion for coarse labels
    Stage 2: Local attention diffusion for fine-grained refinement
    
    This addresses Hahn (2020)'s theoretical result that self-attention
    loses local sensitivity as O(1/n), which is needed for fine-grained
    case distinctions in Sanskrit compounds.
    """
    
    def __init__(
        self,
        device: torch.device,
        # Backbone
        backbone: str = 'google/muril-base-cased',
        dim_model: int = 768,
        freeze_bert: bool = False,
        # Diffusion
        time_steps: int = 1000,
        sampling_steps: int = 50,
        noise_schedule: str = 'cosine',
        snr_scale: float = 1.0,
        # Stage 1 (Global)
        global_depth: int = 6,
        global_num_heads: int = 12,  # Add parameter
        num_coarse_classes: int = 4,
        # Stage 2 (Local)
        local_depth: int = 4,
        local_num_heads: int = 8,  # Add parameter
        local_window_size: int = 5,
        num_fine_classes: int = 86,
        # Label mapping
        fine_to_coarse_map: Optional[Dict[int, int]] = None,
        # Training
        objective: str = 'pred_x0',
        loss_type: str = 'l2'
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
        
        # Backbone encoder
        self.backbone = AutoModel.from_pretrained(backbone)
        if freeze_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Build diffusion schedule
        self._build_diffusion_schedule(time_steps, sampling_steps, noise_schedule)
        
        # Stage 1: Global DiT for coarse labels
        # (You can reuse your existing DiT here)
        from models.dit_discrete import DiT as GlobalDiT
        self.global_dit = GlobalDiT(
            in_channels=self.coarse_bits,
            hidden_size=dim_model,
            depth=global_depth,
            num_heads=global_num_heads,  # Use parameter instead of hardcoded
            num_steps=time_steps,
            time_dim=256
        )
        
        # Stage 2: Local DiT for fine-grained refinement
        self.local_dit = LocalRefinementDiT(
            in_channels=self.fine_bits,
            hidden_size=dim_model,
            depth=local_depth,
            num_heads=local_num_heads,  # Use parameter instead of hardcoded
            window_size=local_window_size,
            num_coarse_classes=num_coarse_classes,
            num_timesteps=time_steps
        )
        
        self.to(device)
    
    def _compute_bits(self, num_classes: int) -> int:
        """Compute number of bits needed to represent num_classes"""
        return max(1, math.ceil(math.log2(num_classes + 1)))
    
    def _build_diffusion_schedule(self, time_steps: int, sampling_steps: int, noise_schedule: str):
        """Build diffusion noise schedule"""
        self.num_timesteps = time_steps
        self.sampling_timesteps = sampling_steps
        
        if noise_schedule == 'cosine':
            betas = cosine_beta_schedule(time_steps)
        else:
            betas = linear_beta_schedule(time_steps)
        
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
    
    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Forward diffusion: add noise to x_start"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Predict x_0 from x_t and predicted noise"""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # -------------------------------------------------------------------------
    # Stage 1: Global Coarse Prediction
    # -------------------------------------------------------------------------
    
    def forward_stage1(
        self,
        features: Tensor,
        coarse_labels: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        Stage 1 training: Predict coarse labels with global attention.
        
        Args:
            features: [batch, seq_len, hidden] - BERT features
            coarse_labels: [batch, seq_len] - ground truth coarse labels
            attention_mask: [batch, seq_len]
        
        Returns:
            loss: scalar
        """
        # Replace -100 (padding) with 0 before bit conversion (we mask these in loss anyway)
        coarse_labels_clean = coarse_labels.clone()
        coarse_labels_clean[coarse_labels_clean == -100] = 0
        
        # Convert labels to bits
        bits = decimal_to_bits(coarse_labels_clean, self.coarse_bits) * self.snr_scale
        
        # Sample timesteps
        batch_size = features.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(bits)
        x_t = self.q_sample(bits, t, noise)
        
        # Predict
        pred = self.global_dit(x_t, t, features, attention_mask)
        
        # Compute loss
        target = bits if self.objective == 'pred_x0' else noise
        mask = (coarse_labels != -100).unsqueeze(-1).expand_as(pred)
        
        # Check if mask has any valid tokens
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.loss_type == 'l2':
            loss = F.mse_loss(pred[mask], target[mask])
        else:
            loss = F.l1_loss(pred[mask], target[mask])
        
        return loss
    
    @torch.no_grad()
    def sample_stage1(
        self,
        features: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        Stage 1 inference: Sample coarse labels.
        
        Returns:
            coarse_preds: [batch, seq_len] - predicted coarse labels
        """
        batch_size, seq_len = features.shape[:2]
        shape = (batch_size, seq_len, self.coarse_bits)
        
        # Start from noise
        x = torch.randn(shape, device=self.device)
        
        # DDIM sampling
        times = torch.linspace(self.num_timesteps - 1, 0, self.sampling_timesteps, dtype=torch.long, device=self.device)
        
        for i, time in enumerate(times):
            t = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            
            pred = self.global_dit(x, t, features, attention_mask)
            
            if self.objective == 'pred_x0':
                x_start = pred
            else:
                x_start = self.predict_start_from_noise(x, t, pred)
            
            x_start = torch.clamp(x_start, -self.snr_scale, self.snr_scale)
            
            if i < len(times) - 1:
                time_next = times[i + 1]
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                
                sigma = 0.0  # Deterministic DDIM
                c = (1 - alpha_next - sigma ** 2).sqrt()
                
                noise = torch.randn_like(x) if time_next > 0 else 0
                x = x_start * alpha_next.sqrt() + c * (x - x_start * alpha.sqrt()) / (1 - alpha).sqrt()
            else:
                x = x_start
        
        # Convert bits to labels
        coarse_preds = bits_to_decimal(x / self.snr_scale, self.coarse_bits)
        return coarse_preds
    
    # -------------------------------------------------------------------------
    # Stage 2: Local Fine-grained Refinement
    # -------------------------------------------------------------------------
    
    def forward_stage2(
        self,
        features: Tensor,
        coarse_labels: Tensor,
        fine_labels: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        Stage 2 training: Predict fine-grained labels with LOCAL attention.
        
        Key: Uses coarse_labels as conditioning and local attention window.
        
        Args:
            features: [batch, seq_len, hidden]
            coarse_labels: [batch, seq_len] - coarse labels (can be GT or predicted)
            fine_labels: [batch, seq_len] - ground truth fine-grained labels
            attention_mask: [batch, seq_len]
        
        Returns:
            loss: scalar
        """
        # Replace -100 (padding) with 0 before bit conversion (we mask these in loss anyway)
        fine_labels_clean = fine_labels.clone()
        fine_labels_clean[fine_labels_clean == -100] = 0
        
        # Convert fine labels to bits
        bits = decimal_to_bits(fine_labels_clean, self.fine_bits) * self.snr_scale
        
        # Sample timesteps
        batch_size = features.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(bits)
        x_t = self.q_sample(bits, t, noise)
        
        # Predict with local attention, conditioned on coarse labels
        pred = self.local_dit(x_t, t, features, coarse_labels, attention_mask)
        
        # Compute loss
        target = bits if self.objective == 'pred_x0' else noise
        mask = (fine_labels != -100).unsqueeze(-1).expand_as(pred)
        
        # Check if mask has any valid tokens
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.loss_type == 'l2':
            loss = F.mse_loss(pred[mask], target[mask])
        else:
            loss = F.l1_loss(pred[mask], target[mask])
        
        return loss
    
    @torch.no_grad()
    def sample_stage2(
        self,
        features: Tensor,
        coarse_labels: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        Stage 2 inference: Sample fine-grained labels conditioned on coarse.
        
        Args:
            features: [batch, seq_len, hidden]
            coarse_labels: [batch, seq_len] - predicted coarse labels from Stage 1
            attention_mask: [batch, seq_len]
        
        Returns:
            fine_preds: [batch, seq_len] - predicted fine-grained labels
        """
        batch_size, seq_len = features.shape[:2]
        shape = (batch_size, seq_len, self.fine_bits)
        
        # Start from noise
        x = torch.randn(shape, device=self.device)
        
        # DDIM sampling with local attention
        times = torch.linspace(self.num_timesteps - 1, 0, self.sampling_timesteps, dtype=torch.long, device=self.device)
        
        for i, time in enumerate(times):
            t = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            
            # Local attention prediction conditioned on coarse
            pred = self.local_dit(x, t, features, coarse_labels, attention_mask)
            
            if self.objective == 'pred_x0':
                x_start = pred
            else:
                x_start = self.predict_start_from_noise(x, t, pred)
            
            x_start = torch.clamp(x_start, -self.snr_scale, self.snr_scale)
            
            if i < len(times) - 1:
                time_next = times[i + 1]
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                
                c = (1 - alpha_next).sqrt()
                x = x_start * alpha_next.sqrt() + c * (x - x_start * alpha.sqrt()) / (1 - alpha).sqrt()
            else:
                x = x_start
        
        # Convert bits to labels
        fine_preds = bits_to_decimal(x / self.snr_scale, self.fine_bits)
        
        # Optional: Constrain fine predictions to valid labels within coarse category
        if self.fine_to_coarse is not None:
            fine_preds = self._constrain_to_coarse(fine_preds, coarse_labels)
        
        return fine_preds
    
    def _constrain_to_coarse(self, fine_preds: Tensor, coarse_labels: Tensor) -> Tensor:
        """
        Constrain fine-grained predictions to be valid within predicted coarse category.
        """
        # Get valid fine labels for each coarse category
        # This ensures consistency between stages
        batch_size, seq_len = fine_preds.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                coarse = coarse_labels[b, s].item()
                fine = fine_preds[b, s].item()
                
                # Check if fine label belongs to predicted coarse category
                if fine < len(self.fine_to_coarse):
                    predicted_coarse = self.fine_to_coarse[fine].item()
                    if predicted_coarse != coarse:
                        # Find nearest valid fine label within coarse category
                        valid_fine = (self.fine_to_coarse == coarse).nonzero(as_tuple=True)[0]
                        if len(valid_fine) > 0:
                            # Pick the first valid one (could be improved with distance metric)
                            fine_preds[b, s] = valid_fine[0]
        
        return fine_preds
    
    # -------------------------------------------------------------------------
    # Combined Forward / Inference
    # -------------------------------------------------------------------------
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        coarse_labels: Optional[Tensor] = None,
        fine_labels: Optional[Tensor] = None,
        stage: str = 'both'  # 'stage1', 'stage2', or 'both'
    ):
        """
        Combined forward pass for training.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            coarse_labels: [batch, seq_len] - for stage1 or stage2 conditioning
            fine_labels: [batch, seq_len] - for stage2
            stage: which stage(s) to train
        
        Returns:
            loss or dict of losses
        """
        # Extract features
        features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        if stage == 'stage1':
            assert coarse_labels is not None
            return self.forward_stage1(features, coarse_labels, attention_mask)
        
        elif stage == 'stage2':
            assert coarse_labels is not None and fine_labels is not None
            return self.forward_stage2(features, coarse_labels, fine_labels, attention_mask)
        
        elif stage == 'both':
            assert coarse_labels is not None and fine_labels is not None
            
            loss1 = self.forward_stage1(features, coarse_labels, attention_mask)
            loss2 = self.forward_stage2(features, coarse_labels, fine_labels, attention_mask)
            
            return {
                'loss': loss1 + loss2,
                'stage1_loss': loss1,
                'stage2_loss': loss2
            }
        
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Full inference: Stage 1 → Stage 2
        
        Returns:
            coarse_preds: [batch, seq_len]
            fine_preds: [batch, seq_len]
        """
        # Extract features
        features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Stage 1: Get coarse predictions
        coarse_preds = self.sample_stage1(features, attention_mask)
        
        # Stage 2: Refine to fine-grained using local attention
        fine_preds = self.sample_stage2(features, coarse_preds, attention_mask)
        
        return coarse_preds, fine_preds


# =============================================================================
# Config for easy instantiation
# =============================================================================

def create_hierarchical_model(
    device: str = 'cuda',
    backbone: str = 'FacebookAI/xlm-roberta-large',
    fine_to_coarse_map: Optional[Dict[int, int]] = None,
    **kwargs
) -> HierarchicalDiffusionNeCTI:
    """
    Factory function to create the hierarchical model.
    
    Example:
        # Define your fine-to-coarse mapping
        fine_to_coarse = {
            0: 0,  # Fine label 0 -> Coarse 0 (Tatpurusa)
            1: 0,  # Fine label 1 -> Coarse 0
            ...
            30: 1,  # Fine label 30 -> Coarse 1 (Bahuvrihi)
            ...
        }
        
        model = create_hierarchical_model(
            device='cuda',
            fine_to_coarse_map=fine_to_coarse,
            num_coarse_classes=4,
            num_fine_classes=86
        )
    """
    return HierarchicalDiffusionNeCTI(
        device=torch.device(device),
        backbone=backbone,
        fine_to_coarse_map=fine_to_coarse_map,
        **kwargs
    )


if __name__ == '__main__':
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = create_hierarchical_model(
        device=device,
        num_coarse_classes=7,
        num_fine_classes=56,
        local_window_size=5
    )
    
    # Dummy inputs
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    coarse_labels = torch.randint(0, 7, (batch_size, seq_len)).to(device)
    fine_labels = torch.randint(0, 56, (batch_size, seq_len)).to(device)
    
    # Test training
    losses = model(input_ids, attention_mask, coarse_labels, fine_labels, stage='both')
    print(f"Stage 1 loss: {losses['stage1_loss'].item():.4f}")
    print(f"Stage 2 loss: {losses['stage2_loss'].item():.4f}")
    
    # Test inference
    model.eval()
    coarse_preds, fine_preds = model.inference(input_ids, attention_mask)
    print(f"Coarse predictions shape: {coarse_preds.shape}")
    print(f"Fine predictions shape: {fine_preds.shape}")