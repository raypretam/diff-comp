"""
Hierarchical Diffusion for NeCTI — Global Coarse→Fine with Scheduled Sampling
==============================================================================

Two-stage hierarchical diffusion, both stages using global attention:

  Stage 1: Global DiT → coarse labels  (3 bits, 6 classes)
  Stage 2: Global DiT → fine labels    (6 bits, 56 classes)
             conditioned on coarse labels via SCHEDULED SAMPLING

Scheduled sampling strategy (fixes teacher forcing exposure bias):
  - Early training (epoch < warmup):  GT coarse labels (pure teacher forcing)
  - Later training (epoch >= warmup):  Increasingly use noisy coarse labels
  - Inference:  Stage 1 predicted coarse labels

This bridges the train-test gap: Stage 2 learns to be robust to
imperfect coarse conditioning, matching what it sees at inference time.
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

from random import random
from models.utils import decimal_to_bits, bits_to_decimal
from models.dit_discrete import DiT as GlobalDiT, sinusoidal_position_embedding


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
# Hierarchical Diffusion Model — Scheduled Sampling
# =============================================================================

class HierarchicalDiffusionNeCTI(nn.Module):
    """
    Hierarchical Diffusion for NeCTI — Global Coarse→Fine with Scheduled Sampling.
    
    Both stages use global attention (proven architecture from flat DiT).
    Stage 2 conditioned on coarse labels via scheduled sampling:
      - Early training:  ground-truth coarse labels (teacher forcing)
      - Later training:  mix of GT and noisy coarse labels (scheduled sampling)
      - Inference:       Stage 1 predicted coarse labels
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
        sampling_steps: int = 100,
        noise_schedule: str = 'cosine',
        snr_scale: float = 2.0,
        # Stage 1 (Global — coarse)
        global_depth: int = 6,
        global_num_heads: int = 16,
        num_coarse_classes: int = 6,
        # Stage 2 (Global — fine, coarse-conditioned)
        fine_depth: int = 12,
        fine_num_heads: int = 16,
        num_fine_classes: int = 56,
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
        
        # Build coarse-to-fine mapping for constrained decoding
        self._build_coarse_to_fine_map(fine_to_coarse_map)
        
        # Backbone encoder
        self.backbone = AutoModel.from_pretrained(backbone)
        if freeze_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Build diffusion schedule
        self._build_diffusion_schedule(time_steps, sampling_steps, noise_schedule)
        
        # Stage 1: Global DiT for coarse labels
        self.global_dit = GlobalDiT(
            in_channels=self.coarse_bits,
            hidden_size=dim_model,
            depth=global_depth,
            num_heads=global_num_heads,
            num_steps=time_steps,
            time_dim=256
        )
        
        # Stage 2: Global DiT for fine-grained labels (depth=12 matches flat baseline)
        self.fine_dit = GlobalDiT(
            in_channels=self.fine_bits,
            hidden_size=dim_model,
            depth=fine_depth,
            num_heads=fine_num_heads,
            num_steps=time_steps,
            time_dim=256
        )
        
        # Coarse conditioning: embed coarse labels into feature space
        self.coarse_embed = nn.Embedding(num_coarse_classes + 1, dim_model)   # +1 for padding
        nn.init.normal_(self.coarse_embed.weight, std=0.02)
        
        # Scheduled sampling config
        self.ss_max_ratio = 0.5    # max probability of using noisy coarse labels
        self.ss_warmup_epochs = 5  # epochs of pure teacher forcing before ramping
        
        self.to(device)
        
        print(f"\n{'='*60}")
        print("HIERARCHICAL DIFFUSION — SCHEDULED SAMPLING")
        print(f"{'='*60}")
        print(f"  Stage 1: Global DiT, depth={global_depth}, heads={global_num_heads}")
        print(f"  Stage 2: Global DiT, depth={fine_depth}, heads={fine_num_heads}")
        print(f"           + coarse conditioning via scheduled sampling")
        print(f"  Coarse classes: {num_coarse_classes} ({self.coarse_bits} bits)")
        print(f"  Fine classes: {num_fine_classes} ({self.fine_bits} bits)")
        print(f"  SNR scale: {snr_scale}")
        print(f"  Scheduled sampling: warmup={self.ss_warmup_epochs}, max_ratio={self.ss_max_ratio}")
        print(f"{'='*60}\n")
    
    def _compute_bits(self, num_classes: int) -> int:
        return max(1, math.ceil(math.log2(num_classes + 1)))
    
    def _build_coarse_to_fine_map(self, fine_to_coarse_map: Optional[Dict[int, int]]):
        if fine_to_coarse_map is None:
            self.coarse_to_fine_map = None
            return
        self.coarse_to_fine_map = {}
        for fine_id, coarse_id in fine_to_coarse_map.items():
            if coarse_id not in self.coarse_to_fine_map:
                self.coarse_to_fine_map[coarse_id] = []
            self.coarse_to_fine_map[coarse_id].append(fine_id)
    
    def _build_diffusion_schedule(self, time_steps: int, sampling_steps: int, noise_schedule: str):
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
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # -------------------------------------------------------------------------
    # Coarse conditioning helpers
    # -------------------------------------------------------------------------

    def _condition_features(self, features: Tensor, coarse_labels: Tensor) -> Tensor:
        """Add coarse label embeddings to BERT features."""
        coarse_clean = coarse_labels.clone()
        coarse_clean[coarse_clean == -100] = self.num_coarse_classes  # padding → last embed
        coarse_clean = coarse_clean.clamp(0, self.num_coarse_classes)
        coarse_emb = self.coarse_embed(coarse_clean)
        return features + coarse_emb

    def _get_scheduled_sampling_ratio(self, epoch: int) -> float:
        """Compute probability of using noisy coarse labels during training.
        
        epoch < warmup  →  0.0  (pure teacher forcing)
        epoch >= warmup →  ramps linearly up to ss_max_ratio
        """
        if epoch < self.ss_warmup_epochs:
            return 0.0
        ramp = (epoch - self.ss_warmup_epochs) / max(1, self.ss_warmup_epochs)
        return min(self.ss_max_ratio, ramp)

    def _get_noisy_coarse_labels(self, coarse_labels: Tensor) -> Tensor:
        """Simulate Stage 1 errors by randomly corrupting coarse labels.
        ~10% token-level corruption (matches typical Stage 1 error rate).
        """
        noisy = coarse_labels.clone()
        valid = coarse_labels != -100
        corrupt_mask = (torch.rand_like(coarse_labels.float()) < 0.10) & valid
        random_labels = torch.randint(0, self.num_coarse_classes, coarse_labels.shape, device=coarse_labels.device)
        noisy[corrupt_mask] = random_labels[corrupt_mask]
        return noisy
    
    # -------------------------------------------------------------------------
    # Stage 1: Global Coarse Prediction
    # -------------------------------------------------------------------------
    
    def forward_stage1(self, features: Tensor, coarse_labels: Tensor, attention_mask: Tensor) -> Tensor:
        coarse_labels_clean = coarse_labels.clone()
        coarse_labels_clean[coarse_labels_clean == -100] = 0
        
        bits = decimal_to_bits(coarse_labels_clean, self.coarse_bits) * self.snr_scale
        
        batch_size = features.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        noise = torch.randn_like(bits)
        x_t = self.q_sample(bits, t, noise)
        
        pred = self.global_dit(x_t, t, features, attention_mask)
        
        target = bits if self.objective == 'pred_x0' else noise
        mask = (coarse_labels != -100).unsqueeze(-1).expand_as(pred)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.loss_type == 'l2':
            loss = F.mse_loss(pred[mask], target[mask])
        else:
            loss = F.l1_loss(pred[mask], target[mask])
        
        return loss
    
    @torch.no_grad()
    def sample_stage1(self, features: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size, seq_len = features.shape[:2]
        shape = (batch_size, seq_len, self.coarse_bits)
        
        x = torch.randn(shape, device=self.device)
        x_start = None
        
        times = torch.linspace(self.num_timesteps - 1, 0, self.sampling_timesteps, 
                               dtype=torch.long, device=self.device)
        
        for i, time in enumerate(times):
            t = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            
            pred = self.global_dit(x, t, features, attention_mask, x_self_cond=x_start)
            
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
        
        coarse_preds = bits_to_decimal(x / self.snr_scale, self.coarse_bits)
        coarse_preds = coarse_preds.clamp(0, self.num_coarse_classes - 1)
        return coarse_preds
    
    # -------------------------------------------------------------------------
    # Stage 2: Global Fine-grained with Scheduled Sampling
    # -------------------------------------------------------------------------
    
    def forward_stage2(
        self,
        features: Tensor,
        coarse_labels: Tensor,
        fine_labels: Tensor,
        attention_mask: Tensor,
        epoch: int = 0
    ) -> Tensor:
        """
        Stage 2 training: Predict fine-grained labels with global DiT.
        Uses scheduled sampling: mix of GT and noisy coarse labels.
        """
        fine_labels_clean = fine_labels.clone()
        fine_labels_clean[fine_labels_clean == -100] = 0
        
        bits = decimal_to_bits(fine_labels_clean, self.fine_bits) * self.snr_scale
        
        batch_size = features.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        noise = torch.randn_like(bits)
        x_t = self.q_sample(bits, t, noise)
        
        # Scheduled sampling: sometimes use noisy coarse labels
        ss_ratio = self._get_scheduled_sampling_ratio(epoch)
        if ss_ratio > 0 and random() < ss_ratio:
            cond_coarse = self._get_noisy_coarse_labels(coarse_labels)
        else:
            cond_coarse = coarse_labels
        
        cond_features = self._condition_features(features, cond_coarse)
        
        # Self-conditioning (50% during training)
        x_self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.fine_dit(x_t, t, cond_features, attention_mask).detach()
        
        pred = self.fine_dit(x_t, t, cond_features, attention_mask, x_self_cond)
        
        target = bits if self.objective == 'pred_x0' else noise
        mask = (fine_labels != -100).unsqueeze(-1).expand_as(pred)
        
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
        attention_mask: Tensor,
        apply_constraint: bool = True
    ) -> Tensor:
        """Stage 2 inference: Conditioned on Stage 1 predicted coarse labels."""
        batch_size, seq_len = features.shape[:2]
        shape = (batch_size, seq_len, self.fine_bits)
        
        cond_features = self._condition_features(features, coarse_labels)
        
        x = torch.randn(shape, device=self.device)
        x_start = None
        
        times = torch.linspace(self.num_timesteps - 1, 0, self.sampling_timesteps,
                               dtype=torch.long, device=self.device)
        
        for i, time in enumerate(times):
            t = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            
            pred = self.fine_dit(x, t, cond_features, attention_mask, x_self_cond=x_start)
            
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
        
        fine_preds = bits_to_decimal(x / self.snr_scale, self.fine_bits)
        fine_preds = fine_preds.clamp(0, self.num_fine_classes - 1)
        
        if apply_constraint and self.coarse_to_fine_map is not None:
            fine_preds = self._constrain_to_coarse(fine_preds, coarse_labels)
        
        return fine_preds
    
    def _constrain_to_coarse(self, fine_preds: Tensor, coarse_labels: Tensor) -> Tensor:
        if self.coarse_to_fine_map is None:
            return fine_preds
        
        batch_size, seq_len = fine_preds.shape
        for b in range(batch_size):
            for s in range(seq_len):
                coarse = coarse_labels[b, s].item()
                fine = fine_preds[b, s].item()
                if coarse in self.coarse_to_fine_map:
                    valid_fine = self.coarse_to_fine_map[coarse]
                    if fine not in valid_fine and len(valid_fine) > 0:
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
        stage: str = 'both',
        epoch: int = 0
    ):
        """Combined forward pass for training."""
        features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        if stage == 'stage1':
            assert coarse_labels is not None
            return self.forward_stage1(features, coarse_labels, attention_mask)
        
        elif stage == 'stage2':
            assert coarse_labels is not None and fine_labels is not None
            return self.forward_stage2(features, coarse_labels, fine_labels, attention_mask, epoch=epoch)
        
        elif stage in ['both', 'joint']:
            assert coarse_labels is not None and fine_labels is not None
            
            loss1 = self.forward_stage1(features, coarse_labels, attention_mask)
            loss2 = self.forward_stage2(features, coarse_labels, fine_labels, attention_mask, epoch=epoch)
            
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
        attention_mask: Tensor,
        apply_constraint: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Full inference: Stage 1 → coarse preds → condition Stage 2 → fine preds."""
        features = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        coarse_preds = self.sample_stage1(features, attention_mask)
        fine_preds = self.sample_stage2(features, coarse_preds, attention_mask, 
                                        apply_constraint=apply_constraint)
        
        return coarse_preds, fine_preds
    
    def get_coarse_from_fine(self, fine_preds: Tensor) -> Tensor:
        if self.fine_to_coarse is None:
            return torch.zeros_like(fine_preds)
        fine_preds_clamped = fine_preds.clamp(0, len(self.fine_to_coarse) - 1)
        return self.fine_to_coarse[fine_preds_clamped]


# =============================================================================
# Factory Function
# =============================================================================

def create_hierarchical_model(
    device: str = 'cuda',
    backbone: str = 'FacebookAI/xlm-roberta-large',
    fine_to_coarse_map: Optional[Dict[int, int]] = None,
    **kwargs
) -> HierarchicalDiffusionNeCTI:
    return HierarchicalDiffusionNeCTI(
        device=torch.device(device),
        backbone=backbone,
        fine_to_coarse_map=fine_to_coarse_map,
        **kwargs
    )