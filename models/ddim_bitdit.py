from random import random
import math
from collections import namedtuple
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from .utils import decimal_to_bits, bits_to_decimal
from data.cws.cws_dataset import LabelSet
from .dit_discrete import DiT

__all__ = ["BitDit"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def identity(t, *args, **kwargs):
    return t


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


class BitDit(nn.Module):
    def __init__(self,
                 device: torch.device,
                 num_classes: int,
                 backbone: str,
                 time_steps: int,
                 sampling_steps: int,
                 depth: int,
                 ddim_sampling_eta: float,
                 self_condition: bool,
                 snr_scale: float,
                 dataset: str,
                 dim_model: int,
                 dim_time: int,
                 max_length: int,
                 num_labels: int,
                 bit_dim: int = 16,
                 noise_schedule: str = 'cosine',
                 objective: str = 'pred_x0',
                 loss_type: str = 'l2',
                 add_lstm: bool = False,
                 freeze_bert: bool = False):
        super().__init__()

        self.device = torch.device(device)

        # entity classes
        self.num_classes = num_classes
        self.dim_model = dim_model
        self.add_lstm = add_lstm
        self.bit_dim = bit_dim
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}

        ####################################
        # 1. BERT/XLM-R Encoder
        ####################################
        self.backbone = AutoModel.from_pretrained(backbone)
        if freeze_bert:
            self._freeze_backbone()

        ####################################
        # 2. Label Embedding + Classifier Head
        ####################################
        self.label_embed = nn.Embedding(num_classes, bit_dim)
        self.classifier = nn.Sequential(
            nn.Linear(bit_dim, bit_dim),
            nn.ReLU(),
            nn.Linear(bit_dim, num_classes)
        )

        ####################################
        # 3. Diffusion Scheduler Parameters
        ####################################
        if noise_schedule == "linear":
            betas = self.linear_beta_schedule(time_steps)
        elif noise_schedule == "cosine":
            betas = self.cosine_beta_schedule(time_steps)
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.num_timesteps = int(time_steps)
        self.sampling_timesteps = default(sampling_steps, time_steps)
        assert self.sampling_timesteps <= time_steps
        self.is_ddim_sampling = self.sampling_timesteps < time_steps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.self_condition = self_condition
        self.scale = snr_scale
        self.loss_type = loss_type

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        ####################################
        # 4. Diffusion Model: DiT on embeddings
        ####################################
        self.model = DiT(
            in_channels=bit_dim,
            hidden_size=dim_model,
            num_steps=self.num_timesteps,
            time_dim=dim_time,
            depth=depth,
            num_heads=8,
            mlp_ratio=4.0
        )

        self.to(self.device)

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    ############################################################
    # Beta Schedules
    ############################################################
    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    ############################################################
    # Diffusion Helpers
    ############################################################
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    ############################################################
    # Model Prediction Wrapper
    ############################################################
    def model_predictions(self, x, t, bert_features, attention_mask, x_self_cond=None, clip_x_start=True):
        """
        Make model predictions, get pred_x_start and pred_noise at the same time
        Args:
            x: [bsz, len, bit_dim]
            t: [bsz,]
            bert_features: [bsz, len, hid]
            attention_mask: [bsz, len]
            x_self_cond: [bsz, len, bit_dim] / None
            clip_x_start: True/False

        Returns:
            (noise, x_start): [bsz, len, bit_dim]
        """
        model_output = self.model(x, t, bert_features, attention_mask, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-self.scale, max=self.scale) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, bert_features, attention_mask, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, bert_features, attention_mask, x_self_cond=x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-self.scale, self.scale)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, bert_features, attention_mask, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, bert_features, attention_mask, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_bits = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_bits, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, bert_features, attention_mask):
        batch, device = shape[0], self.betas.device

        bit_seq = torch.randn(shape, device=device)

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            bit_seq, x_start = self.p_sample(bit_seq, t, bert_features, attention_mask, self_cond)

        return bit_seq

    @torch.no_grad()
    def ddim_sample(self, shape, bert_features, attention_mask, average=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective =\
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        batch_res = []
        bit_seq = torch.randn(shape, device=device)
        batch_res.append(bit_seq)

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(bit_seq, time_cond, bert_features, attention_mask, self_cond, clip_x_start=True)
            
            if time_next < 0:
                bit_seq = x_start
                batch_res.append(bit_seq)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(bit_seq)

            bit_seq = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            batch_res.append(bit_seq)

        return bit_seq, batch_res

    ############################################################
    # Forward Pass (Training)
    ############################################################
    def forward(self, input_ids, attention_mask, seq_labels, words2pieces=None, ensemble=False):

        bsz, seq_len = seq_labels.shape
        label_mask = (seq_labels != -100).long()

        ##############################
        # 1. Encode text
        ##############################
        bert_features = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        ##############################
        # 2. Embed labels
        ##############################
        # shape: [bsz, seq_len, bit_dim]
        x0 = self.label_embed(seq_labels.clamp(min=0))

        ##############################
        # 3. Forward diffusion (sample t)
        ##############################
        t = torch.randint(0, self.num_timesteps, (bsz,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        ##############################
        # 4. Self-conditioning
        ##############################
        self_cond = None
        if self.self_condition and torch.rand(1).item() < 0.5:
            with torch.no_grad():
                pred_noise, pred_x0 = self.model_predictions(x_t, t, bert_features, attention_mask)
                self_cond = pred_x0.detach()

        ##############################
        # 5. Predict noise or x0
        ##############################
        pred_noise, pred_x0 = self.model_predictions(
            x_t, t, bert_features, attention_mask, x_self_cond=self_cond
        )

        ##############################
        # 6. Compute loss
        ##############################
        target = noise if self.objective == 'pred_noise' else x0
        mask = label_mask.unsqueeze(-1).expand_as(target)

        if self.loss_type == 'l2':
            loss = F.mse_loss(pred_noise[mask], target[mask])
        else:
            loss = F.l1_loss(pred_noise[mask], target[mask])

        return loss

    ############################################################
    # Reverse Process (Sampling)
    ############################################################
    @torch.no_grad()
    def sample(self, shape, bert_features, attention_mask):
        """DDIM Sampling"""
        bsz, seq_len = shape
        x = torch.randn(bsz, seq_len, self.bit_dim, device=self.device)
        x0_path = []

        times = torch.linspace(-1, self.num_timesteps - 1,
                               steps=self.sampling_timesteps + 1).int().tolist()
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        for t_cur, t_next in time_pairs:
            t_cond = torch.full((bsz,), t_cur, device=self.device, dtype=torch.long)
            pred_noise, x0_pred = self.model_predictions(x, t_cond, bert_features, attention_mask)

            x0_path.append(x0_pred)

            if t_next < 0:
                x = x0_pred
                break

            alpha = self.alphas_cumprod[t_cur]
            alpha_next = self.alphas_cumprod[t_next]

            sigma = self.ddim_sampling_eta * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()

            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x)

            x = x0_pred * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # decode x0 to label ids
        x0_final = x
        logits = self.classifier(x0_final)
        pred_ids = logits.argmax(dim=-1)

        return pred_ids, x0_path

    def prepare_targets(self, gold_seq_labels):
        """
        Run forward process to get diffused labels and timesteps
        Args:
            gold_seq_labels: [bsz, len]

        Returns:
            diffused_labels: [bsz, len, bit_dim]
            ts: [bsz]
            noise: [bsz, len, bit_dim]
        """
        bsz = gold_seq_labels.shape[0]
        ts = torch.randint(0, self.num_timesteps, (bsz,), device=self.device).long()
        
        # Embed labels first
        x0 = self.label_embed(gold_seq_labels.clamp(min=0))
        noise = torch.randn_like(x0, device=self.device)
        diffused_labels = self.q_sample(x_start=x0, t=ts, noise=noise)

        return diffused_labels, ts, noise

    @torch.no_grad()
    def predict_with_voting(self, input_ids, attention_mask, seq_labels, words2pieces=None, num_votes=5):
        """
        Runs the sampling process multiple times and uses majority voting for the final prediction.
        """
        self.eval()
        batch_size = input_ids.shape[0]
        seq_len = seq_labels.shape[1]
        
        # Store counts: [batch, seq_len, num_classes]
        vote_counts = torch.zeros((batch_size, seq_len, self.num_classes), device=self.device)
        
        # Helper to accumulate votes
        def accumulate_votes(res_tensor):
            for b in range(batch_size):
                curr_len = res_tensor.shape[1]
                for t in range(curr_len):
                    label_id = res_tensor[b, t].long()
                    if label_id < self.num_classes:
                        vote_counts[b, t, label_id] += 1

        # Run num_votes iterations
        for _ in range(num_votes):
            bert_features = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
            
            _, x0_path = self.sample((batch_size, seq_len), bert_features, attention_mask)
            x0_final = x0_path[-1] if x0_path else torch.randn(batch_size, seq_len, self.bit_dim, device=self.device)
            
            logits = self.classifier(x0_final)
            res = logits.argmax(dim=-1)
            accumulate_votes(res)
        
        # Take argmax to get the consensus label
        final_preds = vote_counts.argmax(dim=-1)
        
        return final_preds, None