"""
BiaffineBitDit: Minimal modification of the baseline BitDit model.

DESIGN PRINCIPLE: Change as little as possible from the working baseline.

Changes from BitDit:
    1. Adds a BiaffineConditioner after BERT feature extraction
    2. Enhanced features are passed to the SAME DiT, SAME DDIM sampler

Everything else is IDENTICAL to BitDit:
    - DiT architecture (dit_discrete.py, num_heads=8)
    - DDIM sampling with eta parameter
    - Self-conditioning
    - Weight initialization
    - Noise schedule
    - Loss computation
"""

from random import random
import torch
import torch.nn.functional as F
from .ddim_bitdit import BitDit
from .biaffine_conditioner import BiaffineConditioner
from .utils import decimal_to_bits, bits_to_decimal
from .compound_encoder import extract_compound_masks_from_labels


class BiaffineBitDit(BitDit):
    """
    BitDit + BiaffineConditioner.
    
    Only overrides forward() to insert biaffine enhancement after BERT.
    ALL diffusion mechanics, DiT architecture, DDIM sampling, and 
    self-conditioning are inherited UNCHANGED from BitDit.
    """
    
    def __init__(self, *args,
                 biaffine_rank: int = 64,
                 biaffine_num_heads: int = 8,
                 biaffine_dropout: float = 0.1,
                 biaffine_gate_init: float = -3.0,
                 **kwargs):
        # Initialize the full BitDit (DiT, DDIM, schedules, etc.)
        super().__init__(*args, **kwargs)
        
        # Add biaffine conditioner and move to same device as rest of model
        self.biaffine = BiaffineConditioner(
            hidden_size=self.dim_model,
            rank=biaffine_rank,
            num_heads=biaffine_num_heads,
            dropout=biaffine_dropout,
            gate_init_bias=biaffine_gate_init,
        ).to(self.device)
        
        print(f"\n{'='*60}")
        print(f"BiaffineBitDit initialized")
        print(f"  Biaffine rank: {biaffine_rank}")
        print(f"  Gate init: {biaffine_gate_init} (sigmoid={torch.sigmoid(torch.tensor(biaffine_gate_init)).item():.4f})")
        print(f"  Everything else: IDENTICAL to BitDit baseline")
        print(f"{'='*60}\n")
    
    def _enhance_features(self, features, mask):
        """Apply biaffine enhancement to BERT features."""
        return self.biaffine(features, mask)
    
    def forward(self, input_ids, attention_mask, seq_labels, span_labels=None,
                words2pieces=None, ensemble=False):
        """
        Same as BitDit.forward() but with biaffine enhancement after BERT.
        
        The ONLY difference: features = biaffine_enhance(bert_output)
        """
        # 1. Feature Extraction (same as baseline)
        label_mask = (seq_labels != -100).long()
        bert_output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        if self.add_lstm:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            min_value = torch.min(bert_output).item()
            bert_output = bert_output.unsqueeze(dim=1).expand(
                -1, words2pieces.shape[1], -1, -1
            )
            bert_output = torch.masked_fill(
                bert_output, words2pieces.eq(0).unsqueeze(dim=-1), min_value
            )
            features, _ = torch.max(bert_output, dim=2)
            lengths = words2pieces.sum(dim=-1).gt(0).sum(dim=-1).cpu()
            packed_features = pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False
            )
            packed_outs, (hidden, _) = self.lstm(packed_features)
            features = pad_packed_sequence(
                packed_outs, batch_first=True, total_length=max(lengths)
            )[0]
        else:
            features = bert_output

        # *** THE ONLY CHANGE: biaffine enhancement ***
        features = self._enhance_features(features, label_mask)

        # --- INFERENCE MODE --- (identical to BitDit)
        if not self.training:
            if self.compound_aware and self.label_set is not None:
                shape = (*features.shape[:2], self.bits.item())
                results_pass1 = self.sample(shape, features, label_mask)
                bit_seq_pass1, _ = results_pass1
                predictions_pass1 = bits_to_decimal(bit_seq_pass1, self.bits.item())
                
                try:
                    compound_masks = extract_compound_masks_from_labels(
                        predictions_pass1, self.label_set
                    )
                    if self.use_graph_encoder:
                        compound_features, compound_mask = self.compound_encoder(
                            features, compound_masks, predictions_pass1, self.label_set
                        )
                    else:
                        compound_features, compound_mask = self.compound_encoder(
                            features, compound_masks
                        )
                    compound_shape = (*compound_features.shape[:2], self.bits.item())
                    results_pass2 = self.sample(
                        compound_shape, compound_features, compound_mask
                    )
                    bit_seq_compounds, _ = results_pass2
                    bit_seq = self.compound_decoder(bit_seq_compounds, compound_masks)
                    results = bits_to_decimal(bit_seq, self.bits.item())
                    return results, [results]
                except Exception as e:
                    print(f"Fallback to token inference: {e}")
                    results = bits_to_decimal(bit_seq_pass1, self.bits.item())
                    return results, [results]
            else:
                shape = (*features.shape[:2], self.bits.item())
                results = self.sample(shape, features, label_mask)
                bit_seq, path_x = results
                path_x = torch.stack(
                    [bits_to_decimal(r, self.bits.item()) for r in path_x], dim=1
                )
                results = bits_to_decimal(bit_seq, self.bits.item())
                return results, path_x

        # --- TRAINING MODE --- (identical to BitDit)
        if self.training:
            bits_seq_labels = decimal_to_bits(seq_labels, self.bits)
            bits_seq_labels *= self.scale
            
            total_loss = 0.0
            loss_components = {}
            
            # TASK 1: Compound-Level Diffusion (optional, same as baseline)
            if self.compound_aware and self.label_set is not None:
                try:
                    compound_masks = extract_compound_masks_from_labels(
                        seq_labels, self.label_set
                    )
                    if self.use_graph_encoder:
                        compound_features, compound_mask = self.compound_encoder(
                            features, compound_masks, seq_labels, self.label_set
                        )
                    else:
                        compound_features, compound_mask = self.compound_encoder(
                            features, compound_masks
                        )
                    
                    bsz, max_compounds, _ = compound_masks.shape
                    bits = self.bits.item()
                    compound_labels = torch.zeros(
                        bsz, max_compounds, bits, device=self.device
                    )
                    for b in range(bsz):
                        for c in range(max_compounds):
                            comp_mask = compound_masks[b, c].bool()
                            if comp_mask.any():
                                compound_labels[b, c] = bits_seq_labels[
                                    b, comp_mask
                                ].mean(dim=0)
                    
                    noise_cmp, ts_cmp, noise_err = self.prepare_targets(compound_labels)
                    self_cond = None
                    if random() < 0.5:
                        with torch.no_grad():
                            self_cond = self.model_predictions(
                                noise_cmp, ts_cmp, compound_features, compound_mask
                            ).pred_x_start
                            self_cond.detach_()
                    pred_cmp = self.model(
                        noise_cmp, ts_cmp, compound_features, compound_mask, self_cond
                    )
                    pred_token_from_cmp = self.compound_decoder(pred_cmp, compound_masks)
                    targets = bits_seq_labels
                    targets_mask = label_mask.unsqueeze(dim=-1).expand(-1, -1, bits)
                    if self.objective == 'pred_noise':
                        noise_token_from_cmp = self.compound_decoder(
                            noise_err, compound_masks
                        )
                        targets = noise_token_from_cmp
                    loss_compound = F.mse_loss(
                        pred_token_from_cmp[targets_mask.bool()],
                        targets[targets_mask.bool()],
                    )
                    total_loss += loss_compound
                    loss_components['diff_cmp'] = loss_compound.item()
                except Exception as e:
                    print(f"Compound training skipped (batch error): {e}")

            # TASK 2: Token-Level Diffusion (same as baseline)
            noise_tok, ts_tok, noise_err_tok = self.prepare_targets(bits_seq_labels)

            self_cond_tok = None
            if random() < 0.5:
                with torch.no_grad():
                    self_cond_tok = self.model_predictions(
                        noise_tok, ts_tok, features, label_mask
                    ).pred_x_start
                    self_cond_tok.detach_()

            pred_tok = self.model(
                noise_tok, ts_tok, features, label_mask, self_cond_tok
            )

            targets_tok = bits_seq_labels
            if self.objective == 'pred_noise':
                targets_tok = noise_err_tok
            
            targets_mask = label_mask.unsqueeze(dim=-1).expand(
                -1, -1, self.bits.item()
            )
            loss_token = F.mse_loss(
                pred_tok[targets_mask.bool()], targets_tok[targets_mask.bool()]
            )
            total_loss += loss_token
            loss_components['diff_tok'] = loss_token.item()

            # Contrastive Loss (optional, same as baseline)
            if self.use_contrastive and ts_tok.float().mean() < self.timesteps * 0.3:
                contrastive_l = self.contrastive_loss_fn(
                    pred_tok, seq_labels, label_mask
                )
                total_loss += self.contrastive_weight * contrastive_l
                loss_components['contrastive'] = contrastive_l.item()

            # TASK 3: Span Classification (optional, same as baseline)
            if span_labels is not None:
                span_logits = self.span_classifier(features)
                active_loss = label_mask.view(-1) == 1
                active_logits = span_logits.view(-1, 2)
                active_labels = span_labels.view(-1)
                span_loss = F.cross_entropy(
                    active_logits[active_loss], active_labels[active_loss]
                )
                total_loss += self.span_loss_weight * span_loss
                loss_components['span'] = span_loss.item()

            # Log gate value for monitoring
            loss_components['biaffine_gate'] = self.biaffine.get_gate_value()
            
            self.last_losses = loss_components
            return total_loss
