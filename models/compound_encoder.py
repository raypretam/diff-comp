"""
Compound-Aware Encoding for NeCTI Task
Operates at compound level instead of token level for better consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class CompoundEncoder(nn.Module):
    """
    Pool tokens within compounds into fixed-size compound representations.
    
    This module:
    1. Takes token-level BERT features
    2. Pools them into compound-level representations using compound masks
    3. Returns fixed-size compound embeddings
    """
    
    def __init__(self, input_dim: int, output_dim: int = None, pooling_method: str = 'mean'):
        """
        Args:
            input_dim: Dimension of input token features (actual BERT dimension, e.g., 768 or 1024)
            output_dim: Desired output dimension (target dim_model). If None, uses input_dim
            pooling_method: 'mean', 'max', 'attention', or 'lstm'
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.pooling_method = pooling_method
        
        # Projection layer if input and output dimensions differ
        if self.input_dim != self.output_dim:
            self.projection = nn.Linear(input_dim, self.output_dim)
            print(f"  Added projection layer: {input_dim} → {self.output_dim}")
        else:
            self.projection = None
        
        if pooling_method == 'attention':
            # Learnable attention pooling
            self.attention_weights = nn.Linear(input_dim, 1)
        elif pooling_method == 'lstm':
            # LSTM-based pooling
            self.lstm = nn.LSTM(input_dim, input_dim // 2, bidirectional=True, batch_first=True)
    
    def forward(self, bert_features: torch.Tensor, compound_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool token-level features into compound-level representations.
        
        Args:
            bert_features: [batch_size, seq_len, input_dim] - Token-level BERT features
            compound_masks: [batch_size, max_compounds, seq_len] - Binary masks for each compound
                           mask[b, c, t] = 1 if token t belongs to compound c in batch b
        
        Returns:
            compound_features: [batch_size, max_compounds, output_dim] - Compound representations
            compound_mask: [batch_size, max_compounds] - Binary mask for valid compounds (1=valid, 0=padding)
        """
        batch_size, seq_len, input_dim = bert_features.shape
        _, max_compounds, _ = compound_masks.shape
        
        # Determine which compounds are valid (have at least one token)
        compound_mask = compound_masks.sum(dim=-1) > 0  # [batch_size, max_compounds]
        
        if self.pooling_method == 'mean':
            # Mean pooling over tokens in each compound
            compound_features = self._mean_pooling(bert_features, compound_masks)
        
        elif self.pooling_method == 'max':
            # Max pooling over tokens in each compound
            compound_features = self._max_pooling(bert_features, compound_masks)
        
        elif self.pooling_method == 'attention':
            # Attention-weighted pooling
            compound_features = self._attention_pooling(bert_features, compound_masks)
        
        elif self.pooling_method == 'lstm':
            # LSTM pooling (take last hidden state for each compound)
            compound_features = self._lstm_pooling(bert_features, compound_masks)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Apply projection if dimensions differ
        if self.projection is not None:
            compound_features = self.projection(compound_features)
        
        return compound_features, compound_mask.long()
    
    def _mean_pooling(self, bert_features: torch.Tensor, compound_masks: torch.Tensor) -> torch.Tensor:
        """Mean pooling over compound tokens"""
        # bert_features: [batch_size, seq_len, hidden_size]
        # compound_masks: [batch_size, max_compounds, seq_len]
        
        # Expand features for broadcasting: [batch_size, 1, seq_len, hidden_size]
        features_expanded = bert_features.unsqueeze(1)
        
        # Expand masks for broadcasting: [batch_size, max_compounds, seq_len, 1]
        masks_expanded = compound_masks.unsqueeze(-1).float()
        
        # Masked sum: [batch_size, max_compounds, hidden_size]
        compound_sum = (features_expanded * masks_expanded).sum(dim=2)
        
        # Count tokens per compound: [batch_size, max_compounds, 1]
        compound_lengths = masks_expanded.sum(dim=2).clamp(min=1)  # Avoid division by zero
        
        # Average: [batch_size, max_compounds, hidden_size]
        compound_features = compound_sum / compound_lengths
        
        return compound_features
    
    def _max_pooling(self, bert_features: torch.Tensor, compound_masks: torch.Tensor) -> torch.Tensor:
        """Max pooling over compound tokens"""
        batch_size, seq_len, hidden_size = bert_features.shape
        max_compounds = compound_masks.shape[1]
        
        # Expand features: [batch_size, 1, seq_len, hidden_size]
        features_expanded = bert_features.unsqueeze(1)
        
        # Expand masks: [batch_size, max_compounds, seq_len, 1]
        masks_expanded = compound_masks.unsqueeze(-1).float()
        
        # Apply mask (set non-compound tokens to very negative value)
        masked_features = features_expanded * masks_expanded + (1 - masks_expanded) * (-1e9)
        
        # Max pooling: [batch_size, max_compounds, hidden_size]
        compound_features = masked_features.max(dim=2)[0]
        
        return compound_features
    
    def _attention_pooling(self, bert_features: torch.Tensor, compound_masks: torch.Tensor) -> torch.Tensor:
        """Attention-weighted pooling over compound tokens"""
        batch_size, seq_len, hidden_size = bert_features.shape
        max_compounds = compound_masks.shape[1]
        
        # Compute attention scores: [batch_size, seq_len, 1]
        attention_scores = self.attention_weights(bert_features)
        
        # Expand for compounds: [batch_size, 1, seq_len, 1]
        attention_scores = attention_scores.unsqueeze(1)
        
        # Mask attention scores: [batch_size, max_compounds, seq_len, 1]
        masks_expanded = compound_masks.unsqueeze(-1).float()
        masked_scores = attention_scores * masks_expanded + (1 - masks_expanded) * (-1e9)
        
        # Softmax over sequence dimension: [batch_size, max_compounds, seq_len, 1]
        attention_weights = F.softmax(masked_scores, dim=2)
        
        # Weighted sum: [batch_size, max_compounds, hidden_size]
        features_expanded = bert_features.unsqueeze(1)
        compound_features = (features_expanded * attention_weights).sum(dim=2)
        
        return compound_features
    
    def _lstm_pooling(self, bert_features: torch.Tensor, compound_masks: torch.Tensor) -> torch.Tensor:
        """LSTM-based pooling (process each compound with LSTM, take final state)"""
        batch_size, seq_len, hidden_size = bert_features.shape
        max_compounds = compound_masks.shape[1]
        
        compound_features_list = []
        
        for b in range(batch_size):
            batch_compounds = []
            for c in range(max_compounds):
                # Get tokens for this compound
                mask = compound_masks[b, c]  # [seq_len]
                if mask.sum() == 0:
                    # Empty compound, use zero vector
                    batch_compounds.append(torch.zeros(hidden_size, device=bert_features.device))
                else:
                    # Extract compound tokens
                    compound_tokens = bert_features[b, mask.bool()]  # [comp_len, hidden_size]
                    # Run LSTM
                    _, (h_n, _) = self.lstm(compound_tokens.unsqueeze(0))  # h_n: [2, 1, hidden_size//2]
                    # Concatenate forward and backward final states
                    compound_repr = torch.cat([h_n[0, 0], h_n[1, 0]], dim=0)  # [hidden_size]
                    batch_compounds.append(compound_repr)
            
            compound_features_list.append(torch.stack(batch_compounds))
        
        compound_features = torch.stack(compound_features_list)  # [batch_size, max_compounds, hidden_size]
        
        return compound_features


class CompoundDecoder(nn.Module):
    """
    Broadcast compound-level predictions back to token-level.
    
    This module:
    1. Takes compound-level bit predictions
    2. Broadcasts them back to tokens using compound masks
    3. Returns token-level predictions
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, compound_bits: torch.Tensor, compound_masks: torch.Tensor) -> torch.Tensor:
        """
        Broadcast compound predictions to token level.
        
        Args:
            compound_bits: [batch_size, max_compounds, bits] - Compound-level predictions
            compound_masks: [batch_size, max_compounds, seq_len] - Compound membership masks
        
        Returns:
            token_bits: [batch_size, seq_len, bits] - Token-level predictions
        """
        batch_size, max_compounds, bits = compound_bits.shape
        seq_len = compound_masks.shape[2]
        
        # Expand compound bits: [batch_size, max_compounds, 1, bits]
        compound_bits_expanded = compound_bits.unsqueeze(2)
        
        # Expand masks: [batch_size, max_compounds, seq_len, 1]
        masks_expanded = compound_masks.unsqueeze(-1).float()
        
        # Broadcast: [batch_size, max_compounds, seq_len, bits]
        broadcasted = compound_bits_expanded * masks_expanded
        
        # Sum over compounds (each token should belong to exactly one compound)
        token_bits = broadcasted.sum(dim=1)  # [batch_size, seq_len, bits]
        
        return token_bits


def extract_compound_masks_from_labels(labels: torch.Tensor, label_set, max_compounds: int = None) -> torch.Tensor:
    """
    Extract compound masks from token-level labels.
    
    Compounds are identified by sequences ending with "Comp_root" or labels containing "Comp".
    
    Args:
        labels: [batch_size, seq_len] - Token-level label IDs
        label_set: NeCTILabelSet object with id2label mapping
        max_compounds: Maximum number of compounds per sequence (None = auto-detect)
    
    Returns:
        compound_masks: [batch_size, max_compounds, seq_len] - Binary masks
    """
    batch_size, seq_len = labels.shape
    device = labels.device
    
    # Determine max_compounds if not provided
    if max_compounds is None:
        max_compounds_in_batch = 0
        for b in range(batch_size):
            num_compounds = count_compounds_in_sequence(labels[b], label_set)
            max_compounds_in_batch = max(max_compounds_in_batch, num_compounds)
        max_compounds = max(max_compounds_in_batch, 1)  # At least 1
    
    # Initialize masks
    compound_masks = torch.zeros(batch_size, max_compounds, seq_len, dtype=torch.long, device=device)
    
    # Fill masks for each batch
    for b in range(batch_size):
        masks_b = extract_compound_masks_single(labels[b], label_set)
        num_comps = min(len(masks_b), max_compounds)
        for c in range(num_comps):
            compound_masks[b, c] = torch.tensor(masks_b[c], dtype=torch.long, device=device)
    
    return compound_masks


def extract_compound_masks_single(label_ids: torch.Tensor, label_set) -> List[List[int]]:
    """
    Extract compound masks for a single sequence.
    
    Args:
        label_ids: [seq_len] - Token-level label IDs
        label_set: NeCTILabelSet object
    
    Returns:
        List of masks, each mask is a list of 0/1 for each token
    """
    seq_len = label_ids.shape[0]
    label_ids_cpu = label_ids.cpu().tolist()
    
    compound_masks = []
    current_compound = []
    
    for t, label_id in enumerate(label_ids_cpu):
        label = label_set.id2label(label_id)
        
        # Skip padding tokens (-100)
        if label_id == -100:
            continue
        
        # Check if part of a compound
        if 'Comp' in label and label != 'No_rel':
            current_compound.append(t)
            
            # Check if this is the end of a compound
            if 'Comp_root' in label or 'root' in label.lower():
                # Create mask for this compound
                mask = [1 if i in current_compound else 0 for i in range(seq_len)]
                compound_masks.append(mask)
                current_compound = []
    
    # Handle any remaining unclosed compound
    if current_compound:
        mask = [1 if i in current_compound else 0 for i in range(seq_len)]
        compound_masks.append(mask)
    
    # If no compounds found, create a dummy compound with all tokens
    if not compound_masks:
        mask = [1 if label_ids_cpu[i] != -100 else 0 for i in range(seq_len)]
        compound_masks.append(mask)
    
    return compound_masks


def count_compounds_in_sequence(label_ids: torch.Tensor, label_set) -> int:
    """Count number of compounds in a sequence"""
    count = 0
    for label_id in label_ids:
        if label_id == -100:
            continue
        label = label_set.id2label(label_id.item())
        if 'Comp_root' in label or label == 'root':
            count += 1
    return max(count, 1)  # At least 1 compound
