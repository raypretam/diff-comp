"""
Contrastive Learning Loss for Diffusion-based Sequence Labeling
Implements InfoNCE loss to improve label representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for sequence labeling.
    Pulls embeddings of the same label closer while pushing different labels apart.
    """
    
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Temperature parameter for scaling similarities (default: 0.07)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels, mask=None):
        """
        Compute InfoNCE loss for label embeddings.
        
        Args:
            embeddings: [batch_size, seq_len, embedding_dim] - predicted embeddings at diffusion time t
            labels: [batch_size, seq_len] - ground truth labels
            mask: [batch_size, seq_len] - attention mask (1 for valid tokens, 0 for padding)
        
        Returns:
            loss: scalar tensor - InfoNCE contrastive loss
        """
        batch_size, seq_len, embedding_dim = embeddings.shape
        
        # Flatten batch and sequence dimensions
        embeddings_flat = embeddings.view(-1, embedding_dim)  # [B*L, D]
        labels_flat = labels.view(-1)  # [B*L]
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1).bool()
            embeddings_flat = embeddings_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]
            
            # Skip if no valid tokens
            if embeddings_flat.size(0) == 0:
                return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize embeddings to unit sphere
        embeddings_normalized = F.normalize(embeddings_flat, dim=-1, p=2)
        
        # Compute pairwise cosine similarity matrix
        # similarity[i, j] = cosine_sim(emb_i, emb_j)
        similarity_matrix = torch.matmul(
            embeddings_normalized, 
            embeddings_normalized.T
        ) / self.temperature  # [N, N]
        
        # Create mask for positive pairs (same label, different position)
        labels_equal = labels_flat.unsqueeze(1) == labels_flat.unsqueeze(0)  # [N, N]
        positives_mask = labels_equal.float()
        
        # Remove self-similarity (diagonal)
        positives_mask.fill_diagonal_(0)
        
        # Compute exponential of similarities
        exp_similarity = torch.exp(similarity_matrix)
        
        # Mask out self-similarity for denominator
        exp_similarity_no_diag = exp_similarity.clone()
        exp_similarity_no_diag.fill_diagonal_(0)
        
        # Sum of positive similarities (numerator)
        pos_similarity_sum = (exp_similarity * positives_mask).sum(dim=1)
        
        # Sum of all similarities except self (denominator)
        all_similarity_sum = exp_similarity_no_diag.sum(dim=1)
        
        # InfoNCE loss: -log(sum(pos) / sum(all))
        # Add epsilon for numerical stability
        epsilon = 1e-8
        loss_per_sample = -torch.log(
            (pos_similarity_sum + epsilon) / (all_similarity_sum + epsilon)
        )
        
        # Only compute loss for samples that have at least one positive pair
        has_positives = positives_mask.sum(dim=1) > 0
        
        if has_positives.any():
            loss = loss_per_sample[has_positives].mean()
        else:
            # If no positive pairs exist (e.g., all labels are different), return zero loss
            loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss


class HierarchicalInfoNCELoss(nn.Module):
    """
    Hierarchical InfoNCE loss for nested/hierarchical labels.
    Useful for tasks like NeCTI where labels have coarse-fine structure.
    """
    
    def __init__(self, label_hierarchy, temperature=0.07, coarse_weight=0.5):
        """
        Args:
            label_hierarchy: dict mapping fine labels to coarse labels
                e.g., {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, ...}
                where fine labels 2,3 map to coarse label 1
            temperature: Temperature parameter for scaling
            coarse_weight: Weight for coarse-level positives (0.5 means coarse positives 
                          contribute half as much as fine positives)
        """
        super().__init__()
        self.temperature = temperature
        self.coarse_weight = coarse_weight
        
        # Create label hierarchy mapping as tensor for efficient lookup
        if label_hierarchy is not None:
            max_label = max(label_hierarchy.keys())
            self.hierarchy_tensor = torch.zeros(max_label + 1, dtype=torch.long)
            for fine, coarse in label_hierarchy.items():
                self.hierarchy_tensor[fine] = coarse
        else:
            self.hierarchy_tensor = None
    
    def forward(self, embeddings, labels, mask=None):
        """
        Compute hierarchical InfoNCE loss.
        
        Args:
            embeddings: [batch_size, seq_len, embedding_dim]
            labels: [batch_size, seq_len] - fine-grained labels
            mask: [batch_size, seq_len]
        
        Returns:
            loss: scalar tensor
        """
        if self.hierarchy_tensor is None:
            # Fall back to simple InfoNCE if no hierarchy provided
            return InfoNCELoss(self.temperature)(embeddings, labels, mask)
        
        batch_size, seq_len, embedding_dim = embeddings.shape
        device = embeddings.device
        
        # Move hierarchy tensor to correct device
        if self.hierarchy_tensor.device != device:
            self.hierarchy_tensor = self.hierarchy_tensor.to(device)
        
        # Flatten
        embeddings_flat = embeddings.view(-1, embedding_dim)
        labels_flat = labels.view(-1)
        
        # Apply mask
        if mask is not None:
            mask_flat = mask.view(-1).bool()
            embeddings_flat = embeddings_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]
            
            if embeddings_flat.size(0) == 0:
                return torch.tensor(0.0, device=device)
        
        # Normalize embeddings
        embeddings_normalized = F.normalize(embeddings_flat, dim=-1, p=2)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(
            embeddings_normalized,
            embeddings_normalized.T
        ) / self.temperature
        
        # Create fine-grained positive mask (exact same label)
        fine_labels_equal = labels_flat.unsqueeze(1) == labels_flat.unsqueeze(0)
        fine_positives = fine_labels_equal.float()
        fine_positives.fill_diagonal_(0)
        
        # Create coarse-grained positive mask (same coarse label, different fine label)
        # Get coarse labels for each fine label
        coarse_labels = self.hierarchy_tensor[labels_flat]
        coarse_labels_equal = coarse_labels.unsqueeze(1) == coarse_labels.unsqueeze(0)
        coarse_positives = coarse_labels_equal.float() * (1 - fine_positives)  # Exclude fine positives
        coarse_positives.fill_diagonal_(0)
        
        # Compute exponentials
        exp_similarity = torch.exp(similarity_matrix)
        exp_similarity_no_diag = exp_similarity.clone()
        exp_similarity_no_diag.fill_diagonal_(0)
        
        # Weighted positive similarities
        # Fine positives get full weight (1.0), coarse positives get reduced weight
        pos_similarity_sum = (
            (exp_similarity * fine_positives).sum(dim=1) * 1.0 +
            (exp_similarity * coarse_positives).sum(dim=1) * self.coarse_weight
        )
        
        # All similarities (denominator)
        all_similarity_sum = exp_similarity_no_diag.sum(dim=1)
        
        # InfoNCE loss
        epsilon = 1e-8
        loss_per_sample = -torch.log(
            (pos_similarity_sum + epsilon) / (all_similarity_sum + epsilon)
        )
        
        # Only compute for samples with positives
        has_positives = (fine_positives.sum(dim=1) + coarse_positives.sum(dim=1)) > 0
        
        if has_positives.any():
            loss = loss_per_sample[has_positives].mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss


def create_contrastive_loss(config):
    """
    Factory function to create appropriate contrastive loss based on config.
    
    Args:
        config: Configuration dict or Namespace with:
            - contrastive_type: 'simple' or 'hierarchical'
            - contrastive_temp: temperature parameter
            - label_hierarchy: (optional) dict for hierarchical loss
            - coarse_weight: (optional) weight for coarse positives
    
    Returns:
        ContrastiveLoss instance
    """
    contrastive_type = getattr(config, 'contrastive_type', 'simple')
    temperature = getattr(config, 'contrastive_temp', 0.07)
    
    if contrastive_type == 'hierarchical':
        label_hierarchy = getattr(config, 'label_hierarchy', None)
        coarse_weight = getattr(config, 'coarse_weight', 0.5)
        return HierarchicalInfoNCELoss(
            label_hierarchy=label_hierarchy,
            temperature=temperature,
            coarse_weight=coarse_weight
        )
    else:
        return InfoNCELoss(temperature=temperature)
