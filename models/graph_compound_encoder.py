"""
Graph-Aware Compound Encoder for NeCTI Task
Encodes dependency relationships BETWEEN compounds using Graph Neural Networks

Key Innovation: Unlike the original CompoundEncoder which treats compounds independently,
this version builds a dependency graph between compounds and uses message passing to
propagate information along dependency edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .compound_encoder import CompoundEncoder


class GraphCompoundEncoder(nn.Module):
    """
    Graph-aware compound encoder that:
    1. Pools tokens within compounds (like CompoundEncoder)
    2. Extracts dependency structure BETWEEN compounds from labels
    3. Uses GNN to propagate information along compound dependencies
    4. Encodes relation types (Tatpurusha, Dvandva, etc.)
    """
    
    def __init__(self, input_dim: int, output_dim: int = None, 
                 pooling_method: str = 'mean',
                 num_gnn_layers: int = 2,
                 num_relation_types: int = 50):
        """
        Args:
            input_dim: Dimension of input token features (BERT dimension)
            output_dim: Desired output dimension. If None, uses input_dim
            pooling_method: How to pool tokens within compounds
            num_gnn_layers: Number of graph neural network layers
            num_relation_types: Number of dependency relation types
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_gnn_layers = num_gnn_layers
        
        # Base compound encoder (pools tokens within compounds)
        self.compound_encoder = CompoundEncoder(input_dim, output_dim, pooling_method)
        
        # Relation type embeddings (for different compound dependency types)
        self.relation_embeddings = nn.Embedding(num_relation_types, self.output_dim)
        
        # GNN layers for message passing between compounds
        self.gnn_layers = nn.ModuleList([
            CompoundGNNLayer(self.output_dim) 
            for _ in range(num_gnn_layers)
        ])
        
        print(f"✓ GraphCompoundEncoder initialized:")
        print(f"  - Base pooling: {pooling_method}")
        print(f"  - GNN layers: {num_gnn_layers}")
        print(f"  - Relation types: {num_relation_types}")
    
    def forward(self, 
                bert_features: torch.Tensor, 
                compound_masks: torch.Tensor,
                seq_labels: torch.Tensor,
                label_set) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode compounds with dependency structure.
        
        Args:
            bert_features: [batch_size, seq_len, input_dim] - Token-level BERT features
            compound_masks: [batch_size, max_compounds, seq_len] - Binary masks for compounds
            seq_labels: [batch_size, seq_len] - Token-level label IDs (contains dependency info)
            label_set: NeCTILabelSet object with id2label mapping
        
        Returns:
            compound_features: [batch_size, max_compounds, output_dim] - Graph-aware compound representations
            compound_mask: [batch_size, max_compounds] - Binary mask for valid compounds
        """
        batch_size, seq_len, input_dim = bert_features.shape
        _, max_compounds, _ = compound_masks.shape
        
        # Step 1: Pool tokens within each compound (standard compound encoding)
        compound_features, compound_mask = self.compound_encoder(bert_features, compound_masks)
        # compound_features: [batch_size, max_compounds, output_dim]
        
        # Step 2: Extract dependency graph between compounds
        dependency_graphs = self._extract_compound_dependencies(
            compound_masks, seq_labels, label_set, batch_size, max_compounds, seq_len
        )
        # Returns: adjacency_matrix, relation_types for each batch item
        
        # Step 3: Apply GNN layers to propagate information along dependencies
        for gnn_layer in self.gnn_layers:
            compound_features = gnn_layer(
                compound_features, 
                dependency_graphs['adjacency'], 
                dependency_graphs['relation_types'],
                self.relation_embeddings,
                compound_mask
            )
        
        return compound_features, compound_mask
    
    def _extract_compound_dependencies(self,
                                      compound_masks: torch.Tensor,
                                      seq_labels: torch.Tensor,
                                      label_set,
                                      batch_size: int,
                                      max_compounds: int,
                                      seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Extract dependency graph between compounds from labels.
        
        Labels are encoded as "{distance}:{relation_type}" or "ROOT:{type}"
        Example: "+1:Tatpurusha" means head is 1 position to the right with Tatpurusha relation
        
        Returns:
            Dictionary with:
            - adjacency: [batch_size, max_compounds, max_compounds] - binary adjacency matrix
            - relation_types: [batch_size, max_compounds, max_compounds] - relation type IDs
        """
        device = compound_masks.device
        
        # Initialize adjacency matrix and relation types
        adjacency = torch.zeros(batch_size, max_compounds, max_compounds, dtype=torch.float32, device=device)
        relation_types = torch.zeros(batch_size, max_compounds, max_compounds, dtype=torch.long, device=device)
        
        # Build a mapping from token index to compound index
        token_to_compound = self._build_token_to_compound_map(compound_masks, batch_size, seq_len, max_compounds)
        # token_to_compound[b, t] = compound index that token t belongs to in batch b
        
        for b in range(batch_size):
            for t in range(seq_len):
                label_id = seq_labels[b, t].item()
                if label_id == -100:  # Skip padding
                    continue
                
                label_str = label_set.id2label(label_id)
                
                # Parse label: "{distance}:{relation}" or "ROOT:{relation}"
                if ':' not in label_str:
                    continue
                
                parts = label_str.split(':')
                distance_str = parts[0]
                relation_type = ':'.join(parts[1:])  # Handle colons in relation names
                
                # Get compound index for current token
                compound_idx = token_to_compound[b, t].item()
                if compound_idx == -1:  # Token not in any compound
                    continue
                
                # Determine head compound based on distance
                if distance_str == "ROOT":
                    # Root compound - no parent
                    continue
                else:
                    try:
                        distance = int(distance_str)
                        head_token_idx = t + distance  # Head token index
                        
                        if 0 <= head_token_idx < seq_len:
                            head_compound_idx = token_to_compound[b, head_token_idx].item()
                            
                            if head_compound_idx != -1 and head_compound_idx != compound_idx:
                                # Add edge: head_compound -> current_compound
                                adjacency[b, head_compound_idx, compound_idx] = 1.0
                                
                                # Store relation type (need to map relation string to ID)
                                relation_id = self._relation_to_id(relation_type)
                                relation_types[b, head_compound_idx, compound_idx] = relation_id
                    except ValueError:
                        # Distance is not a number (shouldn't happen with proper labels)
                        continue
        
        return {
            'adjacency': adjacency,
            'relation_types': relation_types
        }
    
    def _build_token_to_compound_map(self, 
                                     compound_masks: torch.Tensor,
                                     batch_size: int,
                                     seq_len: int,
                                     max_compounds: int) -> torch.Tensor:
        """
        Build mapping from token index to compound index.
        
        Returns:
            token_to_compound: [batch_size, seq_len] - compound index for each token (-1 if none)
        """
        device = compound_masks.device
        token_to_compound = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            for c in range(max_compounds):
                mask = compound_masks[b, c]  # [seq_len]
                token_indices = torch.where(mask == 1)[0]
                token_to_compound[b, token_indices] = c
        
        return token_to_compound
    
    def _relation_to_id(self, relation: str) -> int:
        """
        Map relation type string to integer ID.
        Uses simple hash for now - could be replaced with proper vocabulary.
        """
        # Simple hash-based mapping (deterministic)
        # Better: maintain a proper relation vocabulary
        return abs(hash(relation)) % self.relation_embeddings.num_embeddings


class CompoundGNNLayer(nn.Module):
    """
    Single layer of Graph Neural Network for compound dependencies.
    Uses message passing along dependency edges with relation-aware attention.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message function: transforms neighbor features
        self.message_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # [neighbor_feat, relation_emb]
        
        # Attention mechanism for aggregating messages
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)  # [self_feat, neighbor_feat]
        
        # Update function: combines self features with aggregated messages
        self.update_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self,
                compound_features: torch.Tensor,
                adjacency: torch.Tensor,
                relation_types: torch.Tensor,
                relation_embeddings: nn.Embedding,
                compound_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply one layer of graph convolution.
        
        Args:
            compound_features: [batch_size, max_compounds, hidden_dim]
            adjacency: [batch_size, max_compounds, max_compounds] - binary adjacency matrix
            relation_types: [batch_size, max_compounds, max_compounds] - relation type IDs
            relation_embeddings: Embedding layer for relation types
            compound_mask: [batch_size, max_compounds] - valid compound mask
        
        Returns:
            updated_features: [batch_size, max_compounds, hidden_dim]
        """
        batch_size, max_compounds, hidden_dim = compound_features.shape
        device = compound_features.device
        
        # Expand masks for broadcasting
        compound_mask_expanded = compound_mask.unsqueeze(-1).float()  # [batch, compounds, 1]
        
        updated_features = []
        
        for b in range(batch_size):
            batch_features = compound_features[b]  # [max_compounds, hidden_dim]
            batch_adj = adjacency[b]  # [max_compounds, max_compounds]
            batch_rel_types = relation_types[b]  # [max_compounds, max_compounds]
            batch_mask = compound_mask[b]  # [max_compounds]
            
            new_features = []
            
            for i in range(max_compounds):
                if batch_mask[i] == 0:  # Invalid compound
                    new_features.append(torch.zeros(hidden_dim, device=device))
                    continue
                
                self_feat = batch_features[i]  # [hidden_dim]
                
                # Find neighbors (incoming edges: j -> i)
                neighbor_indices = torch.where(batch_adj[:, i] > 0)[0]
                
                if len(neighbor_indices) == 0:
                    # No neighbors, keep self features
                    new_features.append(self_feat)
                    continue
                
                # Aggregate messages from neighbors
                messages = []
                attention_scores = []
                
                for j in neighbor_indices:
                    neighbor_feat = batch_features[j]  # [hidden_dim]
                    
                    # Get relation embedding
                    rel_type_id = batch_rel_types[j, i]
                    rel_emb = relation_embeddings(rel_type_id)  # [hidden_dim]
                    
                    # Compute message: combine neighbor features with relation embedding
                    message_input = torch.cat([neighbor_feat, rel_emb], dim=0)  # [hidden_dim * 2]
                    message = self.message_fc(message_input)  # [hidden_dim]
                    messages.append(message)
                    
                    # Compute attention score
                    attention_input = torch.cat([self_feat, neighbor_feat], dim=0)  # [hidden_dim * 2]
                    attention_score = self.attention_fc(attention_input)  # [1]
                    attention_scores.append(attention_score)
                
                # Aggregate messages with attention
                messages = torch.stack(messages)  # [num_neighbors, hidden_dim]
                attention_scores = torch.stack(attention_scores).squeeze(-1)  # [num_neighbors]
                attention_weights = F.softmax(attention_scores, dim=0)  # [num_neighbors]
                
                aggregated_message = (messages * attention_weights.unsqueeze(-1)).sum(dim=0)  # [hidden_dim]
                
                # Update self features
                update_input = torch.cat([self_feat, aggregated_message], dim=0)  # [hidden_dim * 2]
                updated_feat = self.update_fc(update_input)  # [hidden_dim]
                
                # Residual connection + layer norm
                updated_feat = self.layer_norm(updated_feat + self_feat)
                
                new_features.append(updated_feat)
            
            updated_features.append(torch.stack(new_features))
        
        updated_features = torch.stack(updated_features)  # [batch_size, max_compounds, hidden_dim]
        
        # Apply mask to zero out invalid compounds
        updated_features = updated_features * compound_mask_expanded
        
        return updated_features


def extract_compound_masks_from_labels(labels: torch.Tensor, label_set, max_compounds: int = None) -> torch.Tensor:
    """
    Extract compound masks from token-level labels.
    Re-exported for compatibility with original API.
    """
    from .compound_encoder import extract_compound_masks_from_labels as original_extract
    return original_extract(labels, label_set, max_compounds)
