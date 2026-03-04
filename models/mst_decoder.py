"""
MST (Maximum Spanning Tree) Decoder for NeCTI
Adapted from separ implementation for dependency tree decoding
Used to enforce valid tree structure constraints on compound predictions
"""

import torch
import numpy as np
from typing import Optional, List


def cycles(adjacent: torch.Tensor) -> List[List[int]]:
    """
    Find cycles in a directed graph represented by an adjacency matrix.
    
    Args:
        adjacent: [n, n] boolean adjacency matrix where adjacent[i, j] = True means arc i->j exists
    
    Returns:
        List of cycles, where each cycle is a list of node indices
    """
    n = adjacent.shape[0]
    visited = [False] * n
    stack = []
    result = []
    
    def dfs(node: int):
        if visited[node]:
            if node in stack:
                # Found a cycle
                idx = stack.index(node)
                result.append(stack[idx:])
            return
        
        visited[node] = True
        stack.append(node)
        
        # Visit all outgoing edges
        for next_node in range(n):
            if adjacent[node, next_node]:
                dfs(next_node)
        
        stack.pop()
    
    # Start DFS from each unvisited node
    for node in range(n):
        if not visited[node]:
            dfs(node)
    
    return result


def preprocess(scores: torch.Tensor) -> torch.Tensor:
    """
    Preprocess a scored adjacency matrix to fulfill dependency-tree constraints.
    
    Args:
        scores: [n, n] score matrix where scores[i, j] = score for arc i->j
    
    Returns:
        Preprocessed score matrix
    """
    n = scores.shape[0]
    
    # Suppress diagonal (no self-loops)
    scores[list(range(n)), list(range(n))] = scores.min() - 1
    
    # Only one root (incoming to position 0)
    root = scores[:, 0].argmax(-1)
    scores[:, 0] = scores.min() - 1
    scores[root, 0] = scores.max() + 1
    
    # Allow cycle w0 -> w0 (special handling for root)
    scores[0, 0] = scores.max()
    
    return scores


def contract(scores: torch.Tensor, cycle: List[int]) -> torch.Tensor:
    """
    Contract operation as described in Chu-Liu-Edmonds algorithm.
    Contracts a cycle into a single node.
    
    Args:
        scores: [n, n, 3] matrix of (score, dep_idx, head_idx)
        cycle: List of node indices forming a cycle
    
    Returns:
        Contracted score matrix
    """
    n = scores.shape[0]
    indices = torch.arange(n, device=scores.device).unsqueeze(-1).repeat(1, n)
    scores = torch.stack([scores, indices, indices.T], -1)
    pivot = min(cycle)
    
    # Update rows: for arcs coming INTO the cycle from outside
    for head in range(n):
        view = scores[cycle, head, 0]
        view = torch.where(view == 0, view.min(), view)
        dep = cycle[view.argmax()]
        scores[pivot, head] = scores[dep, head]
    
    # Update columns: for arcs going OUT of the cycle to outside
    for dep in range(n):
        view = scores[dep, cycle, 0]
        view = torch.where(view == 0, view.min(), view)
        head = cycle[view.argmax()]
        scores[dep, pivot] = scores[dep, head]
    
    cycle_copy = cycle.copy()
    cycle_copy.remove(pivot)
    maintain = [i for i in range(n) if i not in cycle_copy]
    
    return scores[maintain][:, maintain]


def expand(scores: torch.Tensor, adjacent: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """
    Expand operation to recover original nodes after contraction.
    
    Args:
        scores: [n, n, 3] indexed matrix of (score, dep, head)
        adjacent: Estimated adjacency matrix after contraction
        prev: Previous expanded adjacency matrix
    
    Returns:
        New expanded adjacency matrix
    """
    expanded = torch.zeros_like(prev, dtype=torch.bool)
    
    for idep, ihead in adjacent.nonzero():
        _, dep, head = scores[idep, ihead]
        expanded[dep.int().item(), head.int().item()] = True
    
    for dep in range(prev.shape[0]):
        if expanded[dep].sum() == 0:
            expanded[dep] = prev[dep]
    
    return expanded


def MST(scores: torch.Tensor) -> torch.Tensor:
    """
    Maximum Spanning Tree algorithm for finding optimal dependency tree.
    Based on Chu-Liu-Edmonds algorithm.
    
    Args:
        scores: [n, n] score matrix where scores[i, j] = score for head i -> dependent j
    
    Returns:
        [n, n] boolean adjacency matrix representing the MST
    """
    scores = preprocess(scores)
    n = scores.shape[0]
    heads = scores.argmax(-1)
    adjacent = torch.zeros(n, n, dtype=torch.bool, device=scores.device)
    adjacent[list(range(n)), heads] = True
    
    # Find and resolve cycles
    for cycle in cycles(adjacent):
        if cycle != [0]:
            # Contract the cycle
            scores = contract(scores - scores.max(-1).values.unsqueeze(-1), sorted(cycle))
            new = MST(scores[:, :, 0])
            adjacent = expand(scores, new, adjacent)
            break
    
    # Remove incoming arcs to root
    adjacent[0, :] = False
    
    return adjacent


class MSTDecoder:
    """
    MST-based decoder for NeCTI compound identification.
    Ensures decoded structures form valid dependency trees.
    """
    
    def __init__(self, label_set, no_rel_label: str = 'No_rel', root_label: str = 'Comp_root'):
        """
        Initialize MST decoder.
        
        Args:
            label_set: NeCTILabelSet object
            no_rel_label: Label for non-compound tokens
            root_label: Label for compound roots
        """
        self.label_set = label_set
        self.no_rel_label = no_rel_label
        self.root_label = root_label
        
        # Get label IDs
        self.no_rel_id = label_set.label2id(no_rel_label)
        self.root_id = label_set.label2id(root_label)
    
    def decode(self, logits: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                use_greedy: bool = True) -> torch.Tensor:
        """
        Decode logits using MST or greedy constraints.
        
        For NeCTI, we predict token-level labels (relation types), not explicit arc scores.
        MST works best when we have arc scores, so we provide two modes:
        1. use_greedy=True (default): Simpler constraint-based decoding
        2. use_greedy=False: Full MST with inferred arc scores
        
        Args:
            logits: [batch, seq_len, num_labels] model output logits
            attention_mask: [batch, seq_len] mask for valid positions
            use_greedy: If True, use simpler greedy constraint-based decoding
        
        Returns:
            predictions: [batch, seq_len] decoded label IDs
        """
        if use_greedy:
            return self.decode_greedy_with_constraints(logits, attention_mask)
        
        batch_size, seq_len, num_labels = logits.shape
        device = logits.device
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        predictions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            if attention_mask is not None:
                valid_len = int(attention_mask[b].sum().item())
            else:
                valid_len = seq_len
            
            if valid_len == 0:
                continue
            
            token_probs = probs[b, :valid_len]  # [valid_len, num_labels]
            
            # Build score matrix [valid_len, valid_len]
            # score[i, j] = probability that token i is the head of token j
            scores = torch.zeros(valid_len, valid_len, device=device)
            
            for j in range(valid_len):
                for i in range(valid_len):
                    if i == j:
                        scores[i, j] = -float('inf')  # No self-loops
                    else:
                        # If token j has a compound label (not No_rel), score the arc i->j
                        # Use the probability of the best compound type for token j
                        label_scores = token_probs[j].clone()
                        label_scores[self.no_rel_id] = 0.0
                        
                        # Prefer nearby heads (local bias for compounds)
                        distance_penalty = abs(i - j) * 0.01
                        scores[i, j] = label_scores.max() - distance_penalty
            
            # Apply MST algorithm
            try:
                # Ensure scores are valid
                if torch.isnan(scores).any() or torch.isinf(scores).all():
                    # Fallback to greedy
                    predictions[b, :valid_len] = token_probs.argmax(dim=-1)
                    continue
                
                # Add small epsilon to avoid numerical issues
                scores = scores + 1e-6
                adjacent = MST(scores)
                
                # Convert adjacency matrix to predictions
                for j in range(valid_len):
                    # Find head of token j
                    incoming = adjacent[:, j].nonzero(as_tuple=True)[0]
                    
                    if len(incoming) > 0:
                        # Has a head, so it's part of a compound
                        # Get the best non-No_rel label
                        label_probs = token_probs[j].clone()
                        label_probs[self.no_rel_id] = -float('inf')
                        predictions[b, j] = label_probs.argmax()
                    else:
                        # No head, so it's either root or No_rel
                        # Check if it's a root (has dependents)
                        outgoing = adjacent[j, :].sum()
                        if outgoing > 0:
                            predictions[b, j] = self.root_id
                        else:
                            predictions[b, j] = self.no_rel_id
            
            except Exception as e:
                # If MST fails, fall back to greedy decoding
                print(f"MST decoding failed for batch {b}: {e}. Using greedy fallback.")
                predictions[b, :valid_len] = token_probs.argmax(dim=-1)
        
        return predictions
    
    def decode_greedy_with_constraints(self, logits: torch.Tensor, 
                                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Greedy decoding with basic tree validity constraints.
        
        This is more reliable for token-level predictions than full MST,
        as it doesn't require inferring arc scores from labels.
        
        Constraints applied:
        1. Greedy selection of most probable label per token
        2. If no Comp_root predicted, assign to highest-confidence compound token
        3. If all No_rel, keep as is (valid for non-compound sentences)
        
        Args:
            logits: [batch, seq_len, num_labels]
            attention_mask: [batch, seq_len]
        
        Returns:
            predictions: [batch, seq_len]
        """
        batch_size, seq_len, num_labels = logits.shape
        device = logits.device
        
        # Greedy prediction
        probs = torch.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        
        # Apply constraints per sequence
        for b in range(batch_size):
            if attention_mask is not None:
                valid_len = int(attention_mask[b].sum().item())
            else:
                valid_len = seq_len
            
            if valid_len == 0:
                continue
            
            seq_preds = predictions[b, :valid_len]
            seq_probs = probs[b, :valid_len]
            
            # Check if we have any compound-related predictions
            has_root = (seq_preds == self.root_id).any()
            has_compound = ((seq_preds != self.no_rel_id) & (seq_preds != self.root_id)).any()
            
            # If we have compound members but no root, assign a root
            if has_compound and not has_root:
                # Find the token with highest confidence in a compound label
                compound_mask = (seq_preds != self.no_rel_id)
                if compound_mask.any():
                    # Get max probability for compound labels
                    compound_probs = seq_probs.clone()
                    compound_probs[:, self.no_rel_id] = 0.0
                    max_probs = compound_probs.max(dim=-1)[0]
                    max_probs[~compound_mask] = -float('inf')
                    
                    # Assign root to highest confidence compound token
                    root_idx = max_probs.argmax()
                    predictions[b, root_idx] = self.root_id
        
        return predictions


def apply_mst_to_predictions(predictions: torch.Tensor, 
                             label_set,
                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Post-process predictions to enforce MST constraints.
    
    Args:
        predictions: [batch, seq_len] predicted label IDs
        label_set: NeCTILabelSet object
        attention_mask: [batch, seq_len] valid token mask
    
    Returns:
        constrained_predictions: [batch, seq_len] with MST constraints applied
    """
    # Convert predictions to one-hot, apply MST, convert back
    batch_size, seq_len = predictions.shape
    num_labels = len(label_set)
    device = predictions.device
    
    # Create pseudo-logits from predictions
    logits = torch.zeros(batch_size, seq_len, num_labels, device=device)
    logits.scatter_(2, predictions.unsqueeze(-1), 1.0)
    
    decoder = MSTDecoder(label_set)
    return decoder.decode(logits, attention_mask)
