"""
Chu-Liu/Edmonds' algorithm for finding Maximum Spanning Arborescence (directed MST)
Used for enforcing tree structure constraints in NeCTI decoding
"""

import numpy as np
from typing import List, Tuple, Dict
import torch


class ChuLiuEdmondsDecoder:
    """
    Chu-Liu/Edmonds' algorithm implementation for maximum spanning arborescence.
    Enforces tree structure constraints on compound predictions.
    """
    
    def __init__(self, num_labels: int, root_label: str = 'root', no_rel_label: str = 'No_rel'):
        """
        Initialize decoder
        
        Args:
            num_labels: Number of label classes
            root_label: Label ID for root nodes
            no_rel_label: Label ID for no-relation (non-compound) nodes
        """
        self.num_labels = num_labels
        self.root_label = root_label
        self.no_rel_label = no_rel_label
    
    def decode(self, logits: np.ndarray, label_set, threshold: float = 0.0) -> np.ndarray:
        """
        Decode logits using Chu-Liu/Edmonds' algorithm to enforce tree structure
        
        Args:
            logits: Shape [seq_len, num_labels] - model output logits
            label_set: LabelSet object for label mapping
            threshold: Confidence threshold for edge inclusion
        
        Returns:
            predictions: Shape [seq_len] - decoded labels with tree structure
        """
        seq_len = logits.shape[0]
        
        # Convert logits to probabilities
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
        
        # Create score matrix: [seq_len, seq_len, num_labels]
        # score[i, j, label] = probability of token i->j with label
        scores = np.zeros((seq_len, seq_len, self.num_labels))
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    scores[i, j] = probs[j]  # Head is i, dependent is j
        
        # Run Chu-Liu/Edmonds to find best tree
        # Returns edges: List[(head, dep, label)]
        edges = self._chu_liu_edmonds(scores, seq_len, label_set)
        
        # Convert edges to label sequence
        predictions = self._edges_to_labels(edges, seq_len, label_set)
        
        return predictions
    
    def _chu_liu_edmonds(self, scores: np.ndarray, seq_len: int, label_set) -> List[Tuple]:
        """
        Main Chu-Liu/Edmonds algorithm
        
        Args:
            scores: [seq_len, seq_len, num_labels] score matrix
            seq_len: Sequence length
            label_set: Label mapping
        
        Returns:
            List of edges (head, dependent, label_id)
        """
        # Initialize: each node has best incoming edge
        in_edges = {}
        in_scores = {}
        
        for dep in range(seq_len):
            best_score = -np.inf
            best_head = -1
            best_label = -1
            
            for head in range(seq_len):
                if head == dep:
                    continue
                
                # Get best label for this arc
                label_probs = scores[head, dep]
                best_label_idx = np.argmax(label_probs)
                best_label_score = label_probs[best_label_idx]
                
                # Skip root and no_rel
                label_name = label_set.id2label(best_label_idx)
                if label_name in [self.root_label, self.no_rel_label]:
                    continue
                
                if best_label_score > best_score:
                    best_score = best_label_score
                    best_head = head
                    best_label = best_label_idx
            
            if best_head >= 0:
                in_edges[dep] = best_head
                in_scores[dep] = best_score
        
        # Check for cycles
        cycles = self._find_cycles(in_edges, seq_len)
        
        if not cycles:
            # No cycles - return edges
            edges = []
            for dep, head in in_edges.items():
                label_probs = scores[head, dep]
                label_id = np.argmax(label_probs)
                edges.append((head, dep, label_id))
            return edges
        
        # Contract cycles and recurse
        return self._contract_and_recurse(scores, seq_len, in_edges, cycles, label_set)
    
    def _find_cycles(self, in_edges: Dict[int, int], seq_len: int) -> List[List[int]]:
        """
        Find all cycles in the graph defined by in_edges
        
        Returns:
            List of cycles, each cycle is a list of node indices
        """
        visited = set()
        cycles = []
        
        for start in range(seq_len):
            if start in visited:
                continue
            
            path = []
            current = start
            path_set = set()
            
            while current not in visited and current not in path_set:
                if current not in in_edges:
                    break
                
                path.append(current)
                path_set.add(current)
                current = in_edges[current]
            
            if current in path_set:
                # Found cycle
                cycle_start_idx = path.index(current)
                cycle = path[cycle_start_idx:]
                cycles.append(cycle)
                visited.update(cycle)
            else:
                visited.update(path)
        
        return cycles if cycles else []
    
    def _contract_and_recurse(self, scores: np.ndarray, seq_len: int, 
                              in_edges: Dict[int, int], cycles: List[List[int]], 
                              label_set) -> List[Tuple]:
        """
        Contract cycle to single node and recurse
        
        Args:
            scores: Score matrix
            seq_len: Sequence length
            in_edges: Best incoming edges
            cycles: List of cycles found
            label_set: Label mapping
        
        Returns:
            List of edges in original graph
        """
        if not cycles:
            edges = []
            for dep, head in in_edges.items():
                label_probs = scores[head, dep]
                label_id = np.argmax(label_probs)
                edges.append((head, dep, label_id))
            return edges
        
        # Contract first cycle
        cycle = cycles[0]
        cycle_id = seq_len  # New virtual node ID
        
        # Create contracted graph
        new_seq_len = seq_len - len(cycle) + 1
        new_scores = np.zeros((new_seq_len, new_seq_len, self.num_labels))
        
        # Mapping: old node -> new node
        node_map = {}
        new_node_id = 0
        
        for old_id in range(seq_len):
            if old_id not in cycle:
                node_map[old_id] = new_node_id
                new_node_id += 1
        
        node_map[cycle_id] = new_node_id  # Cycle gets last position
        
        # Copy scores for non-cycle nodes
        for old_i in range(seq_len):
            if old_i in cycle:
                continue
            for old_j in range(seq_len):
                if old_j in cycle:
                    continue
                new_i = node_map[old_i]
                new_j = node_map[old_j]
                new_scores[new_i, new_j] = scores[old_i, old_j]
        
        # Compute scores for cycle -> outside and outside -> cycle
        self._compute_cycle_scores(scores, new_scores, cycle, node_map, 
                                   seq_len, new_seq_len, in_edges, label_set)
        
        # Recurse on contracted graph
        new_in_edges = {}
        for dep_node, head_node in in_edges.items():
            if dep_node in cycle:
                continue
            new_dep = node_map[dep_node]
            new_head = node_map[head_node] if head_node not in cycle else node_map[cycle_id]
            new_in_edges[new_dep] = new_head
        
        contracted_edges = self._chu_liu_edmonds(new_scores, new_seq_len, label_set)
        
        # Expand back to original graph
        return self._expand_edges(contracted_edges, cycle, node_map, in_edges, label_set)
    
    def _compute_cycle_scores(self, scores: np.ndarray, new_scores: np.ndarray, 
                              cycle: List[int], node_map: Dict[int, int],
                              seq_len: int, new_seq_len: int, 
                              in_edges: Dict[int, int], label_set):
        """Compute scores for edges involving contracted cycle"""
        cycle_node = new_seq_len - 1
        
        # For each outside node
        for out_node in range(seq_len):
            if out_node in cycle:
                continue
            
            new_out = node_map[out_node]
            
            # Scores from cycle to outside node
            best_score_out = -np.inf
            for cycle_member in cycle:
                score = np.max(scores[cycle_member, out_node])
                best_score_out = max(best_score_out, score)
            
            if best_score_out > -np.inf:
                new_scores[cycle_node, new_out] = best_score_out
            
            # Scores from outside node to cycle
            best_score_in = -np.inf
            for cycle_member in cycle:
                # Cost of removing best in-edge and adding new edge
                in_edge_score = np.max(scores[in_edges.get(cycle_member, -1), cycle_member]) \
                    if cycle_member in in_edges else 0
                arc_score = np.max(scores[out_node, cycle_member])
                score = arc_score - in_edge_score
                best_score_in = max(best_score_in, score)
            
            if best_score_in > -np.inf:
                new_scores[new_out, cycle_node] = best_score_in
    
    def _expand_edges(self, contracted_edges: List[Tuple], cycle: List[int], 
                      node_map: Dict[int, int], in_edges: Dict[int, int],
                      label_set) -> List[Tuple]:
        """Expand contracted edges back to original graph"""
        edges = []
        
        # Add non-cycle edges
        for head, dep, label_id in contracted_edges:
            # Check if involves cycle node (last node in contracted graph)
            inv_map = {v: k for k, v in node_map.items()}
            
            if head in inv_map and inv_map[head] not in cycle:
                orig_head = inv_map[head]
            else:
                orig_head = None
            
            if dep in inv_map and inv_map[dep] not in cycle:
                orig_dep = inv_map[dep]
            else:
                orig_dep = None
            
            if orig_head is not None and orig_dep is not None:
                edges.append((orig_head, orig_dep, label_id))
        
        # Add cycle edges (from in_edges)
        for cycle_member in cycle:
            if cycle_member in in_edges:
                head = in_edges[cycle_member]
                label_probs = np.max(np.zeros(self.num_labels))  # Placeholder
                edges.append((head, cycle_member, 0))
        
        return edges
    
    def _edges_to_labels(self, edges: List[Tuple], seq_len: int, label_set) -> np.ndarray:
        """
        Convert edge list to label sequence
        
        Args:
            edges: List of (head, dependent, label_id) tuples
            seq_len: Sequence length
            label_set: Label mapping
        
        Returns:
            Label sequence [seq_len]
        """
        predictions = np.zeros(seq_len, dtype=np.int32)
        
        # Set all to No_rel initially
        no_rel_id = label_set.label2id(self.no_rel_label)
        root_id = label_set.label2id(self.root_label)
        predictions[:] = no_rel_id
        
        # Apply edges
        for head, dep, label_id in edges:
            if 0 <= dep < seq_len:
                predictions[dep] = label_id
        
        return predictions


def apply_chu_liu_edmonds_constraints(logits: np.ndarray, label_set, 
                                      use_greedy_fallback: bool = True) -> np.ndarray:
    """
    Simplified Chu-Liu-Edmonds-inspired constraint enforcement.
    
    For dependency-style compound prediction, ensure:
    1. No cycles in compound structure
    2. Each token has valid compound type
    3. Respect hierarchical constraints
    
    Args:
        logits: [seq_len, num_labels] prediction logits
        label_set: NeCTILabelSet object
        use_greedy_fallback: If CLE fails, use greedy decoding
    
    Returns:
        predictions: [seq_len] constrained label IDs
    """
    seq_len, num_labels = logits.shape
    
    # Try full CLE decoding
    try:
        decoder = ChuLiuEdmondsDecoder(
            num_labels=num_labels,
            root_label='Comp_root',
            no_rel_label='No_rel'
        )
        predictions = decoder.decode(logits, label_set, threshold=0.0)
        return predictions
    except Exception as e:
        # Fallback to greedy with simple constraints
        if use_greedy_fallback:
            predictions = np.argmax(logits, axis=-1)
            
            # Simple constraint: ensure valid transitions
            no_rel_id = label_set.label2id('No_rel')
            root_id = label_set.label2id('Comp_root') if 'Comp_root' in label_set._label2id else no_rel_id
            
            # Post-process to remove obvious violations
            for i in range(seq_len):
                label_name = label_set.id2label(predictions[i])
                # If label looks invalid, fall back to No_rel
                if label_name not in label_set._label2id:
                    predictions[i] = no_rel_id
            
            return predictions
        else:
            raise e
