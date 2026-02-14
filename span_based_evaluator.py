"""
CORRECT Span-Based Evaluation for NeCTI (matching DepNeCTI paper specification)

This implements TRUE span-based USS, LSS, and EM as described in the paper:
- USS: F1 on span BOUNDARIES only (start, end)
- LSS: F1 on span TUPLES (start, end, label)
- EM: Exact match of compounds (all spans correct)
"""

from typing import List, Dict, Set, Tuple
import json


class SpanBasedEvaluator:
    """Span-based evaluation matching DepNeCTI paper specification"""
    
    @staticmethod
    def extract_spans_from_labels(token_ids: List[int], labels: List[str]) -> List[Tuple[int, int, str]]:
        """
        Extract compound spans from token-level labels.
        
        Expected label format: "Comp3_Start", "Comp3_Middle", "Comp_root", etc.
        
        Args:
            token_ids: List of token IDs (1-indexed)
            labels: List of compound labels for each token
        
        Returns:
            List of (start_token, end_token, compound_type) tuples
            E.g., [(1, 3, "Comp3"), (4, 5, "Comp2")]
        """
        spans = []
        compound_stack = {}  # compound_level -> (start_token, compound_type)
        
        for token_id, label in zip(token_ids, labels):
            if label in ['No_rel', 'root']:
                continue
            
            # Extract compound level and position marker
            if 'Comp' in label:
                # Parse label like "Comp3_Start" or "Comp_root"
                parts = label.split('_')
                comp_level = parts[0]  # "Comp3", "Comp2", etc.
                
                if len(parts) > 1:
                    position = parts[1]  # "Start", "Middle", "End"
                else:
                    position = "root"  # Handle "Comp_root"
                
                if position == "Start":
                    # Mark start of compound
                    compound_stack[comp_level] = (token_id, comp_level)
                
                elif position in ["End", "root"] and "Comp_root" not in label:
                    # End of compound (but not the root marker itself)
                    if comp_level in compound_stack:
                        start_token, _ = compound_stack.pop(comp_level)
                        spans.append((start_token, token_id, comp_level))
                
                elif "Comp_root" in label:
                    # This is the root marker - end all open compounds
                    # Close all open compounds at this token
                    for comp_level, (start_token, _) in list(compound_stack.items()):
                        spans.append((start_token, token_id, comp_level))
                        compound_stack.pop(comp_level)
        
        # Close any remaining open compounds
        for comp_level, (start_token, _) in compound_stack.items():
            spans.append((start_token, len(labels), comp_level))
        
        return sorted(spans, key=lambda x: (x[0], x[1]))
    
    @staticmethod
    def extract_spans_from_predictions(predictions: List[int], 
                                      label_set,
                                      token_ids: List[int]) -> List[Tuple[int, int, str]]:
        """
        Extract spans from model predictions (token-level labels).
        
        Args:
            predictions: List of predicted label IDs
            label_set: NeCTILabelSet object with id2label mapping
            token_ids: List of token IDs
        
        Returns:
            List of (start, end, label) spans
        """
        # Convert prediction IDs to label strings
        labels = [label_set.id2label(pred_id) for pred_id in predictions]
        return SpanBasedEvaluator.extract_spans_from_labels(token_ids, labels)
    
    @staticmethod
    def calculate_uss(true_spans: List[List[Tuple]], pred_spans: List[List[Tuple]]) -> float:
        """
        Calculate Unlabeled Span Score (USS).
        
        USS = F1 on (start_token, end_token) boundaries, ignoring labels
        
        Args:
            true_spans: List of true span sets for each sentence
            pred_spans: List of predicted span sets for each sentence
        
        Returns:
            F1 score (0-1)
        """
        total_correct = 0
        total_pred = 0
        total_true = 0
        
        for true_sent_spans, pred_sent_spans in zip(true_spans, pred_spans):
            # Extract only boundaries (start, end), ignore label
            true_boundaries = set((s[0], s[1]) for s in true_sent_spans)
            pred_boundaries = set((s[0], s[1]) for s in pred_sent_spans)
            
            # Count matches
            correct = len(true_boundaries & pred_boundaries)
            total_correct += correct
            total_pred += len(pred_boundaries)
            total_true += len(true_boundaries)
        
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    @staticmethod
    def calculate_lss(true_spans: List[List[Tuple]], pred_spans: List[List[Tuple]]) -> float:
        """
        Calculate Labeled Span Score (LSS).
        
        LSS = F1 on (start_token, end_token, label) tuples
        
        Args:
            true_spans: List of true span sets for each sentence
            pred_spans: List of predicted span sets for each sentence
        
        Returns:
            F1 score (0-1)
        """
        total_correct = 0
        total_pred = 0
        total_true = 0
        
        for true_sent_spans, pred_sent_spans in zip(true_spans, pred_spans):
            true_spans_set = set(true_sent_spans)
            pred_spans_set = set(pred_sent_spans)
            
            # Count matches (requires start, end, AND label to match)
            correct = len(true_spans_set & pred_spans_set)
            total_correct += correct
            total_pred += len(pred_spans_set)
            total_true += len(true_spans_set)
        
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    @staticmethod
    def calculate_exact_match(true_spans: List[List[Tuple]], pred_spans: List[List[Tuple]]) -> float:
        """
        Calculate Exact Match (EM).
        
        EM = percentage of sentences where ALL spans match perfectly
        
        Args:
            true_spans: List of true span sets for each sentence
            pred_spans: List of predicted span sets for each sentence
        
        Returns:
            EM score (0-1)
        """
        total_sentences = len(true_spans)
        correct_sentences = 0
        
        for true_sent_spans, pred_sent_spans in zip(true_spans, pred_spans):
            true_set = set(true_sent_spans)
            pred_set = set(pred_sent_spans)
            
            # Perfect match if both sets are identical
            if true_set == pred_set:
                correct_sentences += 1
        
        return correct_sentences / total_sentences if total_sentences > 0 else 0.0
    
    @staticmethod
    def evaluate_batch(true_spans_list: List[List[Tuple]], 
                      pred_spans_list: List[List[Tuple]]) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Returns:
            Dict with USS, LSS, and EM scores
        """
        uss = SpanBasedEvaluator.calculate_uss(true_spans_list, pred_spans_list)
        lss = SpanBasedEvaluator.calculate_lss(true_spans_list, pred_spans_list)
        em = SpanBasedEvaluator.calculate_exact_match(true_spans_list, pred_spans_list)
        
        return {
            'USS': uss,
            'LSS': lss,
            'EM': em
        }


# Example usage for integration with DiffusionSL

def extract_all_spans_from_conllu_file(filepath: str, label_set) -> Tuple[List[List[Tuple]], List[int]]:
    """
    Extract spans from a CoNLL-U format file.
    
    Args:
        filepath: Path to CoNLL-U file
        label_set: NeCTILabelSet for label ID conversion
    
    Returns:
        (spans_list, token_ids_list) where each element corresponds to a sentence
    """
    all_spans = []
    all_token_ids = []
    
    current_tokens = []
    current_labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                # Empty line indicates sentence boundary
                if current_tokens:
                    token_ids = list(range(1, len(current_tokens) + 1))
                    spans = SpanBasedEvaluator.extract_spans_from_labels(token_ids, current_labels)
                    all_spans.append(spans)
                    all_token_ids.append(token_ids)
                    current_tokens = []
                    current_labels = []
            
            elif not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 9:  # Valid CoNLL-U line
                    current_tokens.append(parts[1])
                    current_labels.append(parts[9])  # Compound label column
        
        # Handle last sentence if file doesn't end with empty line
        if current_tokens:
            token_ids = list(range(1, len(current_tokens) + 1))
            spans = SpanBasedEvaluator.extract_spans_from_labels(token_ids, current_labels)
            all_spans.append(spans)
            all_token_ids.append(token_ids)
    
    return all_spans, all_token_ids


def evaluate_predictions_file(true_file: str, pred_file: str, label_set) -> Dict[str, float]:
    """
    Evaluate predictions using span-based metrics.
    
    Args:
        true_file: Path to true labels in CoNLL-U format
        pred_file: Path to predicted labels in CoNLL-U format
        label_set: NeCTILabelSet object
    
    Returns:
        Dict with USS, LSS, EM scores
    """
    true_spans, _ = extract_all_spans_from_conllu_file(true_file, label_set)
    pred_spans, _ = extract_all_spans_from_conllu_file(pred_file, label_set)
    
    return SpanBasedEvaluator.evaluate_batch(true_spans, pred_spans)


# Test example
if __name__ == '__main__':
    # Example usage
    evaluator = SpanBasedEvaluator()
    
    # Simulated data
    true_labels = ["Comp3_Start", "Comp3_Middle", "Comp3_Middle", "Comp_root", 
                   "Comp2_Start", "Comp2_Middle", "Comp_root"]
    pred_labels = ["Comp3_Start", "Comp3_Middle", "Comp3_Middle", "Comp_root",
                   "Comp2_Start", "Comp3_Middle", "Comp_root"]  # One error
    
    token_ids = list(range(1, len(true_labels) + 1))
    
    true_spans = [evaluator.extract_spans_from_labels(token_ids, true_labels)]
    pred_spans = [evaluator.extract_spans_from_labels(token_ids, pred_labels)]
    
    print("True spans:", true_spans)
    print("Pred spans:", pred_spans)
    
    results = evaluator.evaluate_batch(true_spans, pred_spans)
    print("\nMetrics:")
    print(f"  USS: {results['uss']:.4f}")
    print(f"  LSS: {results['lss']:.4f}")
    print(f"  EM:  {results['exact_match']:.4f}")
