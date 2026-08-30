#!/usr/bin/env python3
"""
Evaluation script for Sanskrit compound identification.

Compares model predictions against gold standard annotations in CoNLL-U format.
"""

import json
import argparse
from typing import List, Dict, Set, Tuple
import re
import logging


class CompoundEvaluator:
    """Evaluates compound identification against gold standard."""
    
    # Map between different compound type notations
    TYPE_MAPPING = {
        'T6': 'Tatpurusha',
        'Tatpurusha': 'Tatpurusha',
        'BvS': 'Bahuvrihi',
        'Bs6': 'Bahuvrihi',
        'Bsu': 'Bahuvrihi',
        'Bahuvrihi': 'Bahuvrihi',
        'K1': 'Dvandva',
        'Ds': 'Dvandva',
        'Dvandva': 'Dvandva',
        'U': 'Avyayibhava',
        'Avyayibhava': 'Avyayibhava'
    }
    
    def __init__(self):
        self.results = {
            'total_sentences': 0,
            'total_gold_compounds': 0,
            'total_predicted_compounds': 0,
            'correct_span': 0,
            'correct_span_and_type': 0,
            'exact_matches': 0,  # Count of sentences with perfect predictions
            'by_type': {},
            'parsed_from_raw': 0,  # Sentences where compounds were parsed from raw_response
            'failed_to_parse': 0    # Sentences with no compounds and failed parsing
        }
    
    def normalize_type(self, compound_type: str) -> str:
        """Normalize compound type notation."""
        return self.TYPE_MAPPING.get(compound_type, compound_type)

    def parse_raw_response(self, raw_response: str) -> List[Dict]:
        """
        Parse compounds from raw response text.

        Handles cases where the response may be:
        - Wrapped in markdown code blocks (```json ... ```)
        - Truncated or incomplete JSON

        Args:
            raw_response: Raw response string that may contain JSON

        Returns:
            List of compound dictionaries, or empty list if parsing fails
        """
        if not raw_response:
            return []

        # Strip markdown code blocks
        response = raw_response.strip()
        if response.startswith('```'):
            # Remove opening ```json or ```
            response = re.sub(r'^```(?:json)?\s*\n?', '', response)
            # Remove closing ```
            response = re.sub(r'\n?```\s*$', '', response)

        response = response.strip()

        # Try to parse as-is first
        try:
            data = json.loads(response)
            return data.get('compounds', [])
        except json.JSONDecodeError:
            pass

        # Try to fix truncated JSON by completing incomplete objects
        # Look for the last complete compound entry
        if '"compounds":' in response:
            try:
                # Find all complete compound entries
                compound_pattern = r'\{[^{}]*"start"[^{}]*"end"[^{}]*"type"[^{}]*"text"[^{}]*\}'
                compounds = []
                for match in re.finditer(compound_pattern, response):
                    try:
                        compound = json.loads(match.group(0))
                        compounds.append(compound)
                    except json.JSONDecodeError:
                        continue
                return compounds
            except Exception as e:
                logging.debug(f"Failed to extract compounds from raw response: {e}")

        return []

    def extract_gold_compounds(self, conllu_lines: List[str]) -> List[Dict]:
        """
        Extract gold compound annotations from CoNLL-U format.
        
        Identifies compounds by tracking Comp2 and Comp3 markers and their boundaries.
        
        Args:
            conllu_lines: Lines for one sentence in CoNLL-U format
            
        Returns:
            List of gold compound dictionaries
        """
        compounds = []
        words = []
        compound_groups = {}  # Track compound groups (C2, C3, etc.)
        
        for line in conllu_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 10:
                continue
            
            # Skip DUMMY nodes
            if fields[1] == 'DUMMY':
                continue
            
            idx = len(words)
            word_form = fields[1]
            clean_word = re.sub(r'_C[0-9N]$', '', word_form)
            words.append(clean_word)
            
            # Check for compound markers
            comp_marker = None
            if '_C2' in word_form:
                comp_marker = 'C2'
            elif '_C3' in word_form:
                comp_marker = 'C3'
            
            if comp_marker:
                # Get compound type from dependency relation (field 7)
                comp_type = fields[7]
                
                # Track position in compound from misc field (field 9)
                misc_field = fields[9]
                
                if comp_marker not in compound_groups:
                    compound_groups[comp_marker] = {
                        'indices': [],
                        'type': self.normalize_type(comp_type),
                        'words': []
                    }
                
                compound_groups[comp_marker]['indices'].append(idx)
                compound_groups[comp_marker]['words'].append(clean_word)
                
                # Update type if this has a more specific type
                if comp_type not in ['Comp_root', 'No_rel']:
                    compound_groups[comp_marker]['type'] = self.normalize_type(comp_type)
        
        # Convert compound groups to compound entries
        for group_id, group_data in compound_groups.items():
            if len(group_data['indices']) < 2:
                continue
            
            indices = sorted(group_data['indices'])
            compounds.append({
                'start': indices[0],
                'end': indices[-1] + 1,  # Exclusive end
                'type': group_data['type'],
                'text': ' '.join(group_data['words'])
            })
        
        return compounds
    
    def compute_metrics(self, gold: List[Dict], predicted: List[Dict]) -> Dict:
        """
        Compute evaluation metrics.

        Args:
            gold: List of gold compound annotations
            predicted: List of predicted compounds

        Returns:
            Dictionary of metrics
        """
        gold_spans = {(c['start'], c['end']) for c in gold}
        pred_spans = {(c['start'], c['end']) for c in predicted}

        gold_full = {(c['start'], c['end'], c['type']) for c in gold}
        pred_full = {(c['start'], c['end'], c['type']) for c in predicted}

        correct_span = len(gold_spans & pred_spans)
        correct_full = len(gold_full & pred_full)

        # Exact match: all compounds in sentence are correct
        exact_match = 1 if gold_full == pred_full and len(gold_full) > 0 else 0
        # If there are no compounds in gold and no predictions, that's also exact match
        if len(gold_full) == 0 and len(pred_full) == 0:
            exact_match = 1

        # Per-type metrics
        type_metrics = {}
        for compound_type in set([c['type'] for c in gold] + [c['type'] for c in predicted]):
            gold_type = {(c['start'], c['end']) for c in gold if c['type'] == compound_type}
            pred_type = {(c['start'], c['end']) for c in predicted if c['type'] == compound_type}

            tp = len(gold_type & pred_type)
            fp = len(pred_type - gold_type)
            fn = len(gold_type - pred_type)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            type_metrics[compound_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': len(gold_type)
            }

        return {
            'gold_count': len(gold),
            'pred_count': len(predicted),
            'correct_span': correct_span,
            'correct_full': correct_full,
            'exact_match': exact_match,
            'span_precision': correct_span / len(pred_spans) if pred_spans else 0,
            'span_recall': correct_span / len(gold_spans) if gold_spans else 0,
            'full_precision': correct_full / len(pred_full) if pred_full else 0,
            'full_recall': correct_full / len(gold_full) if gold_full else 0,
            'type_metrics': type_metrics
        }
    
    def evaluate_file(self, conllu_file: str, predictions_file: str, num_sentences: int = None) -> Dict:
        """
        Evaluate predictions against gold standard.

        Args:
            conllu_file: Path to CoNLL-U file with gold annotations
            predictions_file: Path to JSON file with predictions
            num_sentences: Optional number of sentences to evaluate (None = all)

        Returns:
            Evaluation results
        """
        # Load predictions
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # Read CoNLL-U sentences
        sentences = []
        current_sentence = []
        
        with open(conllu_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    current_sentence.append(line)
        
        if current_sentence:
            sentences.append(current_sentence)

        # Limit number of sentences if specified
        if num_sentences is not None:
            sentences = sentences[:num_sentences]
            predictions = predictions[:num_sentences]

        # Evaluate each sentence
        all_metrics = []

        for i, (sent_lines, pred_data) in enumerate(zip(sentences, predictions), 1):
            gold_compounds = self.extract_gold_compounds(sent_lines)

            # Try to get compounds, falling back to parsing raw_response if needed
            pred_compounds = pred_data.get('compounds', [])
            parsed_from_raw = False

            if not pred_compounds and 'raw_response' in pred_data:
                pred_compounds = self.parse_raw_response(pred_data['raw_response'])
                if pred_compounds:
                    parsed_from_raw = True
                    self.results['parsed_from_raw'] += 1
                else:
                    self.results['failed_to_parse'] += 1

            metrics = self.compute_metrics(gold_compounds, pred_compounds)
            metrics['sentence_id'] = i
            metrics['sentence'] = pred_data.get('sentence', '')
            metrics['parsed_from_raw'] = parsed_from_raw
            
            all_metrics.append(metrics)

            # Update global results
            self.results['total_sentences'] += 1
            self.results['total_gold_compounds'] += metrics['gold_count']
            self.results['total_predicted_compounds'] += metrics['pred_count']
            self.results['correct_span'] += metrics['correct_span']
            self.results['correct_span_and_type'] += metrics['correct_full']
            self.results['exact_matches'] += metrics['exact_match']
        
        # Compute overall metrics
        overall = {
            'span_precision': self.results['correct_span'] / self.results['total_predicted_compounds'] if self.results['total_predicted_compounds'] > 0 else 0,
            'span_recall': self.results['correct_span'] / self.results['total_gold_compounds'] if self.results['total_gold_compounds'] > 0 else 0,
            'full_precision': self.results['correct_span_and_type'] / self.results['total_predicted_compounds'] if self.results['total_predicted_compounds'] > 0 else 0,
            'full_recall': self.results['correct_span_and_type'] / self.results['total_gold_compounds'] if self.results['total_gold_compounds'] > 0 else 0,
        }

        overall['span_f1'] = 2 * overall['span_precision'] * overall['span_recall'] / (overall['span_precision'] + overall['span_recall']) if (overall['span_precision'] + overall['span_recall']) > 0 else 0
        overall['full_f1'] = 2 * overall['full_precision'] * overall['full_recall'] / (overall['full_precision'] + overall['full_recall']) if (overall['full_precision'] + overall['full_recall']) > 0 else 0

        # Add USS, LSS, and EM metrics (aliases for clarity)
        overall['uss'] = overall['span_f1']  # USS = Unlabeled Span Score
        overall['lss'] = overall['full_f1']   # LSS = Labeled Span Score
        overall['lss_precision'] = overall['full_precision']
        overall['lss_recall'] = overall['full_recall']
        overall['exact_match'] = self.results['exact_matches'] / self.results['total_sentences'] if self.results['total_sentences'] > 0 else 0
        
        return {
            'overall': overall,
            'summary': self.results,
            'per_sentence': all_metrics
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Sanskrit compound identification predictions"
    )
    parser.add_argument(
        "conllu_file",
        help="Gold standard CoNLL-U file"
    )
    parser.add_argument(
        "predictions_file",
        help="Predictions JSON file from identifier script"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file for evaluation results (JSON)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sentence results"
    )
    parser.add_argument(
        "-n", "--num-sentences",
        type=int,
        default=None,
        help="Number of sentences to evaluate (default: all)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for parsing issues"
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    evaluator = CompoundEvaluator()
    results = evaluator.evaluate_file(args.conllu_file, args.predictions_file, args.num_sentences)
    
    # Print overall results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nTotal Sentences: {results['summary']['total_sentences']}")
    print(f"Gold Compounds: {results['summary']['total_gold_compounds']}")
    print(f"Predicted Compounds: {results['summary']['total_predicted_compounds']}")

    # Show parsing stats if there were issues
    if results['summary']['parsed_from_raw'] > 0 or results['summary']['failed_to_parse'] > 0:
        print(f"\nParsing Stats:")
        print(f"  Recovered from raw_response: {results['summary']['parsed_from_raw']}")
        print(f"  Failed to parse: {results['summary']['failed_to_parse']}")

    print(f"\nCorrect Spans: {results['summary']['correct_span']}")
    print(f"Correct Span + Type: {results['summary']['correct_span_and_type']}")
    
    print("\n" + "-"*80)
    print("SPAN-ONLY METRICS (USS - Unlabeled Span Score)")
    print("-"*80)
    print(f"Precision: {results['overall']['span_precision']:.4f}")
    print(f"Recall:    {results['overall']['span_recall']:.4f}")
    print(f"F1 Score:  {results['overall']['span_f1']:.4f} (USS)")

    print("\n" + "-"*80)
    print("FULL METRICS (LSS - Labeled Span Score)")
    print("-"*80)
    print(f"Precision: {results['overall']['full_precision']:.4f}")
    print(f"Recall:    {results['overall']['full_recall']:.4f}")
    print(f"F1 Score:  {results['overall']['full_f1']:.4f} (LSS)")

    print("\n" + "-"*80)
    print("EXACT MATCH (EM)")
    print("-"*80)
    print(f"Exact Match: {results['overall']['exact_match']:.4f} ({results['summary']['exact_matches']}/{results['summary']['total_sentences']} sentences)")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"USS (Unlabeled Span F1):    {results['overall']['uss']:.4f}")
    print(f"LSS (Labeled Span F1):      {results['overall']['lss']:.4f}")
    print(f"LSS Precision:              {results['overall']['lss_precision']:.4f}")
    print(f"LSS Recall:                 {results['overall']['lss_recall']:.4f}")
    print(f"Exact Match (EM):           {results['overall']['exact_match']:.4f}")
    print("="*80)
    
    if args.verbose:
        print("\n" + "="*80)
        print("PER-SENTENCE RESULTS")
        print("="*80)
        for sent_metrics in results['per_sentence']:
            print(f"\nSentence {sent_metrics['sentence_id']}: {sent_metrics['sentence']}")
            print(f"  Gold: {sent_metrics['gold_count']}, Predicted: {sent_metrics['pred_count']}")
            print(f"  Correct spans: {sent_metrics['correct_span']}, Correct full: {sent_metrics['correct_full']}")
    
    # Save to file
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()