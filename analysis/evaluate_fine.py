#!/usr/bin/env python3
"""
Fine-Grained Evaluation for Sanskrit Compound Identification

Evaluates predictions at both span level and fine-grained type level.
"""

import json
import argparse
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re
import logging


class FineGrainedEvaluator:
    """Evaluates fine-grained compound identification."""
    
    # Normalize type codes
    TYPE_GROUPS = {
        # Tatpurusha variants
        'T': 'Tatpurusha', 'T1': 'T1', 'T2': 'T2', 'T3': 'T3', 'T4': 'T4', 
        'T5': 'T5', 'T6': 'T6', 'T7': 'T7',
        'Tm': 'Tm', 'Tn': 'Tn', 'Tp': 'Tp', 'Tds': 'Tds', 'Tdt': 'Tdt', 
        'Tdu': 'Tdu', 'Tg': 'Tg', 'Tk': 'Tk',
        
        # Karmadharaya variants
        'K': 'Karmadharaya', 'K1': 'K1', 'K2': 'K2', 'K3': 'K3', 'K4': 'K4',
        'K5': 'K5', 'K6': 'K6', 'K7': 'K7', 'Km': 'Km',
        
        # Dvandva variants
        'D': 'Dvandva', 'Di': 'Di', 'Ds': 'Ds',
        
        # Bahuvrihi variants
        'Bv': 'Bahuvrihi', 'Bs': 'Bs', 'Bs2': 'Bs2', 'Bs3': 'Bs3', 'Bs5': 'Bs5',
        'Bs6': 'Bs6', 'Bs7': 'Bs7', 'Bsu': 'Bsu', 'Bss': 'Bss', 'BvS': 'BvS',
        'Bvp': 'Bvp', 'Bvs': 'Bvs', 'Bsmn': 'Bsmn', 'Bb': 'Bb', 'Bsp': 'Bsp',
        
        # Avyayibhava / Dvigu
        'U': 'U',
        
        # Aluk
        'A1': 'A1', 'A2': 'A2', 'A7': 'A7'
    }
    
    def __init__(self):
        self.results = {
            'total_sentences': 0,
            'total_gold_compounds': 0,
            'total_predicted_compounds': 0,
            'span_correct': 0,
            'fine_grained_correct': 0,
            'coarse_correct': 0,
            'exact_matches': 0,  # Count of sentences with perfect predictions
            'by_type': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
            'parsed_from_raw': 0,  # Sentences where compounds were parsed from raw_response
            'failed_to_parse': 0    # Sentences with no compounds and failed parsing
        }
    
    def get_coarse_type(self, fine_type: str) -> str:
        """Map fine-grained type to coarse category."""
        if fine_type.startswith('T'):
            return 'Tatpurusha'
        elif fine_type.startswith('K'):
            return 'Karmadharaya'
        elif fine_type.startswith('D'):
            return 'Dvandva'
        elif fine_type.startswith('B'):
            return 'Bahuvrihi'
        elif fine_type == 'U':
            return 'Avyayibhava'
        elif fine_type.startswith('A'):
            return 'Aluk'
        else:
            return fine_type

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
        # For fine-grained, we need to handle both "type" and optional "subtype"
        if '"compounds":' in response:
            try:
                # Find all complete compound entries (with or without subtype)
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
        """Extract gold fine-grained compound annotations from CoNLL-U."""
        compounds = []
        words = []
        compound_groups = {}
        
        for line in conllu_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 10:
                continue
            
            if fields[1] == 'DUMMY':
                continue
            
            idx = len(words)
            word_form = fields[1]
            clean_word = re.sub(r'_C[0-9]+$', '', word_form)
            words.append(clean_word)
            
            # Extract compound group
            comp_marker = None
            match = re.search(r'_C([0-9]+)$', word_form)
            if match:
                comp_marker = f'C{match.group(1)}'
            
            if comp_marker:
                comp_type = fields[7]  # Dependency relation has the type
                
                if comp_marker not in compound_groups:
                    compound_groups[comp_marker] = {
                        'indices': [],
                        'type': comp_type,
                        'words': []
                    }
                
                compound_groups[comp_marker]['indices'].append(idx)
                compound_groups[comp_marker]['words'].append(clean_word)
                
                # Update with most specific type
                if comp_type not in ['Comp_root', 'No_rel']:
                    compound_groups[comp_marker]['type'] = comp_type
        
        # Convert to compound list
        for group_id, group_data in compound_groups.items():
            if len(group_data['indices']) < 2:
                continue
            
            indices = sorted(group_data['indices'])
            compounds.append({
                'start': indices[0],
                'end': indices[-1] + 1,
                'type': group_data['type'],
                'text': ' '.join(group_data['words'])
            })
        
        return compounds
    
    def compute_metrics(self, gold: List[Dict], predicted: List[Dict]) -> Dict:
        """Compute fine-grained and coarse metrics."""
        # Span-only matching
        gold_spans = {(c['start'], c['end']) for c in gold}
        pred_spans = {(c['start'], c['end']) for c in predicted}

        # Fine-grained matching (span + exact type)
        gold_fine = {(c['start'], c['end'], c['type']) for c in gold}
        pred_fine = {(c['start'], c['end'], c['type']) for c in predicted}

        # Coarse matching (span + coarse category)
        gold_coarse = {(c['start'], c['end'], self.get_coarse_type(c['type'])) for c in gold}
        pred_coarse = {(c['start'], c['end'], self.get_coarse_type(c['type'])) for c in predicted}

        span_correct = len(gold_spans & pred_spans)
        fine_correct = len(gold_fine & pred_fine)
        coarse_correct = len(gold_coarse & pred_coarse)

        # Exact match: all compounds in sentence are correct (fine-grained)
        exact_match = 1 if gold_fine == pred_fine and len(gold_fine) > 0 else 0
        # If there are no compounds in gold and no predictions, that's also exact match
        if len(gold_fine) == 0 and len(pred_fine) == 0:
            exact_match = 1

        # Per-type fine-grained metrics
        type_metrics = {}
        all_types = set([c['type'] for c in gold] + [c['type'] for c in predicted])

        for comp_type in all_types:
            gold_type = {(c['start'], c['end']) for c in gold if c['type'] == comp_type}
            pred_type = {(c['start'], c['end']) for c in predicted if c['type'] == comp_type}

            tp = len(gold_type & pred_type)
            fp = len(pred_type - gold_type)
            fn = len(gold_type - pred_type)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            type_metrics[comp_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': len(gold_type)
            }

        return {
            'gold_count': len(gold),
            'pred_count': len(predicted),
            'span_correct': span_correct,
            'fine_correct': fine_correct,
            'coarse_correct': coarse_correct,
            'exact_match': exact_match,
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
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

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

            self.results['total_sentences'] += 1
            self.results['total_gold_compounds'] += metrics['gold_count']
            self.results['total_predicted_compounds'] += metrics['pred_count']
            self.results['span_correct'] += metrics['span_correct']
            self.results['fine_grained_correct'] += metrics['fine_correct']
            self.results['coarse_correct'] += metrics['coarse_correct']
            self.results['exact_matches'] += metrics['exact_match']

        # Overall metrics
        overall = {}

        # Span metrics (USS - Unlabeled Span Score)
        overall['span_precision'] = self.results['span_correct'] / self.results['total_predicted_compounds'] if self.results['total_predicted_compounds'] > 0 else 0
        overall['span_recall'] = self.results['span_correct'] / self.results['total_gold_compounds'] if self.results['total_gold_compounds'] > 0 else 0
        overall['span_f1'] = 2 * overall['span_precision'] * overall['span_recall'] / (overall['span_precision'] + overall['span_recall']) if (overall['span_precision'] + overall['span_recall']) > 0 else 0

        # Fine-grained metrics (LSS - Labeled Span Score with fine types)
        overall['fine_precision'] = self.results['fine_grained_correct'] / self.results['total_predicted_compounds'] if self.results['total_predicted_compounds'] > 0 else 0
        overall['fine_recall'] = self.results['fine_grained_correct'] / self.results['total_gold_compounds'] if self.results['total_gold_compounds'] > 0 else 0
        overall['fine_f1'] = 2 * overall['fine_precision'] * overall['fine_recall'] / (overall['fine_precision'] + overall['fine_recall']) if (overall['fine_precision'] + overall['fine_recall']) > 0 else 0

        # Coarse metrics
        overall['coarse_precision'] = self.results['coarse_correct'] / self.results['total_predicted_compounds'] if self.results['total_predicted_compounds'] > 0 else 0
        overall['coarse_recall'] = self.results['coarse_correct'] / self.results['total_gold_compounds'] if self.results['total_gold_compounds'] > 0 else 0
        overall['coarse_f1'] = 2 * overall['coarse_precision'] * overall['coarse_recall'] / (overall['coarse_precision'] + overall['coarse_recall']) if (overall['coarse_precision'] + overall['coarse_recall']) > 0 else 0

        # Add USS, LSS, and EM metrics (aliases for clarity)
        overall['uss'] = overall['span_f1']  # USS = Unlabeled Span Score
        overall['lss'] = overall['fine_f1']   # LSS = Labeled Span Score (fine-grained)
        overall['lss_precision'] = overall['fine_precision']
        overall['lss_recall'] = overall['fine_recall']
        overall['exact_match'] = self.results['exact_matches'] / self.results['total_sentences'] if self.results['total_sentences'] > 0 else 0

        return {
            'overall': overall,
            'summary': self.results,
            'per_sentence': all_metrics
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-grained Sanskrit compound identification"
    )
    parser.add_argument("conllu_file", help="Gold CoNLL-U file")
    parser.add_argument("predictions_file", help="Predictions JSON file")
    parser.add_argument("-o", "--output", help="Output evaluation JSON")
    parser.add_argument("--verbose", action="store_true", help="Print per-sentence results")
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

    evaluator = FineGrainedEvaluator()
    results = evaluator.evaluate_file(args.conllu_file, args.predictions_file, args.num_sentences)

    print("\n" + "="*80)
    print("FINE-GRAINED COMPOUND IDENTIFICATION EVALUATION")
    print("="*80)
    print(f"\nTotal Sentences: {results['summary']['total_sentences']}")
    print(f"Gold Compounds: {results['summary']['total_gold_compounds']}")
    print(f"Predicted Compounds: {results['summary']['total_predicted_compounds']}")

    # Show parsing stats if there were issues
    if results['summary']['parsed_from_raw'] > 0 or results['summary']['failed_to_parse'] > 0:
        print(f"\nParsing Stats:")
        print(f"  Recovered from raw_response: {results['summary']['parsed_from_raw']}")
        print(f"  Failed to parse: {results['summary']['failed_to_parse']}")

    print("\n" + "-"*80)
    print("SPAN-ONLY METRICS (USS - Unlabeled Span Score)")
    print("-"*80)
    print(f"Precision: {results['overall']['span_precision']:.4f}")
    print(f"Recall:    {results['overall']['span_recall']:.4f}")
    print(f"F1 Score:  {results['overall']['span_f1']:.4f} (USS)")

    print("\n" + "-"*80)
    print("COARSE TYPE METRICS (Boundary + Coarse Category)")
    print("-"*80)
    print(f"Precision: {results['overall']['coarse_precision']:.4f}")
    print(f"Recall:    {results['overall']['coarse_recall']:.4f}")
    print(f"F1 Score:  {results['overall']['coarse_f1']:.4f}")

    print("\n" + "-"*80)
    print("FINE-GRAINED TYPE METRICS (LSS - Labeled Span Score)")
    print("-"*80)
    print(f"Precision: {results['overall']['fine_precision']:.4f}")
    print(f"Recall:    {results['overall']['fine_recall']:.4f}")
    print(f"F1 Score:  {results['overall']['fine_f1']:.4f} (LSS)")

    print("\n" + "-"*80)
    print("EXACT MATCH (EM)")
    print("-"*80)
    print(f"Exact Match: {results['overall']['exact_match']:.4f} ({results['summary']['exact_matches']}/{results['summary']['total_sentences']} sentences)")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"USS (Unlabeled Span F1):    {results['overall']['uss']:.4f}")
    print(f"LSS (Fine-grained F1):      {results['overall']['lss']:.4f}")
    print(f"LSS Precision:              {results['overall']['lss_precision']:.4f}")
    print(f"LSS Recall:                 {results['overall']['lss_recall']:.4f}")
    print(f"Coarse F1:                  {results['overall']['coarse_f1']:.4f}")
    print(f"Exact Match (EM):           {results['overall']['exact_match']:.4f}")
    print("="*80)

    if args.verbose:
        print("\n" + "="*80)
        print("PER-SENTENCE RESULTS")
        print("="*80)
        for sent_metrics in results['per_sentence']:
            print(f"\nSentence {sent_metrics['sentence_id']}: {sent_metrics['sentence']}")
            print(f"  Gold: {sent_metrics['gold_count']}, Predicted: {sent_metrics['pred_count']}")
            print(f"  Correct spans: {sent_metrics['span_correct']}, Correct fine: {sent_metrics['fine_correct']}, Correct coarse: {sent_metrics['coarse_correct']}")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()