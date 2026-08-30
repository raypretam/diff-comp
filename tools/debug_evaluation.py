"""
Debug script to check evaluation pipeline step by step
Run after 1 epoch to see where the 0.0 metrics come from
"""

import torch
from options import get_args
from trainers.trainer_necti import NeCTITrainer
from data.ner.necti_dataset import NeCTILabelSet, NeCTIDataset, NeCTICollator
from transformers import AutoTokenizer

# Load args
import sys
sys.argv = ['test', '--config_file', 'necti_finegrain_xlmr.yaml']
args = get_args()

print("="*80)
print("DEBUG: Checking Evaluation Pipeline")
print("="*80)

# Load label set
label_set = NeCTILabelSet(
    data_path=args.data_path,
    granularity=args.granularity,
    use_context=args.use_context
)

print(f"\nLabel set size: {len(label_set)}")
print(f"Sample labels: {list(label_set._label2id.keys())[:10]}")

# Load dataset
dev_dataset = NeCTIDataset(args.data_path, 'dev', label_set, use_context=args.use_context)
print(f"\nDev dataset size: {len(dev_dataset)}")

# Get first sample
sample = dev_dataset[0]
print(f"\nFirst sample:")
print(f"  Tokens: {sample['tokens']}")
print(f"  Labels (IDs): {sample['labels']}")
print(f"  Labels (strings): {[label_set.id2label(lid) for lid in sample['labels']]}")
print(f"  Compounds: {sample['compounds']}")

# Create tokenizer and collator
tokenizer = AutoTokenizer.from_pretrained(args.backbone)
collate_fn = NeCTICollator(tokenizer, max_length=args.max_length, add_lstm=args.add_lstm)

# Create batch
batch = collate_fn([sample])

print(f"\nBatch keys: {batch.keys()}")
print(f"  input_ids shape: {batch['input_ids'].shape}")
print(f"  seq_labels shape: {batch['seq_labels'].shape}")
print(f"  seq_labels (first 20): {batch['seq_labels'][0][:20]}")

# Test decode_relative_to_spans
trainer = NeCTITrainer(args)

labels = sample['labels']
label_strings = [label_set.id2label(lid) for lid in labels]
token_ids = list(range(1, len(labels) + 1))

print(f"\nTesting decode_relative_to_spans:")
print(f"  Token IDs: {token_ids}")
print(f"  Label strings: {label_strings}")

pred_spans = trainer.decode_relative_to_spans(token_ids, label_strings)
print(f"  Decoded spans: {pred_spans}")

# Get true spans using the NEW method - decode from labels (same as predictions)
true_spans_new = trainer.decode_relative_to_spans(token_ids, label_strings)
print(f"  True spans (using decode_relative_to_spans): {true_spans_new}")

# Get true spans using OLD method - from dataset compounds (only maximal)
true_spans_old = []
for comp in sample['compounds']:
    s_start = comp['start'] + 1  # Convert to 1-indexed
    s_end = comp['end'] + 1
    c_types = comp.get('internal_types', [])
    c_label = c_types[0] if c_types else comp.get('type', 'Compound')
    true_spans_old.append((s_start, s_end, c_label))

print(f"  True spans (OLD - from dataset compounds): {true_spans_old}")

print("\n=== Comparison ===")
print(f"Predicted spans:     {pred_spans}")
print(f"True spans (NEW):    {true_spans_new}")
print(f"True spans (OLD):    {true_spans_old}")
print("\nNEW method extracts ALL nested compounds from dependency structure.")
print("OLD method only extracted maximal compounds, causing false positives.")

# Test evaluator with NEW method (correct)
from span_based_evaluator import SpanBasedEvaluator

results_new = SpanBasedEvaluator.evaluate_batch([true_spans_new], [pred_spans])
print(f"\n=== Evaluation with NEW method (decode from labels) ===")
print(f"  USS: {results_new['USS']:.4f}")
print(f"  LSS: {results_new['LSS']:.4f}")
print(f"  EM: {results_new['EM']:.4f}")
print("  Expected: USS=1.0, LSS=1.0, EM=1.0 (predictions match ground truth)")

# Test evaluator with OLD method (incorrect)
results_old = SpanBasedEvaluator.evaluate_batch([true_spans_old], [pred_spans])
print(f"\n=== Evaluation with OLD method (from dataset compounds) ===")
print(f"  USS: {results_old['USS']:.4f}")
print(f"  LSS: {results_old['LSS']:.4f}")
print(f"  EM: {results_old['EM']:.4f}")
print("  Expected: Lower scores due to missing nested compounds in ground truth")

# Check if spans match at all (using NEW method)
true_set = set(true_spans_new)
pred_set = set(pred_spans)
print(f"\n=== Span comparison (NEW method) ===")
print(f"  True spans set: {true_set}")
print(f"  Pred spans set: {pred_set}")
print(f"  Intersection: {true_set & pred_set}")
print(f"  Only in true: {true_set - pred_set}")
print(f"  Only in pred: {pred_set - true_set}")

print("\n" + "="*80)
print("Summary:")
print("- NEW method uses decode_relative_to_spans() for both predictions and ground truth")
print("- This extracts ALL nested compounds from the dependency structure")
print("- Example: [1→2→3] creates spans (1,2) and (2,3), not just (1,3)")
print("- This fix ensures evaluation measures actual model performance correctly")
print("="*80)
