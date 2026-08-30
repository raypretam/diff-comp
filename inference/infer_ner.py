"""
Inference / test-set evaluation for the BitDit NER model (mahanama_ner).

Works for both the base model (trainer_ner.py, base: ner) and the
hyperbolic-encoder variant (trainer_ner_hyp.py, base: ner_hyp). The model
is rebuilt through the same Trainer class used in training, so the
architecture matches the checkpoint exactly.

Reports overall precision / recall / F1 (entity-level, BMES decoding),
token-level label accuracy, and a per-entity-type breakdown.

Usage:
    # uses --config_file (loaded from configs/) + checkpoint
    python infer_ner.py --config_file mahanama_ner_fine.yaml \
        --checkpoint output/mahanama_ner_fine/best_f1_0.1234

    # hyperbolic variant
    python infer_ner.py --config_file mahanama_ner_fine_hyp.yaml \
        --checkpoint output/mahanama_ner_fine_hyp/best_f1_0.1234

    # evaluate the dev split instead of test
    python infer_ner.py --config_file mahanama_ner_fine.yaml \
        --checkpoint <path> --split dev

If --checkpoint is omitted, the script falls back to the trainer's default
location: <output_dir>/<config_file stem>/<model_path>.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm
from prettytable import PrettyTable


def _parse_infer_args():
    """Pull inference-only flags out of sys.argv before options.get_parser runs."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--checkpoint', type=str, default=None,
                     help='Path to the trained checkpoint (state_dict). '
                          'If omitted, uses <output_dir>/<config stem>/<model_path>.')
    pre.add_argument('--split', type=str, default='test', choices=['dev', 'test'],
                     help='Which split to evaluate.')
    pre.add_argument('--output', type=str, default=None,
                     help='Path to write per-sentence predictions as JSON. '
                          'If omitted, defaults to predictions_<split>.json next '
                          'to the checkpoint.')
    infer_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return infer_args


def resolve_checkpoint(args, checkpoint: str) -> str:
    """Resolve the checkpoint path, honouring the trainer's directory convention."""
    if checkpoint and os.path.isfile(checkpoint):
        return checkpoint

    dir_ = '-'.join(args.config_file.split('.')[:-1])
    dir_path = os.path.join(args.output_dir, dir_)

    if checkpoint:
        candidate = os.path.join(dir_path, checkpoint)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint} (also tried {candidate})")

    candidate = os.path.join(dir_path, args.model_path)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"No --checkpoint given and default not found: {candidate}")


@torch.no_grad()
def evaluate(trainer, split: str):
    """Run entity-level + per-class evaluation on the chosen split.

    Returns aggregate stats plus a list of per-sentence prediction records.
    """
    model = trainer.model
    model.eval()

    loader = trainer.test_dataloader if split == 'test' else trainer.dev_dataloader
    # dev/test loaders use shuffle=False, so dataset order is preserved
    raw_data = loader.dataset.data

    total_gold = total_pred = total_tp = 0
    label_true = label_all = 0
    # per entity-type counts: type -> [gold, pred, tp]
    per_class = defaultdict(lambda: [0, 0, 0])
    records = []
    sent_idx = 0

    for batch in tqdm(loader, desc=f'Evaluating ({split})'):
        input_ids, attention_mask, seq_labels = [x.to(trainer.device) for x in batch]
        results, _ = model(input_ids, attention_mask, seq_labels)
        pred_labels = results.cpu()
        gold_labels = seq_labels.cpu()

        labels_mask = gold_labels != -100
        for i in range(gold_labels.shape[0]):
            gl = gold_labels[i][labels_mask[i]].tolist()
            pl = pred_labels[i][labels_mask[i]].tolist()
            cur_idx = sent_idx
            sent_idx += 1
            if len(gl) != len(pl):
                # alignment mismatch — skip defensively
                continue

            gold_ents = trainer._decode(gl)
            pred_ents = trainer._decode(pl)
            gold_set, pred_set = set(gold_ents), set(pred_ents)

            total_gold += len(gold_set)
            total_pred += len(pred_set)
            total_tp += len(gold_set & pred_set)

            for _, etype in gold_ents:
                per_class[etype][0] += 1
            for _, etype in pred_ents:
                per_class[etype][1] += 1
            for _, etype in (gold_set & pred_set):
                per_class[etype][2] += 1

            label_true += sum(1 for a, b in zip(gl, pl) if a == b)
            label_all += len(gl)

            # ---- per-sentence prediction record ----
            sentence = raw_data[cur_idx]['sentence'] if cur_idx < len(raw_data) else []
            n = min(len(sentence), len(gl))  # align (sentence may be truncated)
            gold_names = [trainer.label_set.id2label(x) for x in gl]
            pred_names = [trainer.label_set.id2label(x) for x in pl]
            records.append({
                'id': cur_idx,
                'sentence': sentence[:n],
                'gold_labels': gold_names[:n],
                'pred_labels': pred_names[:n],
                'gold_entities': [{'span': [s, e], 'type': t} for (s, e), t in gold_ents],
                'pred_entities': [{'span': [s, e], 'type': t} for (s, e), t in pred_ents],
                'correct': gold_set == pred_set,
            })

    return total_gold, total_pred, total_tp, label_true, label_all, per_class, records


def prf(gold: int, pred: int, tp: int):
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def report(total_gold, total_pred, total_tp, label_true, label_all, per_class):
    p, r, f = prf(total_gold, total_pred, total_tp)
    acc = label_true / label_all if label_all else 0.0

    print("\n" + "=" * 70)
    print("OVERALL (entity-level, BMES decoding)")
    print("=" * 70)
    overall = PrettyTable()
    overall.field_names = ['precision', 'recall', 'f1',
                           'num_gold', 'num_pred', 'num_tp', 'token_acc']
    overall.add_row([f'{p:.4f}', f'{r:.4f}', f'{f:.4f}',
                     total_gold, total_pred, total_tp, f'{acc:.4f}'])
    print(overall)

    print("\nPER ENTITY TYPE")
    table = PrettyTable()
    table.field_names = ['type', 'precision', 'recall', 'f1',
                         'gold', 'pred', 'tp']
    macro_f = []
    for etype in sorted(per_class.keys()):
        g, pr, tp = per_class[etype]
        cp, cr, cf = prf(g, pr, tp)
        macro_f.append(cf)
        table.add_row([etype, f'{cp:.4f}', f'{cr:.4f}', f'{cf:.4f}', g, pr, tp])
    print(table)

    macro = sum(macro_f) / len(macro_f) if macro_f else 0.0
    print(f"\nMicro-F1: {f:.4f}   Macro-F1: {macro:.4f}")
    print("=" * 70)
    return {'precision': p, 'recall': r, 'micro_f1': f, 'macro_f1': macro,
            'token_acc': acc, 'num_gold': total_gold, 'num_pred': total_pred,
            'num_tp': total_tp}


def save_predictions(path, split, summary, per_class, records):
    """Write per-sentence predictions + metrics to a JSON file."""
    per_class_out = {}
    for etype in sorted(per_class.keys()):
        g, pr, tp = per_class[etype]
        cp, cr, cf = prf(g, pr, tp)
        per_class_out[etype] = {'precision': cp, 'recall': cr, 'f1': cf,
                                'gold': g, 'pred': pr, 'tp': tp}
    payload = {
        'split': split,
        'summary': summary,
        'per_class': per_class_out,
        'predictions': records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} predictions to {path}")


def main():
    infer_args = _parse_infer_args()

    from options import get_parser
    parser = get_parser()
    args = parser.parse_args()

    # Build the model through the same Trainer used in training
    if args.base == 'ner_hyp':
        from trainers.trainer_ner_hyp import Trainer
    else:
        from trainers.trainer_ner import Trainer
    print(f"Using trainer for base='{args.base}'")

    trainer = Trainer(args)

    ckpt_path = resolve_checkpoint(args, infer_args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=trainer.device)
    # checkpoints from trainer_ner.save() are plain state_dicts
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    trainer.model.load_state_dict(state_dict)

    *stats, per_class, records = evaluate(trainer, infer_args.split)
    summary = report(*stats, per_class)

    out_path = infer_args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                                f'predictions_{infer_args.split}.json')
    save_predictions(out_path, infer_args.split, summary, per_class, records)


if __name__ == '__main__':
    main()
