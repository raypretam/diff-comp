"""
Training script for Hierarchical Diffusion NER (Mahanama).

Two-stage diffusion:
- Stage 1: Global attention for coarse entity types (Person, Location, NORP, Misc, Time, O)
- Stage 2: Local attention for fine-grained BMES tags

Usage:
    python train_hierarchial_ner.py --config configs/hierarchial_mahanama_ner.yaml
    python train_hierarchial_ner.py --config configs/hierarchial_mahanama_ner.yaml --stage stage1
    python train_hierarchial_ner.py --config configs/hierarchial_mahanama_ner.yaml \
        --stage stage2 --stage1_checkpoint path/to/stage1.pt
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
from tqdm import tqdm

from models.hierarchial_diffusion import create_hierarchical_model
from data.ner.ner_dataset import LabelSet1D, NERDataset1D, Collator1D


# =============================================================================
# Coarse mapping: fine entity type -> coarse category
# =============================================================================
COARSE_CATEGORIES = ['Person', 'Location', 'NORP', 'Misc', 'Time', 'O']
COARSE2ID = {c: i for i, c in enumerate(COARSE_CATEGORIES)}

ENTITY_TO_COARSE = {
    # Person
    'īśvaraḥ': 'Person',
    'devatā': 'Person',
    'ṛṣiḥ': 'Person',
    'devayoniḥ': 'Person',
    'manuṣyaḥ': 'Person',
    'jantuḥ': 'Person',
    'alaukikaprāṇī': 'Person',
    # Location
    'prākṛtikasthānam': 'Location',
    'alaukikasthānam': 'Location',
    'janapadaḥ': 'Location',
    'mānavanirmitaḥ': 'Location',
    'vyakti_rūpa_sthānam': 'Location',
    # NORP
    'samūhaḥ': 'NORP',
    # Misc
    'calanirjīvaḥ': 'Misc',
    'acalanirjīvavastu': 'Misc',
    'alaukika_acalanirjīvavastu': 'Misc',
    'alaukikacalanirjīvaḥ': 'Misc',
    'śabdaḥ': 'Misc',
    'unknown': 'Misc',
    # Time
    'kālaḥ': 'Time',
}


def fine_label_to_coarse_id(fine_label: str) -> int:
    """Map a BMES fine label like 'B-manuṣyaḥ' or 'O' to a coarse class id."""
    if fine_label == 'O':
        return COARSE2ID['O']
    if '-' in fine_label:
        ent = fine_label.split('-', 1)[1]
    else:
        ent = fine_label
    coarse = ENTITY_TO_COARSE.get(ent, 'Misc')
    return COARSE2ID[coarse]


# =============================================================================
# Trainer
# =============================================================================

class HierarchicalNERTrainer:
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device(
            f"cuda:{config['gpu'].get('gpus', 0)}"
            if torch.cuda.is_available() and config['gpu']['use_gpu'] else 'cpu'
        )

        self.output_dir = Path(config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logging()

        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = 0.0

        if config['gpu'].get('precision', 32) == 16 and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

    # ------------------------------------------------------------------
    def _setup_data(self):
        data_config = self.config['data']
        dataset_name = data_config['dataset']

        self.fine_label_set = LabelSet1D(dataset_name)
        self.fine_to_coarse_map = {
            fid: fine_label_to_coarse_id(self.fine_label_set.id2label(fid))
            for fid in range(len(self.fine_label_set))
        }

        print(f"\nFine classes: {len(self.fine_label_set)}")
        print(f"Coarse classes: {len(COARSE_CATEGORIES)}")
        print("Sample fine -> coarse mappings:")
        for fid in range(min(10, len(self.fine_label_set))):
            fname = self.fine_label_set.id2label(fid)
            cid = self.fine_to_coarse_map[fid]
            print(f"  {fname} (id={fid}) -> {COARSE_CATEGORIES[cid]} (id={cid})")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['backbone'])
        self.collate_fn = Collator1D(self.tokenizer, max_length=data_config['max_length'])

        self.train_dataset = NERDataset1D(dataset_name, 'train', self.fine_label_set)
        self.dev_dataset = NERDataset1D(dataset_name, 'dev', self.fine_label_set)
        self.test_dataset = NERDataset1D(dataset_name, 'test', self.fine_label_set)

        bsz = self.config['training']['batch_size']
        nw = data_config['num_workers']
        self.train_loader = DataLoader(self.train_dataset, batch_size=bsz, shuffle=True,
                                       num_workers=nw, collate_fn=self.collate_fn, pin_memory=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=bsz, shuffle=False,
                                     num_workers=nw, collate_fn=self.collate_fn, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=bsz, shuffle=False,
                                      num_workers=nw, collate_fn=self.collate_fn, pin_memory=True)

        print(f"Train: {len(self.train_dataset)}  Dev: {len(self.dev_dataset)}  Test: {len(self.test_dataset)}")

    # ------------------------------------------------------------------
    def _setup_model(self):
        c = self.config
        self.model = create_hierarchical_model(
            device=str(self.device),
            backbone=c['backbone'],
            dim_model=c['dim_model'],
            freeze_bert=c['freeze_bert'],
            time_steps=c['time_steps'],
            sampling_steps=c['sampling_steps'],
            noise_schedule=c['noise_schedule'],
            snr_scale=c['snr_scale'],
            global_depth=c['stage1']['depth'],
            global_num_heads=c['stage1']['num_heads'],
            num_coarse_classes=c['stage1']['num_classes'],
            local_depth=c['stage2']['depth'],
            local_num_heads=c['stage2']['num_heads'],
            local_window_size=c['stage2']['window_size'],
            num_fine_classes=c['stage2']['num_classes'],
            fine_to_coarse_map=self.fine_to_coarse_map,
            objective=c['objective'],
            loss_type=c['loss_type'],
        )

        if self.args.stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {self.args.stage1_checkpoint}")
            ckpt = torch.load(self.args.stage1_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total:,}  Trainable: {trainable:,}")

    # ------------------------------------------------------------------
    def _setup_optimizer(self):
        t = self.config['training']
        lr_bert = float(t['lr_bert'])
        lr_s1 = float(t['lr_stage1'])
        lr_s2 = float(t['lr_stage2'])

        param_groups = [
            {'params': list(self.model.backbone.parameters()), 'lr': lr_bert, 'name': 'backbone'},
            {'params': list(self.model.global_dit.parameters()), 'lr': lr_s1, 'name': 'stage1'},
            {'params': list(self.model.local_dit.parameters()), 'lr': lr_s2, 'name': 'stage2'},
        ]
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
        self.optimizer = AdamW(param_groups, weight_decay=t['weight_decay'])

        total_steps = len(self.train_loader) * t['max_epochs']
        warmup_steps = t.get('warmup_steps', 500)

        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        main = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-7)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, main], milestones=[warmup_steps])

    # ------------------------------------------------------------------
    def _setup_logging(self):
        log_config = self.config.get('logging', {})
        self.logger = log_config.get('logger', 'None')
        if self.logger == 'wandb':
            import wandb
            wandb.init(
                project=log_config.get('project_name', 'hierarchical-mahanama-ner'),
                name=log_config.get('experiment_name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                config=self.config,
            )
            self.wandb = wandb

    # ------------------------------------------------------------------
    def _get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        coarse = fine_labels.clone()
        valid = fine_labels != -100
        for fid, cid in self.fine_to_coarse_map.items():
            coarse[(fine_labels == fid) & valid] = cid
        return coarse

    # ------------------------------------------------------------------
    def train_epoch(self, stage: str = 'both') -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        s1_sum = 0.0
        s2_sum = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch in pbar:
            input_ids, attention_mask, fine_labels = [x.to(self.device) for x in batch]
            coarse_labels = self._get_coarse_labels(fine_labels)

            self.optimizer.zero_grad()

            def compute_loss():
                if stage == 'both':
                    return self.model(input_ids, attention_mask, coarse_labels, fine_labels, stage='both')
                elif stage == 'stage1':
                    return self.model(input_ids, attention_mask, coarse_labels, None, stage='stage1')
                else:
                    return self.model(input_ids, attention_mask, coarse_labels, fine_labels, stage='stage2')

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    out = compute_loss()
                    if stage == 'both':
                        loss = out['loss']
                        s1_sum += out['stage1_loss'].item()
                        s2_sum += out['stage2_loss'].item()
                    else:
                        loss = out

                if torch.isnan(loss):
                    print(f"NaN loss at step {self.global_step}, skipping")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config['training']['max_grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
            else:
                out = compute_loss()
                if stage == 'both':
                    loss = out['loss']
                    s1_sum += out['stage1_loss'].item()
                    s2_sum += out['stage2_loss'].item()
                else:
                    loss = out

                if torch.isnan(loss):
                    print(f"NaN loss at step {self.global_step}, skipping")
                    continue

                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.config['training']['max_grad_norm'])
                if torch.isnan(gn) or torch.isinf(gn):
                    print(f"Bad grad norm {gn.item():.4f}, skipping")
                    continue
                self.optimizer.step()
                self.scheduler.step()

            total_loss += loss.item()
            n += 1
            self.global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})

            if self.logger == 'wandb' and self.global_step % 50 == 0:
                log = {'train/loss': loss.item(), 'train/lr': self.scheduler.get_last_lr()[0]}
                if stage == 'both':
                    log['train/stage1_loss'] = out['stage1_loss'].item()
                    log['train/stage2_loss'] = out['stage2_loss'].item()
                self.wandb.log(log, step=self.global_step)

        metrics = {'loss': total_loss / max(1, n)}
        if stage == 'both':
            metrics['stage1_loss'] = s1_sum / max(1, n)
            metrics['stage2_loss'] = s2_sum / max(1, n)
        return metrics

    # ------------------------------------------------------------------
    def _decode_bmes(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        """Decode a BMES label sequence into entity spans (start, end, type)."""
        entities = []
        i = 0
        n = len(labels)
        while i < n:
            lab = labels[i]
            if lab.startswith('S-'):
                entities.append((i, i, lab[2:]))
                i += 1
            elif lab.startswith('B-'):
                ent = lab[2:]
                start = i
                j = i + 1
                end = None
                while j < n:
                    if labels[j] == 'M-' + ent:
                        j += 1
                        continue
                    if labels[j] == 'E-' + ent:
                        end = j
                        j += 1
                        break
                    break
                if end is not None:
                    entities.append((start, end, ent))
                i = j
            else:
                i += 1
        return entities

    @torch.no_grad()
    def evaluate(self, loader, name: str = 'dev') -> Dict[str, float]:
        self.model.eval()

        all_fine_preds, all_fine_labels = [], []
        all_coarse_preds, all_coarse_labels = [], []
        total_gold = total_pred = total_tp = 0

        for batch in tqdm(loader, desc=f'Eval {name}'):
            input_ids, attention_mask, fine_labels = [x.to(self.device) for x in batch]
            coarse_labels = self._get_coarse_labels(fine_labels)

            coarse_preds, fine_preds = self.model.inference(input_ids, attention_mask)
            mask = fine_labels != -100

            for i in range(input_ids.shape[0]):
                sm = mask[i]
                if sm.sum() == 0:
                    continue
                pred_ids = fine_preds[i][sm].cpu().tolist()
                true_ids = fine_labels[i][sm].cpu().tolist()
                cpred_ids = coarse_preds[i][sm].cpu().tolist()
                ctrue_ids = coarse_labels[i][sm].cpu().tolist()

                all_fine_preds.extend(pred_ids)
                all_fine_labels.extend(true_ids)
                all_coarse_preds.extend(cpred_ids)
                all_coarse_labels.extend(ctrue_ids)

                num_fine = len(self.fine_label_set)
                pred_names = [self.fine_label_set.id2label(p) if 0 <= p < num_fine else 'O'
                              for p in pred_ids]
                true_names = [self.fine_label_set.id2label(t) if 0 <= t < num_fine else 'O'
                              for t in true_ids]
                gold = set(self._decode_bmes(true_names))
                preds = set(self._decode_bmes(pred_names))
                total_gold += len(gold)
                total_pred += len(preds)
                total_tp += len(gold & preds)

        coarse_acc = float(np.mean(np.array(all_coarse_preds) == np.array(all_coarse_labels))) if all_coarse_labels else 0.0
        fine_acc = float(np.mean(np.array(all_fine_preds) == np.array(all_fine_labels))) if all_fine_labels else 0.0
        p = total_tp / total_pred if total_pred else 0.0
        r = total_tp / total_gold if total_gold else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0

        return {
            'coarse_acc': coarse_acc, 'fine_acc': fine_acc,
            'precision': p, 'recall': r, 'f1': f1,
            'num_gold': total_gold, 'num_pred': total_pred, 'num_tp': total_tp,
        }

    # ------------------------------------------------------------------
    def save_checkpoint(self, is_best: bool = False):
        ckpt = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        torch.save(ckpt, self.output_dir / 'last.pt')
        if is_best:
            torch.save(ckpt, self.output_dir / 'best.pt')
            print(f"Saved best model (f1={self.best_metric:.4f})")

    def train(self):
        t = self.config['training']
        strategy = t.get('strategy', 'joint')
        stage = self.args.stage or strategy

        print(f"\n{'='*60}\nTraining strategy: {stage}\n{'='*60}")
        patience = 0
        for epoch in range(t['max_epochs']):
            self.current_epoch = epoch

            if stage == 'sequential':
                if epoch < t.get('stage1_epochs', 15):
                    cur_stage = 'stage1'
                else:
                    cur_stage = 'stage2'
                    if t.get('freeze_stage1_for_stage2', False):
                        for p in self.model.global_dit.parameters():
                            p.requires_grad = False
            else:
                cur_stage = stage if stage in ('stage1', 'stage2', 'both') else 'both'

            train_m = self.train_epoch(cur_stage)
            print(f"\nEpoch {epoch} train loss: {train_m['loss']:.4f}")

            eval_m = self.evaluate(self.dev_loader, 'dev')
            print(f"Epoch {epoch} dev | coarse_acc={eval_m['coarse_acc']:.4f}  "
                  f"fine_acc={eval_m['fine_acc']:.4f}  "
                  f"P={eval_m['precision']:.4f}  R={eval_m['recall']:.4f}  F1={eval_m['f1']:.4f}")

            if self.logger == 'wandb':
                self.wandb.log({
                    'epoch': epoch,
                    'val/coarse_acc': eval_m['coarse_acc'],
                    'val/fine_acc': eval_m['fine_acc'],
                    'val/precision': eval_m['precision'],
                    'val/recall': eval_m['recall'],
                    'val/f1': eval_m['f1'],
                }, step=self.global_step)

            is_best = eval_m['f1'] > self.best_metric
            if is_best:
                self.best_metric = eval_m['f1']
                patience = 0
            else:
                patience += 1
            self.save_checkpoint(is_best=is_best)

            if patience >= t.get('patience', 10):
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\nBest dev F1: {self.best_metric:.4f}")
        print("Running final test eval...")
        test_m = self.evaluate(self.test_loader, 'test')
        print(f"Test | P={test_m['precision']:.4f}  R={test_m['recall']:.4f}  F1={test_m['f1']:.4f}")

        if self.logger == 'wandb':
            self.wandb.log({'test/f1': test_m['f1'],
                            'test/precision': test_m['precision'],
                            'test/recall': test_m['recall']})
            self.wandb.finish()


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Hierarchical Diffusion NER (Mahanama)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--stage', type=str, default="both",
                        choices=['stage1', 'stage2', 'both', 'sequential'])
    parser.add_argument('--stage1_checkpoint', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = HierarchicalNERTrainer(config, args)

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        trainer.current_epoch = ckpt['epoch']
        trainer.global_step = ckpt['global_step']
        trainer.best_metric = ckpt['best_metric']

    trainer.train()


if __name__ == '__main__':
    main()
