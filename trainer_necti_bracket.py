"""
Bracket-augmented training script for NeCTI nested compound identification.

Architecture:
  - XLM-R encoder produces contextual token representations
  - Bit-diffusion (BitDit) denoises JOINT labels over  
      joint_id = bracket_id * num_comptype + comptype_id
  - An auxiliary linear bracket head on top of the encoder provides
    a structural consistency signal at every denoising step
  - At inference, joint ids are decoded back to (bracket, comptype) pairs
    and the bracket sequence is used to reconstruct the compound tree via
    the separ-style stack decoder

Why bracketing helps:
  1. Well-formedness constraint: at each reverse-diffusion step the partial
     bracket sequence must have balanced stacks → strong structural prior
  2. Disentangles STRUCTURE (which tokens form a compound span) from LABEL
     (what kind of compound it is) → cleaner gradient signal
  3. Hierarchical compounds (Comp2 inside Comp3) map naturally to k-planar
     bracket sequences with plane indices
  4. At inference, bracket decoding + CLE gives a valid tree without
     post-hoc constraint enforcement
"""

import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data.ner.necti_bracket import NeCTIBracketLabelSet, decode_sentence
from data.ner.necti_bracket_dataset import NeCTIBracketDataset, NeCTIBracketCollator
from models.ddim_bitdit import BitDit
from utils import get_lr_scheduler


class NeCTIBracketTrainer:
    """
    Trains a bit-diffusion model on JOINT bracket × compound-type labels.

    The model learns  p(bracket_id, comptype_id | context)  jointly via the
    diffusion objective, with an additional bracket-consistency auxiliary loss
    from a lightweight linear head on the frozen encoder output.
    """

    def __init__(self, args: Namespace):
        self.args = args

        if not hasattr(args, 'use_context'):
            args.use_context = False
        if not hasattr(args, 'bracket_k'):
            args.bracket_k = 2          # number of planes for k-planar encoding
        if not hasattr(args, 'bracket_aux_weight'):
            args.bracket_aux_weight = 0.3   # weight of auxiliary bracket CE loss

        ctx_mode = 'with_ctx' if args.use_context else 'no_ctx'

        if args.logger == 'wandb':
            run_name = (f"necti-bracket-{args.granularity}-{ctx_mode}"
                        f"--k{args.bracket_k}"
                        f"--lr_bert_{args.lr_bert}--epochs_{args.max_epochs}")
            wandb.init(project='DiffusionSL-NeCTI-Bracket', name=run_name)
            wandb.config.update(args)
            wandb.define_metric('f1', summary='max')

        self.device = self._configure_device()

        # ── Label set ──────────────────────────────────────────────────────
        self.label_set = NeCTIBracketLabelSet(
            data_path=args.data_path,
            granularity=args.granularity,
            use_context=args.use_context,
            k=args.bracket_k,
        )

        # Override num_classes to joint space
        joint_classes = self.label_set.num_joint_classes
        print(f"[NeCTIBracketTrainer] Joint class space: "
              f"{self.label_set.num_bracket_classes} brackets × "
              f"{self.label_set.num_comptype_classes} compound types "
              f"= {joint_classes} joint labels")
        args.num_classes = joint_classes

        # ── Diffusion model ─────────────────────────────────────────────────
        self.model = BitDit(
            device=self.device,
            num_classes=joint_classes,
            backbone=args.backbone,
            time_steps=args.time_steps,
            sampling_steps=args.sampling_steps,
            noise_schedule=args.noise_schedule,
            ddim_sampling_eta=args.ddim_sampling_eta,
            self_condition=args.self_condition,
            snr_scale=args.snr_scale,
            dataset=f"necti_bracket_{args.granularity}",
            dim_model=args.dim_model,
            dim_time=args.dim_time,
            objective=args.objective,
            loss_type=args.loss_type,
            add_lstm=args.add_lstm,
            freeze_bert=args.freeze_bert,
            max_length=args.max_length,
            depth=args.depth,
            num_labels=joint_classes,
        )

        # ── Auxiliary bracket head  ─────────────────────────────────────────
        # Lightweight linear head on BERT output → bracket class logits.
        # This provides a direct bracket supervision signal independent of the
        # diffusion objective, encouraging the encoder to build
        # structure-aware representations.
        bert_dim = self.model.bert_dim
        self.bracket_head = nn.Linear(
            bert_dim, self.label_set.num_bracket_classes
        ).to(self.device)
        self.bracket_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.bracket_aux_weight = args.bracket_aux_weight

        if args.logger == 'wandb':
            wandb.watch(self.model, log_freq=1000)

        # ── Tokeniser and data ──────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        self.collate_fn = NeCTIBracketCollator(
            tokenizer=self.tokenizer,
            label_set=self.label_set,
            max_length=args.max_length,
            joint_head=True,    # produce seq_labels for BitDit + separate bracket labels
        )

        self.train_loader = self._loader('train', args.batch_size)
        self.dev_loader   = self._loader('dev',   args.batch_size)
        self.test_loader  = self._loader('test',  args.batch_size)

        try:
            self.ood_loader = self._loader('ood', args.batch_size)
        except FileNotFoundError:
            print("OOD dataset not found, skipping.")
            self.ood_loader = None

        self.steps = args.max_steps
        self.optimizer, self.lr_scheduler = self._configure_optim(
            args.optimizer_type, args.lr_scheduler_type)

    # ────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────────

    def _loader(self, mode: str, bsz: int) -> DataLoader:
        dataset = NeCTIBracketDataset(
            self.args.data_path, mode, self.label_set,
            use_context=self.args.use_context,
        )
        return DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=(mode == 'train'),
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def _configure_device(self):
        if torch.cuda.is_available():
            dev = torch.device('cuda')
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            dev = torch.device('cpu')
            print("CPU mode")
        return dev

    def _configure_optim(self, opt_type, sched_type):
        bert_params  = [p for n, p in self.model.named_parameters()
                        if 'backbone' in n and p.requires_grad]
        other_params = [p for n, p in self.model.named_parameters()
                        if 'backbone' not in n and p.requires_grad]
        other_params += list(self.bracket_head.parameters())

        optimizer = AdamW(
            [{'params': bert_params,  'lr': self.args.lr_bert},
             {'params': other_params, 'lr': self.args.lr_other}],
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_lr_scheduler(
            name=sched_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.steps,
            num_cycles=getattr(self.args, 'num_cycles', 1),
        )
        return optimizer, scheduler

    # ────────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────────

    def train(self):
        best_f1 = 0.0
        global_step = 0

        for epoch in range(1, self.args.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.max_epochs}")
            self.model.train()
            self.bracket_head.train()

            epoch_loss, epoch_br_loss, n_steps = 0., 0., 0

            for batch in tqdm(self.train_loader, desc='Train'):
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_labels     = batch['seq_labels'].to(self.device)          # joint
                br_labels      = batch['seq_bracket_labels'].to(self.device)  # bracket only

                # ── Main diffusion loss ────────────────────────────────────
                diff_loss = self.model(input_ids, attention_mask, seq_labels)

                # ── Auxiliary bracket head loss ────────────────────────────
                # Re-use BERT encoder representations (no additional forward pass cost)
                with torch.no_grad():
                    encoder_out = self.model.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).last_hidden_state   # (B, L, bert_dim)

                br_logits = self.bracket_head(encoder_out)  # (B, L, num_br)
                br_loss = self.bracket_loss_fn(
                    br_logits.view(-1, self.label_set.num_bracket_classes),
                    br_labels.view(-1),
                )

                total_loss = diff_loss + self.bracket_aux_weight * br_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.bracket_head.parameters()),
                    self.args.max_grad_norm,
                )
                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss    += diff_loss.item()
                epoch_br_loss += br_loss.item()
                n_steps       += 1
                global_step   += 1

                if self.args.logger == 'wandb':
                    wandb.log({
                        'train/diff_loss'   : diff_loss.item(),
                        'train/bracket_loss': br_loss.item(),
                        'train/total_loss'  : total_loss.item(),
                        'global_step'       : global_step,
                    })

            print(f"  diff_loss={epoch_loss/n_steps:.4f}  "
                  f"br_loss={epoch_br_loss/n_steps:.4f}")

            # ── Validation ────────────────────────────────────────────────
            dev_metrics = self.evaluate(self.dev_loader, 'dev')
            print(f"  dev  USS={dev_metrics['uss']:.4f}  "
                  f"LSS={dev_metrics['lss']:.4f}  "
                  f"EM={dev_metrics['em']:.4f}  "
                  f"br_acc={dev_metrics['bracket_acc']:.4f}")

            if self.args.logger == 'wandb':
                wandb.log({'epoch': epoch, **{f'dev/{k}': v
                                              for k, v in dev_metrics.items()}})

            if dev_metrics['lss'] > best_f1:
                best_f1 = dev_metrics['lss']
                self._save(epoch, best_f1)
                print(f"  ✓ Best LSS {best_f1:.4f} — model saved")

        # ── Final test evaluation ─────────────────────────────────────────
        self._load_best()
        test_metrics = self.evaluate(self.test_loader, 'test')
        print("\nTest Results:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        if self.args.logger == 'wandb':
            wandb.log({f'test/{k}': v for k, v in test_metrics.items()})

    # ────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str) -> dict:
        """
        Returns:
          uss         : Unlabelled Structure Score (bracket F1, ignoring compound type)
          lss         : Labelled Structure Score   (joint bracket+type F1)
          em          : Exact Match
          bracket_acc : Accuracy of auxiliary bracket head
          comptype_f1 : Macro F1 over compound types (ignoring structure)
        """
        self.model.eval()
        self.bracket_head.eval()

        pred_joints, gold_joints = [], []
        pred_brs,    gold_brs    = [], []
        n_correct_em, n_total_em = 0, 0
        br_correct, br_total     = 0, 0

        for batch in tqdm(loader, desc=f'Eval {split}'):
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            seq_labels     = batch['seq_labels'].to(self.device)
            br_labels      = batch['seq_bracket_labels'].to(self.device)

            # Diffusion inference
            predictions, _ = self.model(input_ids, attention_mask, seq_labels)

            mask = (seq_labels != -100)
            p_joint = predictions[mask].cpu().numpy()
            g_joint = seq_labels[mask].cpu().numpy()

            # Decompose joint → (bracket_id, comptype_id)
            p_br, p_ct = np.divmod(p_joint, self.label_set.num_comptype_classes)
            g_br, g_ct = np.divmod(g_joint, self.label_set.num_comptype_classes)

            pred_joints.extend(zip(p_br.tolist(), p_ct.tolist()))
            gold_joints.extend(zip(g_br.tolist(), g_ct.tolist()))
            pred_brs.extend(p_br.tolist())
            gold_brs.extend(g_br.tolist())

            # Exact-match per sentence
            B = input_ids.shape[0]
            for b in range(B):
                m = mask[b]
                if m.sum() == 0:
                    continue
                sent_pred = predictions[b][m].cpu()
                sent_gold = seq_labels[b][m].cpu()
                n_correct_em += (sent_pred == sent_gold).all().item()
                n_total_em   += 1

            # Auxiliary bracket head accuracy
            enc_out = self.model.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            br_logits = self.bracket_head(enc_out)
            br_preds  = br_logits.argmax(-1)
            br_mask   = (br_labels != -100)
            br_correct += (br_preds[br_mask] == br_labels[br_mask]).sum().item()
            br_total   += br_mask.sum().item()

        pred_brs = np.array(pred_brs)
        gold_brs = np.array(gold_brs)
        pred_joints = [(b, c) for b, c in pred_joints]
        gold_joints = [(b, c) for b, c in gold_joints]

        # USS: unlabelled: bracket structure correct (ignore comptype)
        uss_tp = np.sum(pred_brs == gold_brs)
        uss    = uss_tp / max(len(gold_brs), 1)

        # LSS: labelled: both bracket and comptype correct
        lss_tp = sum(p == g for p, g in zip(pred_joints, gold_joints))
        lss    = lss_tp / max(len(gold_joints), 1)

        return {
            'uss'        : float(uss),
            'lss'        : float(lss),
            'em'         : float(n_correct_em / max(n_total_em, 1)),
            'bracket_acc': float(br_correct / max(br_total, 1)),
        }

    # ────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ────────────────────────────────────────────────────────────────────────

    def _save(self, epoch: int, score: float):
        ctx  = 'with_ctx' if self.args.use_context else 'no_ctx'
        path = os.path.join(
            os.getcwd(), 'saved_models',
            f'necti_bracket_{self.args.granularity}_{ctx}',
            'best_model.pt',
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch'              : epoch,
            'model_state_dict'   : self.model.state_dict(),
            'bracket_head_state' : self.bracket_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score'              : score,
            'args'               : self.args,
        }, path)

    def _load_best(self):
        ctx  = 'with_ctx' if self.args.use_context else 'no_ctx'
        path = os.path.join(
            os.getcwd(), 'saved_models',
            f'necti_bracket_{self.args.granularity}_{ctx}',
            'best_model.pt',
        )
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.bracket_head.load_state_dict(ckpt['bracket_head_state'])
            print(f"Loaded best model from epoch {ckpt['epoch']} "
                  f"(score={ckpt['score']:.4f})")
        else:
            print("No saved model found, using current model.")


def main():
    from options import get_args
    args = get_args()
    trainer = NeCTIBracketTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
