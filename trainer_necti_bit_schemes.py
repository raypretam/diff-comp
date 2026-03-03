"""
Unified trainer for Bit4 / Bit7 / HexaTagging NeCTI experiments.

Pass --scheme bit4 | bit7 | hexa to select the encoding.

HOW THE BIT-DIFFUSION ALIGNMENT WORKS
======================================
DiffusionSL's BitDit internally computes:
    self.bits = ceil(log2(num_labels))

  Scheme | num_joint_labels | self.bits | Meaning
  -------|-----------------|-----------|----------------------------
  bit4   | 16 × C          | 4+ceil(log2 C) | 4 structural bits + label bits
  bit7   | 128 × C         | 7+ceil(log2 C) | 7 structural bits + label bits
  hexa   | 6 × C           | 3+ceil(log2 C) | 3 hexa-structure bits + label bits

For bit4 with C=4 (Coarse): 64 classes → 6 bits.
The diffusion model denoises 6-dimensional bit vectors where the first
4 bits encode compound tree structure and the last 2 bits encode type.

An auxiliary linear head on top of XLM-R separately predicts the structural
class (bit4/bit7/hexa) at every training step, giving the encoder a direct
structural supervision signal.
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

from data.ner.necti_bit_schemes import NeCTIBitSchemeLabelSet
from data.ner.necti_bit_schemes_dataset import NeCTIBitSchemeDataset, NeCTIBitSchemeCollator
from models.ddim_bitdit import BitDit
from utils import get_lr_scheduler


class NeCTIBitSchemeTrainer:
    """
    Trains BitDit on JOINT (structural × compound-type) labels.

    Works identically for bit4, bit7, and hexa — only the label set
    and vocabulary size change.
    """

    def __init__(self, args: Namespace):
        self.args = args

        if not hasattr(args, 'scheme'):
            args.scheme = 'bit4'
        if not hasattr(args, 'use_context'):
            args.use_context = False
        if not hasattr(args, 'struct_aux_weight'):
            args.struct_aux_weight = 0.3

        ctx  = 'with_ctx' if args.use_context else 'no_ctx'

        if args.logger == 'wandb':
            run_name = (f"necti-{args.scheme}-{args.granularity}-{ctx}"
                        f"--lr_bert_{args.lr_bert}--epochs_{args.max_epochs}")
            wandb.init(project=f'DiffusionSL-NeCTI-{args.scheme.upper()}', name=run_name)
            wandb.config.update(args)
            wandb.define_metric('lss', summary='max')

        self.device = self._device()

        # ── Label set ──────────────────────────────────────────────────────
        self.label_set = NeCTIBitSchemeLabelSet(
            data_path=args.data_path,
            granularity=args.granularity,
            use_context=args.use_context,
            scheme=args.scheme,
        )
        joint_classes = self.label_set.num_joint_classes
        args.num_classes = joint_classes
        print(f"[Trainer:{args.scheme}] joint classes = {joint_classes}")

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
            dataset=f"necti_{args.scheme}_{args.granularity}",
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

        # ── Auxiliary structural head ───────────────────────────────────────
        # Predicts structural class (bit4/bit7/hexa) directly from XLM-R
        # output.  This gives a clean gradient toward structural correctness
        # independent of the diffusion noise schedule.
        self.struct_head = nn.Linear(
            self.model.bert_dim, self.label_set.num_struct
        ).to(self.device)
        self.struct_loss_fn    = nn.CrossEntropyLoss(ignore_index=-100)
        self.struct_aux_weight = args.struct_aux_weight

        if args.logger == 'wandb':
            wandb.watch(self.model, log_freq=1000)

        # ── Tokeniser + data loaders ────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        self.collate_fn = NeCTIBitSchemeCollator(
            self.tokenizer, self.label_set, max_length=args.max_length
        )
        self.train_loader = self._loader('train', args.batch_size)
        self.dev_loader   = self._loader('dev',   args.batch_size)
        self.test_loader  = self._loader('test',  args.batch_size)
        try:
            self.ood_loader = self._loader('ood', args.batch_size)
        except FileNotFoundError:
            self.ood_loader = None
            print("OOD dataset not found, skipping.")

        self.steps = args.max_steps
        self.optimizer, self.lr_scheduler = self._optim()

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _device(self):
        if torch.cuda.is_available():
            d = torch.device('cuda')
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return d
        print("CPU mode")
        return torch.device('cpu')

    def _loader(self, mode, bsz):
        ds = NeCTIBitSchemeDataset(
            self.args.data_path, mode, self.label_set, self.args.use_context
        )
        return DataLoader(ds, batch_size=bsz, shuffle=(mode == 'train'),
                          num_workers=self.args.num_workers, drop_last=False,
                          collate_fn=self.collate_fn)

    def _optim(self):
        bert_p  = [p for n, p in self.model.named_parameters()
                   if 'backbone' in n and p.requires_grad]
        other_p = [p for n, p in self.model.named_parameters()
                   if 'backbone' not in n and p.requires_grad]
        other_p += list(self.struct_head.parameters())

        opt = AdamW(
            [{'params': bert_p,  'lr': self.args.lr_bert},
             {'params': other_p, 'lr': self.args.lr_other}],
            weight_decay=self.args.weight_decay,
        )
        sched = get_lr_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=opt,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.steps,
            num_cycles=getattr(self.args, 'num_cycles', 1),
        )
        return opt, sched

    # ────────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────────

    def train(self):
        best_lss = 0.0
        global_step = 0

        for epoch in range(1, self.args.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.max_epochs}")
            self.model.train()
            self.struct_head.train()

            e_diff, e_aux, n = 0., 0., 0

            for batch in tqdm(self.train_loader, desc='Train'):
                input_ids   = batch['input_ids'].to(self.device)
                attn_mask   = batch['attention_mask'].to(self.device)
                seq_labels  = batch['seq_labels'].to(self.device)       # joint
                struct_labs = batch['seq_struct_labels'].to(self.device) # struct only

                # Diffusion loss
                diff_loss = self.model(input_ids, attn_mask, seq_labels)

                # Auxiliary structural head
                with torch.no_grad():
                    enc_out = self.model.backbone(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                    ).last_hidden_state
                struct_logits = self.struct_head(enc_out)
                aux_loss = self.struct_loss_fn(
                    struct_logits.view(-1, self.label_set.num_struct),
                    struct_labs.view(-1),
                )

                total = diff_loss + self.struct_aux_weight * aux_loss

                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.struct_head.parameters()),
                    self.args.max_grad_norm,
                )
                self.optimizer.step()
                self.lr_scheduler.step()

                e_diff += diff_loss.item()
                e_aux  += aux_loss.item()
                n      += 1
                global_step += 1

                if self.args.logger == 'wandb':
                    wandb.log({'train/diff': diff_loss.item(),
                               'train/aux':  aux_loss.item(),
                               'global_step': global_step})

            print(f"  diff={e_diff/n:.4f}  aux_struct={e_aux/n:.4f}")

            dev = self.evaluate(self.dev_loader, 'dev')
            print(f"  dev  USS={dev['uss']:.4f}  LSS={dev['lss']:.4f}  "
                  f"EM={dev['em']:.4f}  struct_acc={dev['struct_acc']:.4f}")

            if self.args.logger == 'wandb':
                wandb.log({'epoch': epoch, **{f'dev/{k}': v for k, v in dev.items()}})

            if dev['lss'] > best_lss:
                best_lss = dev['lss']
                self._save(epoch, best_lss)
                print(f"  ✓ Best LSS {best_lss:.4f}")

        # Final test
        self._load_best()
        test = self.evaluate(self.test_loader, 'test')
        print("\nTest Results:")
        for k, v in test.items():
            print(f"  {k}: {v:.4f}")

        if self.ood_loader:
            ood = self.evaluate(self.ood_loader, 'ood')
            print("OOD Results:")
            for k, v in ood.items():
                print(f"  {k}: {v:.4f}")

        if self.args.logger == 'wandb':
            wandb.log({f'test/{k}': v for k, v in test.items()})

    # ────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader, split='dev') -> dict:
        self.model.eval()
        self.struct_head.eval()

        p_joints, g_joints = [], []
        p_structs, g_structs = [], []
        em_correct, em_total = 0, 0
        sa_correct, sa_total = 0, 0

        for batch in tqdm(loader, desc=f'Eval {split}'):
            input_ids   = batch['input_ids'].to(self.device)
            attn_mask   = batch['attention_mask'].to(self.device)
            seq_labels  = batch['seq_labels'].to(self.device)
            struct_labs = batch['seq_struct_labels'].to(self.device)

            preds, _ = self.model(input_ids, attn_mask, seq_labels)

            mask = seq_labels != -100
            pj = preds[mask].cpu().numpy()
            gj = seq_labels[mask].cpu().numpy()
            ps, _ = np.divmod(pj, self.label_set.num_comptype_classes)
            gs, _ = np.divmod(gj, self.label_set.num_comptype_classes)

            p_joints.extend(zip(ps.tolist(), (pj - ps * self.label_set.num_comptype_classes).tolist()))
            g_joints.extend(zip(gs.tolist(), (gj - gs * self.label_set.num_comptype_classes).tolist()))
            p_structs.extend(ps.tolist())
            g_structs.extend(gs.tolist())

            B = input_ids.shape[0]
            for b in range(B):
                m = mask[b]
                if m.sum() == 0:
                    continue
                em_correct += (preds[b][m] == seq_labels[b][m]).all().item()
                em_total   += 1

            # Auxiliary struct head accuracy
            enc_out = self.model.backbone(
                input_ids=input_ids, attention_mask=attn_mask
            ).last_hidden_state
            st_preds = self.struct_head(enc_out).argmax(-1)
            st_mask  = struct_labs != -100
            sa_correct += (st_preds[st_mask] == struct_labs[st_mask]).sum().item()
            sa_total   += st_mask.sum().item()

        p_structs = np.array(p_structs)
        g_structs = np.array(g_structs)

        uss = float(np.mean(p_structs == g_structs)) if len(g_structs) > 0 else 0.
        lss = float(sum(p == g for p, g in zip(p_joints, g_joints)) / max(len(g_joints), 1))

        return {
            'uss'       : uss,
            'lss'       : lss,
            'em'        : float(em_correct / max(em_total, 1)),
            'struct_acc': float(sa_correct / max(sa_total, 1)),
        }

    # ────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ────────────────────────────────────────────────────────────────────────

    def _save(self, epoch, score):
        ctx  = 'with_ctx' if self.args.use_context else 'no_ctx'
        path = os.path.join(
            os.getcwd(), 'saved_models',
            f'necti_{self.args.scheme}_{self.args.granularity}_{ctx}',
            'best_model.pt',
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch'           : epoch,
            'model_state_dict': self.model.state_dict(),
            'struct_head_state': self.struct_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score'           : score,
            'args'            : self.args,
        }, path)

    def _load_best(self):
        ctx  = 'with_ctx' if self.args.use_context else 'no_ctx'
        path = os.path.join(
            os.getcwd(), 'saved_models',
            f'necti_{self.args.scheme}_{self.args.granularity}_{ctx}',
            'best_model.pt',
        )
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.struct_head.load_state_dict(ckpt['struct_head_state'])
            print(f"Loaded best model (epoch={ckpt['epoch']}, score={ckpt['score']:.4f})")
        else:
            print("No saved model found.")


def main():
    from options import get_args
    args = get_args()
    trainer = NeCTIBitSchemeTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
