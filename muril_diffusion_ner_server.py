"""
Hierarchical Diffusion NER for Sanskrit — Server Version
2-stage: Stage 1 (coarse: PER/LOC/MISC/TIME/NORP/O) + Stage 2 (fine: 37 BIO labels)

Run from inside: ~/muril/muril_with_difussion/diff-comp/
Usage: /home/paramhans-pg/miniconda3/envs/llama_env/bin/python hierarchical_sanskrit_ner_server.py > hier_log.txt 2>&1 &
"""

import json, os, re, math, sys, glob, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer
from tqdm import tqdm
from prettytable import PrettyTable
import indic_transliteration
from indic_transliteration import sanscript

# ── Data paths ────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = "/home/pretam-pg/DiffusionSL/mahanama_ner/mahanama_final_fine_grained_0204_json.json"
TEST_DATA_PATH  = "/home/pretam-pg/DiffusionSL/mahanama_ner/mahanama_labelstudio_cleaned_test_data_0604_updated.json"
JSON_OUT_DIR    = "./ner_data/json"
OUTPUT_DIR      = "./output_hier"
GPU             = "cuda:1"   # change to cuda:0 if needed
MAX_CHECKPOINTS = 1          # keep only best checkpoint

os.makedirs(JSON_OUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)

# ── Memory & disk management ──────────────────────────────────────────────────
# Point HuggingFace cache inside project dir — avoids filling home partition
os.environ["HF_HOME"]            = os.path.abspath("./hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./hf_cache")
os.environ["HF_DATASETS_CACHE"]  = os.path.abspath("./hf_cache/datasets")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs("./hf_cache", exist_ok=True)

def check_disk_space(path=".", min_gb=5.0):
    """Warn if free disk space falls below min_gb."""
    stat    = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        print(f"WARNING: Only {free_gb:.1f}GB free on disk! Consider cleaning up.")
    return free_gb

def cleanup_checkpoints(output_dir, keep=1):
    """Keep only the `keep` newest best checkpoints, delete the rest."""
    ckpts = sorted(glob.glob(os.path.join(output_dir, "best_f1_*.pt")),
                   key=os.path.getmtime, reverse=True)
    for old_ckpt in ckpts[keep:]:
        os.remove(old_ckpt)
        print(f"  Deleted old checkpoint: {old_ckpt}")

def get_dir_size_gb(path):
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, files in os.walk(path) for f in files)
    return total / (1024**3)

print(f"Disk free: {check_disk_space():.1f} GB")
print(f"HF cache : {os.environ['HF_HOME']}")

# ── Label setup ───────────────────────────────────────────────────────────────
IGNORE_LABELS = {'unknown', 'vyakti_rūpa_sthānam'}

fine_labels = [
    "manuṣyaḥ", "īśvaraḥ", "devatā", "devayoniḥ", "ṛṣiḥ",
    "alaukikaprāṇī", "jantuḥ",
    "samūhaḥ",
    "prākṛtikasthānam", "alaukikasthānam", "janapadaḥ", "mānavanirmitaḥ",
    "śabdaḥ", "alaukika_acalanirjīvavastu", "acalanirjīvavastu",
    "alaukikacalanirjīvaḥ", "calanirjīvaḥ",
    "kālaḥ",
]

label_list = ["O"]
for fl in fine_labels:
    label_list.append(f"B-{fl}")
    label_list.append(f"I-{fl}")

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# ── Coarse hierarchy ──────────────────────────────────────────────────────────
COARSE_CATEGORIES = ['O', 'PER', 'LOC', 'MISC', 'TIME', 'NORP']

FINE_TO_COARSE_NAME = {
    'O'                          : 0,
    'manuṣyaḥ'                   : 1,
    'īśvaraḥ'                    : 1,
    'devatā'                     : 1,
    'devayoniḥ'                  : 1,
    'ṛṣiḥ'                       : 1,
    'alaukikaprāṇī'              : 1,
    'jantuḥ'                     : 1,
    'prākṛtikasthānam'           : 2,
    'alaukikasthānam'            : 2,
    'janapadaḥ'                  : 2,
    'mānavanirmitaḥ'             : 2,
    'acalanirjīvavastu'          : 3,
    'alaukika_acalanirjīvavastu' : 3,
    'calanirjīvaḥ'               : 3,
    'alaukikacalanirjīvaḥ'      : 3,
    'śabdaḥ'                     : 3,
    'kālaḥ'                      : 4,
    'samūhaḥ'                    : 5,
}

# ── Helper functions ──────────────────────────────────────────────────────────
def parse_entity_tag(ent_tag):
    if not ent_tag.startswith("("):
        return None
    content = ent_tag.lstrip("(").strip()
    parts   = content.split("-")
    if len(parts) < 3:
        return None
    fine = parts[2].rstrip(")")
    if fine in IGNORE_LABELS:
        return None
    return fine

def iast_to_dev(text):
    return indic_transliteration.sanscript.transliterate(
        text, sanscript.IAST, sanscript.DEVANAGARI
    )

def verse_to_bio(verse_data):
    tokens, ner_tags = [], []
    current_fine, in_entity = None, False
    for token_data in verse_data['tokens']:
        if not isinstance(token_data['id'], int):
            continue
        form    = iast_to_dev(token_data['form'])
        misc    = token_data.get('misc', {}) or {}
        ent_tag = str(misc.get('Entity', ''))
        if ent_tag.startswith("("):
            fine = parse_entity_tag(ent_tag)
            if fine: current_fine = fine
            if ent_tag.endswith(")"):
                bio_tag = f"B-{current_fine}" if current_fine else "O"
                in_entity, current_fine = False, None
            else:
                if current_fine:
                    bio_tag, in_entity = f"B-{current_fine}", True
                else:
                    bio_tag, in_entity, current_fine = "O", False, None
        elif ent_tag.endswith(")") and in_entity:
            bio_tag = f"I-{current_fine}" if current_fine else "O"
            in_entity, current_fine = False, None
        elif in_entity:
            bio_tag = f"I-{current_fine}" if current_fine else "O"
        else:
            bio_tag = "O"
        tokens.append(form)
        ner_tags.append(label2id.get(bio_tag, label2id["O"]))
    return tokens, ner_tags

def labelstudio_to_bio(entry):
    text_orig   = entry['data']['text']
    text        = iast_to_dev(text_orig)
    annotations = entry.get('annotations', [])
    char_ann_id = [-1] * len(text_orig)
    char_labels = ['O'] * len(text_orig)
    for ann_idx, ann in enumerate(annotations):
        fine_list = ann.get('fine_labels', [])
        if not fine_list: continue
        fine = fine_list[0]
        if fine in IGNORE_LABELS: continue
        if f"B-{fine}" not in label2id: continue
        for ci in range(ann['start'], ann['end']):
            char_labels[ci] = fine
            char_ann_id[ci] = ann_idx
    orig_words = text_orig.split()
    dev_words  = text.split()
    tokens, ner_tags = [], []
    cursor, prev_start = 0, -1
    for orig_word, dev_word in zip(orig_words, dev_words):
        start = text_orig.index(orig_word, cursor)
        end   = start + len(orig_word)
        word_char_labels = char_labels[start:end]
        word_char_annids = char_ann_id[start:end]
        if len(set(word_char_labels)) == 1 and word_char_labels[0] != 'O':
            fine   = word_char_labels[0]
            ann_id = word_char_annids[0]
            if prev_start == -1 or char_labels[prev_start] == 'O' \
                    or char_ann_id[prev_start] != ann_id:
                bio_tag = f'B-{fine}'
            else:
                bio_tag = f'I-{fine}'
        else:
            bio_tag = 'O'
        tokens.append(dev_word)
        ner_tags.append(label2id.get(bio_tag, label2id['O']))
        prev_start = start
        cursor     = end
    return tokens, ner_tags

# ── Load and build data ───────────────────────────────────────────────────────
print("Loading data...")
with open(TRAIN_DATA_PATH) as f:
    mahanama = json.load(f)
with open(TEST_DATA_PATH, encoding='utf-8') as f:
    loaded_test_data = json.load(f)

test_ids = set(d['data']['verse_id'] for d in loaded_test_data)

print("Building train data...")
train_tokens, train_tags = [], []
for volume_no, volume_data in mahanama['data']['mahanama_conllu'].items():
    for subchapter_no, subchapter_data in volume_data.items():
        for verse_data in subchapter_data:
            if verse_data['metadata']['sent_id'] in test_ids:
                continue
            tokens, ner_tags = verse_to_bio(verse_data)
            if tokens:
                train_tokens.append(tokens)
                train_tags.append(ner_tags)
print(f"Train verses: {len(train_tokens)}")

print("Building test data...")
test_tokens, test_tags = [], []
for entry in loaded_test_data:
    tokens, ner_tags = labelstudio_to_bio(entry)
    if tokens:
        test_tokens.append(tokens)
        test_tags.append(ner_tags)
print(f"Test entries: {len(test_tokens)}")

# ── Write JSON ────────────────────────────────────────────────────────────────
def write_json(tokens_list, tags_list, outpath):
    data = []
    for tokens, tag_ids in zip(tokens_list, tags_list):
        bio_labels = [id2label[tid] for tid in tag_ids]
        data.append({"sentence": tokens, "label": bio_labels})
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

write_json(train_tokens, train_tags, os.path.join(JSON_OUT_DIR, "train.json"))
write_json(test_tokens,  test_tags,  os.path.join(JSON_OUT_DIR, "test.json"))
write_json(test_tokens,  test_tags,  os.path.join(JSON_OUT_DIR, "dev.json"))
print(f"✓ JSON files written to {JSON_OUT_DIR}")

# ── Sanskrit NER Label Set ────────────────────────────────────────────────────
class SanskritNERLabelSet:
    def __init__(self, json_dir):
        all_labels = set()
        for split in ['train.json', 'dev.json', 'test.json']:
            path = os.path.join(json_dir, split)
            if not os.path.exists(path): continue
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                all_labels.update(item['label'])

        self._labelset = sorted(list(all_labels),
            key=lambda x: (x[2:] if x != 'O' else '', x[:2] if x != 'O' else ''))
        self._label2id = {l: i for i, l in enumerate(self._labelset)}
        self._id2label = {i: l for i, l in enumerate(self._labelset)}

        self._fine_to_coarse = {}
        for i, label in enumerate(self._labelset):
            fine = 'O' if label == 'O' else label[2:]
            self._fine_to_coarse[i] = FINE_TO_COARSE_NAME.get(fine, 0)

        print(f"Fine labels: {len(self._labelset)}, Coarse: {len(COARSE_CATEGORIES)}")
        for cid, cname in enumerate(COARSE_CATEGORIES):
            fine_ids = [i for i, c in self._fine_to_coarse.items() if c == cid]
            print(f"  {cname}: {[self._id2label[i] for i in fine_ids]}")

    def label2id(self, label):
        return self._label2id.get(label, self._label2id.get('O', 0))

    def id2label(self, idx):
        return self._id2label.get(idx, 'O')

    def fine_to_coarse(self, fine_id):
        return self._fine_to_coarse.get(fine_id, 0)

    def get_fine_to_coarse_map(self):
        return self._fine_to_coarse.copy()

    def __len__(self):
        return len(self._labelset)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SanskritNERDataset(Dataset):
    def __init__(self, json_dir, mode, label_set):
        path = os.path.join(json_dir, f"{mode}.json")
        with open(path, encoding='utf-8') as f:
            self.data = json.load(f)
        self.label_set = label_set

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'tokens': item['sentence'],
            'labels': [self.label_set.label2id(l) for l in item['label']]
        }


# ── Collator ──────────────────────────────────────────────────────────────────
class SanskritNERCollator:
    def __init__(self, tokenizer, label_set, max_length=128):
        self.tokenizer  = tokenizer
        self.label_set  = label_set
        self.max_length = max_length
        n = len(label_set)
        self._f2c = torch.zeros(n, dtype=torch.long)
        for fine_id, coarse_id in label_set.get_fine_to_coarse_map().items():
            self._f2c[fine_id] = coarse_id

    def __call__(self, batch):
        sentences = [item['tokens'] for item in batch]
        labels    = [item['labels'] for item in batch]
        enc = self.tokenizer(sentences, is_split_into_words=True, padding=True,
                             truncation=True, max_length=self.max_length, return_tensors='pt')
        word_ids   = [enc.word_ids(i) for i in range(len(batch))]
        seq_labels = []
        for wids, labs in zip(word_ids, labels):
            sl = []
            for i, wid in enumerate(wids):
                if wid is None:
                    sl.append(-100)
                elif i == 0 or wids[i] != wids[i-1]:
                    sl.append(labs[wid] if wid < len(labs) else -100)
                else:
                    sl.append(-100)
            seq_labels.append(sl)
        fine_labels   = torch.as_tensor(seq_labels, dtype=torch.long)
        coarse_labels = fine_labels.clone()
        mask = fine_labels != -100
        coarse_labels[mask] = self._f2c[fine_labels[mask].clamp(min=0)]
        return {
            'input_ids'      : enc.input_ids,
            'attention_mask' : enc.attention_mask,
            'fine_labels'    : fine_labels,
            'coarse_labels'  : coarse_labels,
        }


# ── Trainer ───────────────────────────────────────────────────────────────────
class HierarchicalSanskritNERTrainer:

    def __init__(self, args):
        self.args   = args
        self.device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.label_set = SanskritNERLabelSet(args.json_dir)

        from models.hierarchial_diffusion import HierarchicalDiffusionNeCTI
        self.model = HierarchicalDiffusionNeCTI(
            device             = self.device,
            backbone           = args.backbone,
            dim_model          = args.dim_model,
            freeze_bert        = False,
            time_steps         = args.time_steps,
            sampling_steps     = args.sampling_steps,
            noise_schedule     = args.noise_schedule,
            snr_scale          = args.snr_scale,
            global_depth       = args.global_depth,
            global_num_heads   = args.global_num_heads,
            num_coarse_classes = len(COARSE_CATEGORIES),
            local_depth        = args.local_depth,
            local_num_heads    = args.local_num_heads,
            local_window_size  = args.window_size,
            num_fine_classes   = len(self.label_set),
            fine_to_coarse_map = self.label_set.get_fine_to_coarse_map(),
            objective          = 'pred_x0',
            loss_type          = 'l2',
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        collator = SanskritNERCollator(self.tokenizer, self.label_set, args.max_length)

        train_ds = SanskritNERDataset(args.json_dir, 'train', self.label_set)
        dev_ds   = SanskritNERDataset(args.json_dir, 'dev',   self.label_set)
        test_ds  = SanskritNERDataset(args.json_dir, 'test',  self.label_set)

        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                       shuffle=True,  num_workers=4, collate_fn=collator, pin_memory=True)
        self.dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, collate_fn=collator, pin_memory=True)
        self.test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, collate_fn=collator, pin_memory=True)

        param_groups = [
            {'params': list(self.model.backbone.parameters()),   'lr': args.lr_bert},
            {'params': list(self.model.global_dit.parameters()), 'lr': args.lr_other},
            {'params': list(self.model.local_dit.parameters()),  'lr': args.lr_other},
        ]
        self.optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

        total_steps  = len(self.train_loader) * args.max_epochs
        warmup_steps = args.warmup_steps
        self.scheduler = SequentialLR(self.optimizer, [
            LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
        ], milestones=[warmup_steps])

        self.best_f1 = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, s1_loss, s2_loss, n = 0, 0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels    = batch['fine_labels'].to(self.device)
            coarse_labels  = batch['coarse_labels'].to(self.device)

            self.optimizer.zero_grad()
            losses = self.model(input_ids, attention_mask, coarse_labels, fine_labels, stage='both')
            loss   = losses['loss']
            if torch.isnan(loss): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            s1_loss    += losses['stage1_loss'].item()
            s2_loss    += losses['stage2_loss'].item()
            n          += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss/n, s1_loss/n, s2_loss/n

    @torch.no_grad()
    def eval_epoch(self, loader, desc='Evaluating'):
        self.model.eval()
        num_gold = num_pred = num_tp = num_label_true = num_label_all = 0

        for batch in tqdm(loader, desc=desc):
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels    = batch['fine_labels'].to(self.device)

            _, fine_preds = self.model.inference(input_ids, attention_mask)
            mask = fine_labels != -100

            for i in range(fine_labels.shape[0]):
                gl = fine_labels[i][mask[i]].cpu().tolist()
                pl = fine_preds[i][mask[i]].cpu().tolist()
                pl = [min(max(p, 0), len(self.label_set)-1) for p in pl]

                gold_ents = self._decode_bio(gl)
                pred_ents = self._decode_bio(pl)

                num_gold       += len(gold_ents)
                num_pred       += len(pred_ents)
                num_tp         += len(set(gold_ents) & set(pred_ents))
                num_label_true += sum(g == p for g, p in zip(gl, pl))
                num_label_all  += len(gl)

        p  = num_tp / num_pred if num_pred else 0.
        r  = num_tp / num_gold if num_gold else 0.
        f1 = 2*p*r/(p+r)       if (p+r)   else 0.
        label_acc = num_label_true / num_label_all if num_label_all else 0.

        t = PrettyTable()
        t.title = desc
        t.field_names = ['P', 'R', 'F1', 'num_pred', 'num_gold', 'num_tp', 'label_acc']
        t.add_row([f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}",
                   num_pred, num_gold, num_tp, f"{label_acc:.4f}"])
        print(t)
        return p, r, f1

    def _decode_bio(self, label_ids):
        entities = []
        labels   = [self.label_set.id2label(i) for i in label_ids]
        i = 0
        while i < len(labels):
            if labels[i].startswith('B-'):
                start, end, ent = i, i, labels[i][2:]
                j = i + 1
                while j < len(labels) and labels[j] == f'I-{ent}':
                    end = j
                    j  += 1
                entities.append(((start, end), ent))
                i = j
            else:
                i += 1
        return entities

    def save(self, name):
        path = os.path.join(OUTPUT_DIR, name)
        torch.save(self.model.state_dict(), path)
        print(f"Saved: {path}")
        cleanup_checkpoints(OUTPUT_DIR, keep=MAX_CHECKPOINTS)
        print(f"  Disk free after save: {check_disk_space():.1f} GB")

    def train(self):
        for epoch in range(self.args.max_epochs):
            tl, s1l, s2l = self.train_epoch(epoch)
            print(f"Epoch {epoch+1} | loss={tl:.4f} | stage1={s1l:.4f} | stage2={s2l:.4f}")
            p, r, f1 = self.eval_epoch(self.dev_loader, desc='Dev')
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.save(f"best_f1_{f1:.4f}.pt")
                print(f"  ✓ New best F1: {f1:.4f}")

            # check disk every 5 epochs
            if (epoch + 1) % 5 == 0:
                free = check_disk_space()
                print(f"  [Disk] {free:.1f} GB free | output_hier: {get_dir_size_gb(OUTPUT_DIR):.2f} GB | hf_cache: {get_dir_size_gb('./hf_cache'):.2f} GB")

        print(f"\nTraining complete. Best dev F1: {self.best_f1:.4f}")
        print("Final test evaluation:")
        self.eval_epoch(self.test_loader, desc='Test')


# ── Config & Run ──────────────────────────────────────────────────────────────
from argparse import Namespace

args = Namespace(
    json_dir        = JSON_OUT_DIR,
    backbone        = "google/muril-large-cased",
    dim_model       = 1024,
    gpu             = GPU,
    max_length      = 128,

    # diffusion
    time_steps      = 1000,
    sampling_steps  = 10,
    noise_schedule  = "linear",
    snr_scale       = 2.0,

    # stage 1 (global)
    global_depth    = 6,
    global_num_heads= 8,

    # stage 2 (local)
    local_depth     = 4,
    local_num_heads = 8,
    window_size     = 5,

    # training
    max_epochs      = 20,
    batch_size      = 16,
    lr_bert         = 1e-5,
    lr_other        = 1e-4,
    weight_decay    = 1e-5,
    warmup_steps    = 500,
)

if __name__ == '__main__':
    trainer = HierarchicalSanskritNERTrainer(args)
    trainer.train()