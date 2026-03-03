"""
Bit4, Bit7 and HexaTagging encoding schemes for NeCTI nested compound identification.

Adapted from separ (Gómez-Rodríguez et al., 2023 / Amini et al., 2023) to the
DepNeCTI CoNLL-U dependency format.

WHY THIS WORKS NATURALLY WITH DIFFUSIONSL
==========================================
DiffusionSL's BitDit internally sets:
    self.bits = ceil(log2(num_labels))
and denoises in {-1,+1}^bits space.

  Scheme  | num_labels | self.bits | Comment
  --------|------------|-----------|--------
  Bit4    |     16     |     4     | 4-bit structural label IS the noise vector dim
  Bit7    |    128     |     7     | 7-bit structural label IS the noise vector dim
  Hexa    |     12     |     4     | joint hexa×fence id mapped to 4-bit space

For Bit4 and Bit7 the diffusion model literally denoises the structural encoding
bits — no information is wasted.  The DEPREL (compound type) is predicted by a
separate linear head on the XLM-R output.

SCHEME DEFINITIONS (from separ)
================================
Bit4  (projective, 4 bits per token — Gómez-Rodríguez et al., 2023):
  b0: is a right dependent  (head > dep)
  b1: is the outermost dependent of its parent
  b2: has left children
  b3: has right children
  → 16 distinct bit patterns

Bit7  (2-planar, 7 bits per token — Gómez-Rodríguez et al., 2023):
  b0: is a right dependent
  b1: belongs to the second plane (first plane = 0)
  b2: is the outermost dependent in its plane
  b3: has left children in plane 1
  b4: has right children in plane 1
  b5: has left children in plane 2
  b6: has right children in plane 2
  → 128 distinct bit patterns

HexaTagging (projective, 2 labels per token — Amini et al., 2023):
  HEXA : '>' (leaf opens right subtree) | '<' (leaf closes left subtree)
  FENCE: '>' (fencepost opens new node) | '<' (fencepost merges nodes)
  Constituent label L / R determines head direction.
  → 2×2 = 4 joint states per token (extended to include root boundary markers)
"""

from __future__ import annotations
import torch
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NULL_BITS    = '0' * 4   # Bit4 null (doesn't participate in any arc)
NULL_BITS7   = '0' * 7   # Bit7 null
HEXA_PAD     = '<pad>'
FENCE_PAD    = '<pad>'

HEXA_VOCAB  = ['>', '<']          # token tags
FENCE_VOCAB = ['>', '<', '<pad>'] # fencepost tags
CON_VOCAB   = ['L', 'R', '<pad>'] # constituent labels


# ---------------------------------------------------------------------------
# Bit4 encoder / decoder
# ---------------------------------------------------------------------------

def encode_bit4(sentence_data: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Encode a sentence using the 4-bit projective scheme.

    Args:
        sentence_data: list of token dicts with 'idx', 'head_idx', 'relation'

    Returns:
        bits     : list[str] of 4-char binary strings, one per token
        comptypes: list[str] compound type labels (DEPREL column)
    """
    n = len(sentence_data)

    # Adjacency bookkeeping: [dep_1based] → head_1based
    # Clamp head indices that exceed n (e.g. dangling refs to DUMMY token) to 0 (root)
    head = [min(item['head_idx'], n) if item['head_idx'] > n else item['head_idx']
             for item in sentence_data]   # 1-based, 0=root
    adj  = torch.zeros(n + 1, n + 1, dtype=torch.bool)
    for i, h in enumerate(head):
        dep = i + 1
        if h != dep:
            adj[dep, h] = True

    bits = [[False] * 4 for _ in range(n)]
    for idep, item in enumerate(sentence_data):
        dep = item['idx']
        h   = item['head_idx']

        # b0: right dependent (head is to the left, i.e. head index < dep index)
        bits[idep][0] = h < dep

        # b1: outermost dependent of its parent in that direction
        if h < dep:   # right dependent: we are the rightmost dep of h
            deps_of_h = adj[:, h].nonzero(as_tuple=True)[0]
            if len(deps_of_h):
                bits[idep][1] = deps_of_h.max().item() == dep
        elif h > 0:   # left dependent: we are the leftmost dep of h
            deps_of_h = adj[:, h].nonzero(as_tuple=True)[0]
            if len(deps_of_h):
                bits[idep][1] = deps_of_h.min().item() == dep

        # b2: has any left child
        bits[idep][2] = adj[:dep, dep].any().item()

        # b3: has any right child
        bits[idep][3] = adj[dep:, dep].any().item()

    bit_strs  = [''.join(map(str, map(int, b))) for b in bits]
    comptypes = [item['relation'] for item in sentence_data]
    return bit_strs, comptypes


def decode_bit4(bits: List[str], rels: List[str]) -> Tuple[List[Dict], bool]:
    """
    Decode 4-bit strings back into arcs.

    Returns:
        arcs       : list[{'dep', 'head', 'rel'}]
        well_formed: True if decoding is fully consistent
    """
    left, right = [], [0]   # stacks; right initialised with virtual root
    well_formed = True
    n = len(bits)
    adj = torch.zeros(n + 1, n + 1, dtype=torch.bool)

    for idep, label in enumerate(bits):
        dep = idep + 1
        if len(label) != 4:
            well_formed = False
            continue
        b0, b1, b2, b3 = [bool(int(c)) for c in label]

        if b0:   # right dependent
            if right:
                head = right[-1]
                adj[dep, head] = True
                if b1:
                    right.pop(-1)
            else:
                well_formed = False

        if b2:   # pull left children
            done = False
            while left and not done:
                child, is_outermost = left.pop(-1)
                adj[child, dep] = True
                done = is_outermost

        if not b0:   # left dependent → push onto left stack
            left.append((dep, b1))

        if b3:   # this token will have right children
            right.append(dep)

    well_formed = well_formed and (len(left) == 0) and (len(right) == 0)

    arcs = []
    for dep_t in range(1, n + 1):
        heads = adj[dep_t].nonzero(as_tuple=True)[0]
        if heads.numel() > 0:
            arcs.append({'dep': dep_t, 'head': heads[0].item(), 'rel': rels[dep_t - 1]})
        else:
            arcs.append({'dep': dep_t, 'head': 0, 'rel': rels[dep_t - 1]})

    return arcs, well_formed


def bit4_to_id(bit_str: str) -> int:
    """Convert 4-char binary string to class id 0-15."""
    return int(bit_str, 2) if len(bit_str) == 4 else 0


def id_to_bit4(class_id: int) -> str:
    return format(class_id, '04b')


# ---------------------------------------------------------------------------
# Bit7 encoder / decoder
# ---------------------------------------------------------------------------

def _plane_split(sentence_data: List[Dict]) -> Tuple[set, set, set, set]:
    """
    Partition arcs into two planes (greedy non-crossing assignment).
    Returns: (deps1, heads1, deps2, heads2) — sets of 1-based token indices.
    """
    arcs = [{'dep': item['idx'], 'head': item['head_idx']}
            for item in sentence_data if item['head_idx'] != item['idx']]
    plane1, plane2 = [], []

    for arc in sorted(arcs, key=lambda a: abs(a['dep'] - a['head'])):
        if not any(_arcs_cross(arc, a) for a in plane1):
            plane1.append(arc)
        else:
            plane2.append(arc)

    return (set(a['dep'] for a in plane1), set(a['head'] for a in plane1 if a['head'] > 0),
            set(a['dep'] for a in plane2), set(a['head'] for a in plane2 if a['head'] > 0))


def _arcs_cross(a1, a2) -> bool:
    l1, r1 = sorted([a1['dep'], a1['head']])
    l2, r2 = sorted([a2['dep'], a2['head']])
    return l1 < l2 < r1 < r2 or l2 < l1 < r2 < r1


def encode_bit7(sentence_data: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Encode a sentence using the 7-bit 2-planar scheme.

    Returns:
        bits     : list[str] of 7-char binary strings
        comptypes: list[str] compound type labels
    """
    n = len(sentence_data)
    # Clamp out-of-bounds head indices (DUMMY references) to 0 (root)
    head = [min(item['head_idx'], n) if item['head_idx'] > n else item['head_idx']
             for item in sentence_data]
    adj  = torch.zeros(n + 1, n + 1, dtype=torch.bool)
    for i, h in enumerate(head):
        adj[i + 1, h] = True

    deps1, heads1, deps2, heads2 = _plane_split(sentence_data)

    bits = [[False] * 7 for _ in range(n)]
    for idep, item in enumerate(sentence_data):
        dep = item['idx']
        h   = item['head_idx']

        bits[idep][0] = h < dep            # b0: right dep
        bits[idep][1] = dep in deps2       # b1: in plane 2

        # b2: outermost in its plane
        plane_deps = deps2 if dep in deps2 else deps1
        others = set(adj[:, h].nonzero(as_tuple=True)[0].tolist()) & plane_deps
        if others:
            bits[idep][2] = (max(others) if h < dep else min(others)) == dep

        # b3/b4: has left/right children in plane 1
        if dep in heads1:
            lc1 = set(adj[:dep, dep].nonzero(as_tuple=True)[0].tolist()) & deps1
            rc1 = set(adj[dep:, dep].nonzero(as_tuple=True)[0].tolist()) & deps1
            bits[idep][3] = len(lc1) > 0
            bits[idep][4] = len(rc1) > 0

        # b5/b6: has left/right children in plane 2
        if dep in heads2:
            lc2 = set(adj[:dep, dep].nonzero(as_tuple=True)[0].tolist()) & deps2
            rc2 = set(adj[dep:, dep].nonzero(as_tuple=True)[0].tolist()) & deps2
            bits[idep][5] = len(lc2) > 0
            bits[idep][6] = len(rc2) > 0

    bit_strs  = [''.join(map(str, map(int, b))) for b in bits]
    comptypes = [item['relation'] for item in sentence_data]
    return bit_strs, comptypes


def decode_bit7(bits: List[str], rels: List[str]) -> Tuple[List[Dict], bool]:
    """Decode 7-bit strings back into arcs."""
    left1, right1 = [], [0]
    left2, right2 = [], []
    well_formed = True
    n = len(bits)
    adj = torch.zeros(n + 1, n + 1, dtype=torch.bool)

    for idep, label in enumerate(bits):
        dep = idep + 1
        if len(label) != 7:
            well_formed = False
            continue
        b0, b1, b2, b3, b4, b5, b6 = [bool(int(c)) for c in label]

        # right dep in plane 1
        if b0 and not b1:
            if right1:
                adj[dep, right1[-1]] = True
                if b2: right1.pop(-1)
            else:
                well_formed = False

        # right dep in plane 2
        if b0 and b1:
            if right2:
                adj[dep, right2[-1]] = True
                if b2: right2.pop(-1)
            else:
                well_formed = False

        # pull left children from plane 1
        if b3:
            done = False
            while left1 and not done:
                child, outermost = left1.pop(-1)
                adj[child, dep] = True
                done = outermost

        # pull left children from plane 2
        if b5:
            done = False
            while left2 and not done:
                child, outermost = left2.pop(-1)
                adj[child, dep] = True
                done = outermost

        if not b0 and not b1: left1.append((dep, b2))
        if not b0 and b1:     left2.append((dep, b2))
        if b4: right1.append(dep)
        if b6: right2.append(dep)

    well_formed = well_formed and len(left1) == len(left2) == len(right1) == len(right2) == 0

    arcs = []
    for d in range(1, n + 1):
        hs = adj[d].nonzero(as_tuple=True)[0]
        arcs.append({'dep': d, 'head': hs[0].item() if hs.numel() > 0 else 0, 'rel': rels[d - 1]})
    return arcs, well_formed


def bit7_to_id(bit_str: str) -> int:
    return int(bit_str, 2) if len(bit_str) == 7 else 0


def id_to_bit7(class_id: int) -> str:
    return format(class_id, '07b')


# ---------------------------------------------------------------------------
# HexaTagging encoder / decoder (simplified BHT approach)
# ---------------------------------------------------------------------------
# Each token gets:
#   hexa : '>' or '<'  (2 classes)
#   fence: '>' or '<'  (2 classes, last token gets '<pad>')
# Joint id = hexa_id * 3 + fence_id   (using 0=>, 1=<, 2=pad)
# Total: 6 joint classes → ceil(log2(6))=3 bits in the diffusion model.

HEXA2ID  = {'>': 0, '<': 1}
FENCE2ID = {'>': 0, '<': 1, '<pad>': 2}
ID2HEXA  = {0: '>', 1: '<'}
ID2FENCE = {0: '>', 1: '<', 2: '<pad>'}
NUM_HEXA_CLASSES = 6   # 2 × 3


def hexa_pair_to_id(hexa: str, fence: str) -> int:
    return HEXA2ID.get(hexa, 0) * 3 + FENCE2ID.get(fence, 2)


def id_to_hexa_pair(joint_id: int) -> Tuple[str, str]:
    h_id, f_id = divmod(joint_id, 3)
    return ID2HEXA.get(h_id, '>'), ID2FENCE.get(f_id, '<pad>')


def encode_hexa(sentence_data: List[Dict]) -> Tuple[List[int], List[str]]:
    """
    Encode using hexa tagging via Binary Head Tree construction.

    NOTE: This is a simplified approximation of the full Amini et al. (2023)
    hexa encoding.  The full version requires converting the dependency tree
    to a binary head tree (PTB.Tree) using separ's dep2tree/tree2dep methods.
    For exact hexa round-trips, integrate separ's HexaTaggingDependencyParser
    directly.  Bit4 and Bit7 are exact and recommended for NeCTI.

    Returns:
        joint_ids : list[int] of length n, joint hexa×fence class ids
        comptypes : list[str] compound type labels
    """
    n = len(sentence_data)
    if n <= 1:
        # trivially single token → '>' tag, '<pad>' fence
        return [hexa_pair_to_id('>', '<pad>')] * n, [item['relation'] for item in sentence_data]

    head = [item['head_idx'] for item in sentence_data]

    # Build the BHT via the recursive dep2tree procedure
    hexas, fences = _encode_bht_from_deps(head, n)

    # Extend to n tokens (fence has n-1 real values; pad the last)
    fences = fences + ['<pad>']

    joint_ids = [hexa_pair_to_id(h, f) for h, f in zip(hexas, fences)]
    comptypes = [item['relation'] for item in sentence_data]
    return joint_ids, comptypes


def _encode_bht_from_deps(head: List[int], n: int) -> Tuple[List[str], List[str]]:
    """
    Lightweight hexa encoder using the interleaved traversal described in
    Amini et al. (2023).  We perform an Euler tour of the dependency tree,
    visiting left subtrees first (pre-order), and emit:
      '>' when a leaf opens a new right subtree
      '<' when a leaf closes a left subtree
    Fencepost tags indicate whether the next node opens a new sub-tree ('>') 
    or merges (closes) the current top-of-stack ('<').
    """
    # Build children lists
    left_children  = [[] for _ in range(n + 1)]  # 1-based + root
    right_children = [[] for _ in range(n + 1)]

    root = 0
    for dep_0based, h in enumerate(head):
        dep = dep_0based + 1
        if h == 0:
            root = dep
        elif dep < h:
            left_children[h].append(dep)
        else:
            right_children[h].append(dep)

    for i in range(n + 1):
        left_children[i].sort(reverse=True)  # process rightmost-left first
        right_children[i].sort()             # process leftmost-right first

    hexas, fences = [], []
    stack = [(root, False)]   # (node, is_right_subtree)

    def visit(node):
        # Visit left subtrees first
        for lc in left_children[node]:
            visit(lc)
            if fences:
                fences[-1] = '>'   # there is a next node → open direction
            stack.append((node, False))

        # Emit leaf token
        hexas.append('>' if (not stack or stack[-1][1]) else '<')
        if len(hexas) < n:
            fences.append('<')   # default: merge; overwritten if needed

        # Visit right subtrees
        for rc in right_children[node]:
            if fences:
                fences[-1] = '>'
            stack.append((node, True))
            visit(rc)

    visit(root)

    # Pad / trim to exactly n and n-1
    hexas  = (hexas  + ['>'] * n)[:n]
    fences = (fences + ['<'] * (n - 1))[:(n - 1)]
    return hexas, fences


def decode_hexa(joint_ids: List[int], rels: List[str]) -> Tuple[List[Dict], bool]:
    """
    Decode joint hexa×fence ids back into arcs using stack-based BHT reconstruction.
    Falls back gracefully on malformed sequences.
    """
    n = len(joint_ids)
    if n == 0:
        return [], True

    hexas  = [id_to_hexa_pair(j)[0] for j in joint_ids]
    fences = [id_to_hexa_pair(j)[1] for j in joint_ids]

    # Stack-based reconstruction
    stack = []
    well_formed = True
    parents = [0] * (n + 1)   # parent[dep 1-based] = head 1-based

    right_stack  = [0]   # virtual root on right
    left_stack   = []
    pos = 1

    for i, (h, f) in enumerate(zip(hexas, fences)):
        dep = i + 1
        if h == '>':
            if right_stack:
                parents[dep] = right_stack[-1]
            else:
                well_formed = False
        else:   # '<'
            if left_stack:
                parents[dep] = left_stack.pop(-1)
            else:
                well_formed = False

        if f == '>' and i < n - 1:
            right_stack.append(dep)
        elif f == '<' and i < n - 1:
            if right_stack and right_stack[-1] != 0:
                right_stack.pop(-1)
            left_stack.append(dep)

    arcs = [{'dep': d, 'head': parents[d], 'rel': rels[d - 1]} for d in range(1, n + 1)]
    return arcs, well_formed


# ---------------------------------------------------------------------------
# Unified vocabulary builder
# ---------------------------------------------------------------------------

class NeCTIBitSchemeLabelSet:
    """
    Manages the label vocabulary for a specific bit scheme.

    scheme: 'bit4'  → 16 structural classes + N comptype classes
            'bit7'  → 128 structural classes + N comptype classes
            'hexa'  → 6  structural classes + N comptype classes

    Joint encoding strategy (for single-head BitDit):
        joint_id = struct_id * num_comptypes + comptype_id
    """

    SCHEMES = {'bit4': (16, encode_bit4, decode_bit4),
               'bit7': (128, encode_bit7, decode_bit7),
               'hexa': (NUM_HEXA_CLASSES, None, decode_hexa)}

    def __init__(self, data_path: str, granularity: str = 'Coarse',
                 use_context: bool = False, scheme: str = 'bit4'):
        import os
        assert scheme in self.SCHEMES, f"scheme must be one of {list(self.SCHEMES)}"
        self.scheme          = scheme
        self.granularity     = granularity
        self.use_context     = use_context
        self.num_struct      = self.SCHEMES[scheme][0]

        context_dir  = 'With Context' if use_context else 'Without Context'
        self.data_path = os.path.join(data_path, context_dir, granularity)

        # Collect comptype vocabulary
        comptype_vocab = set()
        for split in ('train', 'dev', 'test'):
            fp = os.path.join(self.data_path, f"{granularity}_{split}_san")
            if not os.path.exists(fp):
                continue
            for sent in self._parse_conllu(fp):
                for item in sent:
                    comptype_vocab.add(item['relation'])

        self._comptype_list = sorted(
            list(comptype_vocab),
            key=lambda x: (0 if x == 'No_rel' else (1 if x == 'Comp_root' else 2), x)
        )
        self._ct2id = {c: i for i, c in enumerate(self._comptype_list)}
        self._id2ct = dict(enumerate(self._comptype_list))

        print(f"[BitSchemeLabelSet:{scheme}] {self.num_struct} struct classes × "
              f"{len(self._comptype_list)} compound types "
              f"= {self.num_joint_classes} joint classes")
        print(f"  CompTypes: {self._comptype_list}")

    @property
    def num_comptype_classes(self) -> int:
        return len(self._comptype_list)

    @property
    def num_joint_classes(self) -> int:
        return self.num_struct * self.num_comptype_classes

    def ct2id(self, c: str) -> int:
        return self._ct2id.get(c, self._ct2id.get('No_rel', 0))

    def id2ct(self, i: int) -> str:
        return self._id2ct.get(i, 'No_rel')

    @staticmethod
    def _sanitize(sentence_data: List[Dict]) -> List[Dict]:
        """Clamp head indices that exceed sentence length (DUMMY token refs → root=0)."""
        n = len(sentence_data)
        cleaned = []
        for item in sentence_data:
            h = item['head_idx']
            cleaned.append({**item, 'head_idx': h if h <= n else 0})
        return cleaned

    def encode_sentence(self, sentence_data: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        Returns:
            struct_ids  : structural class id per token
            comptype_ids: compound type id per token
        """
        sentence_data = self._sanitize(sentence_data)
        if self.scheme == 'bit4':
            bits, ctypes = encode_bit4(sentence_data)
            struct_ids = [bit4_to_id(b) for b in bits]
        elif self.scheme == 'bit7':
            bits, ctypes = encode_bit7(sentence_data)
            struct_ids = [bit7_to_id(b) for b in bits]
        else:   # hexa
            struct_ids, ctypes = encode_hexa(sentence_data)

        comptype_ids = [self.ct2id(c) for c in ctypes]
        return struct_ids, comptype_ids

    def joint_id(self, struct_id: int, ct_id: int) -> int:
        return struct_id * self.num_comptype_classes + ct_id

    def split_joint(self, joint_id: int) -> Tuple[int, int]:
        return divmod(joint_id, self.num_comptype_classes)

    def decode_sentence(self, joint_ids: List[int], ignore_index: int = -100) -> List[Dict]:
        """Decode joint ids back to arcs."""
        valid = [j for j in joint_ids if j != ignore_index]
        if not valid:
            return []
        struct_ids   = [self.split_joint(j)[0] for j in valid]
        comptype_ids = [self.split_joint(j)[1] for j in valid]
        rels = [self.id2ct(c) for c in comptype_ids]

        if self.scheme == 'bit4':
            arcs, _ = decode_bit4([id_to_bit4(s) for s in struct_ids], rels)
        elif self.scheme == 'bit7':
            arcs, _ = decode_bit7([id_to_bit7(s) for s in struct_ids], rels)
        else:
            arcs, _ = decode_hexa(struct_ids, rels)
        return arcs

    def _parse_conllu(self, filepath: str) -> List[List[Dict]]:
        sentences, current = [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current:
                        sentences.append(current)
                        current = []
                elif not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 8:
                        current.append({
                            'idx'       : int(parts[0]),
                            'token'     : parts[1].split('_')[0],
                            'comp_label': parts[3],
                            'head_idx'  : int(parts[6]),
                            'relation'  : parts[7],
                        })
        if current:
            sentences.append(current)
        return sentences

    def __len__(self):
        return self.num_comptype_classes
