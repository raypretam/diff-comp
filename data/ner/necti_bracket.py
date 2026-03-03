"""
Bracketing encoding for NeCTI nested compound identification.

Adapts the bracket encoding from separ (Strzyz et al., 2019 / Ezquerro et al., 2024)
to the DepNeCTI CoNLL-U dependency format.

Each token gets two labels:
  - BRACKET: e.g. '<', '\\', '/', '>', '' (or combinations for k>1 planes)
  - COMPTYPE: compound relation label, e.g. 'Dvandva', 'Tatpurusha', 'No_rel', 'Comp_root'

The bracket sequence encodes the compound tree-structure.  The compound type
label encodes *what kind* of compound the arc represents.  Splitting these two
signals allows the diffusion model to reason about tree structure and arc type
independently at each denoising step.

Bracket symbol convention (matches separ):
  <  = opening, left arc  (dep < head positionally)
  \\ = closing, left arc
  /  = opening, right arc (dep > head positionally)
  >  = closing, right arc

For k-planar structures (nested compounds crossing multiple planes)
a plane index suffix '*^p' is appended (plane 0 = no suffix).
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional

BRACKET_SYMBOLS = ['\\', '>', '<', '/']
PLANE_CHAR = '*'
NULL_BRACKET = ''   # token not involved in any arc


# ---------------------------------------------------------------------------
# Low-level Bracket dataclass
# ---------------------------------------------------------------------------

class CompoundBracket:
    """One bracket symbol at one plane for one arc endpoint."""

    PRIORITY = ['\\', '>', '<', '/']

    def __init__(self, symbol: str, p: int = 0):
        assert symbol in self.PRIORITY, f"Unknown bracket symbol '{symbol}'"
        self.symbol = symbol
        self.p = p

    @property
    def order(self) -> Tuple[int, int]:
        return self.PRIORITY.index(self.symbol), self.p

    def __repr__(self) -> str:
        return self.symbol + PLANE_CHAR * self.p

    def __lt__(self, other: CompoundBracket) -> bool:
        return self.order < other.order

    def is_closing(self)  -> bool: return self.symbol in ('>', '\\')
    def is_opening(self)  -> bool: return self.symbol in ('<', '/')
    def is_left(self)     -> bool: return self.symbol in ('\\', '<')
    def is_right(self)    -> bool: return self.symbol in ('>', '/')
    def is_dep(self)      -> bool: return self.symbol in ('>', '<')
    def is_head(self)     -> bool: return self.symbol in ('\\', '/')

    @classmethod
    def from_string(cls, raw: str) -> List[CompoundBracket]:
        """Parse a bracket string like '<\\/' into a list of CompoundBracket objects."""
        tokens = []
        for c in raw:
            if c in cls.PRIORITY:
                tokens.append(c)
            else:
                tokens[-1] += c
        return [cls(t[0], t.count(PLANE_CHAR)) for t in tokens]


# ---------------------------------------------------------------------------
# Encoding / Decoding
# ---------------------------------------------------------------------------

def encode_sentence(
    sentence_data: List[Dict],
    k: int = 2,
) -> Tuple[List[str], List[str]]:
    """
    Encode one parsed sentence (list of token dicts from _parse_conllu) into
    per-token bracket strings + compound-type labels.

    Args:
        sentence_data: list of dicts with keys
            'idx'       : 1-based token index
            'token'     : surface form
            'comp_label': Comp2 / Comp3 / CompNo etc.
            'head_idx'  : 1-based head token index (0 = root)
            'relation'  : compound type string
        k: number of planes to encode (1 = projective, 2 = most nested compounds)

    Returns:
        brackets  : list[str] of length n, bracket string per token
        comptypes : list[str] of length n, compound type per token (the arc label)
    """
    n = len(sentence_data)
    # per-token lists of CompoundBracket objects (one list per token)
    bracket_lists: List[List[CompoundBracket]] = [[] for _ in range(n)]

    # Build arc list from dependency structure, sorted by span length
    arcs = []
    for item in sentence_data:
        dep = item['idx']       # 1-based
        # Clamp head indices that exceed sentence length (DUMMY token refs → root=0)
        head = item['head_idx'] if item['head_idx'] <= n else 0
        rel  = item['relation']
        if dep != head:         # skip self-loops (shouldn't exist but be safe)
            arcs.append({'dep': dep, 'head': head, 'rel': rel})

    # Decompose arcs into planes (greedy, non-crossing within each plane)
    planes = _assign_planes(arcs, k)

    for p, plane_arcs in enumerate(planes):
        # Clamp to k-1 so the bracket vocab stays bounded (mirror what decode does)
        pp = min(p, k - 1)
        for arc in plane_arcs:
            dep  = arc['dep']
            head = arc['head']
            if dep < head:          # left arc: dep is to the left of head
                # dep gets opening-left '<', head gets closing-left '\\'
                bracket_lists[dep  - 1].append(CompoundBracket('<',  pp))
                if head > 0:
                    bracket_lists[head - 1].append(CompoundBracket('\\', pp))
            else:                   # right arc: dep is to the right of head
                # dep gets closing-right '>', head gets opening-right '/'
                bracket_lists[dep  - 1].append(CompoundBracket('>',  pp))
                if head > 0:
                    bracket_lists[head - 1].append(CompoundBracket('/',  pp))

    # Sort brackets at each position by canonical order, then stringify
    brackets  = [''.join(repr(b) for b in sorted(blist)) for blist in bracket_lists]
    comptypes = [item['relation'] for item in sentence_data]

    return brackets, comptypes


def decode_sentence(
    brackets:  List[str],
    comptypes: List[str],
    k: int = 2,
) -> Tuple[List[Dict], bool]:
    """
    Decode per-token bracket strings back into an arc list.

    Returns:
        arcs      : list of {'dep', 'head', 'rel'} dicts (1-based indices)
        well_formed: True if all bracket stacks closed properly
    """
    planes_brackets = [[[] for _ in range(len(brackets))] for _ in range(k)]

    # Parse each token's bracket string and bin by plane
    for i, raw in enumerate(brackets):
        for br in CompoundBracket.from_string(raw):
            p = min(br.p, k - 1)
            planes_brackets[p][i].append(br)

    all_arcs: List[Dict] = []
    well_formed = True

    for p in range(k):
        arcs_p, wf_p = _decode_plane(planes_brackets[p], comptypes)
        all_arcs.extend(arcs_p)
        well_formed &= wf_p

    # deduplicate (same dep→head may appear in multiple planes)
    seen, unique = set(), []
    for arc in all_arcs:
        key = (arc['dep'], arc['head'])
        if key not in seen:
            seen.add(key)
            unique.append(arc)

    return unique, well_formed


def _decode_plane(
    plane_brackets: List[List[CompoundBracket]],
    comptypes: List[str],
) -> Tuple[List[Dict], bool]:
    """Decode one plane of bracket lists into arcs."""
    right_stack = [0]   # initialised with virtual root (index 0)
    left_stack  = []
    arcs = []
    well_formed = True

    for i, bracks in enumerate(plane_brackets):
        dep = i + 1   # 1-based
        for b in bracks:
            if b.is_opening():
                if b.is_left():
                    left_stack.append(dep)
                else:
                    right_stack.append(dep)
            else:   # closing
                if b.is_right():    # '>' : dep closes arc from right_stack top
                    if right_stack:
                        head = right_stack[-1]
                        if head != 0:
                            right_stack.pop()
                        arcs.append({'dep': dep, 'head': head, 'rel': comptypes[i]})
                    else:
                        well_formed = False
                else:               # '\\': head closes arc, dep from left_stack top
                    if left_stack:
                        d = left_stack.pop()
                        arcs.append({'dep': d, 'head': dep, 'rel': comptypes[d - 1]})
                    else:
                        well_formed = False

    right_stack.pop(0)   # remove virtual root
    if left_stack or right_stack:
        well_formed = False

    return arcs, well_formed


def _assign_planes(arcs: List[Dict], k: int) -> List[List[Dict]]:
    """
    Greedily assign arcs to planes so that arcs in the same plane do not cross.
    (A simplified version of the relaxed k-planar decomposition used in separ.)
    """
    planes: List[List[Dict]] = [[] for _ in range(k)]
    # Sort by span length so shorter arcs go into earlier planes first
    sorted_arcs = sorted(arcs, key=lambda a: abs(a['dep'] - a['head']))

    for arc in sorted_arcs:
        placed = False
        for p in range(k):
            if not _crosses_any(arc, planes[p]):
                planes[p].append(arc)
                placed = True
                break
        if not placed:
            # overflow goes into last plane (may cross — well-formedness degrades gracefully)
            planes[-1].append(arc)

    return planes


def _crosses(a1: Dict, a2: Dict) -> bool:
    """True if two arcs (given as dep/head 1-based indices) cross."""
    l1, r1 = sorted([a1['dep'], a1['head']])
    l2, r2 = sorted([a2['dep'], a2['head']])
    return l1 < l2 < r1 < r2 or l2 < l1 < r2 < r1


def _crosses_any(arc: Dict, plane_arcs: List[Dict]) -> bool:
    return any(_crosses(arc, a) for a in plane_arcs)


# ---------------------------------------------------------------------------
# Label Vocabulary for Bracket + Type joint space
# ---------------------------------------------------------------------------

class NeCTIBracketLabelSet:
    """
    Two-column label vocabulary for the bracket-augmented NeCTI task.

    Column 0 (BRACKET): all observed bracket strings (including NULL_BRACKET '')
    Column 1 (COMPTYPE): all observed compound type strings

    The diffusion model can either:
      (a) treat them as a joint label (bracket_str + SEP + comptype) → single head
      (b) use two separate classification heads (recommended for disentanglement)
    """

    SEP = '|'   # separator when encoding joint labels as a single string

    def __init__(self, data_path: str, granularity: str = 'Coarse',
                 use_context: bool = False, k: int = 2):
        import os
        from data.ner.necti_dataset import NeCTIDataset, NeCTILabelSet

        self.k = k
        self.granularity = granularity
        self.use_context = use_context

        # Collect raw label set from files first, then build bracket vocabs
        context_dir = 'With Context' if use_context else 'Without Context'
        self.data_path = os.path.join(data_path, context_dir, granularity)

        bracket_vocab: set = set()
        comptype_vocab: set = set()

        for split in ['train', 'dev', 'test']:
            filename = f"{granularity}_{split}_san"
            filepath = os.path.join(self.data_path, filename)
            if not os.path.exists(filepath):
                continue
            sentences = self._parse_conllu(filepath)
            for sent in sentences:
                bracks, ctypes = encode_sentence(sent, k=k)
                bracket_vocab.update(bracks)
                comptype_vocab.update(ctypes)

        # Canonical ordering: '' first for brackets; No_rel / Comp_root first for types
        self._bracket_list = [NULL_BRACKET] + sorted(
            [b for b in bracket_vocab if b != NULL_BRACKET])
        self._comptype_list = sorted(
            list(comptype_vocab),
            key=lambda x: (0 if x == 'No_rel' else (1 if x == 'Comp_root' else 2), x)
        )

        self._br2id   = {b: i for i, b in enumerate(self._bracket_list)}
        self._id2br   = {i: b for i, b in enumerate(self._bracket_list)}
        self._ct2id   = {c: i for i, c in enumerate(self._comptype_list)}
        self._id2ct   = {i: c for i, c in enumerate(self._comptype_list)}

        print(f"[BracketLabelSet] {len(self._bracket_list)} bracket types, "
              f"{len(self._comptype_list)} compound types  (k={k})")
        print(f"  Brackets : {self._bracket_list}")
        print(f"  CompTypes: {self._comptype_list}")

    # -- Bracket column --
    def br2id(self, b: str) -> int:
        return self._br2id.get(b, self._br2id.get(NULL_BRACKET, 0))

    def id2br(self, i: int) -> str:
        return self._id2br.get(i, NULL_BRACKET)

    @property
    def num_bracket_classes(self) -> int:
        return len(self._bracket_list)

    # -- CompType column --
    def ct2id(self, c: str) -> int:
        return self._ct2id.get(c, self._ct2id.get('No_rel', 0))

    def id2ct(self, i: int) -> str:
        return self._id2ct.get(i, 'No_rel')

    @property
    def num_comptype_classes(self) -> int:
        return len(self._comptype_list)

    # -- Total classes for joint head --
    @property
    def num_joint_classes(self) -> int:
        return self.num_bracket_classes * self.num_comptype_classes

    def joint2ids(self, joint_id: int) -> Tuple[int, int]:
        """Map a joint label id → (bracket_id, comptype_id)."""
        return divmod(joint_id, self.num_comptype_classes)

    def ids2joint(self, br_id: int, ct_id: int) -> int:
        return br_id * self.num_comptype_classes + ct_id

    # -- Helpers --
    def _parse_conllu(self, filepath: str) -> List[List[Dict]]:
        """Parse a CoNLL-U file into a list of sentences (each = list of token dicts)."""
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

    def __len__(self) -> int:
        return self.num_comptype_classes   # backward-compatible with NeCTILabelSet usage
