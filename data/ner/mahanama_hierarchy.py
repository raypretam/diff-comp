"""
Label hierarchy for the mahanama NER dataset.

The fine-grained labels are BMES tags (e.g. 'B-manuṣyaḥ', 'S-janapadaḥ', 'O').
Each entity type belongs to a coarse category. This module exposes that
two-level hierarchy so models can use it (e.g. hyperbolic hierarchical
contrastive learning).

Coarse categories: Person, Location, NORP, Misc, Time, O
"""

from typing import List, Tuple


COARSE_CATEGORIES = ['Person', 'Location', 'NORP', 'Misc', 'Time', 'O']
COARSE2ID = {c: i for i, c in enumerate(COARSE_CATEGORIES)}

# entity type (BMES suffix) -> coarse category
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
    ent = fine_label.split('-', 1)[1] if '-' in fine_label else fine_label
    return COARSE2ID[ENTITY_TO_COARSE.get(ent, 'Misc')]


def build_fine_to_coarse(label_set) -> Tuple[List[int], int, int]:
    """
    Build the fine -> coarse id mapping for a LabelSet1D.

    Args:
        label_set: object with id2label(int)->str and __len__.

    Returns:
        fine_to_coarse: list of length num_fine, fine id -> coarse id
        num_coarse:     number of coarse categories
        o_label_id:     fine id of the 'O' label (-1 if absent)
    """
    fine_to_coarse = []
    o_label_id = -1
    for fid in range(len(label_set)):
        name = label_set.id2label(fid)
        if name == 'O':
            o_label_id = fid
        fine_to_coarse.append(fine_label_to_coarse_id(name))
    return fine_to_coarse, len(COARSE_CATEGORIES), o_label_id
