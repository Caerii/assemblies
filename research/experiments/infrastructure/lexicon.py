"""
Core Lexicon Builder

Projects each word into its core area in isolation and records compact
indices for later measurement. The lexicon maps words to their core
area, assembly neuron positions, and assembly objects.
"""

from typing import Dict, List, Any

import numpy as np

from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.ops import project, _snap


def build_core_lexicon(
    parser: EmergentParser,
    words: List[str],
    rounds: int,
) -> Dict[str, Dict[str, Any]]:
    """For each word, project phon->core in isolation and record compact indices.

    Uses inhibit_areas (clear activation) rather than reset_area_connections
    (which destroys trained Hebbian weights).  The compact indices correspond
    to positions in the engine's all_inputs vector.

    We also store the mapped Assembly for overlap measurements.
    """
    lexicon = {}
    for word in words:
        core = parser._word_core_area(word)
        phon = parser.stim_map.get(word)
        if phon is None:
            continue
        # Clear activation only — keep trained Hebbian weights
        parser.brain.inhibit_areas([core])
        project(parser.brain, phon, core, rounds=rounds)
        # Get compact indices (these ARE the indices into all_inputs[0:w])
        compact_winners = np.array(
            parser.brain.areas[core].winners, dtype=np.uint32)
        # Also store mapped assembly for reference
        asm = _snap(parser.brain, core)
        lexicon[word] = {
            "core_area": core,
            "compact_winners": compact_winners,
            "assembly": asm,
        }
    return lexicon
