"""
Fuzzy readout: map neural assemblies back to symbolic labels.

Implements the readout step from Mitropolsky & Papadimitriou (2023):
given an assembly and a lexicon of known word→assembly mappings,
find the best-matching word (or return None for an improper parse).

READOUT IS AN EXPERIMENTER'S INSTRUMENT, NOT A BRAIN MECHANISM.  Nothing in
NEMO converts an assembly back into a symbol; there is no area that holds
"the word dog" as a label.  This module exists so that a human can ask what
the network is representing, by comparing the live assembly against snapshots
taken when known words were presented.  It is the equivalent of decoding from
recorded activity, and like any decoder it can be wrong in ways the network
itself is not.

Two consequences follow.  First, the threshold in :func:`fuzzy_readout` is a
DECODER's confidence criterion, not a neural one -- returning ``None`` means
"I cannot tell what this is", not "the network failed".  Second, readout can
only ever name things already in the lexicon, so a novel assembly reads out as
its nearest known neighbour, or as nothing.  Absence of a readout is weak
evidence about what the network computed.

Functions:
    fuzzy_readout   Best-matching word above threshold, or None.
    readout_all     All words with overlaps, sorted descending.
    build_lexicon   Project each word's stimulus and snapshot the assembly.

Reference:
    Mitropolsky, D. & Papadimitriou, C. H. (2023).
    "The Architecture of a Biologically Plausible Language Organ."
    arXiv:2306.15364.
"""

from typing import Dict, List, Optional, Tuple

from .assembly import Assembly, overlap
from .ops import project
from .contracts import LEXICON_BUILD_CONTRACT, READOUT_CONTRACT, LexiconBuildPlan, ReadoutPlan, implements


# Type alias: word string → Assembly snapshot
Lexicon = Dict[str, Assembly]


@implements(READOUT_CONTRACT)
def fuzzy_readout(assembly: Assembly, lexicon: Lexicon,
                  threshold: float = 0.7) -> Optional[str]:
    """Return the best-matching word above *threshold*, or None.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-readout-threshold

    An assembly that doesn't match any known word above the threshold
    indicates an improper parse (no valid symbolic interpretation).

    Args:
        assembly: The neural assembly to read out.
        lexicon: Mapping of word labels to their reference assemblies.
        threshold: Minimum overlap required (0.0 to 1.0).

    Returns:
        The word with highest overlap if it exceeds *threshold*,
        otherwise None (improper parse).

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-readout
    """
    plan = ReadoutPlan(assembly, lexicon, threshold)
    assembly, lexicon, threshold = plan.assembly, plan.lexicon, plan.threshold
    if not lexicon:
        return None

    best_word = None
    best_overlap = -1.0

    for word, ref_assembly in lexicon.items():
        ov = overlap(assembly, ref_assembly)
        if (ov > best_overlap
                or (ov == best_overlap
                    and (best_word is None or word < best_word))):
            best_overlap = ov
            best_word = word

    if best_overlap >= threshold:
        return best_word
    return None


def readout_all(assembly: Assembly,
                lexicon: Lexicon) -> List[Tuple[str, float]]:
    """Return all words with their overlaps, sorted descending.

    Useful for diagnostics and debugging readout quality.

    Args:
        assembly: The neural assembly to read out.
        lexicon: Mapping of word labels to their reference assemblies.

    Returns:
        List of (word, overlap) tuples sorted by overlap descending.
    """
    results = []
    for word, ref_assembly in lexicon.items():
        ov = overlap(assembly, ref_assembly)
        results.append((word, ov))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


@implements(LEXICON_BUILD_CONTRACT)
def build_lexicon(brain, area: str, words: List[str],
                  stimuli_map: Dict[str, str],
                  rounds: int = 10) -> Lexicon:
    """Build a lexicon by projecting each word's stimulus into an area.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-lexicon-build

    For each word, projects the corresponding stimulus into the target
    area for *rounds* steps (with recurrence) and snapshots the result.

    Between words, the area's recurrent connections are reset so each
    word gets an independent assembly (no attractor carryover).

    Args:
        brain: Brain instance with stimuli and target area already added.
        area: Name of the target area.
        words: List of word labels.
        stimuli_map: Maps each word to its stimulus name.
        rounds: Projection rounds per word (default 10).

    Returns:
        Lexicon mapping each word to its Assembly snapshot.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-lexicon-build
    """
    plan = LexiconBuildPlan(area, tuple(words), stimuli_map, rounds)
    plan.preflight(brain)
    area, words, stimuli_map, rounds = plan.area, plan.words, plan.stimuli_map, plan.rounds

    lexicon: Lexicon = {}

    for word in words:
        stim_name = stimuli_map[word]
        assembly = project(brain, stim_name, area, rounds=rounds)
        lexicon[word] = assembly

        # Reset recurrent connections so the next word starts fresh
        brain.reset_area_connections(area)

    return lexicon
