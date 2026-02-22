"""
Brain construction and lexicon operations.

Handles the boilerplate of creating a Brain instance with the correct areas,
stimuli, and word assemblies for language experiments. Experiments call
create_language_brain() once and get back a fully initialized brain.

The key design choice: brain setup is vocabulary-driven. Adding a new word
category to the Vocabulary automatically creates the corresponding core area,
registers its stimuli, and builds its assemblies. No per-experiment wiring.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.core.brain import Brain
from research.experiments.lib.vocabulary import Vocabulary, DEFAULT_VOCAB


@dataclass
class BrainConfig:
    """Neural parameters for brain construction.

    These are the biophysical parameters shared across all language experiments.
    Experiment-specific parameters (training rounds, etc.) belong in the
    experiment's own config.
    """
    n: int = 10000       # neurons per area
    k: int = 100         # assembly size (winners per step)
    p: float = 0.05      # connection probability
    beta: float = 0.10   # Hebbian plasticity rate
    w_max: float = 20.0  # maximum synaptic weight
    lexicon_rounds: int = 20  # rounds for building word assemblies


def activate_word(brain: Brain, word: str, area: str, rounds: int):
    """Activate a word's assembly in its core area via stimulus projection.

    Inhibits the area first (clearing previous activation), then projects
    the word's phonological stimulus with self-recurrence to stabilize
    the assembly.

    Args:
        brain: Brain instance.
        word: Word string (stimulus will be PHON_{word}).
        area: Target core area.
        rounds: Number of projection rounds for stabilization.
    """
    brain.inhibit_areas([area])
    for _ in range(rounds):
        brain.project({f"PHON_{word}": [area]}, {area: [area]})


def create_language_brain(
    bcfg: BrainConfig,
    vocab: Vocabulary = None,
    seed: int = 42,
) -> Brain:
    """Create and initialize a Brain for language experiments.

    Steps:
      1. Create Brain with specified neural parameters
      2. Add all areas (core + role + PREDICTION) from vocabulary
      3. Register phonological stimuli for all words
      4. Build stable word assemblies in each core area

    Args:
        bcfg: Neural parameters.
        vocab: Vocabulary specification (defaults to DEFAULT_VOCAB).
        seed: Random seed for reproducibility.

    Returns:
        Fully initialized Brain ready for training.
    """
    v = vocab or DEFAULT_VOCAB
    brain = Brain(p=bcfg.p, seed=seed, w_max=bcfg.w_max)

    # Create all areas
    for area in v.all_areas:
        brain.add_area(area, bcfg.n, bcfg.k, bcfg.beta)

    # Register stimuli for all words (including novel words)
    for word in v.all_words:
        brain.add_stimulus(f"PHON_{word}", bcfg.k)

    # Build word assemblies in core areas
    # Group words by core area to reset connections per-area
    area_words: Dict[str, List[str]] = {}
    for word in v.all_words:
        area = v.core_area_for(word)
        area_words.setdefault(area, []).append(word)

    for area, words in area_words.items():
        for word in words:
            brain._engine.reset_area_connections(area)
            activate_word(brain, word, area, bcfg.lexicon_rounds)

    return brain


def build_lexicon(
    brain: Brain,
    vocab: Vocabulary = None,
    readout_rounds: int = 5,
) -> Dict[str, np.ndarray]:
    """Build prediction lexicon: stimulus-driven PREDICTION fingerprints.

    For each word, projects its phonological stimulus into PREDICTION
    (with plasticity OFF) and records the resulting assembly. These
    fingerprints are the reference patterns for N400 measurement.

    Args:
        brain: Brain instance (plasticity state is preserved).
        vocab: Vocabulary to build lexicon for.
        readout_rounds: Rounds of stimulus projection per word.

    Returns:
        Dict mapping word -> winner array in PREDICTION area.
    """
    v = vocab or DEFAULT_VOCAB
    lexicon = {}

    for word in v.all_words:
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(readout_rounds):
            brain.project({f"PHON_{word}": ["PREDICTION"]}, {})
        lexicon[word] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

    return lexicon
