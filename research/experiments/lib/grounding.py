"""
Sensory grounding for Assembly Calculus language experiments.

Connects word assemblies to sensory features so that semantically related
words share neural overlap through shared sensory co-projection. This
creates assemblies that encode both phonological identity AND sensory
content, enabling:

  1. Semantic overlap: "dog" and "cat" share SENSE_ANIMAL -> higher overlap
  2. Semantic N400: related words are partially pre-activated by prediction
  3. Cross-modal activation: sensory input alone partially evokes word assemblies

The grounding mechanism uses the same co-projection that training.py uses
for prediction bridges — no new primitives, just additional stimuli.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from src.core.brain import Brain
from research.experiments.lib.vocabulary import Vocabulary, DEFAULT_VOCAB
from research.experiments.lib.brain_setup import BrainConfig, activate_word


# Sensory feature groups for the standard vocabulary.
# Each word maps to a list of shared sensory features.
# Words sharing features will develop overlapping assemblies.
SENSORY_FEATURES: Dict[str, List[str]] = {
    # Nouns: visual features
    "dog": ["SENSE_ANIMAL", "SENSE_VISUAL"],
    "cat": ["SENSE_ANIMAL", "SENSE_VISUAL"],
    "bird": ["SENSE_ANIMAL", "SENSE_VISUAL"],
    "boy": ["SENSE_PERSON", "SENSE_VISUAL"],
    "girl": ["SENSE_PERSON", "SENSE_VISUAL"],
    # Locations: spatial features
    "garden": ["SENSE_PLACE", "SENSE_SPATIAL"],
    "park": ["SENSE_PLACE", "SENSE_SPATIAL"],
    "house": ["SENSE_PLACE", "SENSE_SPATIAL"],
    "field": ["SENSE_PLACE", "SENSE_SPATIAL"],
    "river": ["SENSE_PLACE", "SENSE_SPATIAL"],
    # Verbs: motor features
    "chases": ["SENSE_MOTION", "SENSE_MOTOR"],
    "sees": ["SENSE_PERCEPTION", "SENSE_MOTOR"],
    "eats": ["SENSE_ACTION", "SENSE_MOTOR"],
    "finds": ["SENSE_PERCEPTION", "SENSE_MOTOR"],
    "hits": ["SENSE_ACTION", "SENSE_MOTOR"],
    # Prepositions: spatial features
    "in": ["SENSE_SPATIAL"],
    "on": ["SENSE_SPATIAL"],
    "at": ["SENSE_SPATIAL"],
    # Complementizer: no sensory grounding
    "that": [],
    # Novel words: no grounding (tests generalization)
    "table": [],
    "chair": [],
}

# All unique sensory stimulus names
ALL_SENSORY_STIMULI = sorted(set(
    feat for feats in SENSORY_FEATURES.values() for feat in feats
))

# Semantic groups for testing within-group vs between-group overlap
SEMANTIC_GROUPS = {
    "ANIMAL": ["dog", "cat", "bird"],
    "PERSON": ["boy", "girl"],
    "PLACE": ["garden", "park", "house", "field", "river"],
    "MOTION_VERB": ["chases", "hits"],
    "PERCEPTION_VERB": ["sees", "finds"],
}


def create_grounded_brain(
    cfg: BrainConfig,
    vocab: Vocabulary = None,
    seed: int = 42,
    grounding_rounds: int = 10,
) -> Brain:
    """Create brain with sensory-grounded word assemblies.

    Like create_language_brain() but co-projects sensory features during
    lexicon building, creating assemblies that encode both phonological
    and sensory information.

    Args:
        cfg: Brain configuration.
        vocab: Vocabulary specification.
        seed: Random seed.
        grounding_rounds: Rounds of sensory co-projection per word.

    Returns:
        Brain with grounded word assemblies.
    """
    v = vocab or DEFAULT_VOCAB
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    # Standard areas
    for area_name in v.all_core_areas:
        brain.add_area(area_name, cfg.n, cfg.k, cfg.beta)
    brain.add_area("PREDICTION", cfg.n, cfg.k, cfg.beta)

    # Role areas (standard)
    for area_name in v.all_role_areas:
        brain.add_area(area_name, cfg.n, cfg.k, cfg.beta)

    # Register word stimuli
    for word in v.all_words:
        brain.add_stimulus(f"PHON_{word}", cfg.k)

    # Register sensory stimuli
    for sense_name in ALL_SENSORY_STIMULI:
        brain.add_stimulus(sense_name, cfg.k)

    # Build grounded lexicon: co-project PHON + sensory features -> core area
    for area_name in v.all_core_areas:
        area_words = [w for w in v.all_words if v.core_area_for(w) == area_name]
        for word in area_words:
            brain._engine.reset_area_connections(area_name)

            features = SENSORY_FEATURES.get(word, [])

            # Phase 1: Standard lexicon building (phonological only)
            for _ in range(cfg.lexicon_rounds):
                brain.project(
                    {f"PHON_{word}": [area_name]},
                    {area_name: [area_name]},
                )

            # Phase 2: Sensory grounding (co-project PHON + sensory)
            if features:
                stim_dict = {f"PHON_{word}": [area_name]}
                for feat in features:
                    stim_dict[feat] = [area_name]

                for _ in range(grounding_rounds):
                    brain.project(stim_dict, {area_name: [area_name]})

    return brain


def ground_word(
    brain: Brain,
    word: str,
    core_area: str,
    rounds: int = 10,
) -> None:
    """Co-project word's phonological and sensory features into core area.

    Strengthens the association between the word's assembly and its
    sensory features.
    """
    features = SENSORY_FEATURES.get(word, [])
    if not features:
        # No grounding — just standard activation
        activate_word(brain, word, core_area, rounds)
        return

    stim_dict = {f"PHON_{word}": [core_area]}
    for feat in features:
        stim_dict[feat] = [core_area]

    brain.inhibit_areas([core_area])
    for _ in range(rounds):
        brain.project(stim_dict, {core_area: [core_area]})


def activate_sensory_only(
    brain: Brain,
    sensory_features: List[str],
    core_area: str,
    rounds: int = 3,
) -> np.ndarray:
    """Activate sensory features alone (no phonological input).

    Projects sensory stimuli into core_area without PHON_word.
    Tests whether sensory input alone can evoke word-like assemblies.

    Returns:
        Winner array in core_area after sensory activation.
    """
    brain.inhibit_areas([core_area])
    stim_dict = {feat: [core_area] for feat in sensory_features}
    for _ in range(rounds):
        brain.project(stim_dict, {core_area: [core_area]})
    return np.array(brain.areas[core_area].winners, dtype=np.uint32)


def train_prediction_pair_grounded(
    brain: Brain,
    context_word: str,
    context_area: str,
    target_word: str,
    rounds: int = 5,
) -> None:
    """Train prediction bridge with sensory co-projection.

    Like train_prediction_pair() but also projects the target word's
    sensory features into PREDICTION, so prediction assemblies encode
    semantic content (not just word identity). This means "cat" and "dog"
    produce overlapping PREDICTION assemblies (both carry SENSE_ANIMAL),
    enabling semantic N400 priming.
    """
    activate_word(brain, context_word, context_area, 3)
    brain.inhibit_areas(["PREDICTION"])
    features = SENSORY_FEATURES.get(target_word, [])
    stim_dict = {f"PHON_{target_word}": ["PREDICTION"]}
    for feat in features:
        stim_dict[feat] = ["PREDICTION"]
    for _ in range(rounds):
        brain.project(stim_dict, {context_area: ["PREDICTION"]})


def build_grounded_lexicon(
    brain: Brain,
    vocab: Vocabulary = None,
    readout_rounds: int = 5,
) -> Dict[str, np.ndarray]:
    """Build prediction lexicon with sensory co-projection.

    Like build_lexicon() but co-projects sensory features when reading out
    PREDICTION fingerprints. This matches the grounded prediction training:
    if prediction assemblies were trained with sensory features, the lexicon
    entries must also include them for N400 comparison to be valid.
    """
    v = vocab or DEFAULT_VOCAB
    lexicon = {}
    for word in v.all_words:
        brain.inhibit_areas(["PREDICTION"])
        features = SENSORY_FEATURES.get(word, [])
        stim_dict = {f"PHON_{word}": ["PREDICTION"]}
        for feat in features:
            stim_dict[feat] = ["PREDICTION"]
        for _ in range(readout_rounds):
            brain.project(stim_dict, {})
        lexicon[word] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
    return lexicon


@dataclass
class IntegratedConfig:
    """Configuration for integrated brain combining unsupervised + grounded."""
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    grounding_rounds: int = 10
    # Structural pool
    n_struct_areas: int = 6
    refractory_period: int = 5
    inhibition_strength: float = 1.0
    stabilize_rounds: int = 3
    # Training
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    extra_binding_rounds: int = 5
    instability_threshold: float = 0.3


def create_integrated_brain(
    cfg: IntegratedConfig,
    vocab: Vocabulary = None,
    seed: int = 42,
) -> Tuple[Brain, List[str]]:
    """Create brain combining unsupervised role discovery with sensory grounding.

    Merges the STRUCT pool (LRI + MI + NOUN_MARKER) from the unsupervised
    pattern with sensory co-projection during lexicon building from the
    grounding pattern. The result is a brain that:
    - Discovers structural roles dynamically (no pre-specified role areas)
    - Has word assemblies encoding both phonological and sensory content
    - Has sensory stimuli registered for co-projection into PREDICTION

    Returns:
        Tuple of (brain, struct_area_names).
    """
    v = vocab or DEFAULT_VOCAB
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    # Core areas
    for area_name in v.all_core_areas:
        brain.add_area(area_name, cfg.n, cfg.k, cfg.beta)

    # PREDICTION area
    brain.add_area("PREDICTION", cfg.n, cfg.k, cfg.beta)

    # Generic structural pool with LRI + mutual inhibition
    struct_areas = [f"STRUCT_{i}" for i in range(cfg.n_struct_areas)]
    for name in struct_areas:
        brain.add_area(
            name, cfg.n, cfg.k, cfg.beta,
            refractory_period=cfg.refractory_period,
            inhibition_strength=cfg.inhibition_strength,
        )
    brain.add_mutual_inhibition(struct_areas)

    # Register word stimuli
    for word in v.all_words:
        brain.add_stimulus(f"PHON_{word}", cfg.k)

    # Register sensory stimuli
    for sense_name in ALL_SENSORY_STIMULI:
        brain.add_stimulus(sense_name, cfg.k)

    # Shared category marker
    brain.add_stimulus("NOUN_MARKER", cfg.k)

    # Build grounded lexicon: PHON + sensory -> core area
    for area_name in v.all_core_areas:
        area_words = [w for w in v.all_words if v.core_area_for(w) == area_name]
        for word in area_words:
            brain._engine.reset_area_connections(area_name)

            # Phase 1: phonological lexicon
            activate_word(brain, word, area_name, cfg.lexicon_rounds)

            # Phase 2: sensory grounding
            features = SENSORY_FEATURES.get(word, [])
            if features:
                stim_dict = {f"PHON_{word}": [area_name]}
                for feat in features:
                    stim_dict[feat] = [area_name]
                for _ in range(cfg.grounding_rounds):
                    brain.project(stim_dict, {area_name: [area_name]})

    return brain, struct_areas
