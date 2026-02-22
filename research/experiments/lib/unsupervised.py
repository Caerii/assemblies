"""
Unsupervised role binding from word-order statistics.

Combines the role discovery mechanism (shared category marker + LRI + mutual
inhibition) from test_role_self_organization.py with the prediction/binding
training from training.py, enabling sentence training WITHOUT role labels.

The key insight: structural roles are discovered dynamically during training.
When a noun appears, a shared NOUN_MARKER is projected to a pool of generic
STRUCT areas. Mutual inhibition picks the area with highest drive. LRI
suppresses that area's neurons, forcing the NEXT noun to a DIFFERENT area.
The discovered area then receives standard bidirectional binding training.

Prediction bridges are trained identically to the supervised version —
they only need word order, not role labels.
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from src.core.brain import Brain
from research.experiments.lib.vocabulary import Vocabulary, DEFAULT_VOCAB
from research.experiments.lib.brain_setup import activate_word, BrainConfig
from research.experiments.lib.training import train_prediction_pair, train_binding
from research.experiments.lib.measurement import measure_p600


@dataclass
class UnsupervisedConfig:
    """Configuration for unsupervised brain and training."""
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    n_struct_areas: int = 6
    refractory_period: int = 5
    inhibition_strength: float = 1.0
    stabilize_rounds: int = 3
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10


def create_unsupervised_brain(
    cfg: UnsupervisedConfig,
    vocab: Vocabulary = None,
    seed: int = 42,
) -> Tuple[Brain, List[str]]:
    """Create brain with generic STRUCT pool instead of named role areas.

    Returns:
        Tuple of (brain, struct_area_names).
    """
    v = vocab or DEFAULT_VOCAB
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    # Core areas (from vocabulary)
    for area_name in v.all_core_areas:
        brain.add_area(area_name, cfg.n, cfg.k, cfg.beta)

    # PREDICTION area (standard)
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

    # Shared category marker (fires for every noun)
    brain.add_stimulus("NOUN_MARKER", cfg.k)

    # Build word assemblies in core areas
    for area_name in v.all_core_areas:
        area_words = [w for w in v.all_words if v.core_area_for(w) == area_name]
        for word in area_words:
            brain._engine.reset_area_connections(area_name)
            activate_word(brain, word, area_name, cfg.lexicon_rounds)

    return brain, struct_areas


def discover_role_area(
    brain: Brain,
    marker_name: str,
    struct_areas: List[str],
    stabilize_rounds: int = 3,
    hebbian_stabilize: bool = False,
) -> Optional[str]:
    """Project shared category marker to struct pool; MI + LRI picks winner.

    Phase 1 (competition): plasticity OFF, marker -> all struct areas.
        Mutual inhibition picks area with highest total synaptic drive.
        Then losers' LRI is cleared so only winner is suppressed.

    Phase 2 (stabilization): marker -> winner only.
        Repeated projection builds cumulative LRI in winner area.
        If hebbian_stabilize is True, plasticity is ON for self-recurrence
        in the winner area, strengthening the internal assembly. The marker
        connections remain frozen to prevent winner-take-all collapse
        (where one area dominates regardless of LRI position).

    Returns name of winning structural area, or None.
    """
    brain.disable_plasticity = True
    brain.project({marker_name: list(struct_areas)}, {})

    winner = None
    for name in struct_areas:
        if len(brain.areas[name].winners) > 0:
            winner = name
            break

    if winner is None:
        brain.disable_plasticity = False
        return None

    # Clear losers' LRI
    for name in struct_areas:
        if name != winner:
            brain.clear_refractory(name)

    # Stabilization: build up LRI history in winner
    for _ in range(stabilize_rounds):
        if hebbian_stabilize:
            # Marker projection (plasticity OFF) for LRI buildup
            brain.project({marker_name: [winner]}, {})
            # Self-recurrence (plasticity ON) to strengthen internal assembly
            brain.disable_plasticity = False
            brain.project({}, {winner: [winner]})
            brain.disable_plasticity = True
        else:
            brain.project({marker_name: [winner]}, {})

    brain.disable_plasticity = False
    return winner


def train_sentence_unsupervised(
    brain: Brain,
    sentence: Dict[str, Any],
    vocab: Vocabulary,
    struct_areas: List[str],
    cfg: UnsupervisedConfig,
) -> Dict[str, str]:
    """Train one sentence without role labels.

    Prediction bridges are trained from word order (word[i] -> word[i+1]).
    Role binding is trained using dynamically discovered struct areas:
    each noun gets routed to a struct area via shared marker + MI + LRI.

    Args:
        brain: Brain instance with plasticity ON.
        sentence: Dict with 'words' and 'categories' (no 'roles' needed).
        vocab: Vocabulary for core area lookups.
        struct_areas: List of generic STRUCT area names.
        cfg: Configuration.

    Returns:
        Dict mapping word -> discovered struct area (for nouns that were bound).
    """
    words = sentence["words"]
    categories = sentence["categories"]
    bindings_made = {}

    # Clear LRI on all struct areas at sentence start
    for name in struct_areas:
        brain.clear_refractory(name)
    brain.inhibit_areas(struct_areas)

    for i, (word, cat) in enumerate(zip(words, categories)):
        core_area = vocab.core_area_for(word)

        # Train prediction: previous word -> this word
        if i > 0:
            prev_word = words[i - 1]
            prev_area = vocab.core_area_for(prev_word)
            train_prediction_pair(
                brain, prev_word, prev_area, word,
                cfg.train_rounds_per_pair)

        # For nouns (and locations): discover role and train binding
        if cat in ("NOUN", "LOCATION"):
            activate_word(brain, word, core_area, 3)
            role_area = discover_role_area(
                brain, "NOUN_MARKER", struct_areas, cfg.stabilize_rounds)
            if role_area is not None:
                train_binding(
                    brain, word, core_area, role_area, cfg.binding_rounds)
                bindings_made[word] = role_area

    return bindings_made


def build_role_mapping(
    brain: Brain,
    vocab: Vocabulary,
    struct_areas: List[str],
    test_sentences: List[Dict[str, Any]],
    stabilize_rounds: int = 3,
) -> Dict[str, str]:
    """Identify which struct areas correspond to which positional roles.

    Runs test sentences through the marker mechanism (no training) and
    records which area wins at each noun position.

    Returns:
        Dict mapping struct area name -> dominant position label
        (e.g., "STRUCT_0" -> "pos_0", "STRUCT_1" -> "pos_2").
    """
    area_positions: Dict[str, List[int]] = defaultdict(list)

    brain.disable_plasticity = True

    for sent in test_sentences:
        words = sent["words"]
        categories = sent["categories"]

        # Reset LRI
        for name in struct_areas:
            brain.clear_refractory(name)
        brain.inhibit_areas(struct_areas)

        noun_idx = 0
        for i, (word, cat) in enumerate(zip(words, categories)):
            if cat in ("NOUN", "LOCATION"):
                core_area = vocab.core_area_for(word)
                activate_word(brain, word, core_area, 3)
                winner = discover_role_area(
                    brain, "NOUN_MARKER", struct_areas, stabilize_rounds)
                if winner is not None:
                    area_positions[winner].append(noun_idx)
                noun_idx += 1

    brain.disable_plasticity = False

    # Assign each area its dominant position
    mapping = {}
    for area, positions in area_positions.items():
        if positions:
            dominant = Counter(positions).most_common(1)[0][0]
            mapping[area] = f"pos_{dominant}"

    return mapping


def train_binding_with_feedback(
    brain: Brain,
    word: str,
    core_area: str,
    role_area: str,
    binding_rounds: int,
    extra_rounds: int = 5,
    instability_threshold: float = 0.3,
    n_settling_rounds: int = 10,
) -> float:
    """Train binding with P600-guided retraining.

    Standard binding first, then measure anchored instability (the P600
    signal). If instability exceeds threshold, do extra binding rounds.
    This is error-driven learning using the system's own internal signal.

    Returns:
        Measured instability value.
    """
    train_binding(brain, word, core_area, role_area, binding_rounds)
    brain.disable_plasticity = True
    inst = measure_p600(brain, word, core_area, role_area, n_settling_rounds)
    brain.disable_plasticity = False
    if inst > instability_threshold:
        train_binding(brain, word, core_area, role_area, extra_rounds)
    return inst


def train_sentence_integrated(
    brain: Brain,
    sentence: Dict[str, Any],
    vocab: Vocabulary,
    struct_areas: List[str],
    cfg: "UnsupervisedConfig",
    use_hebbian_routing: bool = False,
    use_grounded_prediction: bool = False,
    use_p600_feedback: bool = False,
    extra_binding_rounds: int = 5,
    instability_threshold: float = 0.3,
) -> Dict[str, str]:
    """Train one sentence with optional interventions for ablation study.

    Extends train_sentence_unsupervised() with three toggleable improvements:
    - hebbian_routing: plasticity ON during role discovery stabilization
    - grounded_prediction: co-project sensory features into PREDICTION
    - p600_feedback: extra binding rounds when instability is high

    Args:
        brain: Brain instance with plasticity ON.
        sentence: Dict with 'words' and 'categories'.
        vocab: Vocabulary for core area lookups.
        struct_areas: List of generic STRUCT area names.
        cfg: Configuration (needs stabilize_rounds, train_rounds_per_pair,
             binding_rounds).
        use_hebbian_routing: Enable Hebbian routing consolidation.
        use_grounded_prediction: Enable sensory co-projection in prediction.
        use_p600_feedback: Enable P600-guided retraining.
        extra_binding_rounds: Extra rounds when P600 threshold exceeded.
        instability_threshold: P600 threshold for triggering extra training.

    Returns:
        Dict mapping word -> discovered struct area (for nouns that were bound).
    """
    # Lazy import to avoid circular dependency
    from research.experiments.lib.grounding import train_prediction_pair_grounded

    words = sentence["words"]
    categories = sentence["categories"]
    bindings_made = {}

    # Clear LRI on all struct areas at sentence start
    for name in struct_areas:
        brain.clear_refractory(name)
    brain.inhibit_areas(struct_areas)

    for i, (word, cat) in enumerate(zip(words, categories)):
        core_area = vocab.core_area_for(word)

        # Train prediction: previous word -> this word
        if i > 0:
            prev_word = words[i - 1]
            prev_area = vocab.core_area_for(prev_word)
            if use_grounded_prediction:
                train_prediction_pair_grounded(
                    brain, prev_word, prev_area, word,
                    cfg.train_rounds_per_pair)
            else:
                train_prediction_pair(
                    brain, prev_word, prev_area, word,
                    cfg.train_rounds_per_pair)

        # For nouns (and locations): discover role and train binding
        if cat in ("NOUN", "LOCATION"):
            activate_word(brain, word, core_area, 3)
            role_area = discover_role_area(
                brain, "NOUN_MARKER", struct_areas, cfg.stabilize_rounds,
                hebbian_stabilize=use_hebbian_routing)
            if role_area is not None:
                if use_p600_feedback:
                    train_binding_with_feedback(
                        brain, word, core_area, role_area,
                        cfg.binding_rounds, extra_binding_rounds,
                        instability_threshold)
                else:
                    train_binding(
                        brain, word, core_area, role_area,
                        cfg.binding_rounds)
                bindings_made[word] = role_area

    return bindings_made
