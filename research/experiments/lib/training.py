"""
Training protocols for prediction bridges and role bindings.

Two fundamental training operations, composed to train arbitrary sentences:

  train_prediction_pair: Builds a Hebbian bridge so that context-area
      activation evokes the target word's assembly in PREDICTION. This is
      the mechanism behind N400 — forward prediction from accumulated context.

  train_binding: Bidirectional co-projection binds a word's assembly to a
      structural role slot. This is the mechanism behind P600 — whether a
      structural pathway can sustain a pattern without continued stimulus.

  train_sentence: Composes prediction pairs and bindings for an entire
      sentence, driven by the grammar output (word list + role annotations).
      Handles sentences of any length — the prediction chain naturally extends
      to each word predicting the next, and each noun binds to its annotated role.

Design principle: training operations are agnostic to sentence structure.
The grammar module decides WHAT to train (which word pairs, which bindings);
this module decides HOW (co-projection rounds, bidirectional binding, etc.).
"""

from typing import Dict, Any

from src.core.brain import Brain
from research.experiments.lib.vocabulary import Vocabulary, DEFAULT_VOCAB
from research.experiments.lib.brain_setup import activate_word


def train_prediction_pair(
    brain: Brain,
    context_word: str,
    context_area: str,
    target_word: str,
    rounds: int = 5,
):
    """Train one prediction bridge: context -> target in PREDICTION area.

    Activates context_word in context_area, then co-projects context_area
    and PHON_target into PREDICTION. After training, context_area activation
    alone will partially evoke the target's PREDICTION assembly.

    Args:
        brain: Brain instance with plasticity ON.
        context_word: Word providing predictive context.
        context_area: Core area of context word.
        target_word: Word being predicted.
        rounds: Co-projection rounds.
    """
    activate_word(brain, context_word, context_area, 3)
    brain.inhibit_areas(["PREDICTION"])
    for _ in range(rounds):
        brain.project(
            {f"PHON_{target_word}": ["PREDICTION"]},
            {context_area: ["PREDICTION"]},
        )


def train_binding(
    brain: Brain,
    word: str,
    core_area: str,
    role_area: str,
    rounds: int = 10,
):
    """Train one role binding: word <-> role slot (bidirectional).

    Activates word in core_area, then runs bidirectional co-projection
    between core_area and role_area with the word's stimulus. After training,
    the structural pathway can sustain the word's pattern in the role area
    without continued stimulus (low anchored instability = low P600).

    Args:
        brain: Brain instance with plasticity ON.
        word: Word to bind.
        core_area: Core area with word's assembly.
        role_area: Target role area.
        rounds: Bidirectional co-projection rounds.
    """
    activate_word(brain, word, core_area, 3)
    brain.inhibit_areas([role_area])
    for _ in range(rounds):
        brain.project(
            {f"PHON_{word}": [core_area, role_area]},
            {core_area: [role_area],
             role_area: [core_area]},
        )


def train_sentence(
    brain: Brain,
    sentence: Dict[str, Any],
    vocab: Vocabulary = None,
    prediction_rounds: int = 5,
    binding_rounds: int = 10,
):
    """Train prediction bridges and role bindings for one sentence.

    Driven entirely by the grammar output: the word list determines prediction
    pairs (each word predicts the next), and the role annotations determine
    which nouns bind to which role slots.

    This function handles sentences of any length — SVO, SVO+PP, or deeper
    structures. No length-specific logic.

    Args:
        brain: Brain instance with plasticity ON.
        sentence: Grammar output dict with words, roles, categories.
        vocab: Vocabulary for area lookups.
        prediction_rounds: Rounds per prediction pair.
        binding_rounds: Rounds per role binding.
    """
    v = vocab or DEFAULT_VOCAB
    words = sentence["words"]
    roles = sentence["roles"]

    # Prediction training: each word predicts the next
    for i in range(len(words) - 1):
        context_word = words[i]
        target_word = words[i + 1]
        context_area = v.core_area_for(context_word)
        train_prediction_pair(
            brain, context_word, context_area, target_word, prediction_rounds)

    # Binding training: bind nouns/locations to their role slots
    bindable_roles = {"AGENT", "PATIENT", "PP_OBJ"}
    for word, role in zip(words, roles):
        if role in bindable_roles:
            core_area = v.core_area_for(word)
            role_area = v.role_area_for(role)
            train_binding(brain, word, core_area, role_area, binding_rounds)
