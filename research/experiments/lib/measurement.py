"""
ERP measurement: N400 (prediction error) and P600 (integration difficulty).

Provides composable measurement functions that work at any sentence position:

  measure_n400: Compare context-driven prediction with a word's lexicon entry.
      Returns 1 - overlap (high = unexpected = large N400).

  measure_p600: Anchored instability when settling a word into a role slot.
      Returns instability score (high = difficult integration = large P600).

  measure_erps_at_position: Composite measurement that runs N400 and P600
      for grammatical, category-violation, and novel conditions at one position.

  generate_test_triples / generate_pp_test_triples: Create matched test
      conditions for systematic comparison.

Design principle: measurement is position-agnostic. The same functions
measure ERPs at the object position, PP-object position, or any future
position. The caller provides the context (which areas are active, which
word is being tested) and the measurement functions do the rest.
"""

from typing import Dict, List, Tuple, Any

import numpy as np

from src.core.brain import Brain
from research.experiments.base import measure_overlap
from research.experiments.metrics.instability import compute_anchored_instability
from research.experiments.lib.vocabulary import (
    Vocabulary, DEFAULT_VOCAB,
    NOUNS, VERBS, LOCATIONS, NOVEL_NOUNS,
)
from research.experiments.lib.brain_setup import activate_word


def measure_n400(
    predicted: np.ndarray,
    lexicon_entry: np.ndarray,
) -> float:
    """Measure N400: prediction error at the lexical level.

    N400 = 1 - overlap(context-driven prediction, word's lexicon entry).
    High values mean the word was unexpected given context.

    Args:
        predicted: Winner array from context-driven forward projection.
        lexicon_entry: Winner array from stimulus-driven lexicon readout.

    Returns:
        N400 value in [0, 1]. 0 = perfectly predicted, 1 = completely unexpected.
    """
    return 1.0 - measure_overlap(predicted, lexicon_entry)


def measure_p600(
    brain: Brain,
    word: str,
    core_area: str,
    role_area: str,
    n_settling_rounds: int = 10,
) -> float:
    """Measure P600: structural integration difficulty via anchored instability.

    Phase A: one round with stimulus co-projection to create anchored pattern.
    Phase B: settle with area-to-area connections only (no stimulus).
    Trained pathways sustain the pattern (low instability = low P600).
    Untrained pathways cannot (high instability = high P600).

    Plasticity must be OFF before calling.

    Args:
        brain: Brain instance.
        word: Word to integrate.
        core_area: Core area with word's assembly.
        role_area: Target role area.
        n_settling_rounds: Total settling rounds (Phase A + Phase B).

    Returns:
        Anchored instability score (higher = more P600).
    """
    result = compute_anchored_instability(
        brain, word, core_area, role_area, n_settling_rounds)
    return result["instability"]


def measure_erps_at_position(
    brain: Brain,
    predicted: np.ndarray,
    lexicon: Dict[str, np.ndarray],
    gram_word: str,
    gram_core: str,
    catviol_word: str,
    catviol_core: str,
    novel_word: str,
    novel_core: str,
    role_area: str,
    n_settling_rounds: int = 10,
) -> Dict[str, float]:
    """Measure N400 and P600 for three conditions at one sentence position.

    This is the core measurement function used at any position (object,
    PP-object, or any future position). The caller sets up the context
    (activates preceding words, does forward prediction), then calls this
    with the predicted assembly and the three test conditions.

    Args:
        brain: Brain instance with plasticity OFF.
        predicted: Context-driven prediction assembly (from forward projection).
        lexicon: Word -> PREDICTION fingerprints.
        gram_word: Grammatical (expected) word.
        gram_core: Core area for grammatical word.
        catviol_word: Category violation word.
        catviol_core: Core area for violation word.
        novel_word: Novel (unseen-in-training) word.
        novel_core: Core area for novel word.
        role_area: Role area for P600 measurement.
        n_settling_rounds: Settling rounds for anchored instability.

    Returns:
        Dict with n400_gram, n400_catviol, n400_novel,
                  p600_gram, p600_catviol, p600_novel.
    """
    return {
        "n400_gram": measure_n400(predicted, lexicon[gram_word]),
        "n400_catviol": measure_n400(predicted, lexicon[catviol_word]),
        "n400_novel": measure_n400(predicted, lexicon[novel_word]),
        "p600_gram": measure_p600(
            brain, gram_word, gram_core, role_area, n_settling_rounds),
        "p600_catviol": measure_p600(
            brain, catviol_word, catviol_core, role_area, n_settling_rounds),
        "p600_novel": measure_p600(
            brain, novel_word, novel_core, role_area, n_settling_rounds),
    }


def forward_predict_from_context(
    brain: Brain,
    context_words: List[str],
    vocab: Vocabulary = None,
    activate_rounds: int = 3,
) -> np.ndarray:
    """Activate a sequence of words, then forward-project into PREDICTION.

    Processes each context word by activating it in its core area,
    then projects the final word's area into PREDICTION.

    Args:
        brain: Brain instance (plasticity should be OFF).
        context_words: Words to activate in sequence.
        vocab: Vocabulary for area lookups.
        activate_rounds: Rounds per word activation.

    Returns:
        PREDICTION winners array for N400 comparison.
    """
    v = vocab or DEFAULT_VOCAB
    last_area = None
    for word in context_words:
        area = v.core_area_for(word)
        activate_word(brain, word, area, activate_rounds)
        last_area = area
    brain.inhibit_areas(["PREDICTION"])
    brain.project({}, {last_area: ["PREDICTION"]})
    return np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)


def measure_role_leakage(
    brain: Brain,
    word: str,
    core_area: str,
    role_area: str,
    activate_rounds: int = 3,
) -> np.ndarray:
    """Activate a word and read out its footprint in a role area.

    Projects word into the role area with one round (no plasticity)
    and returns the resulting winners.  Used to measure whether
    an unintended noun 'leaks' into a role area it was not bound to.

    Args:
        brain: Brain instance (plasticity OFF).
        word: Word to test.
        core_area: Core area with the word's assembly.
        role_area: Role area to probe.
        activate_rounds: Rounds for initial word activation.

    Returns:
        Winner array in the role area after one projection round.
    """
    activate_word(brain, word, core_area, activate_rounds)
    brain.inhibit_areas([role_area])
    brain.project(
        {f"PHON_{word}": [core_area, role_area]},
        {core_area: [role_area]},
    )
    return np.array(brain.areas[role_area].winners, dtype=np.uint32)


# ── Test condition generators ──────────────────────────────────────

def generate_test_triples(
    rng: np.random.Generator,
    n_triples: int = 5,
    vocab: Vocabulary = None,
) -> List[Tuple[str, str, str, str, str]]:
    """Generate matched test triples for the object position.

    Returns list of (agent, verb, gram_obj, catviol_obj, novel_obj).
    Each triple shares the same context (agent + verb) but varies
    the critical word across three conditions.
    """
    v = vocab or DEFAULT_VOCAB
    nouns = v.words_for_category("NOUN")
    verbs = v.words_for_category("VERB")
    novels = list(v.novel_words.keys())

    triples = []
    for i in range(n_triples):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        remaining = [n for n in nouns if n != agent]
        gram_obj = remaining[i % len(remaining)]
        catviol_obj = verbs[(i + 1) % len(verbs)]
        novel_obj = novels[i % len(novels)] if novels else remaining[-1]
        triples.append((agent, verb, gram_obj, catviol_obj, novel_obj))

    return triples


def generate_pp_test_triples(
    rng: np.random.Generator,
    n_triples: int = 5,
    vocab: Vocabulary = None,
) -> List[Tuple[str, str, str, str, str, str, str]]:
    """Generate matched test triples for the PP-object position.

    Returns list of (agent, verb, patient, prep, gram_ppobj, catviol_ppobj,
    novel_ppobj). Each shares the same SVO+P context but varies the PP object.
    """
    v = vocab or DEFAULT_VOCAB
    nouns = v.words_for_category("NOUN")
    verbs = v.words_for_category("VERB")
    preps = v.words_for_category("PREP") if "PREP" in v.categories else []
    locs = v.words_for_category("LOCATION") if "LOCATION" in v.categories else []
    novels = list(v.novel_words.keys())

    if not preps or not locs:
        return []

    triples = []
    for i in range(n_triples):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        patient = nouns[(i + 1) % len(nouns)]
        prep = preps[i % len(preps)]
        gram_ppobj = locs[i % len(locs)]
        catviol_ppobj = verbs[(i + 2) % len(verbs)]
        novel_ppobj = novels[i % len(novels)] if novels else locs[-1]
        triples.append((agent, verb, patient, prep,
                        gram_ppobj, catviol_ppobj, novel_ppobj))

    return triples
