"""
Sentence generation via reverse readout and prediction chain traversal.

Exploits the bidirectional pathways created by train_binding() to go from
structural representations back to words:

  build_core_lexicon:  Capture word assemblies in core areas (for matching)
  readout_from_role:   Activate role area → project to core → identify word
  generate_from_prediction_chain:  Chain forward predictions to produce sequences
  score_generation:    Check category accuracy and grammatical pattern validity
  check_novelty:       Whether a sequence appeared in training data

No new Brain primitives — uses only Brain.project() and area.winners.
The reverse pathway exists because train_binding() does bidirectional
co-projection: {core_area: [role_area], role_area: [core_area]}.
"""

from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from src.core.brain import Brain
from research.experiments.base import measure_overlap
from research.experiments.lib.vocabulary import Vocabulary, RECURSIVE_VOCAB
from research.experiments.lib.brain_setup import activate_word


def build_core_lexicon(
    brain: Brain,
    vocab: Vocabulary = None,
    readout_rounds: int = 5,
) -> Dict[str, np.ndarray]:
    """Build core-area fingerprints for each word.

    For each word, activates its stimulus in its core area (plasticity should
    be OFF) and records the resulting assembly. Used for generation readout
    matching — identifying which word a recovered pattern corresponds to.

    Args:
        brain: Brain instance (plasticity should be OFF).
        vocab: Vocabulary specification.
        readout_rounds: Rounds of stimulus projection per word.

    Returns:
        Dict mapping word -> winner array in its core area.
    """
    v = vocab or RECURSIVE_VOCAB
    fingerprints = {}
    for word in v.all_words:
        core_area = v.core_area_for(word)
        activate_word(brain, word, core_area, readout_rounds)
        fingerprints[word] = np.array(
            brain.areas[core_area].winners, dtype=np.uint32)
    return fingerprints


def readout_from_role(
    brain: Brain,
    role_area: str,
    core_area: str,
    vocab: Vocabulary,
    core_lexicon: Dict[str, np.ndarray],
    activate_rounds: int = 3,
) -> Tuple[str, float]:
    """Read out a word from a role area via reverse projection.

    Activates the role area's current pattern, projects into the core area
    using the trained reverse pathway (role_area → core_area), then matches
    the recovered pattern against the core lexicon to identify the word.

    Args:
        brain: Brain instance (plasticity OFF).
        role_area: Source role area containing bound word pattern.
        core_area: Target core area for reverse projection.
        vocab: Vocabulary for filtering candidates.
        core_lexicon: Word -> core area assembly fingerprints.
        activate_rounds: Rounds of reverse projection.

    Returns:
        (best_word, overlap_score) tuple.
    """
    brain.inhibit_areas([core_area])
    for _ in range(activate_rounds):
        brain.project({}, {role_area: [core_area]})

    core_winners = np.array(brain.areas[core_area].winners, dtype=np.uint32)

    best_word, best_overlap = None, -1.0
    for word, fp in core_lexicon.items():
        if vocab.core_area_for(word) != core_area:
            continue
        overlap = measure_overlap(core_winners, fp)
        if overlap > best_overlap:
            best_overlap = overlap
            best_word = word

    return best_word, best_overlap


def generate_from_prediction_chain(
    brain: Brain,
    start_word: str,
    vocab: Vocabulary,
    prediction_lexicon: Dict[str, np.ndarray],
    max_length: int = 6,
    activate_rounds: int = 3,
) -> List[Tuple[str, float]]:
    """Generate a word sequence by chaining forward predictions.

    Starting from start_word, repeatedly:
      1. Activate current word in its core area
      2. Project core area → PREDICTION
      3. Match PREDICTION winners against prediction lexicon
      4. Best match becomes next word

    Stops at max_length or if confidence drops below threshold.

    Args:
        brain: Brain instance (plasticity OFF).
        start_word: First word in the chain.
        vocab: Vocabulary for area lookups.
        prediction_lexicon: Word -> PREDICTION area fingerprints.
        max_length: Maximum sequence length.
        activate_rounds: Rounds per word activation.

    Returns:
        List of (word, confidence) tuples including start_word.
    """
    chain = [(start_word, 1.0)]
    current_word = start_word

    for _ in range(max_length - 1):
        core_area = vocab.core_area_for(current_word)
        activate_word(brain, current_word, core_area, activate_rounds)

        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {core_area: ["PREDICTION"]})
        predicted = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        best_word, best_overlap = None, -1.0
        for word, fp in prediction_lexicon.items():
            overlap = measure_overlap(predicted, fp)
            if overlap > best_overlap:
                best_overlap = overlap
                best_word = word

        if best_word is None or best_overlap < 0.01:
            break

        chain.append((best_word, best_overlap))
        current_word = best_word

    return chain


def score_generation(
    words: List[str],
    vocab: Vocabulary,
) -> Dict[str, Any]:
    """Score a generated word sequence against grammatical expectations.

    Checks category accuracy (nouns in even positions, verbs/preps in odd),
    SVO pattern validity, and SVO+PP pattern validity.

    Args:
        words: Generated word sequence.
        vocab: Vocabulary for category lookups.

    Returns:
        Dict with is_svo, is_svo_pp, category_accuracy, length.
    """
    if not words:
        return {"is_svo": False, "is_svo_pp": False,
                "category_accuracy": 0.0, "length": 0}

    nouns = set(vocab.words_for_category("NOUN"))
    if "LOCATION" in vocab.categories:
        nouns |= set(vocab.words_for_category("LOCATION"))
    verbs = set(vocab.words_for_category("VERB"))
    preps = set()
    if "PREP" in vocab.categories:
        preps = set(vocab.words_for_category("PREP"))

    # Check SVO pattern
    is_svo = (len(words) >= 3 and
              words[0] in nouns and
              words[1] in verbs and
              words[2] in nouns)

    # Check SVO+PP pattern
    is_svo_pp = (is_svo and len(words) >= 5 and
                 words[3] in preps and
                 words[4] in nouns)

    # Category accuracy: expect N-V-N or N-V-N-P-N pattern
    n_correct = 0
    for i, w in enumerate(words):
        if i == 0 and w in nouns:
            n_correct += 1
        elif i == 1 and w in verbs:
            n_correct += 1
        elif i == 2 and w in nouns:
            n_correct += 1
        elif i == 3 and w in preps:
            n_correct += 1
        elif i == 4 and w in nouns:
            n_correct += 1
        elif i >= 5:
            # Beyond SVO+PP, just check alternation
            if i % 2 == 0 and w in nouns:
                n_correct += 1
            elif i % 2 == 1 and w in (verbs | preps):
                n_correct += 1

    return {
        "is_svo": is_svo,
        "is_svo_pp": is_svo_pp,
        "category_accuracy": n_correct / len(words),
        "length": len(words),
    }


def check_novelty(
    generated_words: List[str],
    training_sentences: List[Dict[str, Any]],
) -> bool:
    """Check if this exact word sequence was never seen in training."""
    gen_tuple = tuple(generated_words)
    for sent in training_sentences:
        if tuple(sent["words"]) == gen_tuple:
            return False
    return True
