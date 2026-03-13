"""
Next-token prediction via assembly overlap readout.

Minimal recipe (from BRIDGE_WEBSCALE_CURRICULUM.md, Section 4):
    1. Build vocabulary as a Lexicon (word -> assembly)
    2. Process context: feed tokens sequentially, let Hebbian dynamics
       form a "context assembly" via inter-token bridges
    3. Readout: measure overlap of context assembly with each vocab
       assembly -> distribution over next token
    4. Score: compare predicted distribution to actual next token

This is a narrow, structured demonstration on a toy corpus.
No gradient training -- pure Hebbian + overlap readout.

Architecture:
    - LEX area: holds word assemblies (one per vocabulary word)
    - Context: sequence of stimulus projections into LEX builds a
      "context assembly" via Hebbian bridges between consecutive words
    - Prediction: overlap of context assembly with each vocab assembly
      forms a distribution over the next token

Reference:
    BRIDGE_WEBSCALE_CURRICULUM.md, Section 4 (Minimal recipe).
"""

from typing import Dict, List, Tuple

from .readout import readout_all, build_lexicon, Lexicon
from .ops import project, sequence_memorize, _snap


def build_next_token_model(brain, area: str, vocab: List[str],
                           stimuli_map: Dict[str, str],
                           rounds: int = 10) -> Lexicon:
    """Build the vocabulary lexicon for next-token prediction.

    Each word gets a stable assembly in the target area via
    stimulus projection and recurrence.

    Args:
        brain: Brain instance with stimuli and area already added.
        area: Name of the LEX area.
        vocab: List of vocabulary words.
        stimuli_map: Maps each word to its stimulus name.
        rounds: Projection rounds per word.

    Returns:
        Lexicon mapping word -> Assembly.
    """
    return build_lexicon(brain, area, vocab, stimuli_map, rounds=rounds)


def train_on_corpus(brain, area: str, corpus: List[List[str]],
                    stimuli_map: Dict[str, str],
                    rounds_per_token: int = 5,
                    repetitions: int = 1) -> None:
    """Train the brain on a corpus of sentences.

    For each sentence, feeds tokens sequentially into the area with
    recurrence, building Hebbian bridges between consecutive token
    assemblies.  This is ``sequence_memorize`` applied to the language
    domain.

    Args:
        brain: Brain instance.
        area: Name of the LEX area.
        corpus: List of sentences, each a list of word strings.
        stimuli_map: Maps words to stimulus names.
        rounds_per_token: Projection rounds per token.
        repetitions: Number of corpus repetitions.
    """
    for _rep in range(repetitions):
        for sentence in corpus:
            stim_sequence = [stimuli_map[w] for w in sentence]
            sequence_memorize(
                brain, stim_sequence, area,
                rounds_per_step=rounds_per_token,
                repetitions=1,
            )


def predict_next_token(brain, area: str, context: List[str],
                       stimuli_map: Dict[str, str],
                       lexicon: Lexicon,
                       rounds_per_token: int = 5) -> List[Tuple[str, float]]:
    """Predict the next token given a context sequence.

    Feeds context tokens sequentially with recurrence, then reads
    out the final assembly's overlap with each vocabulary word.

    Args:
        brain: Brain instance (should be trained via train_on_corpus).
        area: Name of the LEX area.
        context: List of context words (e.g., ["the", "cat"]).
        stimuli_map: Maps words to stimulus names.
        lexicon: Vocabulary lexicon for readout.
        rounds_per_token: Projection rounds per context token.

    Returns:
        List of (word, overlap) sorted by overlap descending.
    """
    for i, word in enumerate(context):
        stim = stimuli_map[word]
        if i == 0:
            # First token: stimulus only
            brain.project({stim: [area]}, {})
        # Stimulus + recurrence to build Hebbian bridges
        for _ in range(rounds_per_token - 1):
            brain.project({stim: [area]}, {area: [area]})

    # One autonomous step to let context settle
    brain.project({}, {area: [area]})

    context_assembly = _snap(brain, area)
    return readout_all(context_assembly, lexicon)


def score_corpus(brain, area: str, corpus: List[List[str]],
                 stimuli_map: Dict[str, str],
                 lexicon: Lexicon,
                 rounds_per_token: int = 5) -> Dict[str, float]:
    """Score next-token prediction accuracy on a corpus.

    For each sentence and each position > 0, predicts the next token
    and checks ranking of the actual next token.

    Args:
        brain: Brain instance (trained).
        area: LEX area name.
        corpus: Test corpus.
        stimuli_map: Word -> stimulus mapping.
        lexicon: Vocabulary lexicon.
        rounds_per_token: Rounds per token during prediction.

    Returns:
        Dict with 'top1_accuracy', 'top3_accuracy', 'mrr',
        'total_predictions'.
    """
    top1_correct = 0
    top3_correct = 0
    total_rr = 0.0
    total_predictions = 0

    for sentence in corpus:
        for pos in range(len(sentence) - 1):
            context = sentence[:pos + 1]
            actual_next = sentence[pos + 1]

            predictions = predict_next_token(
                brain, area, context, stimuli_map, lexicon,
                rounds_per_token=rounds_per_token,
            )

            pred_words = [w for w, _ in predictions]
            if pred_words and actual_next == pred_words[0]:
                top1_correct += 1
            if actual_next in pred_words[:3]:
                top3_correct += 1
            if actual_next in pred_words:
                rank = pred_words.index(actual_next) + 1
                total_rr += 1.0 / rank
            total_predictions += 1

    n = max(total_predictions, 1)
    return {
        "top1_accuracy": top1_correct / n,
        "top3_accuracy": top3_correct / n,
        "mrr": total_rr / n,
        "total_predictions": total_predictions,
    }
