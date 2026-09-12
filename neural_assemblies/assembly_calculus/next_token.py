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

HOW TO READ THE NUMBERS.  Three properties of this setup make the reported
accuracies weaker evidence than they look.  None of them are bugs -- they are
consequences of doing prediction with the same dynamics that do learning --
but a reader comparing against a language-model baseline needs them stated.

1. Prediction is not read-only.  ``predict_next_token`` drives the area
   through ``brain.project`` with plasticity at its default (enabled), so
   every prediction potentiates the context it just fed in.  ``score_corpus``
   calls it once per position, which means the model is being trained on the
   evaluation corpus while it is being scored, and scores are order-dependent.
   To measure a frozen model, set ``brain.disable_plasticity = True`` around
   the call.

2. The "distribution" is an overlap ranking, not a probability.  Readout
   returns ``|context ∩ word| / min(|context|, |word|)`` per vocabulary item.
   These do not sum to 1, are not calibrated, and cannot be compared across
   contexts of different assembly sizes.  Rank metrics (top-1, top-3, MRR)
   are meaningful; anything requiring a likelihood is not.

3. The context has no explicit position code.  All tokens of the prefix are
   projected into the SAME area, so what accumulates is a recency-weighted
   blend rather than an ordered representation.  Word order affects the result
   only through the asymmetry of Hebbian bridges, which decays quickly with
   distance.  Expect near-bigram behaviour, and do not read long-range
   agreement into a good score.

Also note that ``predict_next_token`` does not reproduce the Phase A / Phase B
schedule that ``sequence_memorize`` used during training (see below), so the
inference-time dynamics are not identical to the training-time dynamics.

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
from .ops import sequence_memorize, _snap
from .contracts import NEXT_TOKEN_PREDICTION_CONTRACT, NEXT_TOKEN_SCORE_CONTRACT, NEXT_TOKEN_TRAINING_CONTRACT, NextTokenPredictionPlan, NextTokenScorePlan, NextTokenTrainingPlan, implements


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


@implements(NEXT_TOKEN_TRAINING_CONTRACT)
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

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-next-token-training
    """
    plan = NextTokenTrainingPlan(
        area=area,
        corpus=tuple(tuple(sentence) for sentence in corpus),
        stimuli_map=stimuli_map,
        rounds_per_token=rounds_per_token,
        repetitions=repetitions,
    )
    plan.preflight(brain)
    area, corpus, stimuli_map = plan.area, plan.corpus, plan.stimuli_map
    for _rep in range(plan.repetitions):
        for sentence in corpus:
            stim_sequence = [stimuli_map[w] for w in sentence]
            sequence_memorize(
                brain, stim_sequence, area,
                rounds_per_step=plan.rounds_per_token,
                repetitions=1,
            )


@implements(NEXT_TOKEN_PREDICTION_CONTRACT)
def predict_next_token(brain, area: str, context: List[str],
                       stimuli_map: Dict[str, str],
                       lexicon: Lexicon,
                       rounds_per_token: int = 5,
                       adapt: bool = False) -> List[Tuple[str, float]]:
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

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-next-token-prediction
    """
    plan = NextTokenPredictionPlan(
        area=area,
        context=tuple(context),
        stimuli_map=stimuli_map,
        lexicon=lexicon,
        rounds_per_token=rounds_per_token,
        adapt=adapt,
    )
    plan.preflight(brain)
    area, context, stimuli_map, lexicon = plan.area, plan.context, plan.stimuli_map, plan.lexicon
    rounds_per_token, adapt = plan.rounds_per_token, plan.adapt
    # Prediction is a READ-OUT and must not train. Previously this ran at the
    # brain's default (plasticity ENABLED), so every predicted position
    # potentiated the very bridges it was about to measure. `score_corpus`
    # calls this once per position, which made reported accuracies
    # order-dependent and partly a measurement of adaptation to the test set.
    #
    # Pass ``adapt=True`` for genuine online adaptation; it is off by default
    # because scoring is the overwhelmingly common caller and silently
    # training on the evaluation corpus is never what a caller wants.
    import contextlib
    freeze = brain.frozen() if not adapt else contextlib.nullcontext()
    with freeze:
        return _predict_next_token_inner(
            brain, area, context, stimuli_map, lexicon, rounds_per_token,
        )


def _predict_next_token_inner(brain, area: str, context: List[str],
                              stimuli_map: Dict[str, str],
                              lexicon: Lexicon,
                              rounds_per_token: int) -> List[Tuple[str, float]]:
    """Drive the context and read out. See ``predict_next_token``."""
    for i, word in enumerate(context):
        stim = stimuli_map[word]
        if i == 0:
            # First token gets one stimulus-only step so the context starts
            # from the word itself rather than from whatever the area was
            # holding.  Later tokens deliberately do NOT get this step: their
            # job is to perturb the running context, not to replace it.
            #
            # Note this differs from the Phase A / Phase B split that
            # ``sequence_memorize`` used at training time (which gives EVERY
            # token stimulus-only rounds first).  The asymmetry is real and
            # affects results; recorded rather than changed.
            brain.project({stim: [area]}, {})
        # Stimulus + recurrence: the recurrent fiber is what lets the previous
        # tokens' trace interact with this one, and is where the learned
        # bridges are read.
        for _ in range(rounds_per_token - 1):
            brain.project({stim: [area]}, {area: [area]})

    # One autonomous step with the stimulus removed.  This is the actual
    # prediction: with nothing clamping the area to the last token, the
    # strongest remaining drive is whatever the trained bridges point at,
    # i.e. the successor.  Skipping this step would read back the last input
    # word instead of a prediction.
    brain.project({}, {area: [area]})

    context_assembly = _snap(brain, area)
    return readout_all(context_assembly, lexicon)


@implements(NEXT_TOKEN_SCORE_CONTRACT)
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

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-next-token-score
    """
    plan = NextTokenScorePlan(
        area=area,
        corpus=tuple(tuple(sentence) for sentence in corpus),
        stimuli_map=stimuli_map,
        lexicon=lexicon,
        rounds_per_token=rounds_per_token,
    )
    plan.preflight(brain)
    area, corpus, stimuli_map, lexicon = plan.area, plan.corpus, plan.stimuli_map, plan.lexicon
    rounds_per_token = plan.rounds_per_token
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
