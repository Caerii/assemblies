"""Next-token plans and contracts: the lexicon build, training, scoring and prediction.

Part of neural_assemblies.assembly_calculus.contracts."""
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping


from ..assembly import Assembly

from .contract import OperationContract
from .schedule import _explicit_bool, _positive_rounds, _require_name


@dataclass(frozen=True)
class NextTokenTrainingPlan:
    """Validated ordered corpus schedule for Hebbian next-token training."""

    area: str
    corpus: tuple[tuple[str, ...], ...]
    stimuli_map: Mapping[str, str]
    rounds_per_token: int = 5
    repetitions: int = 1

    def __post_init__(self) -> None:
        _require_name("area", self.area)
        if not isinstance(self.corpus, tuple) or not self.corpus or any(
            not isinstance(sentence, tuple) or not sentence
            or any(not isinstance(word, str) or not word for word in sentence)
            for sentence in self.corpus
        ):
            raise ValueError("training corpus must be a nonempty tuple of nonempty word tuples")
        if not isinstance(self.stimuli_map, Mapping):
            raise TypeError("training stimuli_map must be a mapping")
        words = {word for sentence in self.corpus for word in sentence}
        if any(word not in self.stimuli_map for word in words):
            raise KeyError("training corpus contains a word missing from stimuli_map")
        if any(not isinstance(self.stimuli_map[word], str) or not self.stimuli_map[word] for word in words):
            raise ValueError("training stimuli must be nonempty names")
        if isinstance(self.rounds_per_token, bool) or not isinstance(self.rounds_per_token, Integral) or self.rounds_per_token < 1:
            raise ValueError("rounds_per_token must be a positive integer")
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, Integral) or self.repetitions < 1:
            raise ValueError("repetitions must be a positive integer")
        object.__setattr__(self, "rounds_per_token", int(self.rounds_per_token))
        object.__setattr__(self, "repetitions", int(self.repetitions))

    def preflight(self, brain) -> None:
        if self.area not in brain.areas:
            raise KeyError(f"training area is unknown: {self.area!r}")
        words = {word for sentence in self.corpus for word in sentence}
        missing = [self.stimuli_map[word] for word in words if self.stimuli_map[word] not in brain.stimuli]
        if missing:
            raise KeyError(f"training stimuli are unknown: {sorted(set(missing))}")


@dataclass(frozen=True)
class NextTokenScorePlan:
    """Validated corpus observation schedule for next-token metrics."""

    area: str
    corpus: tuple[tuple[str, ...], ...]
    stimuli_map: Mapping[str, str]
    lexicon: Mapping[str, Assembly]
    rounds_per_token: int = 5

    def __post_init__(self) -> None:
        training = NextTokenTrainingPlan(
            self.area, self.corpus, self.stimuli_map,
            self.rounds_per_token, repetitions=1,
        )
        object.__setattr__(self, "area", training.area)
        object.__setattr__(self, "corpus", training.corpus)
        object.__setattr__(self, "stimuli_map", training.stimuli_map)
        object.__setattr__(self, "rounds_per_token", training.rounds_per_token)
        if not isinstance(self.lexicon, Mapping):
            raise TypeError("score lexicon must map labels to Assembly snapshots")
        if not self.lexicon:
            raise ValueError("score lexicon must be nonempty")
        if any(not isinstance(value, Assembly) for value in self.lexicon.values()):
            raise TypeError("score lexicon must map labels to Assembly snapshots")
        if any(value.area != self.area for value in self.lexicon.values()):
            raise ValueError("score lexicon snapshots must belong to the scoring area")

    def preflight(self, brain) -> None:
        NextTokenTrainingPlan(
            self.area, self.corpus, self.stimuli_map,
            self.rounds_per_token, repetitions=1,
            ).preflight(brain)


@dataclass(frozen=True)
class LexiconBuildPlan:
    """Validated independent stimulus-to-Assembly lexicon schedule."""

    area: str
    words: tuple[str, ...]
    stimuli_map: Mapping[str, str]
    rounds: int = 10

    def __post_init__(self) -> None:
        _require_name("area", self.area)
        if not isinstance(self.words, tuple) or any(not isinstance(word, str) or not word for word in self.words):
            raise ValueError("lexicon words must be nonempty strings")
        if len(set(self.words)) != len(self.words):
            raise ValueError("lexicon words must be unique")
        if not isinstance(self.stimuli_map, Mapping):
            raise TypeError("stimuli_map must be a mapping from words to stimuli")
        if set(self.stimuli_map) != set(self.words):
            raise ValueError("stimuli_map keys must exactly match the lexicon words")
        if any(not isinstance(stimulus, str) or not stimulus for stimulus in self.stimuli_map.values()):
            raise ValueError("lexicon stimuli must be nonempty strings")
        object.__setattr__(self, "stimuli_map", MappingProxyType(dict(self.stimuli_map)))
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))

    def preflight(self, brain) -> None:
        if self.area not in brain.areas:
            raise ValueError(f"unknown lexicon area {self.area!r}")
        missing = [stimulus for stimulus in self.stimuli_map.values() if stimulus not in brain.stimuli]
        if missing:
            raise ValueError(f"unknown lexicon stimuli: {sorted(set(missing))}")


@dataclass(frozen=True)
class NextTokenPredictionPlan:
    """Validated frozen or adapting next-token prediction query."""

    area: str
    context: tuple[str, ...]
    stimuli_map: Mapping[str, str]
    lexicon: Mapping[str, Assembly]
    rounds_per_token: int = 5
    adapt: bool = False

    def __post_init__(self) -> None:
        _require_name("area", self.area)
        if not isinstance(self.context, tuple) or not self.context or any(not isinstance(word, str) or not word for word in self.context):
            raise ValueError("prediction context must be a nonempty tuple of words")
        if not isinstance(self.stimuli_map, Mapping):
            raise TypeError("prediction stimuli_map must be a mapping")
        if any(word not in self.stimuli_map for word in self.context):
            raise KeyError("prediction context contains a word missing from stimuli_map")
        if any(not isinstance(self.stimuli_map[word], str) or not self.stimuli_map[word]
               for word in self.context):
            raise ValueError("prediction stimuli must be nonempty names")
        if isinstance(self.rounds_per_token, bool) or not isinstance(self.rounds_per_token, Integral) or self.rounds_per_token < 1:
            raise ValueError("rounds_per_token must be a positive integer")
        if not isinstance(self.lexicon, Mapping):
            raise TypeError("prediction lexicon must map labels to Assembly snapshots")
        if not self.lexicon:
            raise ValueError("prediction lexicon must be nonempty")
        if any(not isinstance(value, Assembly) for value in self.lexicon.values()):
            raise TypeError("prediction lexicon must map labels to Assembly snapshots")
        if any(value.area != self.area for value in self.lexicon.values()):
            raise ValueError("prediction lexicon snapshots must belong to the prediction area")
        _explicit_bool("adapt", self.adapt)
        object.__setattr__(self, "rounds_per_token", int(self.rounds_per_token))

    def preflight(self, brain) -> None:
        if self.area not in brain.areas:
            raise KeyError(f"prediction area is unknown: {self.area!r}")
        missing = [self.stimuli_map[word] for word in self.context if self.stimuli_map[word] not in brain.stimuli]
        if missing:
            raise KeyError(f"prediction stimuli are unknown: {sorted(set(missing))}")


LEXICON_BUILD_CONTRACT = OperationContract(
    operation_id="lexicon-build-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-lexicon-build",
    plan_type=LexiconBuildPlan,
    inputs=("brain", "target area", "ordered word labels", "stimulus map", "rounds"),
    reads=("phonological stimuli", "target area connectome"),
    mutates=("target area activity", "target recurrent connections between words"),
    regime=("unique words", "exact stimulus-map keys", "reset connections between words"),
    observed_outcome=("word-to-Assembly lexicon",),
    failure_conditions=("unknown area or stimulus", "duplicate/malformed labels", "invalid rounds"),
    constructed_controls=(
        "neural_assemblies/tests/test_readout.py::test_build_lexicon_distinct",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_readout.py::test_build_lexicon_preflights_all_inputs",
    ),
)


NEXT_TOKEN_PREDICTION_CONTRACT = OperationContract(
    operation_id="next-token-prediction-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-next-token-prediction",
    plan_type=NextTokenPredictionPlan,
    inputs=("brain", "area", "ordered context", "stimulus map", "lexicon", "rounds", "adapt"),
    reads=("context stimulus drives", "recurrent area state", "lexicon overlaps"),
    mutates=("temporary activity; weights only when adapt=True",),
    regime=("nonempty context", "positive rounds", "frozen by default", "ranked overlap readout"),
    observed_outcome=("ordered label/overlap scores",),
    failure_conditions=("unknown words/stimuli/area", "invalid rounds", "malformed lexicon", "empty context"),
    constructed_controls=(
        "neural_assemblies/tests/test_next_token.py::test_next_token_after_the",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_next_token.py::test_prediction_rejects_nonpositive_rounds",
    ),
)


NEXT_TOKEN_TRAINING_CONTRACT = OperationContract(
    operation_id="next-token-training-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-next-token-training",
    plan_type=NextTokenTrainingPlan,
    inputs=("brain", "area", "ordered corpus", "stimulus map", "rounds", "repetitions"),
    reads=("corpus token stimuli", "recurrent area state"),
    mutates=("area winners", "recurrent weights", "engine history"),
    regime=("nonempty sentences", "positive rounds and repetitions", "ordered teacher-forced sequence"),
    observed_outcome=("trained brain state",),
    failure_conditions=("unknown words/stimuli/area", "malformed corpus", "invalid schedule"),
    constructed_controls=(
        "neural_assemblies/tests/test_next_token.py::test_lexicon_is_built",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_next_token.py::test_training_rejects_unknown_word_before_mutation",
    ),
)


NEXT_TOKEN_SCORE_CONTRACT = OperationContract(
    operation_id="next-token-score-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-next-token-score",
    plan_type=NextTokenScorePlan,
    inputs=("brain", "area", "ordered corpus", "stimulus map", "lexicon", "rounds"),
    reads=("frozen next-token rankings", "lexicon overlaps"),
    mutates=("temporary observation activity only",),
    regime=("complete corpus preflight", "frozen prediction", "top-k and MRR metrics"),
    observed_outcome=("top1/top3 accuracy, MRR, prediction count",),
    failure_conditions=("unknown words/stimuli/area", "malformed corpus", "invalid rounds"),
    constructed_controls=(
        "neural_assemblies/tests/test_next_token.py::test_above_chance_accuracy",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_next_token.py::test_scoring_rejects_unknown_word_before_prediction",
    ),
)
