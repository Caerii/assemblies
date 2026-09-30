"""Per-stage exit gates for developmental acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Dict, List, Mapping, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..curriculum import StageResult
    from ..parser import EmergentParser

# Minimum metric floors per stage (cheap checks for early stages).
DEFAULT_STAGE_GATE_FLOORS: Dict[str, Dict[str, float]] = {
    "BABBLE": {"babble_forms_min": 12.0},
    "FIRST_WORDS": {"vocab_min": 10.0, "classification_min": 0.0},
    "VOCABULARY_SPURT": {"classification_min": 0.0},
    "TWO_WORD": {"classification_min": 0.0},
    "SENTENCES": {
        "novel_composition_min": 0.33,
        "holdout_bootstrap_min": 0.50,
    },
    "COMPLEX_GRAMMAR": {
        "novel_composition_min": 0.33,
        "holdout_bootstrap_min": 0.50,
    },
    "DIALOGUE": {"prediction_lexicon_min": 10.0},
    "CONVERSATION": {"prediction_lexicon_min": 15.0},
}

# Phases that must appear in ``phases_run`` before advancing.
STAGE_REQUIRED_PHASES: Dict[str, tuple[str, ...]] = {
    "VOCABULARY_SPURT": ("distributional",),
    "TWO_WORD": ("distributional", "roles"),
    "SENTENCES": ("roles", "word_order", "prediction"),
}


@dataclass
class StageGateResult:
    """Outcome of a single stage exit gate."""
    stage: str
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


def _metric_float(metrics: Mapping[str, object], key: str,
                  default: float = 0.0) -> float:
    """Validate an evaluation metric before applying a stage threshold."""
    value = metrics.get(key, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"stage metric {key!r} must be a real number")
    return float(value)


def evaluate_stage_gate(
    parser: "EmergentParser",
    result: "StageResult",
    *,
    holdout_words: Optional[Set[str]] = None,
    floors: Optional[Dict[str, Dict[str, float]]] = None,
) -> StageGateResult:
    """Evaluate whether a stage is safe to exit before advancing."""
    stage = result.stage_name
    gate_floors = (floors or DEFAULT_STAGE_GATE_FLOORS).get(stage, {})
    checks: Dict[str, bool] = {}
    metrics: Dict[str, float] = {}
    failures: List[str] = []

    if stage == "BABBLE":
        n_forms = float(len(getattr(parser, "babble_forms", [])))
        metrics["babble_forms"] = n_forms
        min_forms = gate_floors.get("babble_forms_min", 12.0)
        checks["babble_forms"] = n_forms >= min_forms
        if not checks["babble_forms"]:
            failures.append(f"babble_forms {n_forms:.0f} < {min_forms:.0f}")

    elif stage in STAGE_REQUIRED_PHASES:
        for phase in STAGE_REQUIRED_PHASES[stage]:
            ok = phase in result.phases_run
            checks[f"phase_{phase}"] = ok
            if not ok:
                failures.append(f"missing required phase {phase!r}")

    if stage not in ("BABBLE",):
        metrics["vocab_size"] = float(result.vocab_size)
        metrics["classification_accuracy"] = result.classification_accuracy
        vocab_min = gate_floors.get("vocab_min", 0.0)
        if vocab_min > 0:
            checks["vocab_size"] = result.vocab_size >= vocab_min
            if not checks["vocab_size"]:
                failures.append(
                    f"vocab {result.vocab_size} < {vocab_min:.0f}",
                )
        cls_min = gate_floors.get("classification_min", 0.0)
        if cls_min > 0:
            checks["classification"] = result.classification_accuracy >= cls_min
            if not checks["classification"]:
                failures.append(
                    f"classification {result.classification_accuracy:.1%} "
                    f"< {cls_min:.1%}",
                )

    if stage in ("SENTENCES", "COMPLEX_GRAMMAR"):
        from ..evaluation.generalization import NOVEL_COMPOSITION_PROBES
        from ..evaluation.suite import EvaluationSuite
        from .pos_inference import decompose_holdout_classification

        holdout_map = {
            "bird": "NOUN",
            "finds": "VERB",
            "small": "ADJ",
        }
        if holdout_words:
            holdout_map = {
                w: holdout_map.get(w, "NOUN")
                for w in holdout_words
            }

        suite = EvaluationSuite(parser)
        novel = suite.evaluate_roles(NOVEL_COMPOSITION_PROBES)
        novel_acc = _metric_float(novel, "accuracy")
        metrics["novel_composition"] = novel_acc
        novel_min = gate_floors.get("novel_composition_min", 0.33)
        checks["novel_composition"] = novel_acc >= novel_min
        if not checks["novel_composition"]:
            failures.append(
                f"novel_composition {novel_acc:.1%} < {novel_min:.1%}",
            )

        decomp = decompose_holdout_classification(parser, holdout_map)
        boot = _metric_float(decomp, "accuracy_bootstrapped")
        metrics["holdout_bootstrap"] = boot
        boot_min = gate_floors.get("holdout_bootstrap_min", 0.50)
        checks["holdout_bootstrap"] = boot >= boot_min
        if not checks["holdout_bootstrap"]:
            failures.append(
                f"holdout_bootstrap {boot:.1%} < {boot_min:.1%}",
            )

        if "prediction" in result.phases_run:
            from ..evaluation.generalization import collect_bridge_probes
            from ..evaluation.parity import score_next_token_probes
            from ..core.corpus_index import compile_corpus
            from ..curriculum.data import create_training_sentences

            corpus_index = compile_corpus(
                parser, create_training_sentences(),
            )
            holdout_set = set(holdout_map.keys())
            oov_prefixes, oov_expected = collect_bridge_probes(
                corpus_index,
                parser,
                holdout_set,
                oov_only=True,
                max_probes=10,
            )
            bridge_oov = score_next_token_probes(
                parser, oov_prefixes, oov_expected,
            )
            bridge_top5 = _metric_float(bridge_oov, "top5")
            metrics["bridge_oov_top5"] = bridge_top5
            # Soft metric for now — hard gate once bridge training is stable.

    elif stage in ("DIALOGUE", "CONVERSATION"):
        pred_size = float(len(getattr(parser, "prediction_lexicon", {})))
        metrics["prediction_lexicon_size"] = pred_size
        pred_min = gate_floors.get("prediction_lexicon_min", 10.0)
        checks["prediction_lexicon"] = pred_size >= pred_min
        if not checks["prediction_lexicon"]:
            failures.append(
                f"prediction_lexicon {pred_size:.0f} < {pred_min:.0f}",
            )

    passed = bool(checks) and all(checks.values())
    if not checks:
        passed = True

    return StageGateResult(
        stage=stage,
        passed=passed,
        checks=checks,
        metrics=metrics,
        failures=failures,
    )


def inter_stage_sleep(parser: "EmergentParser", stage_name: str) -> None:
    """Plasticity-off replay of recent exposure before the next stage."""
    from ..training.perf import developmental_curriculum_enabled

    if not developmental_curriculum_enabled():
        return
    if stage_name in ("BABBLE", "CONVERSATION"):
        return

    log = getattr(parser, "_exposure_log", [])
    if not log:
        return

    with parser.brain.frozen():
        for words in log[-15:]:
            parser.ingest_raw_sentence(list(words))
