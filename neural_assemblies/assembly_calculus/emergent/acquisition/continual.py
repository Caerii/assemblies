"""Continual learning stability checks after online chat exposure.

THE RISK BEING MEASURED.  Learning during a chat session is the same Hebbian
update as learning during training -- there is no separate "inference mode"
that leaves the connectome alone.  So a session of conversation writes into
the same weights that carry the grammar, and a long or unusual session can
degrade what was learned earlier.  In connectionist terms this is catastrophic
forgetting; in this model it is more concrete than that, because the mechanism
is visible: new co-firing potentiates synapses in the same areas, and an
assembly is only stable relative to what else that area holds.

Assembly Calculus has one structural defence, which is that plasticity is
multiplicative and bounded per step rather than gradient-driven, so drift is
gradual rather than abrupt.  It is a defence, not immunity.

Hence the before/after snapshot.  The probes chosen are the things that would
break FIRST and silently -- word-order typology, held-out bootstrapping,
prediction lexicon size, novel composition.  None of them is something a chat
session touches directly, which is the point: they are load-bearing structure
that continued exposure should not have moved, so a change in them is
evidence of interference rather than of learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser import EmergentParser


@dataclass
class StabilitySnapshot:
    """Metrics captured before/after an online learning session."""
    word_order_svo: bool = False
    holdout_bootstrap: float = 0.0
    prediction_lexicon_size: float = 0.0
    novel_composition: float = 0.0


@dataclass
class StabilityReport:
    before: StabilitySnapshot
    after: StabilitySnapshot
    regressions: List[str] = field(default_factory=list)
    stable: bool = True


def capture_stability_snapshot(parser: "EmergentParser") -> StabilitySnapshot:
    from ..evaluation.suite import EvaluationSuite
    from ..evaluation.generalization import (
        DEFAULT_LEXICON_HOLDOUTS,
        NOVEL_COMPOSITION_PROBES,
    )

    suite = EvaluationSuite(parser)
    wo = suite.evaluate_word_order(target="SVO")
    novel = suite.evaluate_roles(NOVEL_COMPOSITION_PROBES)

    holdout = 0.0
    try:
        from .pos_inference import decompose_holdout_classification

        holdout = decompose_holdout_classification(
            parser, DEFAULT_LEXICON_HOLDOUTS,
        )["accuracy_bootstrapped"]
    except Exception:
        pass

    return StabilitySnapshot(
        word_order_svo=bool(wo.get("correct")),
        holdout_bootstrap=float(holdout),
        prediction_lexicon_size=float(
            len(getattr(parser, "prediction_lexicon", {}))
        ),
        novel_composition=float(novel.get("accuracy", 0.0)),
    )


def compare_stability(
    before: StabilitySnapshot,
    after: StabilitySnapshot,
    *,
    holdout_tolerance: float = 0.15,
    novel_tolerance: float = 0.20,
) -> StabilityReport:
    regressions: List[str] = []

    if before.word_order_svo and not after.word_order_svo:
        regressions.append("word_order_svo_lost")
    if after.holdout_bootstrap < before.holdout_bootstrap - holdout_tolerance:
        regressions.append("holdout_bootstrap_dropped")
    if after.novel_composition < before.novel_composition - novel_tolerance:
        regressions.append("novel_composition_dropped")
    if after.prediction_lexicon_size < before.prediction_lexicon_size * 0.9:
        regressions.append("prediction_lexicon_shrunk")

    return StabilityReport(
        before=before,
        after=after,
        regressions=regressions,
        stable=len(regressions) == 0,
    )


def replay_corpus_sample(
    parser: "EmergentParser",
    sentences: List[list],
    *,
    max_sentences: int = 20,
) -> int:
    """Light replay of stored sentences to mitigate catastrophic forgetting."""
    count = 0
    for sent in sentences[:max_sentences]:
        known = [w for w in sent if w in parser.stim_map]
        if len(known) < 2:
            continue
        parser.ingest_raw_sentence(known)
        if hasattr(parser, "train_next_token"):
            try:
                parser.train_next_token([known], dedupe_sentences=False)
            except Exception:
                pass
        count += 1
    return count


def stability_gate_after_session(
    parser: "EmergentParser",
    before: StabilitySnapshot,
    *,
    replay_sentences: Optional[List[list]] = None,
) -> StabilityReport:
    """Check stability after chat; optionally replay a small corpus sample."""
    after = capture_stability_snapshot(parser)
    report = compare_stability(before, after)

    if not report.stable and replay_sentences:
        replay_corpus_sample(parser, replay_sentences)
        after_replay = capture_stability_snapshot(parser)
        report = compare_stability(before, after_replay)

    return report
