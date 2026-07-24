"""Wobbly-parse acquisition: episodic memory + POS replay from ERP surprise.

Probe measurement: ``evaluation.erp.runner.run_incremental_erp_probes``.
This package adds ERROR activation, episodic storage, hypothesis competition,
and remedial replay via ``replay_wobbly_episodes``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from ...evaluation.erp import (
    ErpBaseline,
    ErpProbeResult,
    ErpReadiness,
    ErpThresholds,
    probe_word_at_position,
    run_incremental_erp_probes,
)
from .hypotheses import generate_pos_hypotheses
from .memory import WobblyEpisode, WobblyMemory, WobblyProbe, activate_error
from .mining import mine_wobbly_episodes
from .replay import (
    bootstrap_from_wobbly_memory,
    format_wobbly_report,
    mine_and_bootstrap_from_exposure,
    replay_wobbly_episodes,
    resolve_wobbly_hypotheses,
)

if TYPE_CHECKING:
    from ...parser import EmergentParser

__all__ = [
    "WobblyEpisode",
    "WobblyMemory",
    "WobblyProbe",
    "bootstrap_from_wobbly_memory",
    "format_wobbly_report",
    "generate_pos_hypotheses",
    "mine_and_bootstrap_from_exposure",
    "mine_wobbly_episodes",
    "parse_with_wobbly_probes",
    "probe_word_surprise",
    "replay_wobbly_episodes",
    "resolve_wobbly_hypotheses",
]


def parse_with_wobbly_probes(
    parser: "EmergentParser",
    words: List[str],
    *,
    apply_calibration: bool = True,
    baseline: Optional[ErpBaseline] = None,
    readiness: Optional[ErpReadiness] = None,
    thresholds: Optional[ErpThresholds] = None,
) -> Tuple[dict, List[ErpProbeResult]]:
    """Incremental parse with ERP probes and ERROR activation on wobble."""
    return run_incremental_erp_probes(
        parser,
        words,
        apply_calibration=apply_calibration,
        activate_error=True,
        baseline=baseline,
        readiness=readiness,
        thresholds=thresholds,
        error_callback=activate_error,
    )


def probe_word_surprise(
    parser: "EmergentParser",
    words: List[str],
    position: int,
    *,
    baseline: Optional[ErpBaseline] = None,
    readiness: Optional[ErpReadiness] = None,
) -> ErpProbeResult:
    """Backward-compatible alias for ``probe_word_at_position`` with ERROR."""
    return probe_word_at_position(
        parser,
        words,
        position,
        baseline=baseline,
        readiness=readiness,
        activate_error=True,
        error_callback=activate_error,
    )
