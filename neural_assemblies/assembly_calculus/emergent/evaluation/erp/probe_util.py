"""Shared ERP probe helpers for calibration and mining."""

from __future__ import annotations

from typing import Callable, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ...parser import EmergentParser


def critical_probe_measure_fn(
    probe_depth: str = "calibration",
) -> Callable:
    """Measure_fn that probes only the final content word of each sentence."""
    from .runner import run_incremental_erp_probes

    def measure(parser: "EmergentParser", known: List[str], **kw):
        pos = len(known) - 1
        return run_incremental_erp_probes(
            parser,
            known,
            apply_calibration=False,
            probe_depth=probe_depth,
            stop_at_position=pos,
            probe_positions={pos},
        )

    return measure


def target_word_probe_positions(
    known: List[str],
    target_words: Optional[Set[str]],
) -> Optional[Set[int]]:
    """Indices of *target_words* in *known*, or final word if unset."""
    if not known:
        return None
    if not target_words:
        return {len(known) - 1}
    positions = {i for i, w in enumerate(known) if w in target_words}
    return positions or {len(known) - 1}
