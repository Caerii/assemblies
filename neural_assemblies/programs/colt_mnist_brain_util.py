"""Shared helpers for explicit-Brain MNIST training."""

from __future__ import annotations

import numpy as np

from neural_assemblies.core.backend import get_xp


def area_has_active_winners(brain, area: str) -> bool:
    """Whether an area currently supplies drive to a composition step.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-active-source-routing
    """
    return brain.areas[area].active_count > 0


def renorm_connectome_columns(brain, src: str, dst: str) -> None:
    conn = brain.connectomes[src][dst]
    w = conn.weights
    if w.size == 0:
        return
    xp = get_xp()
    col_sums = w.sum(axis=0, keepdims=True)
    conn.weights = w / xp.maximum(col_sums, 1e-12)
    if brain._explicit_engine is not None:
        econn = brain._explicit_engine._area_conns.get(src, {}).get(dst)
        if econn is not None:
            econn.weights = conn.weights


def sync_protocol_weights(
    brain,
    w: np.ndarray,
    a: np.ndarray,
    input_area: str,
    hidden_area: str,
) -> None:
    a32 = a.astype(np.float32)
    w32 = w.astype(np.float32)
    brain.connectomes[input_area][hidden_area].weights = a32
    brain.connectomes[hidden_area][hidden_area].weights = w32
    if brain._explicit_engine is not None:
        brain._explicit_engine._area_conns[input_area][hidden_area].weights = a32
        brain._explicit_engine._area_conns[hidden_area][hidden_area].weights = w32


def set_kcap_winners(brain, area: str, pattern: np.ndarray) -> np.ndarray:
    winners = np.flatnonzero(pattern > 0).astype(np.uint32)
    brain.areas[area].unfix_assembly()
    brain.engine_for(area).set_winners(area, winners)
    brain.areas[area].winners = winners
    return winners


def clear_area_winners(brain, area: str) -> None:
    empty = np.array([], dtype=np.uint32)
    brain.areas[area].unfix_assembly()
    brain.engine_for(area).set_winners(area, empty)
    brain.areas[area].winners = empty


def class_slot_neurons(slot_index: int, k: int) -> np.ndarray:
    return np.arange(slot_index * k, (slot_index + 1) * k, dtype=np.uint32)


def fix_class_slot(brain, area: str, digit: int, k: int) -> None:
    slot = class_slot_neurons(digit, k)
    brain.areas[area].winners = slot
    brain.areas[area].fix_assembly()
    eng = brain.engine_for(area)
    eng.set_winners(area, slot)
    eng.fix_assembly(area)


def reinforce_class_slot(
    brain,
    src_area: str,
    dst_area: str,
    slot_index: int,
    k: int,
    *,
    beta: float | None = None,
) -> None:
    """Hebbian reinforcement from src winners to a fixed digit slot in dst."""
    brain.reinforce_connectome(
        src_area,
        dst_area,
        class_slot_neurons(slot_index, k),
        beta=beta,
    )


def read_slot_scores(winners, n_slots: int, k: int) -> list[float]:
    """Overlap of winners with each contiguous slot [i*k, (i+1)*k)."""
    return [slot_overlap(winners, i, k) for i in range(n_slots)]


def slot_overlap(winners, slot_index: int, k: int) -> float:
    slot = np.arange(slot_index * k, (slot_index + 1) * k, dtype=int)
    if len(winners) == 0:
        return 0.0
    return len(np.intersect1d(winners, slot)) / k
