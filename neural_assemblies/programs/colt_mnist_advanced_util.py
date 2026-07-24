"""Advanced MNIST helpers — CLASS consolidation and connectome readout."""

from __future__ import annotations

import numpy as np

from neural_assemblies.core.backend import to_cpu
from neural_assemblies.programs.colt_mnist_brain_util import (
    class_slot_neurons,
    renorm_connectome_columns,
)


def read_class_connectome_scores(
    high_vec: np.ndarray,
    brain,
    src: str,
    dst: str,
    k: int,
    n_slots: int,
) -> np.ndarray:
    """Sum readout over CLASS digit slots (Papadimitriou et al. Fig. 2)."""
    w = np.asarray(to_cpu(brain.connectomes[src][dst].weights), dtype=np.float64)
    winners = np.flatnonzero(high_vec > 0)
    if winners.size == 0:
        return np.zeros(n_slots, dtype=np.float64)
    return np.array(
        [float(w[winners, i * k:(i + 1) * k].sum()) for i in range(n_slots)],
        dtype=np.float64,
    )


def wire_class_from_prototypes(
    brain,
    prototypes: np.ndarray,
    src: str,
    dst: str,
    k: int,
) -> None:
    """Consolidate src→dst readout weights from learned prototypes (systems replay)."""
    wire_digit_slots_from_prototypes(brain, prototypes, src, dst, k)


def wire_digit_slots_from_prototypes(
    brain,
    prototypes: np.ndarray,
    src: str,
    dst: str,
    k: int,
    *,
    digits: frozenset[int] | None = None,
) -> None:
    """Wire src prototypes to contiguous digit slots in dst (CLASS or SEMANTIC)."""
    n_digits = prototypes.shape[0]
    w = np.asarray(to_cpu(brain.connectomes[src][dst].weights), dtype=np.float64)
    if digits is None:
        w[:] = 0.0
    for digit in range(n_digits):
        if digits is not None and digit not in digits:
            continue
        pre = np.flatnonzero(prototypes[digit] > 0)
        post = class_slot_neurons(digit, k)
        w[np.ix_(pre, post)] = 1.0
    conn = brain.connectomes[src][dst]
    conn.weights = w.astype(np.float32)
    if brain._explicit_engine is not None:
        econn = brain._explicit_engine._area_conns.get(src, {}).get(dst)
        if econn is not None:
            econn.weights = conn.weights
    renorm_connectome_columns(brain, src, dst)


def wire_class_mixed_prototypes(
    brain,
    prototypes: np.ndarray,
    refined: np.ndarray,
    confused_digits: frozenset[int],
    src: str,
    dst: str,
    k: int,
) -> None:
    """Wire CLASS using refined prototypes for confused digits only."""
    w = np.asarray(to_cpu(brain.connectomes[src][dst].weights), dtype=np.float64)
    w[:] = 0.0
    for digit in range(prototypes.shape[0]):
        proto = refined[digit] if digit in confused_digits else prototypes[digit]
        pre = np.flatnonzero(proto > 0)
        post = class_slot_neurons(digit, k)
        w[np.ix_(pre, post)] = 1.0
    conn = brain.connectomes[src][dst]
    conn.weights = w.astype(np.float32)
    if brain._explicit_engine is not None:
        econn = brain._explicit_engine._area_conns.get(src, {}).get(dst)
        if econn is not None:
            econn.weights = conn.weights
    renorm_connectome_columns(brain, src, dst)


def wire_semantic_from_anchors(
    brain,
    prototypes: np.ndarray,
    anchors: dict[str, "Assembly"],
    src: str,
    dst: str,
) -> None:
    """Wire src→dst using anchor support (non-slot semantic assemblies)."""
    w = np.asarray(to_cpu(brain.connectomes[src][dst].weights), dtype=np.float64)
    w[:] = 0.0
    for digit in range(prototypes.shape[0]):
        pre = np.flatnonzero(prototypes[digit] > 0)
        anchor = anchors[str(digit)]
        post = np.asarray(anchor.winners, dtype=np.int64)
        if pre.size and post.size:
            w[np.ix_(pre, post)] = 1.0
    conn = brain.connectomes[src][dst]
    conn.weights = w.astype(np.float32)
    if brain._explicit_engine is not None:
        econn = brain._explicit_engine._area_conns.get(src, {}).get(dst)
        if econn is not None:
            econn.weights = conn.weights
    renorm_connectome_columns(brain, src, dst)


def wire_semantic_from_view_projections(
    brain,
    high_outputs: np.ndarray,
    src: str,
    dst: str,
    k: int,
    *,
    digits: frozenset[int] | None = None,
    reset: bool = True,
) -> None:
    """Wire src->dst from all training-view HIGH assemblies (stronger than prototypes)."""
    n_digits = high_outputs.shape[0]
    w = np.asarray(to_cpu(brain.connectomes[src][dst].weights), dtype=np.float64)
    if reset:
        w[:] = 0.0
    for digit in range(n_digits):
        if digits is not None and digit not in digits:
            continue
        post = class_slot_neurons(digit, k)
        for j in range(high_outputs.shape[1]):
            pre = np.flatnonzero(high_outputs[digit, j] > 0)
            if pre.size:
                w[np.ix_(pre, post)] += 1.0
    conn = brain.connectomes[src][dst]
    conn.weights = w.astype(np.float32)
    if brain._explicit_engine is not None:
        econn = brain._explicit_engine._area_conns.get(src, {}).get(dst)
        if econn is not None:
            econn.weights = conn.weights
    renorm_connectome_columns(brain, src, dst)


def refined_confused_prototypes(
    prototypes: np.ndarray,
    high_outputs: np.ndarray,
    confused_digits: frozenset[int],
    k: int,
) -> np.ndarray:
    """Re-centre prototypes for stroke-confusable digits from view mean activity."""
    out = prototypes.copy()
    for digit in confused_digits:
        mean_act = high_outputs[digit].mean(axis=0)
        top = np.argsort(mean_act)[-k:]
        out[digit] = 0.0
        out[digit, top] = 1.0
    return out


def prototypes_to_lexicon(
    prototypes: np.ndarray,
    area_name: str,
    k: int,
) -> dict[str, "Assembly"]:
    """Build a digit lexicon from prototype support vectors."""
    from neural_assemblies.assembly_calculus.assembly import Assembly

    lexicon: dict[str, Assembly] = {}
    for digit in range(prototypes.shape[0]):
        support = np.flatnonzero(prototypes[digit] > 0).astype(np.uint32)
        if support.size == 0:
            support = prototypes[digit].argsort()[-k:].astype(np.uint32)
        lexicon[str(digit)] = Assembly(area_name, support)
    return lexicon
