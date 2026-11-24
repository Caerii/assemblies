"""Exact full-connectome oracle for native explicit Hebbian projection.

This checker is independent of NumpyExplicitEngine.project_into. It uses the
contracted public learn schedule and copies dense area-to-area matrices BEFORE
the write. Intended for small validation cells, not production inference.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np


def snapshot_area_fibers(brain, areas: Sequence[str]) -> dict[tuple[str, str], np.ndarray]:
    """Copy ALL source/target area-to-area fibers; fail on mixed ownership.

    A complete directed graph is required: a newly added area or missing fiber
    cannot quietly escape validation. Stimulus connections are out of scope;
    this experimental protocol deliberately registers no stimuli.
    """
    names = tuple(areas)
    if not names or len(names) != len(set(names)) or not all(
            isinstance(name, str) and name for name in names):
        raise ValueError("areas must be unique nonempty names")
    engine = brain.engine_for(names[0])
    if any(brain.engine_for(name) is not engine for name in names[1:]):
        raise RuntimeError("mixed engine ownership cannot be audited as one graph")
    graph = getattr(engine, "_area_conns", None)
    if graph is None:
        raise RuntimeError("native explicit fiber registry missing")
    captured: dict[tuple[str, str], np.ndarray] = {}
    for src in names:
        for dst in names:
            try:
                weights = graph[src][dst].weights
            except (KeyError, AttributeError, TypeError) as exc:
                raise RuntimeError(f"missing explicit fiber {src}->{dst}") from exc
            if (not isinstance(weights, np.ndarray) or weights.ndim != 2
                    or weights.dtype != np.dtype("float32")
                    or not np.isfinite(weights).all()):
                raise RuntimeError(f"invalid explicit float32 fiber {src}->{dst}")
            captured[src, dst] = weights.copy(order="C")
    return captured


def graph_digest(graph: Mapping[tuple[str, str], np.ndarray]) -> str:
    """Hash source/destination, dtype, shape, and all synaptic bytes."""
    h = hashlib.sha256()
    for (src, dst), weights in sorted(graph.items()):
        h.update(f"{src}->{dst}|{weights.dtype.str}|{weights.shape}".encode())
        h.update(np.ascontiguousarray(weights).tobytes())
    return h.hexdigest()


def _indices(value, n: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if (result.ndim != 1 or not np.issubdtype(result.dtype, np.integer)
            or len(result) == 0 or (result < 0).any()
            or (result >= n).any() or len(set(map(int, result))) != len(result)):
        raise ValueError(f"invalid {name} IDs")
    return result.astype(np.intp)


def verify_exact_hebbian_write(
    before: Mapping[tuple[str, str], np.ndarray],
    after: Mapping[tuple[str, str], np.ndarray],
    *, source: str, target: str, cue, selected_by_round: Sequence,
    beta: float, ceiling: float | None,
) -> dict:
    """Require exact native float32 multiplication on precisely coactive edges.

    Every other fiber and all off-target edges must be byte-for-byte equal.
    Selected winners, NOT teacher labels, determine eligible synapses.
    """
    if not before or set(before) != set(after) or (source, target) not in before:
        raise AssertionError("incomplete before/after directed fiber snapshots")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("invalid Hebbian beta")
    if ceiling is not None and (not np.isfinite(ceiling) or ceiling <= 0):
        raise ValueError("invalid synaptic ceiling")
    if not selected_by_round:
        raise ValueError("at least one write round required")
    dims = before[source, target].shape
    sources = _indices(cue, dims[0], "source")
    rounds = [_indices(w, dims[1], "winner") for w in selected_by_round]
    changes = 0
    for key, original in before.items():
        observed = after[key]
        if observed.shape != original.shape or observed.dtype != original.dtype:
            raise AssertionError(f"fiber changed shape or dtype: {key}")
        if not np.isfinite(observed).all():
            raise AssertionError(f"fiber contains nonfinite weights: {key}")
        expected = original.copy()
        if key == (source, target) and beta:
            for targets in rounds:
                idx = np.ix_(sources, targets)
                # Match NumPyExplicitEngine: float32 in-place multiplication,
                # followed by indexed float32 clipping and assignment.
                expected[idx] *= (1 + beta)
                if ceiling is not None:
                    piece = expected[idx]
                    np.clip(piece, 0, ceiling, out=piece)
                    expected[idx] = piece
        if not np.array_equal(observed, expected):
            wrong = int(np.count_nonzero(observed != expected))
            raise AssertionError(f"unexpected Hebbian change in {key}: {wrong} mismatched synapses")
        changes += int(np.count_nonzero(original != observed))
    return {
        "rounds": len(rounds),
        "changed_edges_full_graph": changes,
        "before_sha256": graph_digest(before),
        "after_sha256": graph_digest(after),
        "oracle": "native_numpy_explicit_float32_multiplicative",
    }
