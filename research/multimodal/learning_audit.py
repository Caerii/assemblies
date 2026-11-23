"""Fail-closed audit of explicit native Brain source->role synaptic matrices.

For diagnostics only: accesses NumpyExplicitEngine._area_conns, whose storage
is intentionally not a stable public API. The audit refuses missing paths.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _fiber(brain, role: str, source: str = 'LEX') -> np.ndarray:
    try:
        engine = brain.engine_for(role)
        conn = engine._area_conns[source][role]
        data = conn.weights
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(f'Cannot inspect explicit fiber {source}->{role}') from exc
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise RuntimeError('Expected dense 2D NumPy explicit weights')
    if not np.issubdtype(data.dtype, np.floating):
        raise RuntimeError('Expected floating-point weights')
    if not np.isfinite(data).all():
        raise RuntimeError('Nonfinite explicit synaptic weights')
    return data


def weight_fingerprint(brain, roles: tuple[str, ...], source: str = 'LEX') -> str:
    """Hash all source->role matrices, not just the trained submatrix."""
    if not roles or len(roles) != len(set(roles)):
        raise ValueError('Roles must be nonempty and unique')
    digest = hashlib.sha256()
    for role in sorted(roles):
        matrix = _fiber(brain, role, source)
        digest.update(f'{source}->{role}|{matrix.dtype.str}|{matrix.shape}'.encode())
        digest.update(np.ascontiguousarray(matrix).tobytes())
    return digest.hexdigest()


def active_synapses(brain, role, cue, target, source='LEX') -> np.ndarray:
    matrix = _fiber(brain, role, source)
    src, dst = np.asarray(cue), np.asarray(target)
    for label, ids, upper in (('cue', src, matrix.shape[0]), ('target', dst, matrix.shape[1])):
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= upper)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError(f'Invalid {label} IDs for fiber')
    return matrix[np.ix_(src, dst)].copy()


def write_diagnostic(before, after, target, winners):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError('Expected comparable active synapse submatrices')
    win, tgt = set(map(int, winners)), set(map(int, target))
    delta = after.astype(np.float64) - before.astype(np.float64)
    return {'active_edge_delta_sum': float(delta.sum()),
            'active_edge_changed': int(np.count_nonzero(delta)),
            'teacher_target_overlap': len(win & tgt) / len(tgt) if tgt else 0.}
