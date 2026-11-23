"""Independent controlled multiplicative-plasticity reference.

THIS IS NOT THE NATIVE BRAIN ENGINE. It studies one fiber with an explicit
connectome. The strict native-Brain parity gate is separate.
"""
from __future__ import annotations

import numpy as np


def _code(indices, n: int, k: int):
    values = np.asarray(indices)
    if values.ndim != 1 or len(values) != k or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("expected k integer neuron IDs")
    if (values < 0).any() or (values >= n).any() or len(set(map(int, values))) != k:
        raise ValueError("invalid or duplicate neuron IDs")
    return np.sort(values.astype(np.uint32))


class MultiplicativeFiber:
    def __init__(self, weights, k: int, beta: float = .1, w_max: float = 20., *,
                 read_only: bool = False):
        a = np.asarray(weights, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or not 0 < k <= a.shape[0]:
            raise ValueError("expected square matrix and valid k")
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("weights must be finite and nonnegative")
        if not np.isfinite(beta) or beta < 0 or not np.isfinite(w_max) or w_max <= 0:
            raise ValueError("invalid plasticity parameters")
        if (a > w_max).any():
            raise ValueError("initial weights exceed cap")
        self.n, self.k, self.beta, self.w_max = a.shape[0], k, beta, w_max
        self.weights = a.copy()
        self.read_only = read_only
        self.writes = 0

    def drive(self, source):
        src = _code(source, self.n, self.k)
        out = np.zeros(self.n, dtype=np.float32)
        for s in src:
            out += self.weights[int(s)]
        return out

    def predict(self, source):
        scores = self.drive(source)
        ordered = np.lexsort((np.arange(self.n), -scores))
        return np.sort(ordered[:self.k].astype(np.uint32))

    def cofire(self, source, target, rounds: int = 1):
        if self.read_only:
            raise PermissionError("read-only observation cannot mutate")
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        src, dst = _code(source, self.n, self.k), _code(target, self.n, self.k)
        growth = np.float32(1. + self.beta)
        for _ in range(rounds):
            for s in src:
                for t in dst:
                    old = self.weights[int(s), int(t)]
                    if old != 0:
                        self.weights[int(s), int(t)] = np.float32(
                            min(self.w_max, np.float32(old * growth))
                        )
            self.writes += 1


class SparseMultiplicativeFiber:
    """Independent sparse-row implementation with deterministic accumulation."""
    def __init__(self, weights, k: int, beta: float = .1, w_max: float = 20.):
        ref = MultiplicativeFiber(weights, k, beta, w_max)
        self.n, self.k, self.beta, self.w_max = ref.n, ref.k, ref.beta, ref.w_max
        self.rows = {
            i: {int(j): np.float32(v) for j, v in enumerate(row) if v != 0}
            for i, row in enumerate(ref.weights) if np.any(row)
        }
        self.writes = 0

    def drive(self, source):
        src = _code(source, self.n, self.k)
        out = np.zeros(self.n, dtype=np.float32)
        for s in src:
            row = np.zeros(self.n, dtype=np.float32)
            for t, value in self.rows.get(int(s), {}).items():
                row[t] = value
            out += row
        return out

    def predict(self, source):
        ordered = np.lexsort((np.arange(self.n), -self.drive(source)))
        return np.sort(ordered[:self.k].astype(np.uint32))

    def cofire(self, source, target, rounds: int = 1):
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        src, dst = _code(source, self.n, self.k), _code(target, self.n, self.k)
        growth = np.float32(1. + self.beta)
        for _ in range(rounds):
            for s in src:
                row = self.rows.get(int(s))
                if row is None:
                    continue
                for t in dst:
                    tid = int(t)
                    if tid in row:
                        row[tid] = np.float32(min(self.w_max, np.float32(row[tid] * growth)))
            self.writes += 1

    def dense(self):
        out = np.zeros((self.n, self.n), dtype=np.float32)
        for s, row in self.rows.items():
            for t, value in row.items():
                out[s, t] = value
        return out
