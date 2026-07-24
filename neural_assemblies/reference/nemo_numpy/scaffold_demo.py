"""Scaffold vs simple sequence recall (nemo-demo.ipynb protocol)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .areas import RecurrentArea, ScaffoldNetwork, positional_overlap


@dataclass
class ScaffoldRecallResult:
    simple_last: float
    scaffold_last: float
    simple_history: List[float]
    scaffold_history: List[float]
    scaffold_beats_simple: bool


def compare_scaffold_vs_simple(
    *,
    n_presentations: int = 10,
    n: int = 1000,
    k: int = 30,
    seq_len: int = 20,
    seed: int = 42,
    density: float = 0.4,
    plasticity: float = 0.1,
) -> ScaffoldRecallResult:
    """Reference numpy protocol: last-item recall after each presentation."""
    rng = np.random.default_rng(seed)
    cap_size = k
    simple = RecurrentArea(n, n, cap_size, density, plasticity, rng)
    scaffold = ScaffoldNetwork(n, n, cap_size, density, plasticity, rng)

    sequence = np.arange(seq_len * cap_size).reshape(seq_len, cap_size)
    simple_refs = np.zeros((seq_len, cap_size), dtype=int)
    scaff_refs = np.zeros((seq_len, cap_size), dtype=int)
    simple_hist: List[float] = []
    scaff_hist: List[float] = []

    for _j in range(n_presentations):
        for i in range(seq_len):
            simple.forward(sequence[i])
            scaffold.forward(sequence[i])
            if _j == 0:
                simple_refs[i] = simple.read()
                scaff_refs[i] = scaffold.read()

        simple.inhibit()
        simple.forward(sequence[0], update=False)
        simple_curve = []
        for i in range(seq_len):
            if i > 0:
                simple.step(update=False)
            simple_curve.append(positional_overlap(simple.read(), simple_refs[i]))

        scaffold.inhibit()
        scaffold.forward(sequence[0], update=False)
        scaff_curve = []
        for i in range(seq_len):
            if i > 0:
                scaffold.step(update=False)
            scaff_curve.append(positional_overlap(scaffold.read(), scaff_refs[i]))

        simple_hist.append(simple_curve[-1])
        scaff_hist.append(scaff_curve[-1])

    return ScaffoldRecallResult(
        simple_last=simple_hist[-1],
        scaffold_last=scaff_hist[-1],
        simple_history=simple_hist,
        scaffold_history=scaff_hist,
        scaffold_beats_simple=scaff_hist[-1] > simple_hist[-1],
    )
