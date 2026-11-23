"""Native Brain conformance probe. This executes the REAL assembly runtime.

This is deliberately a probe, not proof of parity against the standalone
fiber model. The native engine has recruitment, per-fiber initialization,
plasticity scheduling and inhibition semantics that the standalone fiber lacks.

Run from repository root:
    uv run python research/multimodal/native_contract_probe.py
"""
from __future__ import annotations

import json

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import project, pattern_complete


def run(seed: int = 101, n: int = 256, k: int = 16, rounds: int = 6) -> dict:
    if not (0 < k <= n):
        raise ValueError("k must be between 1 and n")
    brain = Brain(engine="numpy_exact", seed=seed, p=.1, norm_init=False)
    brain.add_stimulus("cue", k)
    brain.add_area("memory", n=n, k=k, beta=.1)
    first = project(brain, "cue", "memory", rounds=rounds, recurrent=True)
    # Check non-mutating readout using the PUBLIC API. The result is a score,
    # not evidence of cognition or cross-modal binding.
    _, recovered = pattern_complete(
        brain, "memory", fraction=.5, rounds=3, seed=seed,
        observation_mode="read-only",
    )
    second = project(brain, "cue", "memory", rounds=rounds, recurrent=True)
    winners = sorted(map(int, second.winners))
    if len(winners) != k or len(set(winners)) != k:
        raise AssertionError("invalid final winner count or duplicate neuron IDs")
    if not 0. <= float(recovered) <= 1.:
        raise AssertionError("readout overlap outside [0,1]")
    return {
        "seed": seed, "engine": "numpy_exact", "n": n, "k": k,
        "rounds": rounds, "first_winners": sorted(map(int, first.winners)),
        "last_winners": winners,
        "half_cue_recovery": float(recovered),
        "native_executed": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
