"""PNAS 2020 extended ops — reciprocal restore and pattern completion."""

from __future__ import annotations

from dataclasses import dataclass

from neural_assemblies.assembly_calculus.assembly import chance_overlap
from neural_assemblies.assembly_calculus.ops import (
    overlap,
    pattern_complete,
    project,
    reciprocal_project,
    _snap,
)


@dataclass
class PnasReciprocalResult:
    reciprocal_restore_overlap: float
    parameters: dict


@dataclass
class PnasPatternCompleteResult:
    pattern_complete_50pct_overlap: float
    parameters: dict


def run_pnas_reciprocal(
    *,
    seed: int = 42,
    n: int = 5000,
    k: int = 80,
    p: float = 0.05,
    beta: float = 0.1,
    rounds: int = 10,
) -> PnasReciprocalResult:
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse")
    brain.add_stimulus("s", k)
    brain.add_area("A", n, k, beta)
    brain.add_area("B", n, k, beta)
    orig = project(brain, "s", "A", rounds=rounds)
    reciprocal_project(brain, "A", "B", rounds=rounds)
    reciprocal_project(brain, "B", "A", rounds=rounds)
    recip_ov = overlap(orig, _snap(brain, "A"))
    params = {"seed": seed, "n": n, "k": k, "p": p, "beta": beta, "rounds": rounds}
    return PnasReciprocalResult(
        reciprocal_restore_overlap=float(recip_ov),
        parameters=params,
    )


def run_pnas_pattern_complete(
    *,
    seed: int = 42,
    n: int = 5000,
    k: int = 80,
    p: float = 0.05,
    beta: float = 0.1,
    rounds: int = 10,
    cue_fraction: float = 0.5,
) -> PnasPatternCompleteResult:
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse")
    brain.add_stimulus("stim", k)
    brain.add_area("A", n, k, beta)
    project(brain, "stim", "A", rounds=rounds)
    _, pc_ov = pattern_complete(
        brain, "A", fraction=cue_fraction, rounds=5, seed=seed,
    )
    params = {
        "seed": seed, "n": n, "k": k, "p": p, "beta": beta,
        "rounds": rounds, "cue_fraction": cue_fraction,
    }
    return PnasPatternCompleteResult(
        pattern_complete_50pct_overlap=float(pc_ov),
        parameters=params,
    )


def chance_overlap_metric(k: int, n: int) -> float:
    return float(chance_overlap(k, n))
