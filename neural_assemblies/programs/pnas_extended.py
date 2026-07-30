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
    rounds: int = 20,
    recurrent: bool = True,
) -> PnasReciprocalResult:
    """PNAS20-M04: A -> B -> A restores A's original assembly.

    SAME PROTOCOL DEFECT as ``run_pnas_pattern_complete``, found the same day.
    This called ``project(...)`` with ``ops.project``'s default
    ``recurrent=False``, so ``A -> A`` was never trained -- and the recovery
    leg leans on exactly that fiber to pattern-complete A from B's return
    drive.  Measured over 8 seeds::

        recurrent=False  rounds=10   0.2188 +/- 0.0380   <- the old default
        recurrent=False  rounds=20   0.7703 +/- 0.0375
        recurrent=True   rounds=10   0.4391 +/- 0.0849
        recurrent=True   rounds=20   0.9922 +/- 0.0124   <- reference protocol

    The reference at its own ``rounds=20`` gives ``0.9988 +/- 0.0033``, so the
    corrected protocol reproduces it.  Both parameters matter and neither is a
    free choice: the reference's build idiom IS
    ``b.project({"stimA": ["A"]}, {"A": ["A"]})`` repeated 20 times
    (``.reference/dmitropolsky-assemblies/overlap_sim.py``), and
    ``(1 + 0.1)^10 = 2.59`` sits below the norm_init retention threshold while
    ``(1 + 0.1)^20 = 6.7`` clears it.

    THE OLD 0.6 THRESHOLD WAS NEVER ACHIEVABLE at rounds=10 -- not by us and
    not by the reference, which reads ``0.5875 +/- 0.1047`` there.  It traced to
    a single-seed reading of 0.740 quoted in an ``ops.py`` docstring.  Set
    thresholds from a multi-seed reference distribution, not from one run.
    """
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse")
    brain.add_stimulus("s", k)
    brain.add_area("A", n, k, beta)
    brain.add_area("B", n, k, beta)
    orig = project(brain, "s", "A", rounds=rounds, recurrent=recurrent)
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
    rounds: int = 20,
    cue_fraction: float = 0.5,
    recurrent: bool = True,
) -> PnasPatternCompleteResult:
    """PNAS20-M05: recover a full assembly from a 50% cue.

    THE PROTOCOL WAS WRONG UNTIL 2026-07-30 and the recorded 1.0 was an
    artifact.  It called ``project(...)`` with ``ops.project``'s default
    ``recurrent=False``, so ``A -> A`` was NEVER TRAINED -- while the completion
    step projects through exactly that fiber.  The theory the operation states
    ("partial activation flows back through STRENGTHENED recurrent
    connections") therefore had nothing to flow through.

    What was actually being measured, and why it read perfectly:

    ==========================================  =================  ============
    protocol                                    overlap, 5 seeds   verdict
    ==========================================  =================  ============
    untrained + self fiber DEAD (pre-fix)       1.0000 +/- 0.0000  preserve-cue
    untrained + self fiber live                 0.0075 +/- 0.0061  chance
    recurrent=True, rounds=10                   0.0700 +/- 0.0573  below floor
    recurrent=True, rounds=20                   0.9725 +/- 0.0255  ATTRACTOR
    recurrent=True, rounds=40                   1.0000 +/- 0.0000  ATTRACTOR
    ==========================================  =================  ============

    An empty ``A -> A`` block delivers zero drive, so ``project_into`` took its
    "zero signal -- preserve current assembly" branch and handed back the cue
    unchanged.  The score then compared the cue against a reference it is part
    of.  See ``_self_fiber_deferred_init`` in the numpy engine.

    ``rounds`` is 20 rather than 10 because ``(1 + 0.1)^10 = 2.59`` sits BELOW
    the norm_init retention threshold while ``(1 + 0.1)^20 = 6.7`` clears it --
    the same lever that explains the reciprocal-restoration shortfall (#53).
    The reference uses the same idiom and the same count:
    ``b.project({"stimA": ["A"]}, {"A": ["A"]})`` x20
    (``.reference/dmitropolsky-assemblies/overlap_sim.py``).

    Note the arithmetic floor: the kept half IS part of the reference, so any
    score at or below ``cue_fraction`` is consistent with recruiting pure noise.
    Only excess above it is evidence of attractor dynamics.
    """
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse")
    brain.add_stimulus("stim", k)
    brain.add_area("A", n, k, beta)
    project(brain, "stim", "A", rounds=rounds, recurrent=recurrent)
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
