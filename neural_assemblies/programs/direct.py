"""
DIRECT directional binding (Kopadi & Kalles 2026).

Trains asymmetric cause → effect coupling in a binding area and measures
forward vs reverse retrieval asymmetry.
"""

from __future__ import annotations

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.ops import _fix, _snap, _unfix, overlap, project


def direct_bind(
    brain,
    cause: str,
    effect: str,
    bind: str,
    cause_stim: str | None = None,
    effect_stim: str | None = None,
    rounds: int = 10,
) -> Assembly:
    """Train directional binding ``cause → effect`` in ``bind``.

    Phase 1 strengthens ``cause → bind``; phase 2 lightly couples
    ``effect → bind`` so reverse retrieval stays weaker than forward.
    """
    if cause_stim:
        project(brain, cause_stim, cause, rounds=rounds)
    if effect_stim:
        project(brain, effect_stim, effect, rounds=rounds)

    _fix(brain, cause)
    brain.project({}, {cause: [bind]})
    for _ in range(rounds - 1):
        brain.project({}, {cause: [bind], bind: [bind]})
    _unfix(brain, cause)

    _fix(brain, effect)
    brain.project({}, {effect: [bind]})
    for _ in range(max(2, rounds // 3)):
        brain.project({}, {effect: [bind], bind: [bind]})
    _unfix(brain, effect)

    return _snap(brain, bind)


def measure_directional_asymmetry(
    brain,
    cause: str,
    effect: str,
    bind: str,
    probe_rounds: int = 5,
) -> tuple[float, float]:
    """Return ``(forward_overlap, reverse_overlap)`` after directional training.

    Forward: cue ``cause`` and project into ``bind``.
    Reverse: cue ``effect`` and project into ``bind``.
  Both are compared to the trained binding assembly snapshot.
    """
    reference = _snap(brain, bind)

    _fix(brain, cause)
    brain.project({}, {cause: [bind]})
    for _ in range(probe_rounds):
        brain.project({}, {cause: [bind], bind: [bind]})
    forward = overlap(_snap(brain, bind), reference)
    _unfix(brain, cause)

    _fix(brain, effect)
    brain.project({}, {effect: [bind]})
    for _ in range(probe_rounds):
        brain.project({}, {effect: [bind], bind: [bind]})
    reverse = overlap(_snap(brain, bind), reference)
    _unfix(brain, effect)

    return forward, reverse


def intervention_bind_overlap(
    brain,
    fixed_area: str,
    cue_area: str,
    bind: str,
    probe_rounds: int = 5,
) -> float:
    """Pearl-style ``do(fixed_area)``: hold one area fixed while cueing another into bind."""
    reference = _snap(brain, bind)

    _fix(brain, fixed_area)
    _unfix(brain, cue_area)
    brain.project({}, {cue_area: [bind]})
    for _ in range(probe_rounds):
        brain.project({}, {cue_area: [bind], bind: [bind]})
    score = overlap(_snap(brain, bind), reference)
    _unfix(brain, fixed_area)
    return score


def validate_direct_do_calculus(
    brain,
    cause: str,
    effect: str,
    bind: str,
    probe_rounds: int = 5,
    min_forward: float = 0.1,
) -> tuple[float, float, float]:
    """Validate DIRECT causal binding plus a ``do(effect)`` intervention readout.

    Kopadi & Kalles 2026: forward retrieval should exceed chance; holding
    *effect* fixed while cueing *cause* should preserve binding strength.
    """
    forward, reverse = measure_directional_asymmetry(
        brain, cause, effect, bind, probe_rounds=probe_rounds,
    )
    do_effect_forward = intervention_bind_overlap(
        brain, effect, cause, bind, probe_rounds=probe_rounds,
    )
    if forward < min_forward:
        raise AssertionError(f"DIRECT forward {forward:.3f} below {min_forward}")
    if reverse <= 0.0:
        raise AssertionError(f"DIRECT reverse overlap must be positive, got {reverse:.3f}")
    if do_effect_forward < min_forward:
        raise AssertionError(
            f"do(effect) forward {do_effect_forward:.3f} below {min_forward}"
        )
    if do_effect_forward < forward * 0.45:
        raise AssertionError(
            f"do(effect) forward {do_effect_forward:.3f} too weak vs {forward:.3f}"
        )
    return forward, reverse, do_effect_forward
