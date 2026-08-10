"""The refraction accumulation rule, in one place, for every backend.

WHAT REFRACTION IS FOR
----------------------
A refracted area accumulates a per-neuron bias that is subtracted from drive
before k-WTA, so neurons that keep winning become progressively harder to fire.
In a conjunction area -- one receiving two fibers, e.g. ``(state, symbol)`` in
Theorem 4's FSM -- this is the force that stops the area collapsing onto
whichever conjunct is exposed more.

It is not a modifier that sharpens an already-working result. Ablated from the
vendored reference FSM (`research/experiments/seq_arc_refraction_reference.py`):

    refraction              across-state   across-symbol   mod-3 task
    drive-proportional          0.000          0.000          3/3
    off                         0.088          0.989          0/3

Chance overlap is k/n = 0.014. Without refraction the arc stops representing
the symbol at all, and the FSM cannot decide. Task #92 measured the mirror
degeneracy (arc collapsed onto the symbol, ignoring state, overlap 0.90-0.99)
and concluded a conjunctive arc "has no operating point"; the direction of
collapse simply follows which conjunct fires more, per
[[role-gain-crowds-not-margins]].

WHY THE RULE IS PROPORTIONAL TO DRIVE
-------------------------------------
The reference accumulates a fraction of the winner's own raw drive::

    bias[winner] += raw_drive[winner] * strength

Every engine in this repo used to accumulate a CONSTANT instead::

    bias[winner] += strength

Both subtract identically before k-WTA; only accumulation differed. The
constant is not an equivalent parameterization, because Hebbian learning
multiplies drive by ``(1 + beta)`` per presentation while a constant increment
grows only linearly -- so refraction fades exactly as training proceeds.
Substituting the constant rule into the reference and sweeping it, the constant
does have an operating point, but it is a knife-edge AND it MOVES with the
drive scale:

    presentations |  const=1  const=3  const=10  const=30  const=100 | proportional
              5   |    0/3      0/3      0/3       0/3       0/3     |     0/3
             15   |    0/3      0/3      3/3       0/3       0/3     |     3/3
             30   |    0/3      0/3      0/3       3/3       0/3     |     3/3

The value that wins at 15 presentations fails at 30. The proportional rule
passes both untouched. This is [[hebbian-mass-follows-frequency]] in another
guise: normalize the rule, do not tune a per-task constant.

Every call site in the repo already passes ``refracted_strength=0.1``, and 0.1
is exactly the reference's ``plasticity``, so correcting the rule makes those
call sites right without changing their arguments.

WHY IT LIVES HERE AND NOT IN AN ENGINE
--------------------------------------
The accumulation existed as three copies -- ``numpy_engine/_sparse.py``,
``torch_engine/_engine.py`` and ``cuda_engine.py`` -- all three carrying the
same wrong rule. That is the shape of [[pricing-law-implemented-twice]]: a law
duplicated per backend drifts, and the copy that is not fixed reports a
different experiment rather than a different number. One owner, three callers.
"""

from __future__ import annotations

import os


def constant_refraction_enabled() -> bool:
    """Whether to accumulate a CONSTANT increment instead of the correct rule.

    Off by default. Set ``ASSEMBLIES_CONSTANT_REFRACTION=1`` to restore the
    pre-correction behaviour for an A/B. Kept reachable rather than deleted for
    the same reason `_fixed_target_plasticity_enabled` keeps its short-circuit:
    a behaviour change wants a switch that can reproduce the old numbers.
    """
    return os.environ.get(
        "ASSEMBLIES_CONSTANT_REFRACTION", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def refraction_increment(net_drive_at_winners, current_bias_at_winners,
                         strength: float):
    """Bias increment for the winners of one projection.

    ``net_drive_at_winners`` is the drive the engine actually ranked, i.e.
    AFTER the bias was subtracted; ``current_bias_at_winners`` is the bias that
    was subtracted from it. The reference charges against the RAW drive
    (`RefractedArea.update` calls ``super().get_total_input()``, which does not
    subtract the bias), so the raw value is reconstructed here as
    ``net + bias``. Doing it in this module rather than at each call site is
    the point: that "+ bias" is the easiest step to get wrong, and getting it
    wrong makes refraction self-limiting instead of homeostatic.

    Both arguments must be array-likes of the backend in use; the arithmetic is
    elementwise and backend-agnostic. Returns an array of the same shape.

    NOTE ON LRI. If an area ALSO runs a sliding-window refractory penalty, the
    reconstruction above does not add that penalty back, so the raw drive is
    underestimated by it. The reference has no LRI and refracted areas in this
    repo do not set one; a target that needs both wants an explicit decision
    about which penalty the charge is computed against, not this default.
    """
    if constant_refraction_enabled():
        return current_bias_at_winners * 0 + strength
    return (net_drive_at_winners + current_bias_at_winners) * strength
