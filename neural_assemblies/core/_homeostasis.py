"""The homeostats, in one place, for every backend.

WHAT THIS MODULE OWNS
---------------------
Three mechanisms in this repository hold a neuron's drive in check, and each
was measured into a law before it was given a home here. They are NOT
interchangeable and one pair of them is incompatible:

    mechanism        acts on                when          arithmetic here
    ---------------  ---------------------  ------------  ---------------------
    refraction       a per-NEURON bias,     every win     `refraction_increment`
                     subtracted pre-k-WTA
    column scaling   a fiber's per-COLUMN   every write   `scaling_setpoint`,
                     incoming mass                        `column_scale`
    norm_init        read-time 1/d_j        once, at      `_pricing.inverse_indegree`
                     per column             init          (a PRICING law; only the
                                                          map lives here)

THE THREE LAWS, each with its register ID
-----------------------------------------
REFRACTION is the exact anti-Hebbian counterweight. With ``w *= (1 + beta)``
on every active synapse into a winner and ``bias += raw * s`` charged at that
winner, a neuron winning repeatedly on the SAME input has::

    net_{t+1} - net_t = (beta - s) * raw_t

so at ``s == beta`` the net drive is CONSTANT under repetition: a neuron
gains nothing from its own history, and is handicapped on every other input
by that whole history. That is what a FEEDFORWARD conjunction area wants (the
reference's only use of ``RefractedArea``) and what a RECURRENT area cannot
survive -- it removes the convergence force, and above ``s ~ 0.75 beta`` the
area churns through every neuron ([[REFRACTION-CANCELS-CONVERGENCE]]). The
charge must be drive-proportional, not constant ([[REFRACTION-PROPORTIONAL]]).

COLUMN SCALING is the BASE-RATE corrector. Raw Hebbian mass counts
co-occurrences, which are dominated by how often the postsynaptic side fires
at all; renormalizing each column to its initial expected sum divides that
out, which is exactly what turns co-occurrence counting into association
learning. Measured on cross-situational word learning: without it every word
aligned to the corpus's most frequent referent; with it, alignment 0.99 and no
head/tail frequency bias ([[cross-situational-needs-homeostasis]]). On the
sequence organ's state area it fires, moves mass, and leaves the soft-defect
census untouched -- those defects are k-WTA bar ties, not mass imbalances.

NORM_INIT is the theorems' homeostasis LINEARIZED AT t=0: the reference
divides each column by its in-degree once, from ``reset()``. After training
the count-based divisor is potentiation-invariant while trained columns carry
``(1+beta)^T`` mass, so 1/d INVERTS the bias toward the sparsest neurons
([[homeostasis-is-a-theorem-hypothesis]]). It is a pricing law -- it exists
so sampled candidates and materialized incumbents are commensurable -- and
its arithmetic stays with the other pricing laws in `core/_pricing.py`.

THE INCOMPATIBILITY, enforced here
----------------------------------
Refraction charges its bias against RAW drive. Column scaling rescales raw
drive between wins. Together the bias ledger is denominated in a moving unit
while it compounds at ``(1 + s)``: a feedforward arc that holds ~100 rounds
under refraction alone and indefinitely under scaling alone loses its
assembly within ~10 presentations under both (late stability 0.12 against
1.00; `research/notes/memory/AUDIT_refraction_scaling.md`). Three S5 studies ran
that combination unscoped and their arc-side results are confounded.
`check_area_homeostasis` makes the combination unspellable at `Brain`.

WHY ONE OWNER
-------------
The refraction rule lived as three engine copies, all wrong together, until
`_refraction.py` (now folded in here) became its owner. The scaling law was
spelled in FIVE places -- the numpy engine, the torch engine, two CSR classes
and the hashed fiber -- each carrying its own setpoint line and its own
zero-mass guard. A law duplicated per backend drifts, and the copy that is
not fixed reports a different experiment rather than a different number
([[pricing-law-implemented-twice]]). Storage-specific work (how to sum a
column out of CSR, dense, or a generated connectome) stays in each engine;
everything that is ARITHMETIC or a GATE is here, and every engine calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Collection, Union

__all__ = [
    "constant_refraction_enabled",
    "refraction_increment",
    "scaling_applies",
    "HomeostasisConfig",
    "scaling_setpoint",
    "column_scale",
    "check_area_homeostasis",
    "HomeostasisConflict",
]

ScalingSpec = Union[bool, Collection[str]]


# ---------------------------------------------------------------------------
# Refraction
# ---------------------------------------------------------------------------

def constant_refraction_enabled() -> bool:
    """Whether to accumulate a CONSTANT increment instead of the correct rule.

    Off by default. Set ``ASSEMBLIES_CONSTANT_REFRACTION=1`` to restore the
    pre-correction behaviour for an A/B. Kept reachable rather than deleted
    for the same reason `_fixed_target_plasticity_enabled` keeps its
    short-circuit: a behaviour change wants a switch that can reproduce the
    old numbers. Read at CALL time, so an experiment can flip it on a running
    process and engines that bound `refraction_increment` at import honour it.
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

    WHY PROPORTIONAL. Hebbian learning multiplies drive by ``(1 + beta)`` per
    presentation while a constant increment grows only linearly, so a
    constant fades exactly as training proceeds; its operating point is a
    knife-edge that MOVES with the drive scale (the value that wins at 15
    presentations fails at 30). The proportional rule passes both untouched
    ([[REFRACTION-PROPORTIONAL]]). At ``strength == beta`` it is the exact
    counterweight described in the module docstring.

    Both arguments must be array-likes of the backend in use; the arithmetic
    is elementwise and backend-agnostic. Returns an array of the same shape.

    NOTE ON LRI. If an area ALSO runs a sliding-window refractory penalty, the
    reconstruction above does not add that penalty back, so the raw drive is
    underestimated by it. The reference has no LRI and refracted areas in this
    repo do not set one; a target that needs both wants an explicit decision
    about which penalty the charge is computed against, not this default.
    """
    if constant_refraction_enabled():
        return current_bias_at_winners * 0 + strength
    return (net_drive_at_winners + current_bias_at_winners) * strength


# ---------------------------------------------------------------------------
# Column scaling (substrate C)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HomeostasisConfig:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-homeostasis-config"""

    norm_init: bool = False
    synaptic_scaling: bool | frozenset[str] = False
    synaptic_scaling_deferred: bool = False

    def __post_init__(self):
        if type(self.norm_init) is not bool or type(self.synaptic_scaling_deferred) is not bool:
            raise ValueError("homeostasis flags must be explicit booleans")
        scope = self.synaptic_scaling
        if type(scope) is not bool:
            if not isinstance(scope, (set, frozenset, tuple, list)) or any(
                    not isinstance(name, str) or not name for name in scope):
                raise ValueError("synaptic_scaling must be boolean or a collection of nonempty area names")
            scope = frozenset(scope) or False
        object.__setattr__(self, "synaptic_scaling", scope)
        if self.synaptic_scaling_deferred and not scope:
            raise ValueError("deferred scaling requires an enabled scaling scope")

    @classmethod
    def from_engine(cls, engine):
        # An absent mechanism is disabled; backends with different storage must
        # override validate_brain_identity rather than claiming this convention.
        return cls(**{name: getattr(engine, name, False) for name in cls.__dataclass_fields__})

    def as_kwargs(self):
        return dict(vars(self))


def scaling_applies(synaptic_scaling: ScalingSpec, target: str) -> bool:
    """Does column scaling act on ``target``?

    ``True`` means every target area (the legacy spelling); a collection of
    area names scopes it to those targets only. Scoping is not cosmetic: it is
    how the sequence organ's refracted arc is kept OUT of scaling (see
    `check_area_homeostasis`), and `Brain` passes the collection through
    as an immutable canonical scope through `HomeostasisConfig`.
    """
    if not synaptic_scaling:
        return False
    if synaptic_scaling is True:
        return True
    return target in synaptic_scaling


def scaling_setpoint(rows: int, p: float) -> float:
    """The mass a scaled column is restored to: its INITIAL EXPECTED sum.

    ``rows * p`` over the presynaptic rows being summed, at the FIBER's own
    density -- two decisions, both measured:

    * The setpoint is the initial expected column sum, NOT 1. Unmaterialized
      neurons are sampled as ~Binomial(active, p) on the unit scale, so
      normalizing materialized columns to 1 would let every fresh candidate
      outbid every incumbent and no assembly could stabilize. The expected
      sum keeps both populations on one scale, and is also the biological
      statement of synaptic scaling: a setpoint, not unity.
    * Priced at the fiber's p, never the brain's. A global-p setpoint
      renormalized a p=0.40 organ fiber inside a p=0.05 brain to 1/8 of its
      natural mass and inverted learning ([[pricing-law-implemented-twice]]).

    "Restore each column to the mass IT started with" (a per-column degree
    setpoint) was tried and is the worst arm ever measured on this substrate
    -- pairwise overlap 26.7x chance against 0.188 -- because it cancels only
    the potentiation and hands the raw in-degree competition back to the hubs.
    The floor keeps an empty fiber from producing a zero setpoint.
    """
    return max(float(rows) * float(p), 1e-12)


def column_scale(sums, setpoint: float, *, xp: Any = None, eps: float = 1e-12):
    """Per-column factors that take ``sums`` to ``setpoint``: ``setpoint/sum``.

    A column whose summed mass is (numerically) zero is left at factor 1.0
    rather than blown up -- it has nothing to rescale, and a ``setpoint/eps``
    factor would be applied to whatever it acquires next.

    Backend-agnostic in the same way as `_pricing.inverse_indegree`: pass
    ``xp`` (numpy or cupy) for an array namespace; without it a torch tensor
    is assumed and ``torch.where`` is used. numpy evaluates both branches of
    ``where``, so the division is done on a guarded copy to keep the
    divide-by-zero warning out of the logs -- the result is identical.
    """
    if xp is not None:
        nonzero = xp.abs(sums) > eps
        safe = xp.where(nonzero, sums, 1.0)
        return xp.where(nonzero, setpoint / safe, 1.0)
    import torch
    return torch.where(sums.abs() > eps, setpoint / sums,
                       torch.ones_like(sums))


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

class HomeostasisConflict(ValueError):
    """Two homeostats that cannot share an area were asked to."""


def check_area_homeostasis(area_name: str, *, refracted: bool,
                           synaptic_scaling: ScalingSpec) -> None:
    """Refuse the one combination the audit showed destroys an area.

    Raised from `Brain.add_area` and `Brain.set_refracted`, so a refracted
    area inside a scaled brain has to be excluded from scaling EXPLICITLY
    (``synaptic_scaling={other areas}``) rather than silently confounded.
    """
    if refracted and scaling_applies(synaptic_scaling, area_name):
        raise HomeostasisConflict(
            f"area {area_name!r} is refracted AND column-scaled. Refraction "
            "charges its bias against raw drive and scaling rescales raw "
            "drive between wins, so the bias ledger is denominated in a "
            "moving unit: measured, a feedforward arc that holds ~100 rounds "
            "under refraction alone and indefinitely under scaling alone "
            "loses its assembly within ~10 presentations under both "
            "(research/notes/memory/AUDIT_refraction_scaling.md). Scope scaling to "
            "the other areas: Brain(synaptic_scaling={...names...}).")
