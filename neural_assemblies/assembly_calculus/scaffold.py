"""
Scaffolded sequence memorization (mdabagia/nemo ScaffoldNetwork port).

THE PROBLEM.  Plain :func:`~.ops.sequence_memorize` stores a chain inside one
area, so every assembly of the sequence competes for the same ``k`` winner
slots and writes into the same recurrent weight matrix.  Long sequences
therefore interfere with themselves: later items overwrite the bridges of
earlier ones, and recall degrades from the front.  Dabagia et al. report that
many presentations are needed before a long sequence is recalled reliably.

THE SCAFFOLD.  Adding a second, bidirectionally coupled area gives the chain
somewhere else to live.  Each item is held jointly by a main-area assembly and
an auxiliary-area assembly, and the main<->aux loop supplies a second,
independent route from item i to item i+1.  The empirical claim being ported
here is that this reduces the number of presentations needed for reliable
recall -- capacity is bought with a second population, not with more training.

Provides:
  - ``compare_scaffold_vs_simple`` — delegates to ``reference.nemo_numpy`` demo
  - ``ScaffoldNetwork`` — Brain API with auxiliary area coupling

Reference:
    Dabagia, Papadimitriou, Vempala. "Computation with Sequences of Assemblies
    in a Model of the Brain." Neural Computation (2025). arXiv:2306.03812.
    Upstream implementation: mdabagia/nemo ``ScaffoldNetwork``.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import List, Sequence as SequenceLike

from neural_assemblies.reference.nemo_numpy import (
    ScaffoldRecallResult,
    compare_scaffold_vs_simple as _compare_scaffold_vs_simple,
)

from .assembly import Assembly
from .ops import _snap
from .sequence import Sequence

__all__ = [
    "ScaffoldRecallResult",
    "ScaffoldNetwork",
    "compare_scaffold_vs_simple",
    "sequence_memorize_scaffold",
]


def compare_scaffold_vs_simple(
    brain=None,
    stimuli: SequenceLike[str] | None = None,
    *,
    n_presentations: int = 10,
    n: int = 1000,
    k: int = 30,
    beta: float = 0.1,
    rounds_per_step: int = 1,
    seq_len: int = 20,
    seed: int = 42,
    density: float = 0.4,
) -> ScaffoldRecallResult:
    """NEMO demo protocol (numpy port of nemo-demo.ipynb cells 21–27).

    Uses reference RecurrentArea / ScaffoldNetwork dynamics.  ``brain`` and
    ``stimuli`` are accepted for API compatibility but ignored.
    """
    del brain, stimuli, rounds_per_step, beta
    return _compare_scaffold_vs_simple(
        n_presentations=n_presentations,
        n=n,
        k=k,
        seq_len=seq_len,
        seed=seed,
        density=density,
    )


def _train_scaffold_step(
    brain,
    stim: str,
    main_area: str,
    scaffold_area: str,
    *,
    rounds_per_step: int = 10,
    phase_b_ratio: float = 0.5,
    beta_boost: float | None = 0.5,
) -> None:
    """One sequence item with bidirectional main ↔ auxiliary coupling.

    Phase A (``stim_rounds``) drives the main area from the stimulus alone, so
    the new item's identity is set by its input rather than by the recurrent
    attractor the previous item left behind.

    Phase B (``recur_rounds``) runs the coupling.  The five ``brain.project``
    calls per round are deliberately SEQUENTIAL rather than one combined call:
    each is a separate time step, so aux->main sees the main assembly from the
    step before, main->aux then sees the aux assembly just updated, and so on.
    Merging them into a single call would make all four fibers read the same
    snapshot and would change the fixed point.  Order matters and is
    load-bearing:

        1. stim -> main        keep the item clamped
        2. aux  -> main        auxiliary support flows in
        3. main -> aux         auxiliary copy is refreshed from it
        4. main -> main        within-area recurrence
        5. aux  -> aux         within-area recurrence in the scaffold

    ``beta_boost`` raises plasticity on both recurrent loops for Phase B only,
    for the same reason as in ``sequence_memorize``: it deepens the
    item-to-item bridges laid down while the previous assembly is still warm.
    """
    recur_rounds = max(1, int(rounds_per_step * phase_b_ratio))
    stim_rounds = max(1, rounds_per_step - recur_rounds)
    main = brain.areas[main_area]
    aux = brain.areas[scaffold_area]
    for _ in range(stim_rounds):
        brain.project({stim: [main_area]}, {})
    if beta_boost is not None:
        main_beta, aux_beta = main.beta, aux.beta
        brain.update_plasticity(main_area, main_area, beta_boost)
        brain.update_plasticity(scaffold_area, scaffold_area, beta_boost)
    try:
        for _ in range(recur_rounds):
            brain.project({stim: [main_area]}, {})
            brain.project({}, {scaffold_area: [main_area]})
            brain.project({}, {main_area: [scaffold_area]})
            brain.project({}, {main_area: [main_area]})
            brain.project({}, {scaffold_area: [scaffold_area]})
    finally:
        if beta_boost is not None:
            brain.update_plasticity(main_area, main_area, main_beta)
            brain.update_plasticity(scaffold_area, scaffold_area, aux_beta)


class ScaffoldNetwork:
    """Brain-native scaffolded sequence network (main + auxiliary areas).

    Mirrors mdabagia/nemo ``ScaffoldNetwork``: the auxiliary area bidirectionally
    couples to the main sequence area.  For literature parity metrics against
    the reference numpy demo, use ``compare_scaffold_vs_simple``.
    """

    def __init__(
        self,
        brain,
        main_area: str,
        scaffold_area: str,
        *,
        n: int | None = None,
        k: int | None = None,
        beta: float = 0.1,
        prefix: str = "_scaffold",
    ):
        self.brain = brain
        self.main_area = main_area
        self.scaffold_area = scaffold_area
        if main_area not in brain.areas:
            brain.add_area(main_area, n or 10000, k or 100, beta)
        if scaffold_area not in brain.areas:
            spec = brain.areas[main_area]
            brain.add_area(scaffold_area, spec.n, spec.k, spec.beta)

    def train(self, stimulus: str, *, rounds_per_step: int = 10) -> Assembly:
        """Present one sequence item through the scaffolded pathway."""
        _train_scaffold_step(
            self.brain, stimulus, self.main_area, self.scaffold_area,
            rounds_per_step=rounds_per_step,
        )
        return _snap(self.brain, self.main_area)

    def memorize(
        self,
        stimuli: SequenceLike[str],
        *,
        rounds_per_step: int = 10,
        repetitions: int = 1,
    ) -> Sequence:
        """Memorize an ordered stimulus list in the main area."""
        return sequence_memorize_scaffold(
            self.brain,
            stimuli,
            self.main_area,
            self.scaffold_area,
            rounds_per_step=rounds_per_step,
            repetitions=repetitions,
        )


def sequence_memorize_scaffold(
    brain,
    stimuli: SequenceLike[str],
    main_area: str,
    scaffold_area: str,
    *,
    rounds_per_step: int = 10,
    repetitions: int = 1,
    phase_b_ratio: float = 0.5,
    beta_boost: float | None = 0.5,
) -> Sequence:
    """Memorize in *main_area* with auxiliary *scaffold_area* coupling."""
    if isinstance(stimuli, (str, bytes)):
        raise TypeError("stimuli must be an ordered collection of stimulus names")
    try:
        stimuli = list(stimuli)
    except TypeError as exc:
        raise TypeError(
            "stimuli must be an ordered collection of stimulus names"
        ) from exc
    if not stimuli:
        raise ValueError("sequence_memorize_scaffold requires a nonempty sequence")
    if main_area not in brain.areas:
        raise KeyError(f"sequence_memorize_scaffold main area is unknown: {main_area!r}")
    if any(stimulus not in brain.stimuli for stimulus in stimuli):
        unknown = [stimulus for stimulus in stimuli if stimulus not in brain.stimuli]
        raise KeyError(f"sequence_memorize_scaffold stimulus name(s) are unknown: {unknown!r}")
    for label, value in (("rounds_per_step", rounds_per_step), ("repetitions", repetitions)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if (isinstance(phase_b_ratio, bool) or not isinstance(phase_b_ratio, Real)
            or not math.isfinite(float(phase_b_ratio))
            or not 0.0 <= float(phase_b_ratio) <= 1.0):
        raise ValueError("phase_b_ratio must be a finite real number in [0, 1]")
    if beta_boost is not None and (
        isinstance(beta_boost, bool) or not isinstance(beta_boost, Real)
        or not math.isfinite(float(beta_boost)) or float(beta_boost) < 0.0
    ):
        raise ValueError("beta_boost must be a finite nonnegative real number")
    if scaffold_area not in brain.areas:
        m = brain.areas[main_area]
        brain.add_area(scaffold_area, m.n, m.k, m.beta)

    assemblies: List[Assembly] = []
    for _ in range(repetitions):
        assemblies = []
        for stim in stimuli:
            _train_scaffold_step(
                brain, stim, main_area, scaffold_area,
                rounds_per_step=rounds_per_step,
                phase_b_ratio=phase_b_ratio,
                beta_boost=beta_boost,
            )
            assemblies.append(_snap(brain, main_area))
    return Sequence(area=main_area, assemblies=assemblies)
