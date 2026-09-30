"""Transient attentional binding (mdabagia/nemo ``AttentionArea`` port).

THE PROBLEM.  Every plasticity rule in this repository is PERMANENT and
COMPOUNDING: ``w *= 1 + beta`` on each co-firing, clipped at ``w_max``, with no
way to retract.  That is right for memory and wrong for attention, which needs
a link that exists while it is being used and is gone afterwards -- otherwise
attending to a pair is indistinguishable from learning it.

THE MECHANISM.  The reference's ``AttentionArea`` writes a link that is both
SATURATING and REVERSIBLE:

    update:        change[pre, post] = plasticity * w[pre, post]
                   w[pre, post]      = (1 + plasticity) * (w[pre, post] > 0)
    decay_weights: w -= change;  change = 0

Two things follow, and both differ from everything else here.

  SET, NOT MULTIPLY.  Potentiation ASSIGNS ``1 + plasticity`` to any positive
  weight instead of scaling it, so a second potentiation of the same pair is a
  no-op.  Weights are effectively binary, ``0`` or ``1 + plasticity``, and no
  clip is needed because nothing compounds.  ``FFArea.update`` -- the path
  ``FSMNetwork`` and therefore every sequence result here rests on -- uses the
  MULTIPLY rule, so this is specific to attention and not a parity problem
  elsewhere (research/literature/CONFORMANCE.md).

  EXACTLY ONE UNDO.  ``change`` is ASSIGNED, not accumulated, so
  ``decay_weights`` reverses the most recent update only.  From baseline it is
  exact; potentiate a pair twice without releasing and the rollback
  over-subtracts, leaving ``1 - plasticity**2`` where ``1`` was.  The
  construction is therefore built for a bind / read / release cycle, and this
  port REFUSES a second bind rather than silently corrupting the baseline --
  the reference does not, and that is recorded rather than hidden.

Provides:
  - ``AttentionArea`` -- the reference construction, faithfully
  - ``attend`` -- one bind / read / release cycle as a single call

Reference:
    Dabagia, Papadimitriou, Vempala. "Computation with Sequences of Assemblies
    in a Model of the Brain." Neural Computation (2025). arXiv:2306.03812.
    Upstream implementation: mdabagia/nemo ``AttentionArea``.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


class AttentionArea:
    """A recurrent area whose links are saturating and reversible.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-attention-area
    """

    def __init__(self, n_neurons: int, cap_size: int, density: float,
                 plasticity: float, *, rng: np.random.Generator | None = None):
        if isinstance(n_neurons, bool) or not isinstance(n_neurons, Integral) or n_neurons <= 0:
            raise ValueError("n_neurons must be a positive integer")
        if isinstance(cap_size, bool) or not isinstance(cap_size, Integral) or not 0 < cap_size <= n_neurons:
            raise ValueError("cap_size must be a positive integer at most n_neurons")
        for name, value in (("density", density), ("plasticity", plasticity)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real number")
        if not 0.0 < float(density) <= 1.0:
            raise ValueError("density must lie in (0, 1]")
        if float(plasticity) < 0.0:
            raise ValueError("plasticity must be nonnegative")
        self.n_neurons = int(n_neurons)
        self.cap_size = int(cap_size)
        self.density = float(density)
        self.plasticity = float(plasticity)
        gen = np.random.default_rng() if rng is None else rng
        #: the reference draws a Bernoulli connectome and never adds synapses
        self.recurrent_weights = (
            gen.random((self.n_neurons, self.n_neurons)) < self.density).astype(np.float64)
        self.recurrent_change = np.zeros_like(self.recurrent_weights)
        self.activations: np.ndarray = np.empty(0, dtype=np.int64)
        self._bound = False

    # -- the reference's own operations ------------------------------------
    def fire(self, activations) -> None:
        """Set the presynaptic set the next bind will read from."""
        idx = np.asarray(activations, dtype=np.int64).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_neurons):
            raise ValueError("activation index outside the area")
        self.activations = idx

    def update(self, new_activations) -> None:
        """The reference's `update`: recurrent only, SET rule, one-step undo.

        Note the reference's own comment -- "only recurrent update" -- so the
        feedforward weights of an attention area never learn.
        """
        new = np.asarray(new_activations, dtype=np.int64).ravel()
        if new.size and (new.min() < 0 or new.max() >= self.n_neurons):
            raise ValueError("activation index outside the area")
        if self._bound:
            raise RuntimeError(
                "this area already holds an unreleased bind. `decay_weights` "
                "reverses ONE update, so binding twice would leave the "
                "baseline corrupted (1 - plasticity**2 where 1 was) rather "
                "than restored. Release first.")
        ix = np.ix_(self.activations, new)
        self.recurrent_change[ix] = self.plasticity * self.recurrent_weights[ix]
        self.recurrent_weights[ix] = (
            (1.0 + self.plasticity) * (self.recurrent_weights[ix] > 0))
        self._bound = True

    def decay_weights(self) -> None:
        """Reverse the most recent update, exactly."""
        self.recurrent_weights -= self.recurrent_change
        self.recurrent_change = np.zeros_like(self.recurrent_weights)
        self._bound = False

    def recurrent_input(self) -> np.ndarray:
        """`RecurrentArea.get_total_input`'s recurrent term: the column sum
        over the currently active rows."""
        if self.activations.size == 0:
            return np.zeros(self.n_neurons, dtype=self.recurrent_weights.dtype)
        return self.recurrent_weights[self.activations].sum(axis=0)

    def select(self) -> np.ndarray:
        """Top `cap_size` by recurrent input, ties to the lower index."""
        drive = self.recurrent_input()
        order = np.lexsort((np.arange(self.n_neurons), -drive))
        return np.sort(order[:self.cap_size])


def attend(area: AttentionArea, query, key) -> np.ndarray:
    """One bind / read / release cycle, which is the protocol the one-step
    undo is built for: link `query` to `key`, read what the link drives, then
    restore the area exactly.
    """
    area.fire(query)
    area.update(key)
    try:
        return area.select()
    finally:
        area.decay_weights()
