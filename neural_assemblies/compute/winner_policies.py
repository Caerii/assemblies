"""
Winner policy definitions for neural assembly selection.

These policy objects provide a stable API for experimenting with
different competition rules without forcing engine code to hard-code
one inhibition mechanism. The existing engines still default to fixed
top-k selection; this module establishes the contract for future
pluggable winner policies such as thresholded or E%-style rules.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TopKPolicy:
    """Select exactly ``k`` highest-scoring winners."""

    k: int
    tie_policy: str = "value_then_index"


@dataclass(frozen=True)
class ThresholdPolicy:
    """Select winners above an absolute threshold, capped at ``k``."""

    k: int
    threshold: float
    tie_policy: str = "value_then_index"


@dataclass(frozen=True)
class RelativeThresholdPolicy:
    """Select winners whose score is a fraction of the max input.

    This is a lightweight foundation for variable-sized competition:
    the winner count depends on the input distribution rather than a
    fixed ``k`` alone. ``min_winners`` prevents empty selections when
    at least one candidate exists, and ``max_winners`` can cap growth.
    """

    fraction_of_max: float
    min_winners: int = 1
    max_winners: int | None = None
    tie_policy: str = "value_then_index"


@dataclass(frozen=True)
class EPercentPolicy:
    """E%-WTA competition (Hoff et al. 2026, Eq. 4-6).

    Models the gamma-cycle selection window: the most-stimulated neuron fires
    first and recruits interneurons, so only neurons within ``epsilon`` of the
    peak fire before inhibition arrives::

        F_t = { j : h_j in [(1 - eps) * h_max, h_max] },  eps = d / tau_m

    with ``d`` the inhibition delay and ``tau_m`` the membrane time constant
    (3 ms and 30 ms in the paper, giving eps = 0.1).

    Assembly size is EMERGENT -- that is the point of the mechanism, so
    ``e_fraction`` defaults to uncapped. Capping it at a small fraction
    reimposes a fixed size and defeats the purpose.

    ``window`` selects how the firing window is measured:

    * ``"epsilon"`` -- the paper's rule, a fixed fraction of ``h_max``.
    * ``"sigma"``   -- ``h_j >= h_max - sigma_c * std(h)``, the window measured
      in spreads of the input distribution.

    The second exists because the paper's rule is sensitive to the SHAPE of the
    input distribution, not just its scale. With ``h ~ mu + sigma*z`` the lower
    edge sits at ``z_low = (1-eps) z_max - eps * mu/sigma``, and since
    ``mu/sigma ~ sqrt(k_s p_s)``, sparse connectivity or small stimuli drive
    ``z_low -> (1-eps) z_max``: the window collapses onto a couple of neurons
    and assemblies fail to form. That is the mechanism behind the paper's own
    Fig. 4a-b limits, which it reports as making the original AC operations
    "unfeasible" under E%-WTA. Measuring in units of sigma cancels the
    ``mu/sigma`` term and restores scale- and shape-invariance -- divisive
    normalization by population activity, in Carandini-Heeger terms.

    Measured formation success (dense reference model, beta=0.01)::

        regime                 epsilon    sigma
        p_s=.5  k_s=200          0.93      0.95   <- the paper's own regime
        p_s=.1  k_s=200          0.35      0.97
        p_s=.05 k_s=60           0.03      0.75
    """

    fraction_of_max: float = 0.9   # 1 - eps, eps = d/tau_m = 3/30
    e_fraction: float = 1.0        # uncapped: size must emerge
    min_winners: int = 1
    tie_policy: str = "value_then_index"
    window: str = "epsilon"        # "epsilon" (paper) | "sigma"
    sigma_c: float = 1.7

    @classmethod
    def from_gamma(cls, d_ms: float = 3.0, tau_m_ms: float = 30.0, **kw):
        """Build from the biophysical constants, eps = d / tau_m."""
        eps = d_ms / tau_m_ms
        return cls(fraction_of_max=1.0 - eps, **kw)

    @property
    def epsilon(self) -> float:
        return 1.0 - self.fraction_of_max


WinnerPolicy = TopKPolicy | ThresholdPolicy | RelativeThresholdPolicy | EPercentPolicy
