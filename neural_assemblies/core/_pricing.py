"""The k-WTA pricing law, in one place, for every backend.

WHAT THIS MODULE IS FOR
-----------------------
``project_into`` chooses winners by ``top-k`` over a vector that concatenates
two populations drawn on *different* scales:

    all_inputs = [ materialized incumbents | sampled candidates ]
                   ^ real summed weights     ^ order statistics of a binomial

Under ``norm_init`` the incumbents are divided per neuron by their in-degree
``d_j`` (see `inverse_indegree`), so the candidates must be divided by a
commensurable quantity (see `candidate_divisor`) or ``top-k`` is comparing two
different units.  Getting that wrong is silent and has a sign:

    candidates OVER-divided   -> no candidate can outbid an incumbent, the area
                                 SEALS at k, and every cap-vs-assembly readout
                                 reads exactly 1.0000 (the perfect-score
                                 signature -- the measurement is dead, not good)
    candidates UNDER-divided  -> candidates always win, the assembly never
                                 stabilizes, and the area materializes toward n

WHY IT LIVES HERE AND NOT IN AN ENGINE
--------------------------------------
It used to live in ``numpy_engine/_sparse.py`` with a hand-written mirror in
``torch_engine/_engine.py`` whose comment read "this is the on-device mirror".
A mirror drifts.  Two fixes (c7ce506, 54e6c00) landed on the numpy copy in
2026-07 and never reached the torch copy, so the two engines priced k-WTA
differently for months.  Measured at k=100, p=0.05, beta=0.05, 15 rounds,
``norm_init=True`` -- engine ``w`` (neurons ever fired) and round-to-round
assembly stability::

    n_src   n_tgt |  numpy (fixed)      |  torch (mirror, pre-unification)
     1000   10000 |  w=154  stab 1.000  |  w=100 SEALED, stab 1.000
     5000    5000 |  w=117  stab 1.000  |  w=114  stab 0.990
    10000    1000 |  w=111  stab 1.000  |  RuntimeError (area exhausted)

Note the first row: the broken engine reads *better* on stability, because a
sealed area is trivially stable.  That is why the divergence survived a parity
suite -- see `neural_assemblies/tests/test_engine_pricing.py`, which asserts the
law directly instead of asserting a threshold a degenerate state also passes.

The storage-specific part (how to count a column's realized in-degree out of a
CSR / dense / 1-D-snapshot connectome) stays in each engine.  Everything that
is *arithmetic* is here, and both engines call it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

__all__ = [
    "candidate_divisor",
    "effective_binomial",
    "inverse_indegree",
    "area_fiber_activity",
]


def candidate_divisor(
    p: Union[float, Sequence[float]],
    tgt_n: int,
    input_sizes: Optional[Sequence[float]] = None,
    src_pops: Optional[Sequence[float]] = None,
) -> float:
    """Scale applied to sampled (unmaterialized) candidate drive.

    Candidates are sampled on the unit-weight scale as top order statistics of
    ``Binomial(sum(input_sizes), p)``.  A materialized neuron's drive is that
    same count divided by its in-degree, so this divisor has to reproduce that
    scale or the two populations ``top-k`` chooses between are not comparable.

    WHY ``n * p`` IS NOT ENOUGH.  A materialized neuron's drive is summed PER
    FIBER, each divided by that fiber's own in-degree (mean ``n_pre * p``)::

        incumbent  ~ sum_f Binomial(a_f, p) / (n_pre_f * p)
                   ~ sum_f a_f / n_pre_f

    while the sampler draws one ``Binomial(sum_f a_f, p)``.  Dividing that by
    ``tgt.n * p`` gives ``(sum_f a_f) / tgt.n``, which agrees only when every
    source population equals the target's ``n``.  When they differ the two
    scales sit a factor ``tgt.n / n_pre`` apart -- the divergence table in the
    module docstring is exactly this, at ratios of 10 in both directions.

    THE FIX.  Choose ``D`` so the sampled total lands on the incumbent scale::

        (sum_f a_f * p) / D  ==  sum_f a_f / n_pre_f
        D = p * (sum_f a_f) / (sum_f a_f / n_pre_f)

    an activity-weighted harmonic mean of the source populations.  When all
    ``n_pre_f == tgt_n`` it collapses to ``tgt_n * p`` exactly, so every
    equal-area configuration is bit-identical to the pre-fix behaviour.  That
    is also why the bug was invisible: the parity suites used a single global
    ``N``, so the heterogeneous case could not arise.

    ``src_pops`` gives the presynaptic population for each entry of
    ``input_sizes``.  For STIMULUS fibers that is the TARGET's ``n``, matching
    `inverse_indegree`'s convention (a stimulus is treated as the active cap of
    an implicit input population of size ``n``).  With either argument omitted
    this falls back to ``tgt_n * p``.

    Candidate in-degree *variance* is deliberately not modelled: an
    unmaterialized neuron has no persistent in-degree yet (it is drawn only when
    the neuron materializes), which is exactly the "no pre-existing hubs"
    property ``norm_init`` exists to enforce.  The variance of the per-fiber
    MIXTURE is handled separately, by `effective_binomial`, which is what the
    sampler's pooled draw needs; this function corrects only the SCALE.

    PER-FIBER ``p``.  ``p`` may be a sequence parallel to ``input_sizes``.  The
    incumbent scale ``sum_f a_f / n_pre_f`` does NOT depend on ``p`` at all --
    dividing a fiber's ``Binomial(a_f, p_f)`` by its in-degree ``n_pre_f * p_f``
    cancels it -- so only the numerator generalizes, from ``p * sum_f a_f`` to
    ``sum_f a_f * p_f``.  With every ``p_f`` equal this is the same expression,
    which is what keeps homogeneous brains bit-identical.
    """
    per_fiber = not isinstance(p, (int, float))
    ps = list(p) if per_fiber else None
    if input_sizes and src_pops and len(input_sizes) == len(src_pops):
        if ps is not None and len(ps) != len(input_sizes):
            raise ValueError(
                f"per-fiber p has {len(ps)} entries for {len(input_sizes)} "
                f"input sizes; they must be parallel")
        total_weighted_p = 0.0
        weighted = 0.0
        for i, (size, pop) in enumerate(zip(input_sizes, src_pops)):
            size = float(size)
            pop = float(pop)
            if size <= 0.0 or pop <= 0.0:
                continue
            total_weighted_p += size * (ps[i] if ps is not None else float(p))
            weighted += size / pop
        if total_weighted_p > 0.0 and weighted > 0.0:
            return max(total_weighted_p / weighted, 1e-12)
    flat = (sum(ps) / len(ps)) if ps else float(p)
    return max(float(tgt_n) * flat, 1e-12)


def effective_binomial(input_sizes: Sequence[float],
                       ps: Sequence[float]) -> Tuple[int, float]:
    """Moment-match ``sum_f Binomial(a_f, p_f)`` to a single ``Binomial(N, P)``.

    The candidate sampler draws one pooled binomial for the whole projection,
    which is exact only when every fiber shares a ``p``.  With per-fiber
    densities the pooled count is a sum of independent binomials with different
    success probabilities -- a Poisson-binomial -- whose first two moments are::

        mu  = sum_f a_f p_f
        var = sum_f a_f p_f (1 - p_f)

    Matching ``N P = mu`` and ``N P (1 - P) = var`` gives ``P = 1 - var/mu`` and
    ``N = mu / P``.  When all ``p_f == p`` this returns ``(sum_f a_f, p)``
    EXACTLY, so the homogeneous path is unchanged rather than approximated --
    that equality is the whole reason this is safe to put on the default path.

    Two moments, not the full distribution: the sampler only needs a location
    and a scale for its truncated-normal tail, and the third moment would not
    survive that approximation anyway.  Returns ``(N, P)`` with ``N`` rounded,
    since a binomial quantile needs an integer trial count.
    """
    sizes = [float(a) for a in input_sizes]
    probs = [float(q) for q in ps]
    # DISPATCH, not arithmetic that happens to agree. Evaluating 1 - var/mu on
    # a homogeneous input returns 0.19999999999999996 for p=0.2, and the
    # sampler's binomial-quantile cache is keyed on that float -- so deriving
    # the answer would change every existing draw by a rounding error. Return
    # the inputs unchanged instead, and homogeneous brains stay bit-identical.
    if not probs or all(q == probs[0] for q in probs):
        return int(round(sum(sizes))), (probs[0] if probs else 0.0)

    mu = sum(a * q for a, q in zip(sizes, probs))
    if mu <= 0.0:
        return 0, 0.0
    var = sum(float(a) * float(q) * (1.0 - float(q))
              for a, q in zip(input_sizes, ps))
    p_eff = 1.0 - var / mu
    # Degenerate only if some p_f == 1 (var 0) or the match leaves the unit
    # interval; fall back to the activity-weighted mean, which is the same
    # location with a slightly wrong scale rather than an invalid draw.
    if not 0.0 < p_eff <= 1.0:
        p_eff = min(max(mu / max(sum(float(a) for a in input_sizes), 1e-12),
                        1e-12), 1.0)
    return int(round(mu / p_eff)), p_eff


def inverse_indegree(deg, n_pre: int, rows_known: int, p: float, xp=None,
                     floor=1.0):
    """``1 / d_j`` for one fiber, given that fiber's realized degree counts.

    Reproduces the reference implementation's ``norm_init``
    (``.reference/mdabagia-nemo/brain.py``: ``FFArea.normalize`` /
    ``RecurrentArea.normalize``, called ONLY from ``reset()`` -- a ONE-TIME
    initialization, not ongoing homeostasis).  There, each fiber's weight matrix
    is divided by its own column sums exactly once at init, when every present
    weight is 1.  The column sum is therefore the postsynaptic neuron's
    IN-DEGREE ``d_j``, and normalization is precisely "initialize every incoming
    synapse of neuron j to ``1/d_j``".

    WHY A READ-TIME SCALE RATHER THAN SCALED WEIGHTS.  These engines materialize
    neurons lazily, so neuron j's full incoming column does not exist when it
    would need to be divided.  But plasticity is purely MULTIPLICATIVE
    (``w *= 1 + beta``), so::

        (w_0 / d_j) * prod_t (1 + beta_t) == (w_0 * prod_t (1 + beta_t)) / d_j

    i.e. dividing a neuron's *summed drive* by a per-neuron constant is
    algebraically identical to having initialized its incoming weights at
    ``1/d_j``.  Storage therefore stays on the unit scale -- which also preserves
    ``w_max``'s "multiples of the initial weight" semantics and keeps the lazily
    sampled candidate drive commensurable -- and the division happens at read
    time.

    ``deg`` must be the count of PRESENT synapses over the rows that exist so
    far, counted and not summed, so the divisor is potentiation-invariant
    (matching the reference's take-it-once-at-init semantics).  Rows that do not
    exist yet are charged at the ambient rate::

        d_j = deg + p * (n_pre - rows_known)

    an unbiased running estimate of the full-population in-degree: each row that
    later materializes contributes a present synapse with probability ``p``,
    exactly the rate the second term assumes.  So ``d_j`` stays centred on
    ``n_pre * p`` while tracking each neuron's own realized excess.

    Sampling ``d_j ~ Binomial(n_pre, p)`` independently of realized wiring was
    tried first and does nothing (measured: winner in-degree z 1.66 -> 1.53,
    overlap 0.940 -> 0.916).  It cannot work -- an independent draw does not
    cancel a neuron's ACTUAL degree advantage, it only adds noise.

    Accepts and returns whatever array type ``deg`` is.  Only the floor is
    backend-specific: pass ``xp`` (the caller's array namespace, e.g. numpy or
    cupy) when there is one, otherwise a torch tensor is detected by its
    ``clamp_min`` method.  The floor is what keeps a neuron whose whole fiber is
    unwired from dividing by ~0 and drowning out every real competitor.
    """
    unknown = max(int(n_pre) - int(rows_known), 0)
    d = deg + float(unknown) * float(p)
    if xp is not None:
        d = xp.maximum(d, xp.float32(floor))
    else:
        clamp_min = getattr(d, "clamp_min", None)
        if clamp_min is None:
            raise TypeError(
                "inverse_indegree needs an `xp` namespace for array types "
                f"without .clamp_min (got {type(deg).__name__})")
        d = clamp_min(float(floor))
    return 1.0 / d


def area_fiber_activity(winners_size: int, k: int, norm_init: bool) -> int:
    """Presynaptic activity ``a_f`` to charge for one AREA fiber.

    Under ``norm_init`` the candidate scale is derived from actual activity, so
    the count must be the number of neurons actually firing, which is not always
    ``k``: a fixed assembly, an explicit source, or a partially-converged area
    can all fire a different number.  Without ``norm_init`` the historical
    behaviour charges the nominal ``k``, and is kept so that runs recorded on
    the unnormalized substrate stay reproducible.

    The two engines disagreed here as well -- torch charged ``k`` unconditionally
    -- which biases the divisor whenever an area's cap is not exactly ``k``.
    """
    return int(winners_size) if norm_init else int(k)
