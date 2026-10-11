"""The numpy engine's semantic switches, read from the environment each time they are
asked (so an A/B can flip one between runs on the same seed), and the prune's threshold.

    _fixed_target_plasticity_enabled  a projection INTO a fixed area still trains its
                                      afferents (the reference semantics; one owner, read
                                      by both engines)
    _warn_fixed_target_enabled        warn on the old silent no-op instead
    _explicit_src_norm_enabled        norm_init on an explicit (dense) source
    _strict_drive_enabled             a zero-signal round raises instead of keeping the assembly
    _PRUNE_MAX_FRACTION               the share of columns above which k-WTA pruning declines

Moved from _sparse.py unchanged, which re-exports them."""
import os

#: Above this share of the materialised columns the prune saves
#: nothing and still pays for its own decision, so it declines.
#: Not a tuning knob for accuracy -- the answer is exact at any
#: value; it only decides when the fast path is worth taking.
_PRUNE_MAX_FRACTION = 0.5


def _warn_fixed_target_enabled() -> bool:
    """Whether to warn on a plasticity-bearing projection INTO a fixed area.

    Off by default (behaviour-neutral); set ``ASSEMBLIES_WARN_FIXED_TARGET=1``
    to surface the silent-no-op footgun during development. See the check at
    the fixed-assembly short-circuit in ``project_into``.
    """
    return os.environ.get("ASSEMBLIES_WARN_FIXED_TARGET", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _explicit_src_norm_enabled() -> bool:
    """Whether an EXPLICIT source area's drive is norm_init-scaled like any other.

    ON by default, because leaving it off silently defeats ``norm_init`` for
    every fiber whose presynaptic area is explicit.

    THE DEFECT.  ``project_into`` accumulates drive from three kinds of source.
    The stimulus path and the general area path both multiply their contribution
    by ``_norm_scale`` (the per-neuron ``1/d_j``).  The explicit-dense path did
    not.  Meanwhile sampled candidates for unmaterialized neurons are ALWAYS
    divided by ``_norm_candidate_divisor`` (``n * p``).  So under ``norm_init``
    the two populations that top-k chooses between were on scales a factor of
    ``n * p`` apart::

        incumbent (explicit source) ~ Binomial(a, p)              # raw counts
        candidate                   ~ Binomial(a, p) / (n * p)    # normalized

    Measured at the [COLT22] parameters (n=1e3, k=1e2, p=0.1, a=165 active):
    incumbent drive min 23.10 / median 25.30 against a candidate max of 0.43 --
    a factor of ~100, exactly ``n * p``.

    TWO CONSEQUENCES, both silent.

    1. **The target seals.**  Once it has materialized ``k`` neurons, no
       candidate can ever outbid an incumbent again, whatever the input.  The
       area's cap becomes constant, so every readout that compares a cap against
       a stored assembly reads exactly 1.0000 -- the perfect-score signature.
    2. **Fiber weighting is corrupted even with no recruitment.**  An area
       driven by BOTH an explicit and a non-explicit source weights the explicit
       one ``n_pre * p`` times too heavily, because only the other fiber is
       divided by its own in-degree.  Normalized, a fiber contributes about
       ``k / n_pre``; unnormalized it contributes about ``k * p`` regardless of
       ``n_pre``, which erases the reference's deliberate geometry.

    The first assembly is affected too: ``_bootstrap_from_explicit_dense``
    selected the initial cap by RAW in-degree, which is precisely the
    "pre-existing hubs" that ``norm_init`` exists to eliminate.

    Set ``ASSEMBLIES_EXPLICIT_SRC_NORM=0`` to restore the unnormalized
    behaviour, which is needed to reproduce results recorded before this fix.

    THE TORCH ENGINE HAD THE SAME DEFECT and did not inherit this fix, because
    it carried a hand-written copy of the pricing math rather than calling a
    shared one.  Both engines now go through ``core/_pricing.py``; the flag
    above is numpy-only and covers only reproduction of pre-fix runs.
    """
    return os.environ.get(
        "ASSEMBLIES_EXPLICIT_SRC_NORM", "1",
    ).strip().lower() in ("1", "true", "yes", "on")


def _fixed_target_plasticity_enabled() -> bool:
    """Whether a projection INTO a fixed area still potentiates its afferents.

    ON by default, because this is what the reference implementation does and
    the divergence was silently breaking its central idiom.

    [ACREF] (``.reference/dmitropolsky-assemblies/brain.py``) handles a fixed
    target by
    pinning the winners and skipping recruitment::

        if target_area.fixed_assembly:
          target_area._new_winners = target_area.winners
          target_area._new_w = target_area.w
          num_first_winners_processed = 0

    and then FALLS THROUGH to the plasticity section, which sits outside that
    branch, so ``from_area -> fixed_target`` synapses are still multiplied by
    ``(1 + beta)`` onto the frozen winners. Holding an area fixed means "do not
    let the winners move", not "do not learn".

    That is the whole mechanism behind the reference's reciprocal idiom --
    ``parser.py``'s "reciprocal until stable, LEX frozen", and
    ``simulations.fixed_assembly_recip_proj``, which freezes A and runs
    ``{"A": ["B"], "B": ["A", "B"]}`` so that B->A is written against a
    stationary A and can later restore it. We short-circuited before plasticity,
    so the back-fiber was never written and restoration was impossible.

    MEASURED against the reference at its own defaults (n=1e5, k=317, p=0.01,
    beta=0.05): first B->A restores 0.246 of A, rising to 0.344 -- which matches
    the expectation recorded in that function's header comment ("first B->A gets
    only 25% ... restore up to 42%"). At the parameters
    ``tests/test_assembly_calculus.py`` uses it reaches 0.75, so that file's
    ``> 0.6`` assertion was calibrated correctly all along; what it was testing
    was broken.

    Set ``ASSEMBLIES_FIXED_TARGET_PLASTICITY=0`` to restore the short-circuit,
    kept so the two can be A/B'd on one seed. Note that the old behaviour makes
    such a projection a silent no-op, which is what
    ``ASSEMBLIES_WARN_FIXED_TARGET`` exists to surface.
    """
    return os.environ.get(
        "ASSEMBLIES_FIXED_TARGET_PLASTICITY", "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _strict_drive_enabled() -> bool:
    """Whether to warn when a projection delivers NO drive to its target.

    Off by default (behaviour-neutral); set ``ASSEMBLIES_STRICT_DRIVE=1``, which
    the test suite and the research scripts should do.

    WHY THIS EXISTS. The recurring failure mode in this project is not a wrong
    number, it is a mechanism that silently does not fire, because a projection
    with no drive still returns k winners and looks like it worked. Three
    instances are on record and each cost days:

      * `reset_area_connections` zeroing a connectome, so every candidate had
        equal input and the deterministic index tie-break returned the SAME k
        winners for every item -- bit-identical stored assemblies, retrieval at
        exactly chance, invariant to every parameter.
      * a second source area into an already-grown target whose weight block
        was never sized, so it delivered zero and the target silently kept its
        previous assembly. Whichever source projected first was the only one
        that ever worked.
      * projection INTO a fixed area, whose inputs are discarded (that one has
        its own switch above).

    All three have the same signature in results -- "mechanism X turns out to
    have surprisingly little effect" -- which is indistinguishable from a real
    negative result by looking at the numbers. This makes the engine say so.
    """
    return os.environ.get("ASSEMBLIES_STRICT_DRIVE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
