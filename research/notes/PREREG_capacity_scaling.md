# PREREG: how does the M-ceiling scale with n?

Registered before running. Bars stated below; the data does not exist yet.

## Why this, now

`PREREG_substrate_ceiling.md` closed with an explicit "What is NOT established":

> Both ceilings sit at or past the tiling limit of an n=2000 area (rows/n
> 0.97-1.00). This is a capacity result for THIS area size. ... What is NOT
> established: how any of this scales with n.

The blocker was compute: the question needs many M, many seeds, and n an order
of magnitude larger, all with TRAINING. `batched_project_hashed` now runs 16-64
independent brains with a generated connectome at ~0.02 ms per brain per round,
verified against `numpy_sparse` to float32 precision on all four substrate arms
(`test_hashed_substrate_parity.py`).

This project has RETRACTED a capacity-scaling claim TWICE -- `n^1.49` withdrawn
([[critical-load-alpha-star]]) and `n^0.29` retracted as a hub artifact
([[core-areas-near-tiling-limit]]). The guards below exist because of that
history, not as ceremony.

## Protocol, and how it DIFFERS from the registered ceiling study

One recurrent area. M assemblies stored sequentially in the SAME area, so they
compete. Assembly a: set the winners to a fixed random k-subset (the cue), run
T rounds of recurrence with plasticity, store the resulting winner set.

**The difference, stated up front.** `PREREG_substrate_ceiling.md` drove each
assembly with a STIMULUS whose own fiber also learns. This path has no stimulus
fiber, so the cue is applied by fixing the initial winners. **Absolute ceilings
are therefore NOT comparable to that study's M* = 41 (B) / 104 (G) at n=2000.**
The deliverable is the SCALING with n, measured within one protocol, with
n=2000 included as an internal anchor.

Readout, identical to the registered study: half-cue rank-1 retrieval. Take
k/2 neurons of stored assembly a, run T rounds of recurrence ALONE (frozen --
no plasticity, the `brain.probe()` equivalent), and score a hit when the
recalled set overlaps stored[a] more than any other stored assembly. Gated by
distinctness exactly as `_gated_half` does: a cell whose assemblies are not
distinct has not stored M things and contributes 0.0.

    HALF_BAR      0.50    completion required to call it an attractor
    DISTINCT_GATE 3.0     x chance; pairwise above this is not a stored set

## Grid

    k = 60, p = 0.50  ->  kp = 30
    n in {1000, 2000, 4000, 8000}   floors 3 ln n = 20.7 / 22.8 / 24.9 / 26.9
                                    ALL CLEARED by kp = 30
    T = 8, beta = 0.10 (the value all three arms chose in the ceiling study)
    M in {8, 16, 32, 64, 128, 256}
    arms: B (norm_init) primary, G (norm_init + synaptic_scaling) secondary
    16 independent brains per cell, each its own seed, run as one batch

Recall samples min(32, M) stored assemblies per checkpoint, so every cell has
at least 32 x 16 = 512 trials, clearing the harness's MIN_TRIALS = 96.

## Bars, stated now

**CAP1.** At each n, the curve must have an interior point (an accuracy strictly
between 0.10 and 0.90) for `ceiling_from_curve` to call `.supported`. A cell
without one is reported as UNSUPPORTED and its `m_star` is not quoted.

**CAP2 (THE CLAIM).** Capacity is EXTENSIVE: `M*(n)` grows linearly in n at
fixed k. Operationalised as a fit of `log M* = a + b log n`, with

    b's 95% CI contains 1      -> extensive, SUPPORTED
    b's 95% CI excludes 1      -> NOT extensive; the sign says which way

**CAP3.** The censoring guard, which is the whole reason the n=2000 result could
not be scaled. At each n report `rows/n`, the fraction of the area's neurons
that ever fired, AT the ceiling. If `rows/n >= 0.95` the ceiling is bounded by
the TILING LIMIT of the area, not by the substrate, and that n is reported as
CENSORED and EXCLUDED from the CAP2 fit. A fit run over censored points would
recover the tiling limit's slope (exactly 1) and call it extensivity -- which is
how a capacity law gets retracted a third time.

**CAP4.** Per-seed distributions, never bare means ([[ensemble-not-realization]]).
Ceilings come from the CURVE via `ceiling_from_curve`, never from a grid point.

## Interpretation, stated now

* **CAP2 supported and CAP3 clean at >= 3 uncensored n**: capacity is extensive
  over the measured range; `M* ~ alpha n/k` and alpha is quotable.
* **CAP2 excluded, b < 1**: capacity is SUBLINEAR -- the finding, and the wall
  a large assembly model has to design around.
* **CAP2 excluded, b > 1**: superlinear. Given this project's history, treat as
  an artifact until a mechanism is identified; do NOT quote an exponent.
* **Fewer than 3 uncensored n**: the question is not answered at this k and p.
  Report that, do not fit a line through two points.

## Committed in advance

The `b` fit is reported with its CI whatever it says. If CAP2 fails, that is
the result. If most n are censored, the study reports its own inconclusiveness
rather than fitting the surviving points -- the failure mode that produced the
`n^0.29` retraction.

Protocol difference from the registered ceiling study is restated in the RESULT
section, so the two sets of numbers cannot be read as comparable later.
