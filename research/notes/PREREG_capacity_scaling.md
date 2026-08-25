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

---

## Amendment 1: the smoke run found the protocol substitution is NOT equivalent

`--smoke` is an API check whose numbers are void by construction, but two of its
observations are structural rather than statistical and are recorded here
BEFORE any real run, because both change the registered protocol.

### 1. Removing the stimulus removed the ANCHOR, not just the fiber

The registered protocol trains with `project({s: [AREA]}, {AREA: [AREA]})` for T
rounds -- the stimulus fires EVERY round. The substitution above applies the cue
once, as the initial winner set, and lets recurrence run. Those are not the same
protocol: without a persistent input the assembly has nothing to converge
toward. Smoke (4 brains, void numbers, shape only):

    B n=1000 M=4   rank1 0.188   fill 0.767
    B n=2000 M=4   rank1 0.250   fill 0.600
    B n=1000 M=8   rank1 0.125   fill 0.852   pw/chance 4.84

Four assemblies are already barely retrievable and the area is 77-85% churned.
A ceiling measured on this protocol would be a ceiling on a system that does not
form stable assemblies, so **CAP1-CAP4 are not evaluated and no result is
claimed**. This is the anchored regime the project has named before
([[minibatch-training-equivalence]] calls it "the stable stim-anchored regime");
the anchor was doing work the substitution silently dropped.

**The study is BLOCKED on a hash-generated stimulus fiber**, which needs its own
`norm_init` pricing to stay commensurable with the area drive -- adding a raw
stimulus count (~k*p = 30) to an area drive divided by d_j (~0.06) would let the
stimulus decide every winner, which is the documented failure mode of getting
`_pricing` wrong.

### 2. Substrate C's clip guard was too conservative, and is fixed

`batched_project_hashed` refuses to run substrate C when the `w_max` clip could
bind, because column scaling and `min()` do not commute. The first bound used
`tab[-1]`, the deepest value the table can hold; the second used `tab[elapsed]`.
Both are wrong in the same direction: they assume some cell was potentiated on
EVERY round so far, so once `elapsed` passes ~31 at beta=0.10 the table
saturates at `w_max` and the guard trips regardless of the data. Measured, the
actual column scale is 1.10 and stays there:

    t=0  mass min 450.30 max 563.20  scale max 1.1104
    t=7  mass min 452.70 max 543.30  scale max 1.1045

`column_mass` itself is exact -- on an untrained state it reproduces the
in-degree with `max|diff| 0` at W=1 and W=2. The guard needs the ACTUAL maximum
count among scaled columns, which the mass kernel already computes per cell and
could return; that is the fix, and it is not done yet.

### What this costs

Nothing measured is retracted: the substrate parity against `numpy_sparse`
stands (`test_hashed_substrate_parity.py`, all four arms). What is retracted is
the assumption that the capacity protocol could be run without a stimulus
fiber. The registered bars stand unchanged and unevaluated.
