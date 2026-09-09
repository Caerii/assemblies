# An area's size and a fiber's column extent are one invariant

#69, resolving #41 and reframing #47. The three tasks were filed separately and
are one thing: a lazily materialized fiber has columns for the target's neurons,
and nothing in the engine or the API enforced that they agree.

## The ambiguity was silent and load-bearing

Three different numbers describe "how much of this fiber is real", and callers
were choosing between them by hand:

| | meaning | value on one fiber |
|---|---|---|
| `area.w` | neurons the engine materialized -- **except** on the explicit engine, where it is `len(winners) == k` and is not an extent at all | 247 |
| `conn._log_cols` | the fiber's logical column watermark | 213 |
| `weights.shape[1]` | PHYSICAL capacity; growth doubles, so it over-runs both | 367 |

All three at the same instant, on the same fiber, at n=1000. The quantity being
measured through them read **1.87 / 1.65 / 2.84**. Choosing wrong is not a
rounding error, and the API offered no way to make the choice explicit.

This is [[same-name-two-meanings]] for the fourth time, and I walked into it
myself while writing the harness for this note: slicing the dense arm by `w`
kept only the ~k²/n assembly neurons whose global id happened to be below k,
which is why that arm's error bar was 5x the others'.

## What the measurement says

Protocol: train a recurrent assembly on S1, then drive an INDEPENDENT stimulus
S2 and ask how much recurrent drive the STORED assembly collects relative to the
area mean. Ground truth is 1.0 -- an unrelated input has no reason to prefer a
stored assembly. Anything above it is a hair-trigger attractor.

`beta = 0`, so there is no potentiation and any excess is **pure wiring**.
12 seeds, probe under `read_only()`, intervals from `diagnostics.ensemble`
(t-based 95% CI, not a standard error):

    sparse, norm_init=False    1.676 +/- 0.085   [1.591, 1.761]   DISJOINT from truth
    sparse, norm_init=True     1.065 +/- 0.050   [1.015, 1.115]   overlaps truth
    dense (explicit area)      1.058 +/- 0.036   [1.022, 1.094]   ground truth

The defect is `_expansion_col`, which returns `win - prior_w` rather than `win`,
so from the second projection onward every recruit dumps its afferents onto the
OLDEST columns -- exactly the neurons most likely to be in the surviving
assembly. Its docstring already described this. What was not known is that the
`norm_init=True` correction **fully closes it**: 1.065 against 1.058, with
overlapping confidence intervals.

Two things make the reading conservative rather than generous. Beta contributes
almost nothing (1.646 -> 1.831 at beta=0.05 for the broken arm), so this is
structural, not dynamical. And the fixed sparse arm carries 3.5x the
stored/probe overlap of the dense arm (0.180 vs 0.052), which should INFLATE its
ratio; it still lands within noise of ground truth.

**`_expansion_col` remains gated on `norm_init`.** That is not a recommendation
to leave it there -- the docstring's own judgement, "a defect worth fixing
globally on its own, with its own regression sweep", is now backed by a number.
But `norm_init=False` is the LITERATURE-PARITY path ([[norm-init-substrate-vs-reference]]),
so flipping it globally moves every paper reproduction in the repository. That
is a decision to take deliberately, not a side effect of this change.

## #41 is resolved, and its recorded status was stale

The task read "CONFIRMED: read_only() does not roll back connectome
materialization". Measured now, over 6 trials:

    read_only()   d(w)  +0.0   d(extent)  +0.0   connectome BYTE-IDENTICAL 6/6
    frozen()      d(w) +26.2   d(extent) +26.2   connectome changed          0/6

`read_only()` is clean. #36 fixed it and #41 was never re-checked.

The live defect is the other half: **`frozen()` is used as a probe context at 50
sites**, and it stops plasticity but NOT recruitment. A single `frozen()` probe
added **750 nonzeros to the self fiber, every one of value 1.0** -- materialization
wiring, not Hebbian potentiation, arriving while plasticity was off.

It breaks the invariant in two distinct ways depending on whether the fiber is
driven, and both are pinned:

* fiber NOT a source -- the area grows and the fiber is not expanded with it, so
  its last columns are uninitialised zeros. Measured 82 -> 101 materialized with
  the extent stuck at 82: **19 neurons with no recurrent column at all**.
* fiber IS a source -- it expands, writing new synapses under frozen().

## How widespread: run the suite with the instrument on

`NEURAL_ASSEMBLIES_STRICT_PROBES=1` raises when a projection RECRUITS while
plasticity is disabled -- the signature of a probe written against `frozen()`
that meant `read_only()`. Opt-in, same discipline as `_VERIFY_NNZ`: it turns
"which of these 50 `frozen()` sites is contaminating?" from an argument into
something you can run.

Over the fast suite:

    baseline                        1 failed, 1206 passed
    NEURAL_ASSEMBLIES_STRICT_PROBES=1    66 failed, 42 errors, 1104 passed
                                         214 strict-probe raises

It is not a corner case. The hits cluster in exactly the places results come
from: **the whole ERP path** (calibration, probes, integration, frames),
**the emergent parser** (overall accuracy, neural role assignment, incremental),
**next_token**, **checkpoint/fork**, the conversation curriculum, and the TACL
parser suite. Every one of those numbers was read through a probe that grew the
brain while reading it.

**This does not say those results are wrong.** It says the probe is not
isolated, so the reported value is a property of the measurement order as well
as of the model -- and that two probes run in different orders are reading
structurally different brains. #32 already named `frozen()` probe contamination
as half the P600 root cause; this is the same defect, and it is everywhere.

WHY STRICT MODE IS NOT THE DEFAULT, and why the 50 sites were not mass-edited:
switching a site to `read_only()` **changes its numbers**. Suppressing
recruitment changes what the probe can select from -- stored/probe overlap moved
0.038 -> 0.180 on this protocol, because an area that cannot recruit must answer
from the neurons it already has. That is the CORRECT semantics for a readout,
but it is a re-measurement, not a rename. Each site needs its before/after
recorded. Some `frozen()` uses are also legitimate: `word_order_learner` uses it
to mirror the reference's `no_plasticity`, and `next_token` uses it as the
don't-learn arm of an A/B.

## What landed

`ComputeEngine.fiber_extent(source, target)` and `materialized_count(area)`, both
returning `Optional[int]` where **None means NOT APPLICABLE and must not be read
as zero** -- a dense fiber has no watermark to disagree with, which is different
from having no columns. `FiberState.extent_desync` reports the gap and is 0
whenever the question is vacuous, so a caller can sum it over a census without
special-casing engines.

Verdicts stay behind `driven=`, matching the existing census contract. Appending
them unconditionally changed what a plain census RETURNS and broke four callers
doing `[f for f in fiber_census(b) if f.dead]` -- a Verdict has no `.dead`.
That regression is now itself a test.

`test_frozen_probe_holds_the_invariant` is an `xfail(strict=True)`: the defect is
PINNED rather than asserted away. If it starts XPASSing, frozen()'s semantics
changed and every probe written against it needs re-reading.

## Two engine defects found on the way

`Brain(engine="numpy_explicit")` raised `TypeError` on the first `add_area`,
because `Brain.add_area` forwards `winner_policy` and `input_noise_std` to the
primary engine unconditionally and that signature had neither. **The dense
engine -- the ground truth every sampler result is checked against -- was
unreachable through the public API.** It survived because the route people
actually used (an `explicit=True` area inside a sparse brain, a different call
with a different kwarg set) worked fine.

And `refractory_period` / `inhibition_strength` WERE in that signature and went
nowhere: `ExplicitAreaState` has no field for either, so LRI was silently off
while the caller's configuration said it was on. They now raise. Winner policies
are implemented rather than refused, through the shared
`compute.winner_selection` path so the engines cannot come to disagree about
what a policy means ([[pricing-law-implemented-twice]]).

## An observation, not a result

E%-WTA on exact dense drive does **not** converge under self-recurrence at
n=300: over 14 rounds it wanders and trends down, seed 5 69->19, seed 6 75->11,
seed 7 128->61. `numpy_exact` at n=2000/k=50 was recorded flat at 184
([[epercent_wta_was_measuring_recruitment]]). Different scale and different
norm_init, so the two are not in contradiction -- but neither is "E%-WTA settles"
a settled fact, and the test deliberately does not assert it.
