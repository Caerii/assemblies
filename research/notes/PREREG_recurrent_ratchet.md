# PREREG: does per-round mass renormalization lift the RATCHET ceiling?

Registered before running. This is the follow-up
`PREREG_substrate_c_homeostasis.md` promised ("register a follow-up to
re-measure the M-ceiling under C -- the multi-assembly question reopens")
and never ran, now motivated much more sharply by what the theorem-regime
study turned out NOT to test.

## Why the theorem-regime result settles nothing about homeostasis

`438d547`: 80/80 cells perfect, both arms, and TR3 passed through a ceiling
(0 <= 0) because the unbounded control was equally perfect. The reason is now
clear: **the S5 organ has no recurrence.** `train_transition` and `step` issue
only feed-forward projections, `recurrent_projection` is False, so the arc's
self-fiber is never potentiated. Homeostasis exists to make RECURRENCE safe;
it was tested where there is none, so it had no job. That is a statement about
the test, not about homeostasis.

## The specific gap this attacks

[[self-recurrence-stability-window]] records TWO collapse mechanisms:

1. **Degree bias** -- k-WTA elects the random graph's hubs. Fixed by
   `norm_init`, a ONE-TIME initial normalization.
2. **The ratchet** -- past ~32 items in one area the competitor is no longer a
   hub but the ALREADY-POTENTIATED assemblies of earlier items. The recorded
   verdict is: norm_init normalizes INITIAL weights and says nothing about
   learned ones, and no parameter fixes this.

**Substrate C normalizes LEARNED mass.** It is precisely and only the
ratchet's substrate, and it did not exist as a working mechanism when that
verdict was reached -- the tool available then was `norm_init`, whose divisor
we showed this session is potentiation-INVARIANT and therefore inverts the
intended bias after training (e73e493). So "no parameter fixes this" was
concluded without the parameter that addresses it.

Substrate C is now known to work mechanically: `438d547` measured column mass
pinned to the setpoint to four figures, and max stored weight 15.7 against 490
unbounded.

## Design

`numpy_sparse`, n=2000, k=50, beta=0.10, p=0.05, recurrence ON
(`recurrent_projection=True`) -- self-recurrence is the DEFINITION of an
assembly, so this is the configuration the calculus actually describes. n=2000
brackets the ~32 ratchet ceiling cheaply; capacity is extensive
(M_max ~ 1.15 n/k = 46 here), so the ratchet ceiling and the capacity limit
are close enough to tell apart.

Arms (the three substrates):

    NONE   norm_init=False, synaptic_scaling=False
    B      norm_init=True,  synaptic_scaling=False    (initial weights only)
    C      norm_init=False, synaptic_scaling=True     (current mass, per-round)

* **Part A (the M-ceiling):** M in {8, 16, 32, 48, 64} at T=8, seeds 42-44.
* **Part B (the T-window):** T in {5, 8, 12, 20} at M=16, seeds 42-44.
  Merge's recurrent window is T~8 and collapses by T=12
  ([[merge-recall-does-not-hold]]), while the theorems demand T >= 56.6.
  Recurrence wants SHALLOW training and the theorems want DEEP; Part B asks
  whether C widens that window.

## Readouts -- four, and three exist to stop a silent pass

For each stored item i, cue and project, then:

1. **rank-1 identity**: does the retrieved assembly's BEST match over all M
   stored assemblies equal i? Primary readout.
2. **pattern completion**: the same, cued with HALF of assembly i. This is
   the noise-robustness property that DEFINES an assembly, and that a
   feed-forward readout cannot have.
3. **pairwise distinctness**: mean overlap between distinct stored assemblies,
   against the chance floor k/n.
4. **self-overlap**: reported for every cell ONLY as a silent-failure guard.

**Why 4 is not a bar.** The documented failure mode is that recall fails
SILENTLY: at M=128 each item still re-cued to self-overlap 0.68 -- so any
probe reading self-overlap alone reports success -- while rank-1 identity was
0.018 against a chance of 0.008 (the recurrence caution in `brain.py`). Any
cell with self-overlap > 0.6 and rank-1 < 0.2 is FLAGGED in the output as a
reproduction of that trap.

## Bars, stated now

* **RC1 (instrument sanity):** at M=8, T=8, ALL THREE arms reach rank-1
  identity > 0.90. If the instrument cannot see success at trivial load,
  nothing below it is interpretable. *Prediction: PASSES (~85%).*
* **RC2 (reproduce the ratchet):** the NONE arm's rank-1 identity falls below
  0.50 by M=48. *Prediction: PASSES (~80%).* Failing it means the ratchet does
  not reproduce here and the rest of the study is void.
* **RC3 (THE CLAIM):** C's M-ceiling exceeds B's, where ceiling is the largest
  M with mean rank-1 identity >= 0.50. *Prediction: PASSES (~50%) -- genuinely
  open, and the reason this study exists.*
* **RC4 (pattern completion):** at the largest M where C passes RC3, C's
  HALF-cue rank-1 identity exceeds 0.50. A ceiling that holds only for exact
  cues is not an attractor. *Prediction: PASSES (~40%)* -- deliberately
  separate from RC3 so a full-cue win cannot be reported as robustness.
* **RC5 (the window):** at M=16, C's rank-1 identity at T=20 exceeds B's.
  *Prediction: PASSES (~45%).*

## Interpretation, stated now

* RC3 passes: per-round mass renormalization is the ratchet's missing fix and
  the multi-assembly recurrent regime reopens. Register the arc-with-
  recurrence rebuild at organ scale and re-test noise robustness there.
* RC3 fails with C ~ B: the ratchet is not a normalization problem at all, and
  the capacity/robustness trade is structural to this substrate. That is a
  LARGER result than a pass and gets written up as one, not buried.
* RC3 passes but RC4 fails: C buys capacity without buying attractors --
  report as capacity, never as robustness.
* RC2 fails: the ratchet did not reproduce; fix the instrument before reading
  anything else.
* Any bar passing while the claim it operationalizes dies is reported as BOTH.
  That has now happened twice in this line of work (the zipf slope, and TR3),
  so it is the default expectation rather than a caveat.

## Committed in advance

1. Bars before data; this file lands before any cell runs.
2. All four readouts reported for every cell whatever they say; the
   silent-failure flag is printed, never filtered.
3. Per-seed values, never bare means
   ([[report-distributions-not-point-estimates]]).
4. No default changes from this unit regardless of outcome.

---

## Result (2026-08-24, 81 cells)

    RC1 FAIL  M=8 rank1: NONE 0.417, B 1.000, C 1.000
    RC2 PASS  NONE at M=48 rank1 0.021 < 0.50
    RC3 PASS  C ceiling 64 > B ceiling 48   (NONE 0)      <- THE CLAIM
    RC4 FAIL  half-cue rank1 at C's ceiling 0.109 < 0.50
    RC5 FAIL  M=16,T=20: C 0.167 < B 0.188

### RC3 holds, and the effect is not marginal

    arm   M=8    M=16   M=32   M=48   M=64
    NONE  0.417  0.062  0.031  0.021  0.016
    B     1.000  1.000  1.000  0.944  0.104     <- cliff between 48 and 64
    C     1.000  1.000  1.000  0.993  0.896

Per-seed at the decisive cell, M=64: B [0.094, 0.125, 0.094], C [1.000,
1.000, 0.688]. The arms do not overlap. **Per-round mass renormalization
lifts the ratchet ceiling that one-time initial normalization cannot** --
which is what "norm_init normalizes INITIAL weights and says nothing about
learned ones" predicted would be needed, and the parameter that addresses it
did not exist as a working mechanism when that verdict was recorded.

**C's ceiling is UNBRACKETED.** 64 was the top of the sweep and C had not
collapsed there. The true ceiling is >= 64 and unmeasured; a wider sweep is
the immediate follow-up, and no capacity NUMBER should be quoted from this
study, only the ordering.

### Two bars that fail without damaging RC3, and one that matters

**RC1 FAIL is not an instrument failure.** B and C both reach 1.000 at M=8,
so the instrument sees success fine. It is the NONE arm that collapses at
trivial load (0.417, pairwise overlap 0.663 against a chance of 0.025 -- 26x).
That is DEGREE BIAS, the first collapse mechanism, which norm_init exists to
fix ([[recurrence-needs-norm-init]]). The bar was mis-specified: it assumed
all three arms would work at trivial load, and without any normalization
recurrence never works at all.

**RC2 passes through the wrong mechanism.** It was registered to reproduce
the RATCHET, a phenomenon past ~32 items. But NONE is already collapsed at
M=8, so its failure at M=48 is degree bias continuing, not the ratchet. The
ratchet is visible elsewhere in the table -- B's cliff from 0.944 at M=48 to
0.104 at M=64 -- so the phenomenon DID reproduce, just not in the arm the bar
pointed at. Reported as both, per the commitment.

**RC5 FAIL is a real cost, in the opposite direction from RC3.** At M=16:

    arm   T=5    T=8    T=12   T=20
    B     1.000  1.000  1.000  0.188
    C     1.000  1.000  0.458  0.167

C NARROWS the training-depth window relative to B (0.458 vs 1.000 at T=12).
So the two normalizations trade against each other: **C buys capacity in M, B
buys depth in T.** Neither reaches the theorems' T floor (>= 56.6); both are
collapsed by T=20. The tension between merge's shallow recurrent window and
the theorems' depth requirement is unresolved and is now measured on both
substrates.

### RC4 is the one that matters most, and it fails everywhere

Half-cue rank-1 identity, the pattern-completion property that DEFINES an
assembly and is the source of noise robustness:

    C:  0.042 (M=8)  0.083 (16)  0.052 (32)  0.083 (48)  0.109 (64)
    B:  0.167         0.104       0.052       0.062       0.010

Against a chance of 1/M, these are at or barely above chance in every arm at
every load. **No substrate produces attractors.** Full-cue retrieval is
excellent (1.000) while half-cue retrieval is chance -- so what these areas
implement is a stimulus-keyed LOOKUP, not an attractor basin. Removing half
the cue destroys it.

That is the honest answer to "aren't assemblies supposed to be robust to
noise": in this substrate, at these settings, they are not, and the failure
is not about capacity or normalization. It survives every arm that fixes
capacity.

### Interpretation, applied as registered

The rule for "RC3 passes but RC4 fails" was *C buys capacity without buying
attractors -- report as capacity, never as robustness*. That is what
happened and that is how it is reported. The multi-assembly recurrent regime
reopens for CAPACITY; the robustness question is untouched by it and is now
the sharper open problem, since it is the property the calculus is named for.

Follow-ups this earns, in order:
1. Bracket C's ceiling (sweep past M=64) -- cheap, and no number should be
   quoted until it is done.
2. Ask why pattern completion fails at ALL loads including M=8, where
   capacity cannot be the explanation. That is a mechanism question about the
   self-fiber, not a load question.
3. The T-window/theorem-depth conflict, now measured on both substrates.

---

## Amendment (post-data): RC4 IS NOT INTERPRETABLE AS WRITTEN

Two defects in the RC4 reading, found while building the regime warning.

**1. The whole study ran OUT OF REGIME.** n=2000, k=50, p=0.05 gives
`kp = 2.5` against the [[SEQ-REGIME]] floor of `3 ln n = 22.8` -- **9.1x
below**. The design section above asserted the sweep would "bracket the ~32
ratchet ceiling", and it does, but it never checked the connectivity
precondition. `regime_audit`'s own docstring says exactly what this costs: "a
null there is not evidence about the mechanism." RC4 is a null. It was read
as evidence about the substrate.

This is the SECOND time in one session (the S5 organ was 1.06x below and that
cost a whole study). The warning added alongside this amendment exists so it
cannot be the third.

**2. Raising kp into regime did NOT rescue completion, and the number is
below chance.** Sweeping p at M=8, T=8, everything else fixed:

    p     kp    reach    rank1_full   rank1_half
    0.05   2.5  0.7226     1.000        0.125
    0.15   7.5  0.9828     1.000        0.000
    0.30  15.0  0.9999     1.000        0.000
    0.50  25.0  1.0000     1.000        0.000     <- in regime

Reachability goes to 1.0000 and half-cue rank-1 goes to ZERO. Chance is
1/M = 0.125, so 0.000 is BELOW chance, systematically. A clean null sits AT
chance; sitting below it, at every setting, is the signature of an instrument
defect, not of an absent attractor -- the same shape as
[[two-index-spaces-compact-vs-neuron-id]], where a broken comparison read as
exactly chance and was mistaken for a real negative for months.

**So the claim "these areas implement a stimulus-keyed lookup, not an
attractor basin" is WITHDRAWN.** It may well be true; this study cannot
support it. What stands is RC3, which is a comparison BETWEEN arms measured
by the same instrument at the same settings, and is therefore robust to a
constant instrument bias in a way an absolute null is not.

**Before RC4 is re-read, in order:**
1. Validate the half-cue probe against a positive control -- an assembly
   formed and immediately half-cued with no competitors (M=1). If completion
   fails at M=1 the probe is broken, full stop.
2. Check whether `project({}, {AREA: [AREA]})` survives the `a != target`
   filter under `recurrent_projection=True`, since a silently dropped
   self-projection would produce exactly this reading
   ([[silent-no-op-dead-fibers]]).
3. Only then re-run in regime.
