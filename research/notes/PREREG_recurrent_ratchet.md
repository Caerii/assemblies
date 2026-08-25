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
