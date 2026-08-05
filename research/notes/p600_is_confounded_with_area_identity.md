# The P600 effect is confounded with area identity, and the control is already in the data

**Status:** confound measured and understood; the FIX is built but NOT adopted,
because the suite that should adjudicate it leaks state across tests. See
"Outcome" and "Why it is not adopted". 2026-08-05.
**Bears on:** #108, #104, #23, #27, #32, and every P600 magnitude in the repo.

## The claim

`p600_auc ~= 1.0` is the headline ERP result. It cannot be attributed to the
grammatical/violation contrast, because **area identity alone reproduces the
same AUC with the condition held constant**.

This does not show the condition effect is zero. It shows the shipped number
does not isolate it.

## Why the arms differ

Three lines, two files:

    runner.py:108     cat, ... = parser._advance_incremental_word(word, ...)
    adapters.py:381   role_area = structural_role_area(cat, verb_seen=verb_seen)
    adapters.py:295   VERB -> VP ;  NOUN/PRON -> ROLE_PATIENT (if verb_seen)

The probed area is dispatched on the **observed word's category**. But a
category violation *is* a word whose observed category differs from the
expected one. So the violation arm probes a different area from its grammatical
control **by construction** -- in every frame set, at every seed, under every
definition of energy.

Measured (`research/experiments/erp_which_area_per_arm.py`), both sets:

    grammatical         ROLE_PATIENT x3
    category_violation  VP x3
    novel_noun          ROLE_PATIENT x3

    grammatical vs violation : MISMATCHED -- {ROLE_PATIENT} vs {VP}
    grammatical vs novel     : MATCHED

## The control

Take **only grammatical sentences**. No violation anywhere. Group their words
by the area each happened to probe:

    grammatical, VERB -> VP            .9925  .9926  .9919
    grammatical, NOUN -> ROLE_PATIENT  .9880  .9918  .9881     AUC = 1.000

Compare with the reported effect:

    category_violation (VP)            .9934  .9938  .9943
    grammatical        (ROLE_PATIENT)  .9880  .9918  .9881     AUC = 1.000

Same AUC, no manipulation. `phrase_stability` makes it starker -- it is
*perfectly* determined by area and never by condition:

    every ROLE_PATIENT probe -> 0.0049      (all conditions)
    every VP probe           -> 0.0000      (all conditions)

## This explains the two rejected candidates

- The shipped self-recurrent probe reads a constant 1.0 for VP because
  `VP -> VP` is shape (0,0) -- VP has no self-fiber (#108).
- `afferent_energy` reads AUC exactly 0.000 on four independent seeds with
  **zero variance**. Zero variance across seeds is the signature of a constant
  structural difference, not a sentence effect (`erp_afferent_vs_recurrent.log`).

Both are the same fact in different clothes: a quantity that differs between
the two AREAS dominates whatever differs between the two CONDITIONS.

It also dissolves the "VP has no self-fiber" puzzle. VP was never the expected
object slot, so nothing ever trained it as one. That is a consequence of the
architecture, not a defect to repair -- which is why `_pregrow_phrase_pathways`
built the fiber, passed every targeted test, and still inverted seed 42 (97438ec,
reverted in 5c1ca77).

## #23 is not a regression

`AREA_MATCHED_CALIBRATION_FRAMES` (frames.py) area-matches by putting every
critical word in object position: `cat`, `dog`, `bird` are all NOUNs, all ->
ROLE_PATIENT. That genuinely fixes the **novel-noun** arm, which is #27's scope.
It cannot touch the grammatical/violation arm, because `finds`/`runs`/`eats` are
VERBs and dispatch to VP wherever you put them. **#23 fixed the arm it was
scoped to; the other arm was never coverable by frame design.**

## What survives

- **Rank/ordering results over the live (ROLE_PATIENT) arm** are unaffected --
  they never depended on the violation arm carrying signal.
- **Every P600 magnitude** is confounded and should not be quoted as an effect
  size.
- The novel-noun contrast (`grammatical vs novel_noun`) IS area-matched and is
  the one ERP contrast currently entitled to a magnitude.

## The fix this points to

Probe the area the parse **expects**, not the area the observed word implies.
After a transitive verb the slot is ROLE_PATIENT whether the next word is `cat`
or `finds`. Both arms then read the same area and the contrast becomes: *did the
word deliver drive into the slot the grammar predicted?*

Two reasons this is the right shape rather than another metric swap:

1. The predicate already exists -- `_verb_takes_an_object` / `transitive_verbs`,
   built for the #24 empty-project detector. This unifies with it instead of
   adding a parallel mechanism.
2. It is what P600 *is* in the ERP literature: a prediction-violation response.
   Dispatching on the observed word measures the word; dispatching on the
   expectation measures the violation.

## Outcome: the P600 is REAL, and was inflated by ~0.19 AUC

`erp_expected_slot_ab.py`, 10 seeds (11/12/13/42/7/19/23/31/37/101), paired per
seed via `diagnostics.ensemble` / `paired_delta`:

    metric     obs [shipped]         exp [adopted]         delta
    p600_auc   0.9056 +/- 0.0268     0.7167 +/- 0.0805     -0.1889 +/- 0.0627   CHANGED
    p600_span  0.0080 +/- 0.0006     0.0064 +/- 0.0008     -0.0017 +/- 0.0004   CHANGED
    n400_auc   0.9111 +/- 0.0902     0.9111 +/- 0.0902      IDENTICAL

    per-seed exp: 1.000, 0.667 x8, 0.833   none below chance, not constant

THE DROP IS THE RESULT. Area-matched, the effect survives at 0.7167 with CI
0.636-0.797 -- above the 0.5 null. So the contrast is genuine and the shipped
0.9056 was inflated by the confound, by about 0.19 AUC.

`n400_auc` identical is CORRECT, not a plumbing failure: N400 comes from
`measure_lexical_surprise`, which this dispatch does not touch.

Pre-registered before the run, all three met: arms area-match; every seed above
chance (42 included -- it is the seed that inverted the last structural change);
violation arm no longer constant.

GRANULARITY CAVEAT. 3 grammatical x 3 violation = 9 pairs, so AUC moves in steps
of 1/9 and 8 of 10 seeds read exactly 6/9. The estimate is coarse by
construction. Widen the frame set before reading finer differences into it.

## Why it is not adopted: the ERP suite leaks state across tests

Flipping the default failed 4 ERP tests. That looked like the 97438ec pattern
(targeted runs green, full suite inverted) and I rolled back. But the failures
DO NOT REPRODUCE IN ISOLATION -- all with `ERP_EXPECTED_SLOT=1`:

    pytest test_erp_metric_range.py                       3 passed, 1 xfailed
    pytest test_erp_calibration.py                        3 passed, 1 xpassed
    pytest test_erp_calibration.py test_erp_metric_range.py
                                                          6 passed, 1 xfailed
    pytest -k erp   (full selection)                      4 FAILED

And a cold single-arm process (`erp_cold_vs_warm_arm.py`) reads seed 11 exp raw
p600 AUC **1.000** on gram [0.988, 0.9923, 0.9881] vs catv [0.9927, 0.993,
0.9927] -- where the in-suite failure read gram [0.9932, 0.9928, 0.9945] vs catv
[0.9927, 0.9925, 0.9924]. A different parser state entirely.

TWO EXPLANATIONS RULED OUT, recorded so they are not re-derived:

- **raw vs excess.** `separation["p600_auc"]` ranks p600_excess and the test
  ranks raw p600, but `p600_excess(v) = max(0, v - p600_median)` is MONOTONE:
  clipping can only create ties (AUC -> 0.5), never invert an ordering. So this
  cannot produce 0.000 vs 1.000.
- **warm-up order in my own A/B harness.** `erp_expected_slot_ab.py` runs obs
  first and exp second in one process, and a cold process's first probes read
  ~0.43 where warm reads ~0.998 -- a real effect, and a real flaw in the
  harness. But it is not this one: the COLD exp arm reads 1.000, not 0.000.

So the blocker is CROSS-TEST STATE LEAKAGE in the ERP suite (#102/#36 family).
Until it is fixed the suite cannot adjudicate this change, and **any suite-level
A/B in this area is unreliable**. The default stays OFF because a change should
not be adopted on contested evidence -- not because the dispatch is known bad.
Three independent measurements (in-process A/B, cold single-arm, isolated test)
agree it is fine on seed 11; only the contaminated full-selection run disagrees.

## A tidy hypothesis that was WRONG, recorded so it is not re-derived

`test_fast_calibration_preserves_the_ordering` XPASSed after the `hits` fix, and
`fast` calibration consumes SWEEP_CALIBRATION_FRAMES, which contained `hits`. A
dead item reads p600 0.0000 -- the minimum -- so including it in the violation
arm drags that arm BELOW grammatical, which is precisely the inversion the xfail
documents. Coherent, and false: the test XPASSes with `hits` restored too. The
xfail is simply unstable, as its own message says (0.889, 0.889, 0.722, 0.444
across trainings). No claim about #80/#104 follows from it.

## Third thing, found in passing: the first frame of a process reads differently

The first two probes of a cold process read `p600 ~0.43` with
`phrase_stability 1.0000` -- the latter is the empty-`phrase_areas` fallback, so
nothing had an active assembly yet. The same frame read warm later in the same
process gives `0.9983 / 0.0000`.

Verified **pre-existing**: identical values under the shipped dispatch and under
the expected-slot flag (`0.4343`, `0.4454`, critical `0.9907`). Not introduced
by either change. It is the same family as #102 (parsing is not idempotent) and
means probe ORDER moves magnitudes, which is another reason not to quote them.

Incidentally this is also the check that the expected-slot dispatch does what it
claims: the **grammatical arm is byte-identical** across both dispatches, because
a noun after a verb already probed ROLE_PATIENT. Only the violation arm moves.

## Second confound, found in passing, NOT fixed

`verb as object` uses `finds` as its critical word, and `finds` is a
`DEFAULT_LEXICON_HOLDOUT`. That item is simultaneously a category violation and
a novel word -- which is precisely the contrast the `novel_noun` arm exists to
isolate. Swapping it changes what the violation arm means, so it wants a
measurement, not an edit.

## Fixed here

`verb as object 3` read `["she", "hits", "the", "eats"]`. `hits` occurs in no
curriculum sentence and no holdout list -- it existed only in `frames.py` -- so
it categorised UNKNOWN and that item's MAIN VERB was unrecognised before its
critical word was reached (p600 0.0000, stability 1.0000: the degenerate
no-parse reading). A third of the violation arm was not a violation. Now
`chases`, which is trained and makes the item a minimal pair of its control.
