# PREREG: does word learning obey the anchor law, and does Zipf attack the corrector?

Registered before running. Two laws already measured make predictions about
the cross-situational learner (`PREREG_unaligned_scenes.md`, U1-U3 PASS at
P = 3 bundles per scene on a flat corpus):

    CAP-ANCHOR-RATIO   what an assembly forms as is set by its anchor's share
                       of the drive at formation
    base-rate law      raw Hebbian mass follows a referent's BASE RATE, and
                       column normalization is what corrects it
                       ([[cross-situational-needs-homeostasis]])

The toy that passed is 13 word types, flat frequencies, 3 bundles per scene.
That is not evidence about language, and this is the test of how far the
mechanism reaches.

## The two manipulations

**LOAD.** Distractor bundles are added to each scene from the corpus's own
inventory: objects the learner perceives but nobody names. P = 2, 3, 5, 8
bundles per scene. A word's referent is then 1/P of the grounding drive on any
one exposure. The scorer's target is unchanged (the word's own deleted
grounding); per-occurrence chance is 1/P.

**FREQUENCY.** Word types are given Zipfian frequencies (exponent 1.0) by
resampling sentences, holding the total number of presentations fixed. Head
words are seen many times, tail words near the exposure floor.

## Predictions, with bars

Registered seeds (42, 1, 2, 3, 4). Per-occurrence alignment is the metric U1
was judged on. `x chance` means accuracy divided by 1/P.

    L1  GRACEFUL, NOT CLIFF. Alignment declines with P but stays well above
        chance at every P.
        PASS   every P has a lower confidence bound above 2x chance, AND
               P=8 mean >= 0.60
        FAIL   any P at or below 1.5x chance, or P=8 below 0.35
        (A cliff would make this a different mechanism from the graded
        drive-share story, and would be reported as such.)

    L2  THE ANCHOR LAW'S DIRECTION. Accuracy falls monotonically in P.
        PASS   mean accuracy is non-increasing across P = 2, 3, 5, 8 (ties
               allowed), and P=2 exceeds P=8 by more than the pooled seed CI
        FAIL   non-monotone beyond seed noise, or P=8 >= P=2

    L3  HOMEOSTASIS IS LOAD-BEARING, AND MORE SO UNDER ZIPF. The scaling-ON
        minus scaling-OFF gap is measured at P = 3 and P = 8, on both the flat
        and the Zipf corpus.
        PASS   the gap is positive in all four cells, AND the Zipf gap exceeds
               the flat gap at the same P (both P)
        FAIL   any gap <= 0, or the Zipf gap smaller at both P
        Rationale: without correction the learner aligns to the most frequent
        bundle; Zipf makes "most frequent" a stronger attractor, so removing
        the corrector should hurt MORE, not less.

    Z1  TAIL WORDS (reported, no bar). Alignment split by frequency decile
        under Zipf, scaling ON vs OFF. Expected: with correction the head-tail
        gap is small; without it, head words win everything.

## What is NOT claimed

* Distractors are drawn from the same 13-bundle inventory, so this raises
  load, not vocabulary. Nothing here is a claim about a realistic lexicon.
* Zipf is imposed by resampling a synthetic corpus. It is the SHAPE being
  tested, not a real distribution ([[synthetic-corpus-is-flat]] is why the
  shape has to be imposed at all).
* Word order (U2) is not re-run per cell; it is measured once at the worst
  passing load to check the pipeline still closes.

## Interpretation, stated now

* L1, L2, L3 pass -> the anchor law reaches into word learning, and
  homeostasis is confirmed as the enabling primitive rather than a setting.
  The next rung is a real lexicon, and CHILDES becomes the honest target
  (cite-only, never committed).
* L2 fails -> alignment does not track drive share; the anchor law is about
  assembly formation and does NOT govern conjunction learning, which narrows
  CAP-ANCHOR-RATIO's scope and is worth knowing before anything is built on it.
* L3 fails -> base-rate correction is not the mechanism, or not the only one;
  re-examine what column normalization actually did in the flat toy.

---

## Amendment 1 (pre-data, 2026-09-03): P=2 TRIMS, so L2 is judged on the added-distractor loads

The API smoke (one seed, numbers void) showed the flaw: P = 2 is reached by
DELETING a bundle from a 3-bundle scene, while P = 5 and 8 are reached by
ADDING distractors. Those are different manipulations -- trimming removes a
real participant and with it the contrast that identifies the remaining ones,
so a low P=2 cell would say nothing about drive share.

L2's monotonicity is therefore judged over the ADD-ONLY loads, P = 3, 5, 8,
whose only difference is the number of unnamed distractors. P = 2 is still run
and reported, labelled as a trimmed cell, because "less scene" is itself worth
seeing -- but it is not evidence for or against the anchor law and cannot
fail L2. L1 (above chance at every load) still includes it. L3 and Z1
unchanged.

---

## Result (2026-09-03): L2 PASS, L1 INCONCLUSIVE (one clause was ill-posed), L3 half-passes, Z1 is the clean one

Registered seeds, log `alignment_load.log`. Per-occurrence alignment:

    corpus  P=2      P=3      P=5      P=8       (scaling ON)
    flat    0.977    0.994    0.917    0.639
    zipf    0.985    0.959    0.770    0.493
    chance  0.500    0.333    0.200    0.125

**L2 -- PASS, both corpora.** Over the add-only loads the means are
0.994 / 0.917 / 0.639 (flat) and 0.959 / 0.770 / 0.493 (zipf): monotone
decreasing, with P3-P8 gaps of +0.355 and +0.466, each larger than the pooled
seed CI. **Alignment tracks the referent's share of grounding drive**, which
is the anchor law's direction, now measured in word learning rather than in
assembly formation.

**L1 -- INCONCLUSIVE, and one clause of it was MY error.** The bar "lower
bound above 2x chance" is unachievable at P=2, where 2x chance = 1.000; no
accuracy can clear it. That clause was ill-posed at registration, and the P=2
cells (0.977, 0.985) neither pass nor fail it -- they are the highest
accuracies in the table. On the add-only loads every cell clears the bound.
The second clause (P=8 mean >= 0.60) is met on the flat corpus (0.639) and
missed under Zipf (0.493) -- but the registered FAIL zone was "P=8 below 0.35,
or at/below 1.5x chance", and 0.493 is 3.94x chance, so that cell falls
BETWEEN the bars. Reported as inconclusive rather than argued either way. The
script printed a binary because it did not implement the registered three-way;
that is a reporting bug and is fixed.

**L3 -- first clause PASS, second clause FAIL.** The scaling ON-minus-OFF gap
is positive in all four cells and large: flat +0.342 (P=3) and +0.480 (P=8),
zipf +0.420 (P=3) and +0.314 (P=8). Homeostasis is load-bearing everywhere.
But the prediction that the gap GROWS under Zipf holds only at P=3; at P=8 it
shrinks. The reason is visible in the numbers and is a defect in how I framed
the prediction: at P=8 the OFF arm is already near its floor (0.160 flat,
0.179 zipf, chance 0.125), so the gap is bounded above by the ON arm, and the
ON arm itself falls under Zipf (0.639 -> 0.493). A difference of two
compressed quantities cannot test the claim. It needs a floor-independent
statistic, which is what Z1 turns out to be.

**Z1 -- reported without a bar, and it is the cleanest evidence in the run.**
Head versus tail words under Zipf at P=3:

    scaling ON    head 0.959   tail 0.977   head - tail  -0.019
    scaling OFF   head 0.557   tail 0.394   head - tail  +0.162

Without column normalization the learner is FREQUENCY-BIASED: common words are
aligned far better than rare ones. With it the bias is gone, indeed slightly
reversed within noise. That is the base-rate mechanism measured directly and
independently of any floor: homeostasis is what makes a RARE word learnable
from the same evidence as a common one.

## What this establishes

The anchor law reaches into word learning (L2), and homeostasis is the
primitive that makes cross-situational learning work at all (L3 first clause,
Z1) rather than a substrate flag. What is NOT established is the
Zipf-amplification claim, framed as a difference of two quantities that both
compress at high load. A future registration should test it as a head-tail
statistic at fixed load, whose shape Z1 already suggests.

## Post-result note (2026-09-03): L3 re-judged on the repo's instrument

The methodology ratchet caught the first draft of `alignment_load.py`
comparing two bare seed MEANS for L3's gap. Re-judged as it should have been
-- the PAIRED per-seed gap (same seed, scaling ON minus OFF) through
`ensemble_from_values`, on the lower confidence bound -- from the committed
per-seed log (`--replay`):

    flat P=3   +0.342 +/- 0.076   (0.298..0.449)   beats 0
    flat P=8   +0.480 +/- 0.123   (0.307..0.560)   beats 0
    zipf P=3   +0.420 +/- 0.188   (0.273..0.653)   beats 0
    zipf P=8   +0.314 +/- 0.204   (0.078..0.500)   beats 0

Same verdict on the first clause, now on the bound rather than the mean. The
bars now live in `judge()`, which a committed log can drive, so a verdict is a
function of the per-seed numbers and can be re-derived without re-running.
