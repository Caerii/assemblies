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
