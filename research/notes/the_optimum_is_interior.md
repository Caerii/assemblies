# The optimum is interior — the split closed one degradation channel and E17 exposes a second

**Task #146 (E17). Experiment: `research/experiments/clean_substrate_400.py` (bars R/V/V' registered after the E16 revert, before any clean 400-frame cell trained). ZIPF × {200, 400} × seeds 42–51, split + feed-forward labels (post-revert code).**

## Bars

- **R PASSES EXACTLY**: zipf-200 overlap = 0.7000 ± 0.0373, replicating
  E15 to the third decimal. The revert is verified by measurement; the
  substrate is the known-best one. (Third independent measurement of
  this configuration at this value — 0.700 is now the program's most
  reproduced number.)
- **V FAILS, and not narrowly**: zipf-400 overlap = 0.570 ± 0.023 —
  BELOW zipf-200. **V' agrees** (fixed 30-form exam: 0.590), so the
  decline is per-form degradation, not exam growth. MI declines less
  (0.637 → 0.613), so the evidence readout is the more scale-fragile of
  the two.
- **The exposure law is broken at 400 on the CLEAN substrate**: "men"
  at 14 episodes reads 0.60, "foods" at 5 reads 0.10 — which
  re-attributes E16's flattening: the recurrence damage was real
  (−0.090 paired at 200) but it was NOT the cause of the 400-frame
  failure. That cause survives the revert.

## The shape of the arc's answer

The budget curve on the best-known configuration is NON-MONOTONE:

    0.528 @ 50  ->  0.700 @ 200  ->  0.570 @ 400

E14 named channel one (label-image merging in a shared area; closed by
the split — uniform-200 recovered and stayed recovered). E17 shows a
SECOND channel that the split does not close: something inside a single
value area degrades as its OWN episode count grows (the PL area sees
only PL episodes — 49 at 200, 107 at 400 — and gets worse at more).
Candidates, in registered-suspect order for E18:

1. **Label-image DRIFT**: the stim image is re-drawn against weights
   that keep changing; words trained EARLY bound to an image that no
   longer exists at readout ([[consolidation-resets-the-index-space]]
   rhymed at the weight level). Discriminating measurement: accuracy vs
   the position of a form's LAST training episode (the corpus is
   deterministic, so episode order is recoverable); drift predicts
   late-trained forms read better.
2. **Value-area crowding**: recruitment grows with episodes; the k-WTA
   competition over a larger materialized pool dilutes both probe and
   image ([[beta-opposes-capacity-and-depth]]'s crowding signature —
   check spread/margin diagnostics before believing it).

## Standing after seventeen experiments

The 0.75 bar was not cleared. The honest terminal numbers and laws:

- **Best configuration, thrice-replicated**: split per-value areas,
  feed-forward label training, within-area overlap readout, Zipfian
  corpus at ~200 frames: **0.700 ± 0.037** (from 0.500 at the arc's
  start — +0.20 of measured, mechanism-attributed improvement).
- Mass -> exposure -> budget×allocation (levels 1–3) hold BELOW the
  interior optimum; ABOVE it, per-area training-scale degradation
  (channel two) dominates, and no corpus statistic fixes a substrate
  dynamic. The scaling roadmap (CHILDES included) now waits on E18's
  drift-vs-crowding verdict, not on more corpus.
- The paper's inter-area inhibition is fully characterized here: a
  commit/hold device (E16's L, exact), with accuracy living in
  within-area evidence.
