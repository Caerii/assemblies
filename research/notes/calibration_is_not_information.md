# Calibration is not information — the readout campaign closes at the mean level, and the next unit moves to what training writes

**Task #151, readout campaign (four decision rules, every gate
registered before its data, every failure committed as evidence for the
next step). Substrate throughout: Brown slice, split areas, no scaling,
n ∈ {3000, 10000}, seeds 42–46, all rules read from ONE exam pass —
item-level paired by construction.**

## The measured ladder (balanced number, n=3000 / n=10000)

| rule | sg (3k/10k) | pl (3k/10k) | bal (3k/10k) | failure mode |
|---|---|---|---|---|
| MI drive | .513 / .215 | .709 / .872 | .611 / .543 | boosted-typical vs selected-extreme: inverts with n |
| raw mass | .770 / .328 | .413 / .761 | .592 / .545 | shared-row CLASS-mass inflation: bias flips class with n |
| **excess mass** | .564 / .499 | .752 / .722 | **.658 / .610** | mean-calibrated; SG still under .60 |
| z (√baseline) | .559 / .493 | .760 / .717 | .659 / .605 | changes nothing — noise is not Poisson-scaled |

Each amendment removed exactly the artifact its predecessor's failed
gate diagnosed, and each diagnosis held: fixed columns killed the
extreme-value inversion; base-rate subtraction killed the class flip.
The excess readout is the best Brown number this substrate has
produced, and its n-dependence is the smallest (|Δ| 0.048 vs MI 0.073).

## Why the family closes here

- **Z1 CONFIRMED**: the shared-row noise variance into the frequent
  class's image exceeds the rare class's by 2.43 ± 0.46 at n=10000
  (1.64 ± 0.65 at n=3000) — the variance asymmetry is real.
- **Z2/Z3 REFUTED**: the √baseline normalizer moves nothing. The
  remaining readout-side option is an EMPIRICAL per-word permutation
  null (Monte-Carlo random row-sets at recall) — noted, not run,
  because the detection-theory reading says the family is near its
  ceiling: a calibrated comparison can symmetrize errors but cannot
  create signal, and the per-word signal is a (1+β)^~3 Hebbian boost
  sitting inside noise whose variance the class imbalance sets. The
  attribution unit's G3 already measured the signal thinning (own-drive
  SG/PL 0.80 → 0.34 with n).

## Adoption status

`morph_readout="mass"` (excess form) stays available and measured-best
ON BROWN; it is NOT adopted as default — the synthetic guard read
−0.059 ± 0.073 paired against MI, i.e. the readouts remain
regime-dependent, which is itself the finding: no decision rule fixes
what the images do not contain.

## The registered continuation — write word identity into the image

Every rule fails the same ultimate limit: value-area winners are
selected almost entirely by the LABEL stimulus during training, so the
image is the class attractor and word identity lives only in weight
boosts of size (1+β)^exposure. The measured-precedent lever
([[semantic-drive-share-is-the-lever]]: phon_weight=6 fixed exactly
this shape at the LEX level) is to raise the WORD-CORE fiber's drive
share during train_number so winners become word-conditioned — per-word
sub-assemblies inside each value area, separable by ANY readout. Next
unit: sweep the core:label drive ratio during value-area training
(implementation via the existing per-fiber gain bracket), bars: Brown
balanced ≥ 0.70 at both n with SG ≥ 0.60, synthetic guard non-negative,
and the E14 merging counter must NOT regress (word-conditioning must
not re-open cross-class merging — that is what the split exists for).
