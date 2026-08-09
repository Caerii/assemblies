# The real corpus inverts the collapse — scaling hands the decision to the rare class, and the capacity story is refuted by its own control

**Task #150 (#30). Census `70c0705` (C1 Zipf −1.096, C2 40859, C3 plural
share 0.102); teacher mechanism `ebfffbe`; Phase 1 harness `e7f28cb`
(bars registered before any substrate cell); 30 registered cells + three
labeled post-hoc probes (drive decomposition, arm D, arm E), each with
its prediction stated before its data. Corpus: Brown (Eng-NA) via
childes-db, 85002 adult utterances; slice = first 1000 teachable
(1386 noun episodes, 409 exam forms, 0 ambiguous).**

## Registered verdicts — and what each turned out to measure

- **GUARD SG >= 0.8 FAILS CATASTROPHICALLY, and the failure is the
  headline.** The production configuration (split + deferred scaling,
  K=40) answers PL for nearly everything: SG 0.036 (A), 0.0003 (B),
  vs 0.517 shared (C). The synthetic corpus's SG-default collapse
  (E12: 20/24 failures answer SG) INVERTS on the real 90/10
  distribution.
- **F1 as registered is confounded**: spearman(exposure, correctness) =
  −0.047 ± 0.041 on arm A — but under the collapse "correct" ≈
  "label is PL", which anti-correlates with exposure. The instrument
  assumed a working readout (assert-the-claim, instrument edition).
  Re-measured on arm D (working readout): **+0.069 ± 0.044, CI
  excludes 0 — the exposure law transfers**, and the buckets reproduce
  its shape exactly: 1 exposure = 0.4995 (chance), 2–3 = 0.560,
  4+ = 0.575. Same sign at n=10000 (+0.043 ± 0.017, buckets monotone).
- **F2 unclaimable** (+0.006 ± 0.006 on a degenerate metric).
- **F3 "passes" (+0.257 ± 0.087) but is an ARTIFACT**: arm A's PL is
  inflated by answering PL for everything. The corrected comparison
  (D vs C, both working): split-no-scaling ≈ shared on Brown
  (SG .513/.517, PL .709/.720). The synthetic PL rescue did not
  transfer at this budget — because Brown's PL forms average ~3
  episodes each, ALREADY in the reliable regime, so shared does not
  collapse PL here in the first place.

## The mechanism, pinned by intervention

1. **Drive probe (arm A, seed 42): the PL area is a CONSTANT
   responder.** drive_PL ≈ 0.9–1.2 for every word regardless of label
   or exposure; drive_SG ≈ 0.1–0.7. Median PL/SG ratio 3.5 for SG
   words, 3.9 for PL words — nearly label-independent. 99.7% of SG
   words invert. The decision is made by AREA-level mass statistics,
   not word identity — the P600 lesson's shape ([[erp-p600-sign-inverted]]),
   produced by homeostasis under class imbalance.
2. **Arm D (no scaling): SG 0.036 → 0.513 ± 0.057.** Per-column
   homeostatic normalization is the CAUSE, not form-load dilution:
   scaling equalizes the two areas' column masses while their form
   loads differ ~8x (363 vs 46 forms), which converts the rare class's
   area into a high-gain responder. Homeostatic scaling was adopted to
   stop the frequent class from swamping the rare one (E1–E19); on a
   real 90/10 distribution it OVERCORRECTS and hands the decision to
   the rare class. The E-series never tested this regime — its
   imbalance was 70/30, and C3 flagged the difference before Phase 1
   ran.
3. **Arm E (n=10000, capacity control): the over-capacity account is
   REFUTED.** Prediction was SG recovery (alpha* ≈ 1.15·n/k puts
   n=3000 3x over load for the SG area, n=10000 under). Measured: SG
   0.513 → **0.215 ± 0.013**, PL 0.709 → 0.872. Bigger n AMPLIFIES the
   rare-class bias. Whatever produces the residual asymmetry grows
   with area size at fixed k — pointing at within-area
   recruitment/attractor dynamics (the label-stimulus attractor gets
   1247 vs 139 writes; [[reinforcement-tradeoff]],
   [[norm-init-stability-threshold]]), not capacity. The asymmetry is
   NOT per-form exposure: both classes average ~3 episodes/form.

## What the graduation bought

The corpus level of the law survives contact with real data: CDS is
Zipfian at slope −1.096 (measured, not assumed), the ~2–4-episode
reliability threshold transfers (chance at 1 exposure, rising after),
and exposure predicts correctness wherever the readout works. What
breaks is exactly what the synthetic corpus could not test: **cross-area
drive comparison under real class imbalance**, where (a) homeostatic
scaling inverts the decision (measured, causal), and (b) a second
n-scaling bias remains unattributed. The #24 weak-primitive finding now
has its strongest form: the MI competition is only calibrated when the
compared areas carry comparable training mass, and both adopted
remedies (scaling, larger n) make it WORSE.

## Adoption consequences (production_configuration.md updated next unit)

- Deferred scaling + K=40 stays measured-best ON THE SYNTHETIC 70/30
  corpus; on real-imbalance corpora it is measured-harmful pending the
  fix. The production note must scope it.
- The registered continuation: **per-form (learning-rule)
  normalization** — the direction
  [[hebbian-mass-follows-frequency]] already named — and a drive
  decomposition of arms D/E to attribute the n-scaling bias before any
  new mechanism is proposed.
