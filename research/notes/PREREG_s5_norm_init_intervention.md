# PREREG: does norm_init delete the soft transitions?

Registered before implementing or running. Follows
`the_cliff_is_a_soft_transition_census.md` (bb5195a): ~0.25-0.6% of trained
transitions emit 69/70 of the right block with exactly ONE intruder neuron,
and every lattice exit is the first visit to such a pair (40/40,
zero-parameter).

## The hypothesis under test

The intruder is a STATE-AREA HUB. These organs run `norm_init=False`,
inherited from A1's parity configuration on the stated ground that "neither
area has a self fiber, which is the only thing norm_init exists to stabilise"
(`mod3_fsm.py`). The soft-transition finding challenges exactly that clause:
[[recurrence-needs-norm-init]] localised collapse to high-degree hubs on
SELF-fibers, and today's A3 toy showed hub formation on a FEED-FORWARD fiber.
If initial-weight column dispersion lets a few state neurons start with
anomalously high afferent mass, 15 presentations of multiplicative Hebbian
amplify the head start, and the strongest such neuron edges out the weakest
member of a target block in a handful of contexts -- one intruder, sometimes
shared across an organ's soft pairs.

This is an INTERVENTION, and it changes the substrate: every number is a
re-measurement, and the control arm re-establishes the baseline inside the
same study rather than citing it.

## Arms

Same groups, seeds 42..51, words, parameters as the registered S5 study.

  control        norm_init=False   (the registered substrate)
  intervention   norm_init=True    (production substrate)

## Readouts

Per organ: the live soft census (snapped inside the probe) NOW RECORDING
IDENTITIES -- for each soft pair, the intruder neuron ID and the displaced
neuron ID; the replayed word's first_dev / first_bad; exact@{10,50,100,500}.

Hub diagnostics on every organ with >= 1 soft pair, both arms:

  * whether the organ's soft pairs SHARE an intruder;
  * the intruder's afferent weight-mass percentile: column sum of the
    arc -> state weights into the intruder, ranked against all state-area
    columns.

## Bars, stated now

* **N1 (validity):** the control arm reproduces the registered soft counts
  per organ exactly. Same seeds, same construction, deterministic engine --
  anything else means the instrument moved. *Prediction: PASSES (~90%).*
* **N2 (the intervention):** the norm_init=True soft rate is ZERO across all
  40 organs. *Prediction: PASSES at ~55%* -- the hub account is coherent but
  circumstantial: nothing yet shows the intruder is weight-anomalous, and
  norm_init could shrink dispersion without eliminating the tail.
* **N3 (consequence, conditional on N2):** exact@500 is 10/10 on every group
  under norm_init=True.
* **N4 (mechanism, judged on the CONTROL arm):** intruders sit above the
  99th percentile of afferent column mass. *Prediction: PASSES (~65%).* If
  N4 fails while N2 passes, norm_init fixed the softness through some other
  channel and the hub story is wrong even if the fix works.
* **N5 (no new pathology):** the intervention arm has zero HARD defects and
  its label census is clean -- norm_init must not break the basic machine.

## Interpretation, stated now

* N2+N4 pass: the parity clause "norm_init exists only for self-fibers" is
  FALSIFIED in this repo's register, [[SEQ-EXACT-RECOVERY]]'s caveat is
  amended (soft spots are a substrate artifact, not intrinsic), and the SSM
  contrast can be re-run on the repaired substrate with real power.
* N2 fails: hubs survive normalization or were never the cause; N4's
  percentile data says which; the defect mechanism stays open and the
  hitting-time account (which does not depend on the mechanism) stands.
* N5 fails: norm_init harms teacher-forced organs; that constrains the
  production substrate claim and is reported as such, not buried.

## Committed in advance

1. Bars before data. The control arm runs FIRST; if N1 fails nothing else is
   interpreted.
2. Per-organ values printed; both arms' full exact@L tables reported.
3. Intruder identities recorded in the artifact for both arms, so the hub
   analysis is auditable offline.
