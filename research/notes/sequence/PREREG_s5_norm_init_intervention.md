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

---

## Result (2026-08-10, fixed engine, run 351e1cb): N1/N5 PASS, N2/N3/N4 FAIL — in the INFORMATIVE direction

    N1 PASS   control reproduces the registered census exactly (30 soft pairs,
              same organs, same counts). The engine fixes did not move the
              norm_init=False substrate.
    N2 FAIL   intervention soft rate is not zero -- it is ~100x the control:
              5392 soft pairs across 40 organs (~90% of transitions;
              107-116/120 on the order-60 groups, 198-216/240 on S5).
    N3 FAIL   exact@500 is 0/10 on every group under norm_init=True.
    N4 FAIL   control intruders are NOT hubs: afferent-mass percentiles spread
              0.106..0.985, min 0.106 vs the >=0.99 bar; 0/19 organs share an
              intruder across their soft pairs.
    N5 PASS   zero hard defects in either arm. Labels stay perfectly correct;
              only codeword integrity degrades.

Intervention defect anatomy: median 2 intruders per soft pair (max 9),
overlap down to 0.871; intruders almost never repeat within an organ (max
recurrence 2% of an organ's soft pairs) -- diffuse displacement, not a
shared hub.

## Interpretation (per the pre-stated rules)

The hub hypothesis is DEAD on both arms: control intruders sit anywhere in
the afferent-mass distribution (N4), and norm_init=True makes the softness
~100x worse rather than deleting it (N2). Per the prereg's own clause, "the
defect mechanism stays open and the hitting-time account stands."

The failure DIRECTION is explained by what `_norm_scale` actually is, which
its docstring states plainly: the divisor d_j COUNTS present synapses --
potentiation-invariant -- reproducing the reference's one-time init-time
normalize(). After 15 presentations, trained block members carry ~(1.1)^15
= 4.2x initial mass but are divided by their UNPOTENTIATED count; untrained
sparse neurons carry ~1x and are divided by a SMALL count. 1/d does not
remove the degree bias at read time -- it inverts it, handing the advantage
to the sparsest neurons. Intruders in the 10th percentile of afferent mass
are the signature of a stale divisor, derivable in advance.

The literature check this triggered (papers, not reference code): PNAS'20,
COLT'22 Thm 6, and the sequences paper's Thms 1-2 all assume ONGOING
homeostasis -- "after each round ... each neuron's incoming weights sum
to 1" is a stated hypothesis of the sequence-memorization guarantees, with
COLT'22 renormalizing per class presentation. The reference code's
init-only normalize() is a simulation shortcut, not the model. No substrate
this repo has measured satisfies the theorems' precondition:

    A  norm_init=False        no normalization        soft ~0.5%   (registered)
    B  norm_init=True         1/d once, count-based   soft ~90%    (this study)
    C  rows -> sum 1 PER ROUND (the theorems' actual hypothesis)   NEVER BUILT

Also surfaced while checking preconditions: the organ sits BELOW the
sequences paper's regime floor kp >= 3 ln n in all four groups (kp=28.0 vs
floors 29.71 / 31.79) -- deepest in S5, the group with the worst horizon.
A live confound for the group ordering, addressable at organ_p=0.5.

Registered next steps: build substrate C (per-round row renormalization),
re-run this census under it; re-run the M-ceiling table under C on
numpy_exact; clear the regime floor and re-measure the group ordering.
