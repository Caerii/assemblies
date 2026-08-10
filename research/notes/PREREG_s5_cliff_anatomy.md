# PREREG: anatomy of the cliff

Registered before implementing or running. Follows `6276685`, where the S5
per-step readout falsified both amendment bars and showed the failure mode is
a CLIFF: `step@500` equals `first_bad/500` to within 0.004 on every group --
exact until derailment, ~0% correct after, never recovering.

## What the existing data already constrains

1. `run()` executes inside `probe()`: no plasticity, no recruitment, no
   refraction charging. The machine is FROZEN, so each step is a deterministic
   function of the current state assembly. If the state assembly snapped
   EXACTLY onto its stored block every step, the dynamics would be a function
   on a finite set, and a pair (q, g) correct at step 10 would be correct at
   step 350. A cliff after hundreds of correct steps is then impossible.
2. After derailment, TRANSITION accuracy (measured from the observed previous
   state) is ~0. A failure that merely sent the machine to a wrong-but-valid
   state would leave later transitions correct. So the first failure ejects
   the machine OFF THE LATTICE into assemblies that drive nothing -- an
   absorbing garbage regime.
3. Derailed-seed first_bad means: Z60 ~215 (M=120 pairs), S5 ~201 (M=240).
   A random walk on a group is uniform over states, so ONE bad pair is hit in
   ~M steps on average. Right order of magnitude for a defect-set account,
   not exact.

## The two hypotheses

**H-defect.** A small static set of transitions is trained WRONG (per seed).
The sequence is exact until the word first exercises a member of that set;
that step ejects off-lattice; everything after is garbage. "Horizon" is not a
time constant -- it is the first-hitting time of the defect set, and the L=500
exact-trajectory number is just the probability the word avoids the set.

**H-drift.** No transition is individually broken. The live state assembly
deviates slightly from the stored block, differently on different VISITS to
the same state; deviation eventually exits a basin. The census (cued from
exact blocks) comes back clean, and the cliff is a property of trajectories,
not edges.

## Instruments

**E1 determinism.** The same word run twice on the same trained organ must
give identical trajectories. Frozen machine + no sampler draws on materialised
areas -> no randomness channel is known. If this fails everything else is
reinterpreted.

**E2 on-block trajectory.** Along the sequence, at each pre-derailment step,
overlap of the LIVE state assembly with the stored block of the state the
readout labels. H-defect predicts exactly 1.000 until the defect step;
H-drift predicts values below 1.000 with structure.

**E3 transition census.** For every (state, symbol) pair: cue the stored
block, step once, record (a) whether the labelled successor matches the table,
(b) the readout MARGIN (best minus second overlap), (c) the arc assembly.
Cheap: two projections per pair after the 50x fix.

**E4 per-seed exact prediction.** From the census defect set and the word's
ground-truth path, predict first_bad EXACTLY: the first step i whose
(true_state_i-1, word_i) is in the defect set (or "clean", predicting no
derailment). Compare per seed. This is a zero-parameter prediction; there is
nothing to fit.

**E5 mechanism on defects (exploratory, no bar).** For failing pairs: max arc
overlap against every other pair's arc (collision), their margin against the
clean pairs' margin distribution, and WHERE the failed step actually lands
(overlap to intended block vs to best block vs off-lattice).

## Bars, stated now

* **C1 (determinism):** identical trajectories on rerun, 10/10 seeds per
  group. *Prediction: PASSES (~95%).*
* **C2 (on-block):** pre-derailment live state overlap with the labelled
  stored block is exactly 1.000 at every step. *Prediction: PASSES (~60%) --
  A1's limit-cycle closure argues for exactness, but it was measured on
  constant input at 5 states, not on a 500-step random word at 60-120.*
* **C3 (the decisive one):** measured first_bad equals the E4 prediction on
  EVERY derailed seed, and every clean seed is predicted clean over its word.
  *Prediction: PASSES (~65%).* C2 and C3 stand or fall together: exact
  on-block dynamics make the census a complete simulator of the sequence.
* **C4:** censused defect count per seed is small (0-2 for order-60 groups).
  *Prediction: PASSES.* Implied by the first-hitting arithmetic above.

## Interpretation, stated now

* C3 passes -> the cliff is FULLY explained by a static defect set. The
  scientific object becomes the DEFECT RATE per trained transition, and
  [[SEQ-EXACT-RECOVERY]] is repaired: recovery is exact on the non-defective
  machine, and the "horizon" was never a dynamical quantity. The next
  question is why specific pairs train wrong (E5: collision vs margin), which
  is where a fix would live.
* C3 fails with a clean census -> H-drift; the on-block trajectory (E2)
  becomes the primary data, and the basin-exit statistics are the object.
* C3 fails with a nonempty census but wrong predictions -> both effects at
  once; report the split (defect-predicted vs earlier-than-predicted
  derailments) without forcing one story.
* If determinism (C1) fails, stop and find the randomness channel first;
  nothing else is interpretable.

## Committed in advance

1. Bars before data; no parameter changes after seeing results. The trained
   organs are IDENTICAL to the registered S5 study (same builds, same seeds,
   same words); only readouts are added.
2. All four groups run; per-seed values printed; the defect-set SIZES are
   reported for clean seeds too, not only derailed ones (a defect the word
   never visits is still a defect).
3. E5 is exploratory and generates hypotheses, not conclusions.
