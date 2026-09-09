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

---

## Result of the first run, and Addendum: the memory channel

C1 PASS (40/40 deterministic). C4 PASS. **C2 FAIL, C3 FAIL** -- and the
census is EMPTY: zero defective transitions on any seed of any group, margin
1.0000 on every censused pair. H-defect is falsified. 11/40 trajectories
derail anyway, with pre-derailment on-block minima as low as 0.086 while the
readout still labels correctly; clean seeds dip to 69/70 and recover.

**The paradox this leaves.** Census: exact-in -> exact-out, universally. The
step has no declared memory: no arc self-fiber, no state self-fiber, weights
frozen under `probe()`. Exact single steps composed memorylessly cannot leave
the lattice, by induction. They do. Therefore the step has UNDECLARED memory,
and the only structural difference between census and sequence is RESIDUAL
CONTENT: the census inhibits the arc (and state) before every step; the
sequence carries the previous step's arc winners into the next projection.

**E6 (bisection), registered before running.** On each derailed seed, replay
(deterministic, so faithful), find the FIRST step d where the live state
deviates from the block of its own label (state at d-1 exact by minimality).
Then three single-step probes of the same (state_{d-1}, sym_d):

  P-census   inhibit arc+state, cue the block, step.       Expect exact.
  P-residual inhibit, cue the block, then SET the arc to its step-(d-1)
             winners before stepping. If this reproduces the in-sequence
             off-block output, the arc residual is the memory channel.
  P-arc      compare the ARC assembly at step d in-sequence against the
             census arc for the same pair: does deviation enter at arc
             SELECTION or at arc -> state?

**Predictions.** P-census exact (anything else contradicts the census just
taken). P-residual reproduces the deviation, ~70% -- elimination leaves arc
residual as the only candidate, but "the only candidate I can see" has been
wrong once already today (H-defect). If P-residual is exact too, the channel
is in how the state ARRIVES (projection vs `activate_assembly`), which is a
different engine finding.

**Interpretation, stated now.** Whatever channel E6 names is an ENGINE
SEMANTICS finding, not a science result: the model's declared step is
memoryless and the implementation's is not. The fix lands in the engine with
a regression test, the S5 study re-runs after it, and [[SEQ-EXACT-RECOVERY]]
is re-evaluated on the repaired substrate -- the current horizon numbers may
be an artifact of the leak.

---

## Addendum 2: the census instrument was DEAD, and the paradox dissolves

**Retraction.** The claim in `79399a3` that "the step has undeclared memory"
is WITHDRAWN. Its premise -- exact-in -> exact-out, universally -- came from a
broken instrument.

**The instrument bug.** `probe()` RESTORES WINNERS ON EXIT (verified directly:
after == before, not after == inside). The census called `fsm.run`, which
wraps its own probe, and then `_snap`ped AFTER it returned -- so every margin
measured the restored pre-census residue, the same winners 240 times per
organ. Margin 1.0000, constant, everywhere: the exact dead-probe signature
this project already documented ([[fake-perfect-probe-signatures]]: "any
margin that never varies is a dead probe"), printed 40 times and believed.
`diagnostics.verify_probe` exists precisely to refuse such an instrument and
was not used. The census LABELS remain valid (computed inside `run`).

**What is actually established.** exact-in -> correct-LABEL-out. Assembly
exactness of single steps was NEVER measured. A transition may emit 69/70 of
the right block; deviation seeds there; composition amplifies or corrects.
E6's one valid reading -- A5 seed 44 deviates at STEP 0, where there is no
residual at all -- already shows deviation without memory. Memoryless step +
SOFT defects is now the parsimonious account, and no engine-semantics anomaly
is needed.

## E7: the soft census, with a live instrument

Single-step census as before, but snapped INSIDE the probe, recording the
OUTPUT ASSEMBLY's overlap with its intended block per pair. Soft defect :=
correct label, overlap < 1.0.

Bars, stated before running:

* **V1:** the valid margins VARY (the instrument is alive), and organs of
  deviating seeds contain at least one soft pair.
* **V2 (zero-parameter, again):** per seed, the first deviation step
  `first_dev` equals the first step at which the word's TRUE path visits a
  soft pair; seeds whose trajectory never deviates visit none.
  *Prediction: PASSES, ~75%.* This is the same prediction shape that
  falsified H-defect, now aimed at the quantity the composition actually
  iterates.
* **V3 (dynamics, exploratory):** whether a seeded deviation derails or
  recovers is NOT barred here; if V2 passes it becomes the next question
  (correction-radius curve).

## Addendum 3 (2026-09-09, before running): the census at width, on the EXPLICIT substrate

E7 ran on the SAMPLED arc (only STATE is materialized in `build`). GATE-3
of the sequence port (DESIGN_sequence_port.md) then found that A1's one
short horizon at p = 0.3 was the sampler's: the same seed materialized runs
2000 digits exact. The soft defect -- correct label, one intruder neuron --
is exactly the kind of one-neuron event a sampled arc could seed. So E7 is
re-run on `HashedArcFSM` (== the materialized engine), same four groups,
same ten seeds, same census and the same 500-symbol words, brains batched.
The hashed organ is not the numpy organ bit for bit (stimulus model,
sampler), so seeds pair by protocol, not by trajectory.

    W1  THE SOFT PAIRS WERE THE SAMPLER'S.  >= 90% of the 40 organs have no
        soft and no hard pair and run their word 500 steps without a wrong
        label.  PREDICTION: PASSES (GATE-3's reading).
        FAIL: soft pairs persist at ~0.25-0.6% of pairs -> the soft map is
        substrate-intrinsic, the hitting-time law stands as measured, and
        the hashed organ inherits it; report the rate against E7's.
    W2  V2 RESTATED (zero-parameter).  On every organ with a bad pair, the
        word's first deviation equals its first true-path visit to a bad
        pair; vacuous on clean organs.
    W3  reported: the soft overlap values (E7: always 69/70 = 0.986).

If W1 passes, [[SEQ-EXACT-RECOVERY]]'s "AND THE MAP HAS SOFT SPOTS at
scale" is re-scoped to the sampled engine in the register; the hitting-time
mechanism is kept as the account of what a soft pair does when one exists.
