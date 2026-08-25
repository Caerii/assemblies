# PREREG: refraction is the stability/conjunctivity tradeoff

Registered before running. Follows the batchability measurement
([[arc-training-is-not-batchable]]): the arc's identical-assembly fraction
peaks ~0.57 at presentation 5-8 and DECAYS to 0.12 by 12. Reading the code
supplies a mechanism, and it is not a bug.

## The mechanism (from code, before data)

`_refraction.refraction_increment` returns `(net_drive + current_bias) *
strength`, and the bias is NEVER cleared during training -- an earlier
version cleared it per `train_transition` and that was deliberately removed
("the reference clears activations between transitions and never clears the
bias", nemo_fsm docstring). So for a neuron that keeps winning:

    bias_{t+1} = bias_t * (1 + strength) + net_t * strength

a GEOMETRIC recursion with ratio 1.1. After 64 presentations a persistently
winning arc neuron carries ~(1.1)^64 = 456x its own drive contribution as
penalty and is forced out of the winner set. Refraction burns out its own
winners.

That single mechanism predicts BOTH measured facts: assemblies drift (winners
are permanently penalised, so the assembly must keep moving) and the arc
saturates at 19999/20000 (every neuron is eventually used and burned).

Refraction exists to keep the arc conjunctive
([[refraction-is-the-anti-swamping-force]]). So the hypothesis under test is
that **conjunctivity and stability are traded against each other**, and
nobody has measured the curve.

## Design

Z60, organ_p=0.5, T=16, w_max=None, synaptic_scaling=False, seeds 42-44
(ensemble, not realization). Cells: `refracted_strength` in {0, 0.05, 0.1}
x {geometric (default), constant (ASSEMBLIES_CONSTANT_REFRACTION=1)}.
Strength 0 is mode-independent, so 5 distinct cells x 3 seeds.

BOTH readouts every cell, because either alone is misleading:
* STABILITY: identical-assembly fraction per presentation, arc winner sets
  re-probed inside `brain.probe()` (no recruitment) -- the instrument from
  the batchability measurement.
* FUNCTION: trajectory accuracy at L=100 against `true_trajectory`.
  A machine that is perfectly stable because it has stopped discriminating
  would pass a stability bar and be worthless.

## Bars, stated now

* **R1 (mechanism):** terminal identical-assembly fraction at strength=0
  exceeds 0.90. If refraction is the burn-out force, removing it should let
  assemblies settle. *Prediction: PASSES (~75%).*
* **R2 (monotone):** terminal stability is ordered 0 > 0.05 > 0.1 in the
  geometric mode. *Prediction: PASSES (~70%).*
* **R3 (geometry matters):** at equal strength, constant mode ends ABOVE
  geometric -- linear accumulation should burn winners slower.
  *Prediction: PASSES (~65%).*
* **R4 (the tradeoff, the interesting one):** trajectory accuracy at
  strength=0 is BELOW accuracy at strength=0.1. *Prediction: PASSES (~55%)
  -- genuinely open.* If R4 FAILS (accuracy equal or better without
  refraction) then refraction is not earning its keep on THIS organ, which
  would be a bigger result than R1-R3 and would reopen the arc's design.

## Interpretation, stated now

* R1+R2+R3 pass: the geometric bias accumulation is the stability mechanism;
  the tradeoff curve is real and the organ's operating point becomes a
  DESIGN choice to register rather than a default to inherit.
* R1 fails: something other than refraction drives the drift; the batchability
  and low-rank findings need a different explanation and this note is wrong.
* R4 fails: refraction costs stability and buys nothing measurable here --
  register a follow-up on whether the arc still needs it at this load.
* Any bar passing while its claim dies is reported as both
  (the E-series lesson; it has already happened once on the zipf slope).

## Committed in advance

1. Bars before data; this file lands before any cell runs.
2. Both readouts reported for every cell whatever they say.
3. Per-seed values, never bare means ([[report-distributions-not-point-estimates]]).

---

## Amendment 1 (pre-data): the stability signal is already recorded

Reading the harness before running found the probe pass is redundant. With
`save_winners=True` (already set in `build`), every projection appends its
winner set to `area.saved_winners`, and the arc is the TARGET of exactly one
of the two projections per transition. Verified at T=4: the arc has 480
entries = 120 transitions x 4 presentations, exactly, so

    saved_winners[t*120 + i]  IS  the arc assembly for transition i at
                                  presentation t

and they are stored as NEURON IDs (mapped through `get_neuron_id_mapping`),
which is the stable space -- no compact-index conversion, so
[[two-index-spaces-compact-vs-neuron-id]] cannot bite.

The registered design re-probed 120 transitions x 16 presentations = 1,920
extra projections per cell inside `brain.probe()`, ~75s of a ~110s cell, to
recompute quantities training had already produced.

**Instrument change, stated before data.** Stability is now read from
`saved_winners`. This is a DIFFERENT statistic from the registered one and
the difference is not cosmetic:

* PROBE (registered): all 120 transitions evaluated at ONE frozen weight
  state, plasticity and recruitment off. Order-independent.
* RECORDED (this amendment): transition i at presentation t sees weights
  already updated by transitions 1..i-1 of that same presentation. Sequential.

The recorded form is arguably the more faithful one -- it is the assembly the
organ ACTUALLY formed, not a counterfactual read -- but it is not the same
number, so the swap is documented rather than assumed.

**Validation, run in the same script:** one cell (strength=0.1, geometric,
seed 42) computes BOTH, and the per-presentation identical-assembly fractions
are reported side by side. If they disagree materially the recorded
instrument is reported as its own quantity and the probe numbers are the ones
the bars are read against.

Cost: ~110s -> ~35s per cell; 15 cells at 14 workers, ~1.5 min total.
No bar changes.

---

## Amendment 2 (pre-data): the design did not reproduce its own phenomenon

The API smoke exposed an error in this registration, not in the code.

This note explains a decay measured in [[arc-training-is-not-batchable]]:
identical-assembly fraction peaking ~0.57 at presentation 5-8 and falling to
0.12 by 12. **That measurement ran with `synaptic_scaling=True`.** The design
above registered `synaptic_scaling=False`. So as written the study could not
reproduce the phenomenon it was built to explain -- and the smoke cell
(strength=0.1, geometric, seed 42, scaling OFF) came back at terminal
identical 0.842 and RISING, the opposite direction.

Per the standing rule the smoke's numbers are VOID; what is not void is the
structural fact that the registered factor set omits a variable already known
to differ between the motivating measurement and this design.

**Change:** `synaptic_scaling` becomes a FACTOR, {False, True}, crossed with
the existing 5 refraction cells. 10 cells x 3 seeds = 30. Cost ~35s/cell at
14 workers, ~2 min.

**Bars R1-R4 now read against the scaling=True arm**, since that is the arm
whose decay motivated them; the scaling=False arm is the control that says
whether the phenomenon needs refraction at all. R1-R4 are otherwise unchanged
and their thresholds are untouched.

**New bar, stated now:**

* **R5 (which mechanism):** terminal identical-assembly fraction is LOWER
  with scaling ON than OFF, at equal refraction. *Prediction: PASSES (~70%)
  on the smoke's direction.* If R5 passes, the decay is a HOMEOSTASIS effect
  and this note's geometric-refraction mechanism is not the explanation --
  in which case R1-R3 may well pass while the story that motivated them is
  wrong, and both get reported.

This is the second time in this session that a mechanism read out of the code
predicted the wrong thing (cf. the CUDA-graph hypothesis, refuted at 1.2x).
Reading a mechanism is a hypothesis, not a measurement.

---

## Amendment 3 (pre-data): aggregation only -- bars judged on confidence bounds

Recorded BEFORE the study has been run even once. No data from this design
exists, so nothing here can have been chosen to suit a result.

**Why.** `test_methodology_ratchet` flagged this experiment for six
hand-rolled seed statistics. Four of them were load-bearing: `term()` and
`acc()` returned a bare `np.mean` over seeds, and R1-R5 were judged on those
point estimates. That is the exact pattern the ratchet exists to stop -- this
project published a "conserved budget" law computed with `statistics.mean` over
seeds and no interval, then retracted it.

**What changed.** Only the aggregation across seeds:

* `term()` and `acc()` now return `diagnostics.ensemble_from_values` -- mean,
  95% CI, and the per-seed values -- instead of a float.
* **R1** (threshold) is judged with `Ensemble.beats(0.90)`, i.e. the CI LOWER
  BOUND must clear 0.90, not the mean.
* **R2, R3, R4, R5** (all orderings) are judged with `paired_delta`, the
  per-seed difference. That is what an A/B actually asks: comparing two
  independent intervals is a different and weaker test, and comparing a
  difference against a single arm's sd understates the spread by ~sqrt(2) --
  which is how a 1.49-sd difference once got reported here as 2.10 sd.
* The summary table prints mean +/- CI alongside the per-seed values it already
  printed.

**What did NOT change.** The cells, the per-cell statistic, the thresholds
(0.90 for R1; strict ordering for R2/R3/R4/R5), the directions, the arms, the
seeds, and which arm each bar reads. Nothing about what is measured moved.

**This makes the bars STRICTER, and that is deliberate.** Tightening a bar
before any data exists is always safe; the hazard pre-registration guards
against is loosening one after seeing a result. Two consequences to state now
rather than discover later:

1. With 3 seeds the t multiplier is 4.303, so intervals are wide. A bar can now
   read FAIL where a point estimate would have read PASS. **A delta whose
   interval straddles zero prints INCONCLUSIVE**, which is reported as a
   not-pass but is a different fact from a measured reversal, and the printed
   delta +/- CI says which.
2. If a bar comes back INCONCLUSIVE, the honest response is MORE SEEDS, not a
   looser bar. Seed count is a sampling limit, not a statistic-choice problem.

**One `np.mean` deliberately survives**, in `_stability_from_saved`: it
averages assembly overlap over the TRANSITIONS of a single presentation of a
single seed. It forms that seed's value, which the ensembles then aggregate.
Putting a confidence interval over transitions inside one brain would be a
different and wrong claim -- the "mean over CONDITIONS rather than over seeds"
case the ratchet's own advice names.

**Verification.** Both ratchets pass. The changed aggregation was executed
end-to-end against synthetic per-cell results -- every ensemble, paired delta,
verdict branch and the JSON dump -- so the edit is known to run without the
study having been run. The synthetic numbers are meaningless and were discarded.
