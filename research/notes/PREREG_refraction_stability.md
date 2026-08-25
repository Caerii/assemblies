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
