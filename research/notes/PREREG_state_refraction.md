# PREREG: does refracting the STATE area stop the induced-state collapse?

Registered before implementing or running. Follows the A3 measurement in
`af136d0`, which is the reason this exists.

## What is already measured

A3's organ has a refracted arc and an UNREFRACTED state. At toy scale, over 5
seeds, at the registered beta:

| statistic | value |
| --- | ---: |
| arc overlap, position 0 | 0.035 +/- 0.035 |
| state overlap, position 0 | **0.985 +/- 0.028** |
| arc overlap, position 1 | 0.975 +/- 0.022 |

The conjunction separates two prefixes cleanly and the state it writes does
not, so history does not survive one step. The state area has no self fiber,
so this is not [[recurrence-is-the-collapse-channel]]; setting beta to 0 on
arc -> state alone changes it, so it is hub formation on the FEED-FORWARD
fiber. Nothing in the organ opposes that, because refraction was only ever
applied to the arc.

## Why refraction is the candidate, and not a knob

The engine applies refraction as `all_inputs -= bias`, with bias accumulating
on whichever neurons just won, drive-proportionally. That is a
usage-balancing rule: it suppresses the winners-so-far in favour of everything
else. Hub formation is the failure it is shaped to oppose.

The same object appears in deep learning as the fix for VECTOR-QUANTIZED
CODEBOOK COLLAPSE, where all inputs map to one code and the standard remedies
(dead-code restarts, EMA usage statistics, commitment penalties) are all
usage-balancing. Our state collapse and VQ codebook collapse look like one
phenomenon reached from two directions. That is the reason to try refraction
here rather than tune beta: it is the mechanism the failure names, not the
parameter nearest to hand.

## The design, and the trap it is built around

LOW OVERLAP IS NOT THE GOAL. A state area that emitted a fresh random assembly
every step would score a perfect separation and carry no information at all --
the [[fake-perfect-probe-signatures]] shape. So every cell reports TWO
statistics and a cell must win BOTH:

* **separation** = 1 - mean state overlap across DIFFERENT prefixes at the same
  position. Collapse drives this to 0.
* **determinism** = mean state overlap for the SAME prefix run TWICE. A random
  state drives this to chance; a working state must hold it near 1.

Reads run inside `brain.probe()`, so no bias is charged while measuring and
determinism is a fair question ([[probe-isolation-required]]).

## Parameters

Inherited from A3 and FIXED: n=10000, k=200, p=0.05 ambient, organ_p=0.20,
beta=0.10, n_arc=10000, 3 train rounds per pair, seeds 42..51, the same
generator and vocabulary.

Reduced for cell cost, and declared: **n_train = 50** sentences rather than 200.
This is a mechanism experiment, not a rerun of A3.

Swept, as design axes rather than knobs:

* `state_refracted_strength` in {0.0, 0.02, 0.05, 0.10, 0.20}. 0.0 is the
  control and reproduces A3.
* `n_state` in {2000, 10000}. [[REFRACTION-NEEDS-LOAD]] puts refraction's
  operating window in load M*k/n, and the number of distinct states M is
  emergent and unknown in advance -- it is part of the question. The achieved
  load is MEASURED per cell and reported, and the whole grid is reported, not
  its best cell.

## Gate

**G0, runs first.** The control arm (strength 0.0) must reproduce the collapse
at the reduced corpus: separation upper bound < 0.2. If it does not, the
mechanism scale does not exhibit the thing being studied and the study stops
rather than measuring a rescue of a problem that is not there.

## Hypotheses

**R1 (primary).** Some refraction level clears BOTH bars: separation lower
bound > 0.5 and determinism lower bound > 0.9.
*Prediction: roughly even odds, leaning against.* Refraction opposes exactly
this failure, which is the case for. The case against is that the state area
is probably UNDER-LOADED -- refraction was measured inert below load ~0.2, and
an emergent state with few distinct values sits there.

**R2.** The control fails R1. *Prediction: PASSES*, by G0.

**R3 (mechanism, reported regardless).** Separation rises with achieved load
across the grid. This is the prediction that distinguishes "refraction fixed
it" from "refraction fixed it WHERE THE LAW SAYS IT CAN", and it is the reason
n_state is swept at all.

## Committed in advance

1. G0 first. No parameter changes after seeing results.
2. The full grid is reported, never its best cell.
3. Determinism is reported beside separation for every cell, including the
   cells that lose. A separation number alone is not a result here.
4. If R1 passes, the winning cell is re-run at A3's full corpus (n_train=200)
   before any claim leaves this note. A mechanism-scale result is a lead.
