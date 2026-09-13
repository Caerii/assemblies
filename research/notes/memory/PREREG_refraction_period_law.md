# Registration: is the relocation period the clip arithmetic, or a coincidence at one operating point?

> **Status (2026-09-13): registered, not yet run.** Bars fixed before the
> instrument exists.

## Why

`PREREG_refraction_convergence.md` Amendment 2 measured the relocation period
of a refracted recurrent assembly at 41.52 rounds against a predicted 40.93,
an error of 1.4%, with the first relocation landing on round 42 on all forty
brains of two disjoint seed blocks. The prediction is

    period(w_max, beta) = ln(w_max) / ln(1 + beta) + (1 - 1 / w_max) / beta

whose two terms are the climb to the weight clip and the rounds the refraction
bias then needs to erode the member's margin once raw drive stops growing.

That is one operating point: `w_max = 20`, `beta = 0.10`. The entry's caveat
says so, and says the dependence on both parameters is predicted and not
measured. A single point cannot distinguish a law from a number that happened
to land. This registration measures the surface.

For small beta the expression is within half a percent of
`(ln w_max + 1) / beta`, so the law makes two qualitatively different
predictions that a sweep can separate: the period is **linear in 1/beta** and
**logarithmic in w_max**. If both hold, the formula is a design rule for how
long a rehearsed assembly keeps its address. If the period moves but not as
predicted, the clip account of relocation is wrong and the register entry's
mechanism sentence has to be withdrawn.

## What runs

`python -m research.runner refraction-period-law --tag UNIQUE`

Engine `hashed_assembly_memory`. One hashed area (n = 4000, k = 100, p = 0.5,
normalized initialization, no synaptic scaling) written from one stimulus with
recurrence, in one-round episodes so every round's winners are observed, as in
the convergence protocol. Seeds 42 to 61, twenty brains, paired across cells.
The event definition, the stability threshold and the formula are imported
from `research/experiments/refraction_convergence.py`, which owns them.

**Cells.** Five, chosen to sweep each parameter separately through the
measured point:

    w_max   beta    predicted period   rounds run
    20      0.20    21.2               240
    20      0.10    40.9               240   <- the measured point
    20      0.05    80.4               442
    5       0.10    24.9               240
    100     0.10    58.2               320

Rounds per cell are fixed here as `max(240, ceil(5.5 x predicted))` so every
cell can show at least five relocations; they are a declared function of the
prediction, not a quantity read off the data.

**Arms.** Two per cell. `refracted` at strength 0.5 beta, the arm that
relocates. `control` at strength 0, which is the true negative: its weights
clip too, so if relocation were the clip alone the control would relocate as
well. It must not.

**Retained per brain and cell.** The whole consecutive-round overlap curve;
its relocation events (start and length of each maximal run below 0.95);
spacings between successive event starts; the count of events after the
formation event; `stable_fraction` over rounds 20 to the end; and `fill`.

Smoke (`--smoke --seeds 1 2 3`) runs two cells at 60 rounds and is VOID.

## Bars

- **PL-1, the law holds in every cell.** In each of the five cells the mean
  spacing between relocation starts on the refracted arm is within 15% of
  `period(w_max, beta)`.
  PREDICTION: passes. At the measured point the error was 1.4% and the bar
  allows ten times that, because nothing outside that point has been measured.
- **PL-2, beta is inverse.** On the three `w_max = 20` cells the measured mean
  spacing is strictly decreasing in beta, and the ratio between the beta =
  0.05 and beta = 0.20 cells is at least 3.0 (the law predicts 3.8).
- **PL-3, w_max is logarithmic and weak.** On the three `beta = 0.10` cells the
  measured mean spacing is strictly increasing in w_max, and the ratio between
  the `w_max = 100` and `w_max = 5` cells is at most 3.0 (the law predicts
  2.34) despite w_max changing twentyfold.
- **PL-4, the levers are ordered.** The beta ratio of PL-2 exceeds the w_max
  ratio of PL-3, even though beta moves fourfold and w_max twentyfold. This is
  the bar that makes "beta is the lever and the ceiling is not" falsifiable.
- **PL-5, relocation is refraction, not the clip alone.** In every cell the
  control arm has no relocation event after formation on any brain, and
  `stable_fraction` at least 0.95 on every brain, while the refracted arm has
  at least three relocations on every brain. The control's weights clip on the
  same schedule; without a bias there is nothing to erode the margin.
  PREDICTION: passes. Failure would mean the clip alone moves an assembly and
  the two-term account is wrong.

A bar that fails is recorded with its numbers, and if PL-1 fails the register
entry's period sentence is amended to the operating point it was measured at.

## Part 2, registered here and run separately: does the tenure explain the episode penalty?

`PREREG_presentation_schedule.md` found that a refracted memory stores much
better in four short episodes than in one long one (0.195 against 0.846 at
M = 256, 16 rounds per item either way). The reading offered there, and not
adopted, is that a 16-round episode is a large fraction of one tenure, so the
assembly selected at the start of a long episode is one the accumulated bias
no longer selects by its end. At the measured point 16 rounds is 39% of a
tenure.

If that is the mechanism, the penalty is a function of `episode_rounds /
period` and not of `episode_rounds` itself, which is testable two ways.

- **PL-6, shorter episodes at a fixed period.** At `w_max = 20, beta = 0.10`,
  the refracted arm's rank-1 at M = 256 is non-decreasing across the episode
  structures 1 x 16, 2 x 8, 4 x 4 and 8 x 2, and the 1 x 16 arm is strictly
  worst on at least 18 of 20 brains.
- **PL-7, the same episode at a longer period.** At `w_max = 20, beta = 0.05`,
  where the period doubles and 16 rounds becomes 20% of a tenure, the penalty
  `massed minus single` at M = 256 is at most half the penalty measured at
  `beta = 0.10`.
  PREDICTION: uncertain. Lowering beta also changes capacity, so this bar
  compares a penalty, not a recall, and even so the two cells are not matched
  on load. If PL-7 fails while PL-6 passes, the tenure account survives as a
  within-cell statement only.

Part 2 needs the presentation-schedule instrument parameterised by `w_max` and
`beta` and is registered now so its bars precede that work.
