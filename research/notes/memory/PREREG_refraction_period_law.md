# Registration: is the relocation period the clip arithmetic, or a coincidence at one operating point?

> **Status (2026-09-13): run; ALL FIVE BARS PASS.** The period follows the
> clip arithmetic across a fourfold range in beta and a twentyfold range in
> w_max, to 4.1% in the worst cell. Bars were fixed before the instrument
> existed. Part 2 is still unrun.

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

## Result (2026-09-13): the formula is a law across the surface, not a number that landed

Artifact
`research/results/runs/memory.refraction-period-law/period-law-20260913/results.json`
(seeds 42 to 61, twenty brains per cell, pinned worktree at 091ad241; smoke
`research/results/runs/memory.refraction-period-law/pl-smoke-20260913/results.json`,
VOID). Verdict PASS on PL-1 to PL-5.

    cell        w_max  beta   predicted  measured  error  relocations/brain  spacings
    w20b0.2     20     0.20   21.18      22.05     4.1%   10                 21 to 24
    w20b0.1     20     0.10   40.93      41.60     1.6%   5                  40 to 44
    w20b0.05    20     0.05   80.40      79.92     0.6%   5 or 6             78 to 85
    w5b0.1      5      0.10   24.89      25.06     0.7%   9                  23 to 28
    w100b0.1    100    0.10   58.22      58.98     1.3%   5                  57 to 61

**PL-1 PASS.** Every cell is within 4.1% of its prediction against a 15% bar,
and four of the five are within 1.6%. The spacings are tight: across 681
measured spacings there is exactly ONE below 10 rounds, a doublet in the
`w20b0.05` cell where an event reopens two rounds after closing, the same
structure the feedforward arm showed in `PREREG_refraction_convergence.md`.
Excluding it moves that cell's error from 0.6% to 0.4%; nothing else changes.

**PL-2 PASS.** Spacing falls strictly with beta, 79.92, 41.60, 22.05 at beta
0.05, 0.10, 0.20, and the 0.05-over-0.20 ratio is 3.625 against a bar of 3.0
and a predicted 3.80.

**PL-3 PASS.** Spacing rises strictly with w_max, 25.06, 41.60, 58.98 at
w_max 5, 20, 100, and the 100-over-5 ratio is **2.354 against a predicted
2.34**, which is the closest agreement in the run.

**PL-4 PASS.** The fourfold beta change moves the period by 3.625; the
twentyfold w_max change moves it by 2.354. The lever ordering is measured, not
asserted: beta is inverse and the weight ceiling is logarithmic, so buying
tenure with the ceiling costs twenty times the parameter change for two thirds
the effect.

**PL-5 PASS, and it is the bar that names the mechanism.** In every one of the
five cells the unrefracted control had **zero** relocations on **every** brain,
with `stable_fraction` exactly 1.000 throughout. The control's weights clip on
the same schedule as the refracted arm's, so the clip alone does not move an
assembly. What moves it is the bias continuing to charge after the clip has
stopped the raw drive from growing, which is the second term of the formula.
Refraction is the whole of the relocation.

### What this establishes

The relocation period of a refracted recurrent assembly is a closed-form
function of the weight arithmetic, verified on a surface rather than at a
point. For small beta it is within half a percent of `(ln w_max + 1) / beta`,
so the design rule is: **tenure is bought with beta and not with the weight
ceiling.**

### Scope

One area size (n = 4000, k = 100, p = 0.5), one refraction strength
(0.5 beta), one stimulus. The formula has no strength term and the run did not
vary strength, so whether the period is independent of s, as the algebra
suggests and as the 0.5-beta agreement is consistent with, is NOT measured. A
strength sweep is the obvious extension and is not claimed here.

## Correction and re-run (2026-09-13): the estimand is per brain, and the intervals change what can be said

The version-1 statistic pooled every relocation spacing of every brain into
one mean. The methodology ratchet refused it, correctly: pooling weights a
brain by how often it happened to relocate, and it carries no interval. The
estimand is now the mean of the per-brain means over brains, with a confidence
bound and the seeds as keys (protocol version 2, artifact
`research/results/runs/memory.refraction-period-law/period-law-v2-20260913/results.json`,
same seeds 42 to 61, pinned worktree at 1813fb19). All five bars pass again.

    cell        measured   95% interval      formula   error   formula in CI?
    w20b0.2     22.05      [22.02, 22.08]    21.18     4.1%    no
    w20b0.1     41.60      [41.54, 41.66]    40.93     1.6%    no
    w20b0.05    80.05      [78.66, 81.43]    80.40     0.4%    yes
    w5b0.1      25.06      [25.01, 25.10]    24.89     0.7%    no
    w100b0.1    58.98      [58.93, 59.03]    58.22     1.3%    no

Only the `w20b0.05` cell moved at all between the two statistics, 79.92 to
80.05, because it is the one cell where brains differ in how many times they
relocated. Every bar's verdict is unchanged.

**What the intervals add.** They are extraordinarily tight: 0.12 rounds wide
at the measured point, on twenty brains. That precision is itself a finding,
and it is the same determinism the convergence study saw when the first
relocation landed on round 42 on all forty brains. It also changes what can
honestly be claimed. In four of the five cells **the formula's prediction
falls OUTSIDE the measured interval**, by 1 to 4%. The law is an excellent
approximation and PL-1's 15% bar is met with room to spare, but it is not
exact, and the measurement is now sharp enough to say so. The residual is
positive in four cells and negative in one, so it is not a clean bias either.

**A post hoc observation, recorded as post hoc.** The first term counts rounds
to reach the clip, and rounds are integers: a weight that needs 31.43 rounds
of growth actually clips on round 32. Taking the ceiling of that term gives

    cell        measured   plain formula   ceiled formula
    w20b0.2     22.05      4.10%           1.38%
    w20b0.1     41.60      1.63%           0.24%
    w20b0.05    80.05      0.44%           1.18%
    w5b0.1      25.06      0.68%           0.22%
    w100b0.1    58.98      1.31%           0.14%

which improves four cells substantially and worsens the fifth, the one with
the wide interval. It still misses the interval in four of five. **This was
found by looking at the residual of a run that had already happened, so it is
a hypothesis and not a result**, and Amendment 1 tests it where it cannot have
been fitted.

## Amendment 1 (2026-09-13, registered before running): the discretised form, on cells the refinement has never seen

Three cells, sharing no coordinate with the original five, chosen so that the
two candidate forms are far enough apart to tell apart. The intervals above
are about 0.1 rounds wide, so a separation under about 0.3 rounds would decide
nothing:

    w_max   beta    plain    ceiled   gap (rounds)   gap as % of the period
    8       0.25    12.82    13.50    0.68           5.3%
    12      0.18    20.11    21.09    0.99           4.9%
    25      0.15    29.43    30.40    0.97           3.3%

> **Correction, before running.** The first three cells written here (50/0.15,
> 10/0.08, 200/0.10) were chosen from an arithmetic slip: their real gaps are
> 0.01, 0.08 and 0.41 rounds, so two of the three could not have separated the
> forms at all and the amendment would have been unable to fail. They are
> replaced above. No data existed under either set.

Rounds per cell as before, `max(240, ceil(5.5 x plain))`. Same twenty seeds,
same arms including the unrefracted control.

- **PD-1, the discretised form is closer.** In at least 2 of the 3 cells the
  ceiled form's relative error is smaller than the plain form's.
  PREDICTION: passes. It was smaller in four of five cells above, and the
  mechanism is a rounding argument rather than a fitted parameter.
- **PD-2, and it is closer by enough to matter.** The mean absolute relative
  error of the ceiled form across the three cells is below 1.0%, and below the
  plain form's.
- **PD-3, neither form is exact.** Reported, not barred: whether either
  prediction falls inside the 95% interval in each cell. The expectation from
  above is that neither usually does, and saying so keeps the entry from
  claiming an exactness the intervals refuse.
- **PD-4, the control still never relocates.** Zero relocations on every brain
  of every new cell, as PL-5.

If PD-1 fails, the discretisation is dropped and the entry keeps the plain
form with its measured 1 to 4% residual stated. If it passes, the entry states
the ceiled form as the better approximation and still records that neither is
exact.

## Amendment 1 result (2026-09-13): the discretised form is confirmed on cells it was never fitted on

Artifact
`research/results/runs/memory.refraction-period-law/period-law-amendment2-20260913/results.json`
(seeds 42 to 61, pinned worktree at ad8c94dc). PL-1, PL-5, PD-1 and PD-2 all
pass. PL-2, PL-3 and PL-4 name cells this run does not contain and are omitted
rather than failed, which is a fix to the instrument recorded below.

    cell        measured   95% interval      plain    err     ceiled   err
    w8b0.25     13.57      [13.55, 13.59]    12.82    5.86%   13.50    0.52%
    w12b0.18    20.83      [20.80, 20.86]    20.11    3.59%   21.09    1.26%
    w25b0.15    30.23      [30.18, 30.27]    29.43    2.71%   30.40    0.56%

**PD-1 PASS, 3 of 3.** The discretised form is closer in every cell, not two
of three.

**PD-2 PASS.** Mean relative error falls from **4.05% to 0.78%**, a fivefold
reduction, on cells chosen before the run and sharing no coordinate with the
five the refinement was noticed on. That is the test the observation needed:
it was found in a residual and it held where it could not have been fitted.

**PD-3, reported as registered.** Neither form lands inside a 95% interval in
any of the three cells. The intervals are 0.02 to 0.09 rounds wide, so this
says the measurement is sharper than either approximation and not that either
is wrong. The honest statement is that the ceiled form predicts the period to
under one per cent and the plain form to about four, and that the residual of
the better one is still resolvable.

**PD-4 PASS.** Zero control relocations on every brain of every new cell, as
in all five original cells. Across eight cells now, the unrefracted control
has never relocated once.

### What the entry should say

The relocation period is `ceil(ln(w_max) / ln(1 + beta)) + (1 - 1/w_max)/beta`
to under one per cent, across eight cells spanning beta 0.05 to 0.25 and w_max
5 to 100. The ceiling is not a fitted correction: the first term counts rounds
to the clip and a weight needing 31.43 rounds of growth clips on round 32.
The design consequence is unchanged, because the ceiling moves the period by
at most one round: tenure is bought with beta and not with the weight ceiling.

### An instrument fix, recorded

The first amendment run reported PL-2, PL-3 and PL-4 as FAILED when its cell
set simply does not contain the cells those bars name. A missing test is not a
failed one. Bars naming specific cells are now evaluated only when those cells
are present, the omission is recorded in the run, and PL-1 and PL-5 still
quantify over whatever cells ran so a present-but-unmeasured cell fails them.
The superseded first run is not retained; its numbers are identical because
the protocol is deterministic on fixed seeds, and only the bar bookkeeping
changed.

## Amendment 2 (2026-09-13, registered before running): does the period depend on the refraction STRENGTH?

Every cell so far ran at one refraction strength, `s = 0.5 beta`. The formula
carries no strength term, and the entry records that as an unmeasured scope
limit. It is worse than unmeasured: the two candidate forms **coincide exactly
at the strength we happened to use.**

The second term counts the rounds the bias needs to erode a clipped member's
margin. Written as `(1 - 1/w_max) / beta` it says the erosion rate is set by
beta. But the mechanism is `bias += s x raw` at every win, so the rate ought to
be set by `s`, which gives `(1 - 1/w_max) / 2s`. At `s = 0.5 beta` those are
the same number, and every measurement in this registration was taken there.

This is the repository's own recorded lesson from the wrong retraction: when
two forms agree everywhere you have looked, a test they both pass verifies
neither. The second term is 9.5 of 41.5 rounds at the measured point, about
23% of the period, and the intervals are a tenth of a round wide, so a
strength sweep separates them decisively.

### What runs

Four arms at fixed `w_max = 20, beta = 0.10`, differing only in `s / beta`,
281 rounds each (5.5 periods at the slowest candidate). Twenty seeds, plus the
unrefracted control as before.

    s/beta   s        form A (no strength term)   form B (erosion at 2s)   gap
    0.250    0.0250   41.50                        51.00                   22.9%
    0.375    0.0375   41.50                        44.67                    7.6%
    0.500    0.0500   41.50                        41.50                    0.0%
    0.625    0.0625   41.50                        39.60                    4.6%

Strengths above about 0.7 beta are excluded by construction: the convergence
study found the area churns there with no period to measure.

### Bars

- **ST-1, the period is independent of strength.** The four measured periods
  all lie within 5% of their mean.
  PREDICTION: uncertain. This is form A and it is what the register currently
  implies by carrying no strength term.
- **ST-2, the period follows 1/2s.** Each arm's measured period is within 5%
  of `ceil(ln(w_max)/ln(1+beta)) + (1 - 1/w_max) / 2s`.
  PREDICTION: uncertain. This is form B.
- **ST-3, the sweep decides.** Exactly one of ST-1 and ST-2 passes. If both
  fail, neither closed form survives and the second term is something else,
  which is reported as measured values with no adopted form. If both pass the
  sweep was too narrow to separate them and the amendment is void.
- **ST-4, every arm is in the relocating regime.** At least three relocations
  on every brain of every arm, and zero on every control brain. An arm that
  churns has no period and cannot be scored; its bar fails rather than being
  skipped.

Whichever form survives, the register entry's period sentence gains an
explicit strength condition, because the present sentence has none.

## Amendment 2 result (2026-09-13): the period depends STRONGLY on strength, and NEITHER closed form survives

Artifact
`research/results/runs/memory.refraction-period-law/period-law-strength-20260913/results.json`
(four strengths at w_max = 20, beta = 0.10, seeds 42 to 61, pinned worktree at
0b5013a3). ST-4 passes; **ST-1, ST-2 and ST-3 all fail.**

    s/beta   measured   95% interval      form A    err      form B    err      relocations
    0.250    59.49      [59.46, 59.51]    41.50     43.34%   51.00     16.64%   4
    0.375    47.15      [47.10, 47.20]    41.50     13.61%   44.67      5.56%   5
    0.500    41.60      [41.54, 41.66]    41.50      0.24%   41.50      0.24%   5
    0.625    40.78      [38.98, 42.58]    41.50      1.73%   39.60      2.98%   5 to 7

**ST-1 FAILS decisively.** The four periods spread 25.9% about their mean
against a 5% bar, falling monotonically with strength: 59.49, 47.15, 41.60,
40.78. The period is not independent of the refraction strength. **The formula
as the register states it, with no strength term, is only correct at the
strength every earlier cell was measured at.**

**ST-2 FAILS too.** The `1/2s` form is closer than the flat form in three of
the four arms and is still wrong by 16.6% at the slowest one. Reading the
erosion term alone (measured period minus the 32-round climb) gives 27.49,
15.15, 9.60 and 8.78 rounds against the form's 19.00, 12.67, 9.50 and 7.60.
The shape is right in direction and wrong in magnitude.

**ST-3 FAILS, which the registration anticipated and named.** Both forms fail,
so neither closed form survives and the second term is reported as measured
values with no adopted form. No replacement is fitted here: a form chosen to
match these four points would be exactly the post hoc move that Amendment 1
was written to avoid.

### What this costs and what it does not

The period law is unchanged where it was measured. At `s = 0.5 beta` the
discretised form predicts every one of the eight (w_max, beta) cells to under
one per cent, and that strength is not an arbitrary choice: it is the adopted
operating point of [[REFRACTION-ANTI-MERGING]]. What the register may no
longer say is that the period is a function of `w_max` and `beta` alone. It
gains an explicit condition.

A dissociation worth recording. The register already says refraction strength
is a SWITCH for capacity, with one plateau across 0.3 to 0.6 beta. Tenure is
not flat there at all: between `s = 0.375 beta` and `s = 0.5 beta`, both
inside that plateau, the period moves 13%. **Capacity and tenure respond
differently to the same knob**, so a strength chosen for capacity is not
thereby a strength chosen for how long an address survives.

Two instrument notes. The `0.625` arm has by far the widest interval, 3.6
rounds against about 0.05 elsewhere, and 5 to 7 relocations per brain rather
than an identical count: it sits near the transition where the convergence
study found the area begins to churn, so its point is the least trustworthy of
the four. And PD-1 and PD-2 were evaluated on these strength cells in this
run, where both strength-free forms are wrong by construction; the bars are
now scoped to default-strength cells, and their verdicts here are void rather
than informative.

### What would settle the form

The erosion term is the margin at the clip divided by the rate it is spent.
Both depend on strength: a smaller `s` lets net drive grow further before the
clip, giving a bigger margin, and then spends it more slowly. That predicts a
term in `(beta - s)/s` rather than in `1/s`, which the four points do not
cleanly fit either. Settling it needs the margin measured directly rather than
inferred from the period, which is a different instrument and is not attempted
here.

## Amendment 3 (2026-09-13, registered before running): Part 2, with strength as the tenure knob instead of beta

Part 2 above asks whether the schedule study's episode penalty is the tenure.
Its PL-7 proposed halving beta to double the period, and admitted the
confound in the same breath: beta also changes capacity, so the two cells
would not be matched on load.

Amendment 2 supplies a better knob. Strength moves the tenure a long way,
59.49, 47.15 and 41.60 rounds at `s/beta` of 0.25, 0.375 and 0.5, and the
register holds that strength is a capacity SWITCH with one plateau across 0.3
to 0.6 beta. So 0.375 and 0.5 differ in tenure by 13% while sitting inside one
capacity plateau, which is the controlled comparison PL-7 could not make.
Expressed as what a 16-round episode costs:

    s/beta   tenure   16 rounds as a fraction of one tenure   inside the capacity plateau
    0.250    59.49    26.9%                                   no, below it
    0.375    47.15    33.9%                                   yes
    0.500    41.60    38.5%                                   yes

PL-7 is withdrawn and replaced by PL-8 and PL-9. PL-6 stands as written.

### What runs

`python -m research.runner episode-tenure --tag UNIQUE`. All arms are the
refracted memory at `w_max = 20, beta = 0.10`, massed, read with the bias
masked, 16 rounds per item, checkpoints M = 64, 128, 256 where the penalty was
seen. Seeds 42 to 61. Eight arms: the four episode structures 1x16, 2x8, 4x4
and 8x2 at `s = 0.5 beta`, and the two extremes 1x16 and 8x2 at `s = 0.375
beta` and `s = 0.25 beta`.

The write primitive is `ScheduledMemory` imported from the schedule study,
which owns it, and whose single-episode write is bit-identical to
`AssemblyMemory.store`.

- **PL-6, shorter episodes help at a fixed tenure.** At `s = 0.5 beta` and
  M = 256, mean rank-1 is non-decreasing across 1x16, 2x8, 4x4, 8x2, and the
  1x16 arm is strictly worst on at least 18 of 20 brains.
  PREDICTION: passes. The schedule study already saw 1x16 against 4x4.
- **PL-8, the penalty tracks the episode's share of a tenure.** Define the
  penalty as `rank1(8x2) - rank1(1x16)` at M = 256. It is strictly larger at
  `s = 0.5 beta` than at `s = 0.375 beta`, and strictly larger at `s = 0.375
  beta` than at `s = 0.25 beta`.
  PREDICTION: uncertain, and this is the whole test. A 16-round episode is
  38.5%, 33.9% and 26.9% of a tenure at those three strengths.
- **PL-9, and it is not capacity in disguise.** Between `s = 0.5 beta` and
  `s = 0.375 beta`, both inside the register's capacity plateau, the 8x2 arm's
  mean rank-1 at M = 256 differs by less than 0.05 while the penalty differs
  by more. If the 8x2 arms differ by more than that, the two strengths are not
  matched on capacity after all and PL-8 cannot be read as a tenure effect.
- **PL-10, instrument.** Every arm spends exactly 16 rounds per item and
  presents every item exactly its registered number of episodes.

If PL-8 fails while PL-6 passes, the episode penalty is real and the tenure
does not explain it, and the reading offered in the schedule study is
withdrawn rather than amended.

## Amendment 3 result (2026-09-13): the tenure does NOT explain the episode penalty, and the direction is reversed

Artifact
`research/results/runs/memory.episode-tenure/episode-tenure-20260913/results.json`
(eight arms, seeds 42 to 61, pinned worktree at c57939b3; smoke
`research/results/runs/memory.episode-tenure/et-smoke-20260913/results.json`,
VOID). PL-9 and PL-10 pass; **PL-6 and PL-8 fail.**

    arm             episode rounds   share of a tenure   M=64    M=128   M=256
    e1x16-s0.5      16               38.5%               0.762   0.433   0.185
    e2x8-s0.5        8               19.2%               0.893   0.517   0.180
    e4x4-s0.5        4                9.6%               0.990   0.954   0.856
    e8x2-s0.5        2                4.8%               0.982   0.980   0.987
    e1x16-s0.375    16               33.9%               0.716   0.409   0.162
    e8x2-s0.375      2                4.2%               0.981   0.982   0.995
    e1x16-s0.25     16               26.9%               0.680   0.379   0.136
    e8x2-s0.25       2                3.4%               0.998   0.997   1.000

**PL-8 FAILS, and the sign is backwards.** The penalty, `rank1(8x2) -
rank1(1x16)` at M = 256, against the episode's share of one tenure:

    s/beta   share    penalty
    0.500    38.5%    +0.8016  [+0.7768, +0.8263]
    0.375    33.9%    +0.8328  [+0.8154, +0.8503]
    0.250    26.9%    +0.8641  [+0.8524, +0.8757]

The tenure account predicts the penalty SHRINKS as the episode becomes a
smaller share of one. It grows, monotonically, and the intervals for the
0.5 and 0.25 arms do not overlap. **The reading offered in
`PREREG_presentation_schedule.md`, that the episode penalty is the assembly's
tenure, is WITHDRAWN**, as Amendment 3 said it would be if this bar failed.

**PL-9 PASSES, which is what makes the refutation clean.** The two strengths
inside the register's capacity plateau have short arms differing by 0.0078
against a 0.05 bar, so they are matched on capacity and the contrast really is
a tenure contrast. The tenure moved 13% between them and the penalty moved the
wrong way.

**PL-6 FAILS too, and says the effect is a threshold rather than a gradient.**
The ladder at `s = 0.5 beta` reads 0.185, 0.180, 0.856, 0.987 for episodes of
16, 8, 4 and 2 rounds. It is not monotone: the 8-round episode is marginally
worse than the 16-round one. What the numbers show is a step between 8 and 4
rounds per episode, with the two long structures alike and the two short
structures alike, not a smooth function of anything.

### A better candidate, stated as post hoc and not adopted

An episode begins with an inhibit, so its FIRST round has no recurrent
contribution from the area and is driven by the stimulus alone; its remaining
rounds have both. An item trained in `E` episodes therefore spends `E` of its
16 rounds anchored on the stimulus without recurrence, and `16 - E` with it.
The four structures give 1, 2, 4 and 8 such rounds.

That is the quantity [[CAP-ANCHOR-RATIO]] already identifies as setting
capacity at formation: the ratio of the stimulus anchor to the trained
recurrent pull. It predicts the effect depends on the NUMBER of episodes
rather than on their length or on any relation to the tenure, and the step
between 2 and 4 episodes is where that ratio doubles from an eighth to a
quarter.

It also fits the strength direction that defeated the tenure account: the long
arm degrades as strength falls (0.185, 0.162, 0.136) while the short arm does
not (0.987, 0.995, 1.000), and refraction is the force that keeps a
recurrence-dominated write from collapsing together.

Note that the pairwise overlap statistic is at or below chance in every arm
(0.79 to 0.95 times chance), so whatever the long episode is doing it is NOT
showing up as merged assemblies, which is a constraint on any account.

Testing this needs episodes held fixed while total rounds vary, which
separates the episode count from the episode length. It is not attempted here
and nothing above is adopted.
