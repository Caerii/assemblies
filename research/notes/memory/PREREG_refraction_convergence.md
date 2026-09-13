# Registration: refraction strength against convergence of a recurrent assembly, twenty seeds

> **Status (2026-09-13): run on twenty brains; RC-1 and RC-3 pass, RC-2, RC-4
> and RC-5 fail as registered (Result below). Amendment 1 registers the
> version-2 instrument, which retains the whole consecutive curve, and new
> bars for what the marks showed: stability BETWEEN relocations.** Bars
> were fixed before the run. `REFRACTION-CANCELS-CONVERGENCE` rests on `seq_refraction_wander.py`,
> a post hoc diagnostic (16 brains, identities `42 + b`, one mutable
> procedure mixing recurrent, feedforward and scaling audits) with no run
> record. This registration splits its arms, gives them runner seed
> identities, and retains per-brain convergence so the register can check
> the mechanism-disabled contrast.

## Why

The entry says: at refraction strength s = beta the identity net drive =
drive - bias cancels the Hebbian convergence force on a recurrent
assembly, so the winners reshuffle every round (churn), while at s = 0.5
beta the assembly converges (relocating once when the clip binds) and a
feedforward area at s = beta holds because its input ranking is fixed.
The adopted numbers (converge at 0.5 and 0.7 beta, never at 0.8 and
above; feedforward late stability 0.993) are prose from the diagnostic.

## What runs

`python -m research.runner refraction-convergence --tag UNIQUE`

- Engine `hashed_assembly_memory`: one `HashedArea` (n = 4000, k = 100)
  written from one stimulus (k rows, p = 0.5, beta = 0.10, w_max = 20,
  normalized initialization) with recurrence from an `AreaFiber` of the
  same p, exactly the diagnostic's construction, for 240 one-round
  episodes so every round's winners are observed. The organ profile
  declares the inference schedule `training-trajectory`: the observation
  is the write itself; there is no recall.
- Seeds 42 to 61 (twenty brains), the runner's identities, paired across
  arms.
- Arms: `control` (unrefracted, recurrent); `feedforward` (refracted at
  s = beta, no recurrence); `recurrent` at s/beta in {0.5, 0.7, 0.8, 0.9,
  1.0}.
- Per brain and arm the run retains: the consecutive-round overlap curve;
  `late` (mean consecutive overlap over rounds 200 to 240); `conv` (the
  first round after which consecutive overlap stays at least 0.95, or -1);
  the overlap with the round-10 winners at rounds 10, 20, 30, 40, 60, 100,
  150, 200, 240; and `fill`.

Smoke (`--smoke --seeds 1 2 3`) runs 40 episodes on the control and the
0.5 and 1.0 arms and is VOID.

## Bars

Priors: the diagnostic's 16-brain reading (converge 16/16 at 0.5 and 0.7
with conv rounds 45 to 48; 0 of 16 at 0.8, 0.9, 0.95 and 1.0 with late
stability at most 0.22; control converges at round 4; feedforward late
0.993).

- **RC-1, the control converges.** `conv >= 0` and `late >= 0.95` on every
  brain of the control arm, with median `conv` at most 20.
- **RC-2, half beta converges.** On the 0.5 arm `late >= 0.95` on every
  brain and `conv <= 100` on at least 18 of 20.
- **RC-3, full beta churns.** On the 1.0 arm `late <= 0.5` on every brain
  and `conv = -1` on at least 18 of 20.
- **RC-4, feedforward holds.** On the feedforward arm `late >= 0.95` on
  every brain.
- **RC-5, the transition sits between 0.7 and 0.8.** The 0.7 arm converges
  (`late >= 0.95`) on at least 18 of 20 and the 0.8 arm on at most 2 of
  20. PREDICTION: uncertain at 0.7 (the entry records "at 0.7 beta most
  brains no longer converge" in one reading and "0.5, 0.7 converge" in
  another; this bar settles which).

Retained sensitivity for the register: `late`, the 0.5 arm against the
1.0 arm, every brain greater by at least 0.5.

A bar that fails is recorded with its numbers, and the entry's claim is
amended to what twenty brains support.

## Result (2026-09-13): the churn contrast stands; "converges and holds" does not

Artifact `research/results/runs/memory.refraction-convergence/refraction-convergence-20260913/results.json`
(engine `hashed_assembly_memory`, protocol `memory.refraction-convergence`
version 1, seeds 42 to 61 paired across the seven arms, 240 one-round
episodes, from the pinned worktree at 6efd3654; smoke
`research/results/runs/memory.refraction-convergence/rc-smoke-20260913/results.json`,
three seeds, 40 episodes, VOID).

    arm           converged   late (mean, min, max)     conv (per brain)                 fill
    control       20/20       1.000  1.000  1.000       3 or 4 on every brain            0.034
    feedforward    1/20       0.927  0.912  0.975       -1 on 8; 218 to 240 on 12        0.237
    s = 0.5 beta   0/20       0.919  0.904  0.932       217 to 220 on every brain        0.278
    s = 0.7 beta   1/20       0.820  0.632  0.966       -1 on 14; 230 to 240 on 6        0.911
    s = 0.8 beta   0/20       0.114  0.039  0.277       -1 on every brain                1.000
    s = 0.9 beta   0/20       0.008  0.005  0.010       -1 on every brain                1.000
    s = 1.0 beta   0/20       0.005  0.003  0.007       -1 on every brain                1.000

`late` is the mean consecutive overlap over rounds 200 to 240; `conv` the
first round after which consecutive overlap stays at or above 0.95.

* **RC-1 PASS.** The control converges on every brain at round 3 or 4.
* **RC-2 FAIL.** No brain of the 0.5 arm has `late` at or above 0.95
  (0.904 to 0.932) and none converges by round 100 (`conv` 217 to 220).
* **RC-3 PASS.** At s = beta every brain churns (`late` at most 0.007,
  `conv` -1 on 20 of 20).
* **RC-4 FAIL.** The feedforward arm's `late` is 0.912 to 0.975; one brain
  clears 0.95.
* **RC-5 FAIL.** The 0.7 arm converges on 1 of 20 (bar 18); the 0.8 arm
  on 0 of 20 (bar at most 2: that half holds). The transition lies between
  0.5 and 0.7 beta, not between 0.7 and 0.8.

What the retained marks say about the 0.5 arm and the feedforward arm.
Consecutive overlap at every one of the nine marks is 1.000 on every brain
of the 0.5 arm, while the overlap with the round-10 winners falls from 1.000
at round 40 to 0.000 at round 60 and stays there: the assembly is
wholesale ELSEWHERE by round 60, yet identical from one round to the next
at every mark. The `late` deficit (0.08 over 40 rounds) and the `conv`
values clustered at 217 to 220 on all twenty brains are what a relocation
of two to four rounds recurring at a fixed period looks like when sampled
at marks that happen to miss the events. The feedforward arm shows the
same signature earlier (overlap with round 10 is 0.006 at round 40) with
less regular timing (consecutive overlap at marks 30, 60 and 150 is 0.86
to 0.90 in the mean: some brains are mid-event at those rounds). The
reading, stated as a hypothesis because version 1 kept only the marks: at
s at or below 0.5 beta, and in a feedforward area at s = beta, the
assembly is stable BETWEEN relocations that recur each time the clip binds
on the current occupants (the mechanism recorded for the arc in
`PREREG_s5_cliff_anatomy.md`, Addendum 5); above about 0.7 beta the area
churns every round, a different regime. "Converges, relocates once when
the clip binds and holds" and "feedforward holds" were readings of a
shorter window.

Retained sensitivity for the register, as registered: `late`, the 0.5 arm
against the 1.0 arm, every brain greater by at least 0.5 (observed 0.897
to 0.929). PASS.

## Amendment 1 (2026-09-13, registered before running version 2): stability between relocations

Version 1 retained the consecutive curve at nine marks only, so the reading
above cannot be checked on it. Version 2 of the instrument
(`refraction_convergence.py`, protocol version 2, same arms, seeds,
sizes and episodes) retains per brain the whole consecutive-round overlap
curve (239 values), its relocation events (maximal runs of rounds with
consecutive overlap below 0.95: start round and length), the count of
events, the spacings between successive starts, the `stable_fraction`
(rounds 20 to 240 with consecutive overlap at or above 0.95) and the
`relocated_fraction` (rounds with overlap below 0.5). RC-1 and RC-3 are
kept; RC-2, RC-4 and RC-5 are reported as superseded version-1 bars. The
new bars are derived from version 1's marks (first relocation of the 0.5
arm between rounds 40 and 60, `late` deficit of about three rounds per
forty, last event ending at 217 to 220 on every brain: about six events
at a spacing near 35) and are therefore not independent of that run; the
version-2 run is their test on the full curve.

- **RC-6, half beta is stable between relocations.** On the 0.5 arm
  `stable_fraction` at least 0.85 on every brain.
- **RC-7, the relocations are periodic.** On the 0.5 arm every brain has
  between 4 and 9 events, the first starting between rounds 30 and 70, and
  every spacing between successive starts between 20 and 50 rounds.
- **RC-8, the feedforward area relocates too.** On the feedforward arm
  every brain has at least 3 events and `stable_fraction` at least 0.8.
- **RC-9, churn is not relocation.** On the 1.0 arm `stable_fraction` at
  most 0.1 on every brain.
- **RC-10, the transition is below 0.8 beta.** On the 0.8 arm
  `stable_fraction` at most 0.4 on every brain. (The 0.7 arm is reported,
  not barred: version 1 puts it in the transition, `late` 0.63 to 0.97.)

Run: `python -m research.runner refraction-convergence --tag UNIQUE`
(smoke `--smoke --seeds 1 2 3`, VOID). A bar that fails is recorded with
its numbers.

## Amendment 1 result (2026-09-13): the periodic structure is there, and three of the five new bars FAILED

Artifact
`research/results/runs/memory.refraction-convergence/refraction-convergence-v2-20260913/results.json`
(protocol version 2, seeds 42 to 61, the same twenty brains as the version-1
run, pinned worktree at 35132c47; smoke
`research/results/runs/memory.refraction-convergence/rc-v2-smoke-20260913/results.json`,
VOID). Version 2 changes no parameter, so every version-1 quantity is
reproduced exactly: control late 1.000, feedforward 0.927, 0.5 beta 0.919,
0.7 beta 0.820, 0.8 beta 0.114, 0.9 beta 0.008, 1.0 beta 0.005.

    bar                                                     verdict
    RC-1  control converges, median conv <= 20              PASS
    RC-3  s = beta churns                                   PASS
    RC-6  0.5 arm stable_fraction >= 0.85 every brain       FAIL
    RC-7  0.5 arm 4..9 events, first start 30..70,          FAIL
          every spacing 20..50, every brain
    RC-8  feedforward n_events >= 3 and stable_fraction     FAIL
          >= 0.8 every brain
    RC-9  s = beta stable_fraction <= 0.1 every brain       PASS
    RC-10 s = 0.8 beta stable_fraction <= 0.4 every brain   PASS

**What the full curve shows.** The event structure of the 0.5 arm is more
regular than the reading predicted, not less:

    quantity (0.5 arm, twenty brains)                value
    number of events                                 6 on every brain
    first event start                                round 2 on every brain
    first event length                               3 or 4 rounds
    SECOND event start                               round 42 on every brain
    spacings between successive starts               40 to 44, mean 41.6
    event lengths excluding the first                6 to 10, mean 8.02
    stable_fraction (rounds 20 to 240)               0.805 to 0.828

So the assembly forms, then relocates five times on a fixed period of about
42 rounds, each relocation taking about eight rounds, and holds its winners
from round to round in between. The control's event list is the giveaway for
the first entry: the control has exactly one event, at round 2, of length 1
or 2. **Event one is the initial FORMATION, not a relocation.**

**Why each bar failed, stated as an instrument fault and not a mechanism
one.** RC-7 required the first event to start between rounds 30 and 70; on
every brain it starts at round 2, because formation registers as an event.
Its other two clauses hold (6 events is inside 4 to 9; every spacing 40 to 44
is inside 20 to 50). RC-6 required `stable_fraction >= 0.85`; five events of
about eight rounds is about 40 unstable rounds in the 221 counted, which is
0.82, so the bar was unreachable given an event length the marks had put at
two to four. RC-8 failed only on its `stable_fraction >= 0.8` clause, on 4 of
20 feedforward brains (range 0.760 to 0.887); its `n_events >= 3` clause holds
with a minimum of 10.

These are mis-set thresholds and a missing definition, and saying so does not
convert a failed bar into a passed one. RC-6, RC-7 and RC-8 are recorded
FAILED and are not amended in place. The structure above is a measurement made
after the bars were set on the same twenty brains, so it is post hoc, and any
bar written on it is a new prediction that needs data it has not seen.

**The period and the clip.** The measured period, 41.6 rounds with the first
relocation at exactly 42 on every brain, matches the saturation arithmetic
this registration's entry already carries:
`ln(w_max) / ln(1 + beta) + (1 - 1 / w_max) / beta = ln(20)/ln(1.1) + 0.95/0.1
= 40.93` rounds. The 16-brain diagnostic saw it once, as "a transient
re-ranking at rounds 44-48"; on the full curve it is not a transient but the
period of a repeating process. That the second event starts at 42 on all
twenty brains and not at a spread of rounds is what a weight-arithmetic
deadline looks like: the clip time depends on the potentiation schedule, which
is identical across brains, not on the random connectome.

Also measured, and reported rather than barred: the 0.7 arm is a third regime.
Its `stable_fraction` is 0.253 to 0.312 with a `relocated_fraction` of 0.602,
and every brain has one event lasting 94 to 119 rounds, so it churns for
roughly half the run and restabilises, rather than either holding between
short relocations or churning throughout. The 0.8, 0.9 and 1.0 arms each show
exactly one event covering all 239 comparisons, `stable_fraction` 0.000.

## Amendment 2 (2026-09-13, registered before running): the corrected bars, on a FRESH seed block

The quantities above were measured on seeds 42 to 61 after Amendment 1's bars
were set, so they cannot be confirmed on those brains. Amendment 2 states them
as predictions and tests them on the fresh block **seeds 62 to 81**, which no
run of this protocol has used. Version 3 of the instrument changes no retained
quantity -- a version-2 and a version-3 artifact are directly comparable -- and
only replaces which bars are evaluated. It adds `formation_round` and
`clip_period` to the recorded parameters, and accepts either seed block.

RC-1, RC-3, RC-9 and RC-10 carry over unchanged. RC-2, RC-4, RC-5, RC-6, RC-7
and RC-8 are reported on every run as superseded bars, evaluated and printed
but not counted in the verdict, so their failures stay visible.

- **RC-11, the first event is the formation.** On the control and the 0.5 arm
  the first event starts at round 2 on every brain and lasts at most 5 rounds.
  PREDICTION: passes. This is the definition that broke RC-7.
- **RC-12, the relocation period is fixed.** On the 0.5 arm, excluding the
  formation event, every brain has exactly 5 relocations, the first starting
  between rounds 38 and 46, and every spacing between successive event starts
  lies between 35 and 50 rounds.
- **RC-13, the period is the clip arithmetic.** The mean spacing on the 0.5
  arm is within 10% of `ln(w_max) / ln(1 + beta) + (1 - 1 / w_max) / beta`
  (40.93 rounds). PREDICTION: passes at 41.6 on the first block; this is the
  bar that makes the mechanism claim falsifiable.
- **RC-14, stability between relocations.** On the 0.5 arm `stable_fraction`
  is at least 0.78 on every brain and the mean relocation length is at most
  12 rounds.
- **RC-15, the feedforward area relocates oftener and never settles.** On the
  feedforward arm every brain has at least 8 events and `stable_fraction`
  between 0.70 and 0.92.
- **RC-16, the 0.7 arm is neither regime.** On the 0.7 arm `stable_fraction`
  is between 0.15 and 0.40 on every brain and the longest single event lasts
  at least 60 rounds. PREDICTION: uncertain. This arm sits in the transition
  and is the one most likely to move between seed blocks.

Run: `python -m research.runner refraction-convergence --seeds 62 63 ... 81
--tag UNIQUE`. A bar that fails is recorded with its numbers.

## Amendment 2 result (2026-09-13): all ten bars pass on the fresh block, and the period is the clip arithmetic

Artifact
`research/results/runs/memory.refraction-convergence/refraction-convergence-fresh-20260913/results.json`
(protocol version 3, seeds 62 to 81, pinned worktree at dd342620). Verdict
PASS: RC-1, RC-3, RC-9, RC-10 and all six Amendment 2 bars RC-11 to RC-16.
The six superseded bars are evaluated and printed on this run too, and fail
exactly as they did on the first block.

    quantity (0.5 arm)               seeds 42..61      seeds 62..81 (fresh)
    relocations per brain            5 on every brain  5 on every brain
    first relocation                 round 42          round 42
    spacings                         40 to 44          40 to 43
    mean spacing                     41.60             41.52
    relocation length                6 to 10 (8.02)    6 to 10 (7.98)
    stable_fraction                  0.805 to 0.828    0.801 to 0.837

    other arms (fresh block)         n_events          stable_fraction
    control                          1                 1.000 on every brain
    feedforward at s = beta          11 to 32          0.796 to 0.860
    s = 0.7 beta                     4 to 8            0.249 to 0.290
    s = 0.8, 0.9, 1.0 beta           1                 0.000 on every brain

**RC-13, the bar that carries the mechanism.** The predicted period is

    ln(w_max) / ln(1 + beta) + (1 - 1 / w_max) / beta = 40.93 rounds

and the measured mean spacing on the fresh block is 41.52, an error of
**1.44%** against a registered tolerance of 10%. On the first block it was
41.60, an error of 1.6%. The first relocation lands on round 42 on all forty
brains of both blocks, with no spread at all: the deadline is set by the
potentiation schedule, which is identical across brains, not by the random
connectome each brain draws.

This is a closed-form quantity predicted from the weight arithmetic and then
matched by an independent measurement, which is the second such number in
this repository (the first is the clip presentation c* of
`PREREG_s5_cliff_anatomy.md`, Addendum 5). Both are the same physics: a
refracted assembly is stable only until its own weights hit the clip, after
which the raw drive stops growing, the bias keeps charging, and the
best-connected members fall first.

**What the entry now says.** A refracted recurrent area below the transition
does not converge and hold. It forms, then relocates on a fixed period equal
to the clip arithmetic, holding its winners from round to round in between,
and a feedforward area at s = beta does the same thing more often rather than
holding. Above about 0.8 beta there is no stability to relocate from: one
event covers the whole run. The 0.7 arm is a third regime, churning for
roughly half the run in one event of 97 to 120 rounds and restabilising.

**Scope.** One operating point (n = 4000, k = 100, p = 0.5, beta = 0.10,
w_max = 20), one stimulus, 240 rounds. The period's dependence on w_max and
beta is predicted by the formula and is NOT measured here: a sweep over w_max
and beta that moves the period as the arithmetic says is the obvious next
test and is not claimed.
