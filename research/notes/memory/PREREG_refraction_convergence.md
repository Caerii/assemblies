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
