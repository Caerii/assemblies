# Registration: autonomous chain recall on the refracted arc, and where the sequence-length limit actually lives

> **Status (2026-09-13): run; ALL FIVE BARS PASS on twenty fresh brains.** Bars below were
> informed by a four-seed exploratory probe on seeds 42 to 45 and are therefore
> tested on the FRESH block, seeds 62 to 81, which no probe has touched.

## Why

`PREREG_ordered_recall_reproduction.md` Amendment 1 established two things.
This repository's `ordered_recall` advances zero steps and is NOT in either
reference implementation; and what the reference actually builds sequences from
is a feedforward `RefractedArea` arc driven by `(symbol, state)`, which this
repository ported as the hashed refracted arc and measured at 2000 steps.

But those 2000 steps are EXTERNALLY DRIVEN: a symbol arrives every step and
carries the information. The limit the sequences paper reports, and that
`mitropolsky2025simulated` cites as the reason language needs hierarchy --
"it is difficult to maintain sequences of assemblies longer than a limit ...
ranges between 20 and 40" -- is about a chain recalled AUTONOMOUSLY, with no
information arriving. Neither this repository nor either reference had that.

**It is expressible on the mechanism that works.** Train a chain
`(q_i, tick) -> q_{i+1}` on a SINGLE constant symbol. The tick carries no
information, so every step's advance comes from the state through the arc. That
is autonomous ordered recall built on the reference's own construction.

It is also a maximally sharp test of [[ARC-CONJUNCT-EXPOSURE]], which says a
conjunction collapses onto its more-exposed conjunct: here the tick appears in
EVERY transition while each state appears in one, so the tick is the most
over-exposed conjunct the construction admits.

## What runs

`python -m research.runner autonomous-chain --tag UNIQUE`

Engine `hashed_arc_fsm`. One chain of `L + 1` states over one symbol,
`n_arc = 10000, k = 100, p = 0.2, beta = 0.10, w_max = 20`, refracted strength
`s = beta` as the reference runs it, gain table sized to the episode
(`max_potentiations = 96`; the default 4096 overflows float32 at this beta and
the engine says so). Seeds 62 to 81, twenty brains. Training is teacher-forced
exactly as the reference's `FSMNetwork.train` is.

Scored per brain: consecutive correct visits from the start, where visit `t` is
correct when the state read after step `t` is `q_{t+1}`.

Retained per brain and arm: the consecutive-correct count, the full visited
sequence, and the ARC assembly of every state (for the collapse measurement).

Smoke (`--smoke --seeds 1 2 3`) runs `L = 8` and is VOID.

## Bars

- **CL-1, it exceeds the reported band.** At `L = 32`, the top of the paper's
  20-to-40 range, every brain recalls all 32 visits correctly; and at
  `L = 128`, four times that, every brain recalls all 128.
  PREDICTION: passes (probe: 32/32 and 128/128 on 4 of 4 brains).
- **CL-2, refraction is necessary, and totally.** With `refracted_strength = 0`
  and everything else held, consecutive correct is 0 on every brain at
  `L = 32`.
  PREDICTION: passes (probe: 0 on 4 of 4). This is the mechanism-disabled null.
- **CL-3, the perfect score is reachable and not free.** At 5 and at 10
  presentations, consecutive correct is at most 5 on every brain; at 20 it is
  exact. A score that cannot be moved by undertraining would not be a
  measurement.
- **CL-4, the arc needs density.** At `p = 0.05` and `p = 0.02`, consecutive
  correct is at most 5 on every brain, against exact at `p = 0.2`.
- **CL-5, the mechanism is anti-swamping, measured not inferred.** With
  refraction off, the arc assemblies of distinct states COLLAPSE onto each
  other: mean pairwise overlap between the arcs of different states is at least
  0.5 with `strength = 0`, and at most 3x the chance level `k / n_arc` with
  `strength = beta`, on every brain.
  PREDICTION: uncertain. This is the bar that says WHY CL-2 happens, and if it
  fails while CL-2 passes then refraction is necessary for some other reason
  and the anti-swamping reading is withdrawn.

## What this does and does not settle about the paper's limit

If CL-1 passes, then in the reference's own construction, with states
teacher-forced onto disjoint blocks, autonomous chain recall is exact far past
the reported band. That localizes the reported limit: it is **not** in the arc
and not in the sequence mechanism.

It does NOT establish where the limit is. The obvious remaining candidate is
state FORMATION: the paper's chain forms its assemblies by projection, where
they can overlap and interfere, while ours are disjoint by construction. That
is the next study and it is not attempted here. Any claim that this repository
has refuted or explained the 20-to-40 limit before that study would be
unsupported, and is not made.

## Scope

One operating point, one chain topology (a simple path, no branching or
revisiting), teacher-forced states, and a single symbol. Chains that revisit a
state, or branch, are a different problem: the arc's refraction is per-neuron
and cumulative, so a revisited state's arc has already paid bias.

## Result (2026-09-13): autonomous recall is exact to 128, four times the reported band, and refraction is the whole of it

Artifact
`research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json`
(seeds 62 to 81, pinned worktree at 9b0be3bb; smoke
`research/results/runs/sequence.autonomous-chain/ac-smoke-20260913/results.json`,
VOID). Verdict PASS on CL-1 to CL-5.

    arm                  L     exact brains   consecutive correct   arc overlap
    L32                  32    20 / 20        32 on every brain     0.0000
    L128                 128   20 / 20        128 on every brain    0.0108
    L32-no-refraction    32     0 / 20        0 on every brain      1.0000
    L32-pres5            32     0 / 20        1 on every brain      0.0000
    L32-pres10           32     0 / 20        2 to 4                0.0000
    L32-p0.05            32     0 / 20        2 to 3                0.0000
    L32-p0.02            32     0 / 20        1 to 2                0.0001

Chance arc overlap is `k / n_arc = 0.01`.

**CL-1 PASS.** A chain driven only by a constant tick is recalled exactly, on
every brain, at length 32 and at length 128. The paper reports that sequences
of assemblies are difficult to maintain beyond 20 to 40.

**CL-2 PASS, and it is total.** With refraction off, not one brain takes a
single correct step. Not a degradation: zero.

**CL-5 PASS, and it says why, at the extreme.** The arc assemblies of distinct
states are **identical** without refraction, overlap 1.0000 on every brain, and
**disjoint** with it, 0.0000 at L = 32 and 0.0108 at L = 128 against a chance
level of 0.0100. Every state drives the same arc when refraction is off,
because the tick appears in all 32 transitions while each state appears in one,
so the conjunction collapses onto the more exposed conjunct exactly as
[[ARC-CONJUNCT-EXPOSURE]] says. This construction is the extreme case of that
entry, and it is the mechanism measured rather than inferred from the outcome.

**CL-3 and CL-4 PASS: the perfect score moves four ways.** Undertraining at 5
and at 10 presentations, and thinning the arc to p = 0.05 and p = 0.02, each
collapse it to a handful of steps while leaving the arcs disjoint. So the
failure modes are distinguishable: refraction off collapses the ARCS, while
undertraining and sparsity leave them separate and break the chain some other
way.

### What this settles, and what it does not

The reported limit is **not in the arc and not in the sequence mechanism**. In
the reference's own construction, with states teacher-forced onto disjoint
blocks, autonomous recall is exact at four times the top of the reported band
with no sign of degradation.

It does not say where the limit is. Our states are disjoint by construction and
the paper's are formed by projection, where they can overlap and interfere.
That difference is now the whole of the remaining question, and it is the next
study. Nothing here refutes or explains the 20-to-40 limit, and no such claim
is made.
