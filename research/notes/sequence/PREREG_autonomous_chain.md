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

## Amendment 1 (2026-09-13, diagnostic + registration): where the chain DOES break, and an instrument confound found on the way

The study above stops at L = 128 and finds no degradation. Pushing further
finds the break, and finding it required fixing a confound in my own probe.

**The confound.** `HashedArcFSM` sizes the state area as
`max(n_arc, len(states) * k)`, so varying the chain length silently varies
`n_state` and redraws the whole connectome. A first sweep looked non-monotone
because of it -- exact at 512 but broken at 384 for the same arc. Holding
`n_state` FIXED at 64000 makes it monotone. Any chain-length sweep must pin
`n_state`, and the registered study below does.

**A second constraint the engine enforces.** `topk_select` packs a 16-bit
index, so `n_state <= 65536`, which caps the chain at `65536 / k - 1` (639 at
k = 100). The engine raises with that message rather than producing a wrong
answer.

**Diagnostic, three seeds, `n_state = 64000`, `p = 0.3` so that `kp = 30`
clears `3 ln n` in every cell (the study above ran at `kp = 20`, BELOW that
sufficient condition, and was exact anyway, which is consistent with
[[SEQ-REGIME]] being sufficient and not necessary).**

    n_arc   n/k   0.40(n/k)^2    L=160    L=256           L=384          L=512
    2000    20    160            exact    one glitch      dies at 379    dies at 146
    4000    40    640            exact    exact           exact          dies at 508

Total correct equals consecutive correct in every broken cell, so **once the
chain dies it does not recover**: this is a chain death, not a dropped step.
The single exception is L = 256 at `n_arc = 2000`, where one brain misses one
visit and continues.

**What the numbers do and do not say.** The usable length clearly scales with
the arc's size. It does NOT cleanly follow the refracted square law: at
`n/k = 20` the law says 160 and 160 is the largest exactly-recalled length
tested, but at `n/k = 40` the law says 640 while the chain is exact at 384 and
dead by 512. Doubling `n/k` bought a factor between 2.4 and 3.2, against 4 for
a square law and 2 for a linear one -- which is where the register already puts
the exponent, drifting and "not a power law". **Two cells cannot fit an
exponent and no fit is attempted.**

## Amendment 1 bars, registered before running

Fixed grid, no search: `n_state = 64000`, `k = 100`, `p = 0.3`, `beta = 0.10`,
`w_max = 20`, `s = beta`, 20 presentations, seeds 62 to 81. Cells
`n_arc` in {2000, 3000, 4000, 6000}; lengths {160, 256, 384, 512}.

The estimand is `exact_length`: the largest tested length at which EVERY brain
recalls every visit. It is a grid quantity, not an interpolation, and cells
whose `exact_length` is the top of the grid are CENSORED and reported as such.

- **AL-1, it breaks, and the break is real.** In at least one cell, at least
  one length has every brain below its full length, and in every such cell
  total correct equals consecutive correct on every brain (a death, not a
  dropped step).
- **AL-2, it is monotone in the arc.** `exact_length` is non-decreasing in
  `n_arc` across the four cells.
- **AL-3, it is superlinear but not square.** `exact_length(6000) /
  exact_length(2000)` lies strictly between 3 and 9, the values a linear and a
  square law in `n/k` would give for a threefold change.
  PREDICTION: uncertain, and this is the point of the amendment.
- **AL-4, the state area is not what limits it.** `n_state` is identical in
  every cell, so a monotone result in `n_arc` cannot be the state area. Checked
  by construction and asserted in the record.
- **AL-5, refraction still carries it.** At the smallest cell and the shortest
  length, `refracted_strength = 0` gives zero correct steps, as CL-2.

If AL-2 fails, the limit is not an arc property and the account above is
withdrawn. If AL-3 fails high the square law survives; if it fails low the
limit is linear in `n/k` and is a different mechanism from capacity.

## Instrument note (2026-09-13): the collapse measure now SAMPLES pairs

The arc-overlap measurement enumerated every pair of states, which is quadratic
in the chain length: 8128 pairs at L = 128 and 130816 at L = 512. The limit
grid spent thirteen minutes on a single cell before this was found, and was
stopped. It now samples a fixed 512 pairs with a pinned generator, as the
capacity protocol does for the same reason.

The first study's arc-overlap numbers (0.0000 at L = 32, 0.0108 at L = 128)
were EXHAUSTIVE. Sampled and exhaustive values are comparable in expectation
but not identical, so that quantity is comparable within a protocol version
and not across the change. No bar in the first study depended on a difference
smaller than the sampling error: its arms read 0.0000 against 1.0000.

## Amendment 2 (2026-09-13, registered before running): let the STATES collide, which is the one difference from the papers

Every sequence result in this repository -- the finite-state machine, the
transducer, the temporal carry, and the chain above -- assigns each state its
own DISJOINT block of neurons before training. `HashedArcFSM` builds
`blocks = arange(n_states * k).view(n_states, k)` and `read_state` decodes by
integer division, so two states can never share a neuron and state collision is
impossible by construction.

The papers form state assemblies by projection, where two states can land on
overlapping neurons. That is the single remaining difference from their
setting, and it is not a detail: it means the whole sequence line has been
measured in a world where one of the two failure modes cannot occur. Amendment
1 already shows the other one biting -- when the ARC crowds, the chain dies --
and states are the other half of that conjunction.

**The variable is state crowding**, `L * k / n_state`. At `n_state = L * k` the
disjoint blocks exactly fill the area; below that, any assignment must overlap.

### What runs

The chain of Amendment 1 at a cell it recalls exactly (`L = 160`,
`n_arc = 3000`, 20 of 20 exact), with the state code replaced by RANDOM
`k`-subsets of `n_state` and `n_state` swept. Because the assigned code is no
longer contiguous, the block readout cannot be used: states are decoded by
MAXIMUM OVERLAP against the code, which is the readout the papers' setting
needs anyway. The disjoint-block arm is run through the same decoder so the
two are compared on one readout and not on two.

`n_state` in {64000, 32000, 16000, 8000, 4000}; `L * k = 16000`, so the last
two force overlap and the first two do not.

Retained per brain and arm: consecutive correct, total correct, the mean
pairwise overlap between STATE codes, and the mean pairwise overlap between the
arcs.

- **SC-1, the decoder is not the treatment.** With disjoint blocks, the
  overlap decoder reproduces the block decoder's result: 20 of 20 exact at
  `L = 160, n_arc = 3000`. If this fails, every later comparison is confounded
  by the readout and the amendment is void.
- **SC-2, a roomy random code is as good as a disjoint one.** At
  `n_state = 64000`, where random `k`-subsets overlap at about chance, 20 of 20
  brains are exact. PREDICTION: passes. Random assignment per se must not be
  the thing that breaks it.
- **SC-3, crowding the states breaks the chain.** At `n_state = 4000`, where
  the code cannot be disjoint, fewer than 5 of 20 brains are exact.
- **SC-4, it is the states and not the arc.** `n_arc` is identical in every
  arm, and the mean pairwise ARC overlap differs by less than 0.05 between the
  roomiest and most crowded state arms, while the mean pairwise STATE overlap
  differs by more than 0.05.
  PREDICTION: uncertain. If the arc overlap moves too, the arms are not
  isolating the state and SC-3 cannot be read as a state effect.
- **SC-5, the break is a chain death.** In every broken arm, total correct
  equals consecutive correct on every brain, as AL-1.

### What a pass would mean, stated in advance

That the sequence line's exact results depend on a construction choice the
papers do not make, and that state collision is a failure mode we have never
measured. It would NOT by itself establish that the papers' 20-to-40 limit is
state collision: that needs the limit measured as a function of crowding and
compared against their parameters, which is a further study.

A fail of SC-3 would be the stronger result: it would say the chain tolerates
state collision, that teacher-forcing is not doing the work, and that our
sequence results carry over to the papers' setting unchanged.
