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

Fixed grid, no search: `n_state = 51200`, `k = 100`, `p = 0.3`, `beta = 0.10`,
`w_max = 20`, `s = beta`, 20 presentations, seeds 62 to 81. Cells
`n_arc` in {1000, 2000, 3000, 4000}; lengths {160, 256, 384, 512}. (Both
numbers were revised before any artifact existed; see the instrument note
above.)

The estimand is `exact_length`: the largest tested length at which EVERY brain
recalls every visit. It is a grid quantity, not an interpolation, and cells
whose `exact_length` is the top of the grid are CENSORED and reported as such.

- **AL-1, it breaks, and the break is real.** In at least one cell, at least
  one length has every brain below its full length, and in every such cell
  total correct equals consecutive correct on every brain (a death, not a
  dropped step).
- **AL-2, it is monotone in the arc.** `exact_length` is non-decreasing in
  `n_arc` across the four cells.
- **AL-3, it is superlinear but not square.** `exact_length(4000) /
  exact_length(1000)` lies strictly between 4 and 16, the values a linear and a
  square law in `n/k` would give for a FOURfold change.
  PREDICTION: uncertain, and this is the point of the amendment.
- **AL-4, the state area is not what limits it.** `n_state` is identical in
  every cell, so a monotone result in `n_arc` cannot be the state area. Checked
  by construction and asserted in the record.
- **AL-5, refraction still carries it.** At the smallest cell and the shortest
  length, `refracted_strength = 0` gives zero correct steps, as CL-2.

If AL-2 fails, the limit is not an arc property and the account above is
withdrawn. If AL-3 fails high the square law survives; if it fails low the
limit is linear in `n/k` and is a different mechanism from capacity.

## Instrument note (2026-09-13): the grid is bounded by device memory, and the cells moved

The organ fiber allocates a count matrix of `n_state x n_arc x brains` bytes.
The registered grid's largest cell, `n_arc = 6000` at `n_state = 64000` and 20
brains, needs 8.0 GiB and the engine refuses it by name rather than failing on
the device. Eleven cells completed before it was reached.

Two changes, both before any artifact exists. `n_state` drops from 64000 to
**51200**, which is exactly `L_max * k` and so the smallest area that still
holds disjoint blocks for the longest chain; the surplus was pure memory cost.
And the arcs become **{1000, 2000, 3000, 4000}**, a FOURFOLD span whose largest
cell is one that has already run, instead of a threefold span whose largest
cannot.

AL-3's band moves with the span, on the same principle it was written on: for
an `r`-fold change in `n/k`, a linear law predicts `r` and a square law `r^2`.
At `r = 3` that was the registered 3 to 9; at `r = 4` it is **4 to 16**.

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

## Amendment 1 result (2026-09-13): the grid ran, three bars stand, and the instrument was wrong in five ways

Run `chain-limit-v3-20260913`, seeds 62..81, twenty brains a cell, sixteen
cells plus the mechanism-disabled null. Artifact
`research/results/runs/sequence.autonomous-chain/chain-limit-v3-20260913/results.json`.

**The bars as they fell, unamended.**

    FAIL  AL-1 it breaks, and the break is a chain DEATH not a dropped step
    PASS  AL-2 exact_length is non-decreasing in n_arc
    FAIL  AL-3 superlinear but not square: the 4x arc ratio lands strictly between 4 and 16
    PASS  AL-4 the state area cannot be the cause: n_state identical in every cell
    PASS  AL-5 refraction still carries it: 0 correct with strength 0

    exact_length by arc: {1000: 160, 2000: 160, 3000: 256, 4000: 384}; ratio 2.4

**AL-3 fails and the failure is informative.** A fourfold arc ratio buys 2.4x
the exact length. That is sublinear, not superlinear, and it is well below the
registered 4-to-16 band. Whatever sets the chain limit here, it is not bought
back in proportion to arc neurons.

**AL-2 passes but the column beneath it has a hole.** exact/20 by length:

    n_arc=1000   [20, 0, 0, 0]
    n_arc=2000   [20, 14, 0, 0]
    n_arc=3000   [20, 20, 5, 18]     <- L = 384 is WORSE than L = 512
    n_arc=4000   [20, 20, 20, 2]

`exact_length` takes the largest all-exact cell, so the hole at n3000-L384 is
invisible in the scalar. Exactness is not monotone in L, and a "largest length
at which every brain is exact" is only a limit when it is.

### Five instrument defects, all found by reading the code against the data

1. **The chance denominator was the module constant.** `chance = K / N_ARC`
   used `N_ARC = 10000` for every limit cell, whose real arc sizes are 1000 to
   4000. Reported multiples 20.9x down to 8.2x; true multiples 2.09x up to
   3.27x. The correction **reverses the trend**: relative crowding RISES with
   arc size rather than falling.
2. **AL-4 was a tautology.** It compared `a["n_state"]`, which was the
   requested constant `N_STATE_FIXED` written into every arm, against itself.
   The set always had one element, so the bar could not fail. It is retained
   and printed as VOID; AL-8 asks the same question of the organ.
3. **And the control it claimed did not hold.** A chain of length L has L + 1
   states. `N_STATE_FIXED = 51200` is `L_max * k`, one block short, so
   `HashedArcFSM` widened the L = 512 column to 51300 via
   `max(n_state, n_states * k)`. The connectome is hashed on
   `(row, col, pair_seed)` with `pair_seed` from the area NAMES, so every
   column below 51200 is bit-identical and the only real difference is 100
   never-potentiated distractor neurons in the state k-WTA. That can only
   hurt, and L = 512 is the cell that did better, so this does not explain the
   hole -- but the control was still broken in exactly the column that
   behaves oddly, and it is now `(L_max + 1) * k`.
4. **`total_correct` is position-locked, so AL-1 cannot see what it names.**
   Both `correct` and `total_correct` ask whether visit t reads `q_{t+1}`,
   which is a question about PHASE. The retained traces show chains that run
   380 steps, fall back to an early state and then keep stepping correctly:
   `[378, 379, 380, 2, 3, 4, 5]`. That is a chain still obeying the transition
   table, and both statistics call it dead.
5. **AL-1's FAIL is three coincidences.** It fired on
   `total_correct > correct`, which needs one visit anywhere to land on
   `t + 1`. All three brains that failed it were thrashing
   (`[101, 102, 286, 326, 101, 286, 1]`) and hit `t + 1` exactly once in
   hundreds of visits. AL-1 as written is a coincidence detector; its verdict
   carries no information about dropped steps.

Pinned by `test_autonomous_chain.py`: the area holds every chain in the grid,
chance is per cell, a wrap is not a death, a stall is not a wrap,
`total_correct` cannot separate the two, and one coincidence flips AL-1.

### What the traces actually show

Recomputed over the retained per-brain traces, in the four EDGE cells where
some brains are exact and some are not:

    cell          exact   median first error   median landing   kinds
    n2000-L256    14/20   0.996 of L           33 of 256        wrap 5, stall 1
    n3000-L384     5/20   0.995 of L           26 of 384        wrap 6, stall 6, scatter 3
    n3000-L512    18/20   0.998 of L           95 of 512        wrap 2
    n4000-L512     2/20   0.996 of L          141 of 512        wrap 9, stall 3, scatter 6

The chain does not degrade along its length. It runs correctly to within a
handful of steps of the END and then jumps BACKWARD to an early state. In a
plurality of failures it then keeps stepping through the table correctly from
where it landed. A constant per-step hazard cannot produce a median first
error at 0.996 of L, so the failure is position-specific and `exact_length`
was the wrong instrument for it.

**No mechanism is claimed for the backward jump.** Why the last steps of a
chain are the fragile ones, and why the landing is early rather than random,
are not answered by these traces.

## Amendment 3 (2026-09-13, registered before running): where and how the chain breaks

AL-1 to AL-5 are unchanged and their verdicts above stand. These bars are
stated from structure the first grid's traces showed, so they are POST HOC
with respect to that grid and are to be confirmed on a FRESH SEED BLOCK, as
the convergence study's Amendment 2 was. Same cells, same protocol, corrected
instrument.

- **AL-6, the break is at the END.** In every edge cell -- a cell where at
  least one brain is exact and at least one is not -- the median first-error
  position is at or past 0.90 of L. The null is a constant per-step hazard,
  which puts that median near 0.5.
- **AL-7, the break is a BACKWARD jump.** In every edge cell, every first
  error lands on a state strictly earlier than the expected one, and the
  median landing is before 0.5 L.
- **AL-8, the state area control HOLDS.** The organ's actual `n_state`, read
  off the constructed FSM rather than the requested constant, is identical in
  every cell. This is the claim AL-4 could not make. On the first grid it is
  FALSE.
- **AL-9, relative arc crowding RISES with arc size.** At matched L = 160, the
  overlap-over-chance multiple at `n_arc = 4000` exceeds that at
  `n_arc = 1000`. This is the corrected reading of the measure whose
  denominator was wrong; the uncorrected numbers said the opposite.

A failure of AL-6 or AL-7 on fresh seeds would mean the end-of-chain structure
is a property of seed block 62..81 and not of the construction, which would
matter more than the bars passing: it would say the first grid's most
distinctive finding does not replicate.

## Amendment 2 corrections, recorded BEFORE any data (2026-09-13)

Two, both found while wiring the arms.

**The crowding split was off by one, the same way the limit grid's state area
was.** A chain of length L has L + 1 states, so a disjoint code needs
`(L + 1) k = 16100` neurons, not `L k = 16000`. The registration says "the last
two force overlap and the first two do not". In fact **three of the five force
overlap**: 16000, 8000 and 4000 are all below 16100, and only 64000 and 32000
are roomy. The split is now computed from `(STATE_L + 1) * K` rather than
written down, so it cannot drift again. No bar changes: SC-2 names the largest
area and SC-3 the smallest, and both are on the correct side of the line.

**SC-5 inherits AL-1's defect.** It is AL-1's statistic applied to the state
arms, and Amendment 1's result shows that statistic is position-locked: it
cannot tell a chain that wraps to an early state and keeps stepping from one
that dies, and it fires on a single coincidental hit. SC-5 is kept and answered
as registered. **SC-6** asks the question it was reaching for, using the
wrap/stall/scatter classifier: no broken arm is mostly wraps. If SC-5 and SC-6
disagree, SC-6 is the one that describes the chains.

**One readout, verified at the unit level before the study runs.** SC-1's
premise is that the membership decoder and the integer-division decoder are the
same function on a disjoint code. `test_hashed_fsm_state_code.py` pins that
directly on the device: identical cued winners and identical decoded states for
every state, and an identical run through a trained chain. Both arms of this
amendment pass an explicit code -- the disjoint one included -- so both are
read out by membership. SC-1 remains as the end-to-end check.

## Amendment 2, third correction recorded BEFORE any data: SC-4's threshold is unreachable

SC-4 requires the mean pairwise STATE overlap to differ by more than 0.05
between the roomiest and most crowded arm. Two random `k`-subsets of `n`
neurons share `k^2 / n` on average, which is `k / n` as a fraction of `k`:

    n_state    pairwise overlap / k    states sharing the average neuron
     64000                  0.0016                                0.25
     32000                  0.0031                                0.50
     16000                  0.0063                                1.01
      8000                  0.0125                                2.01
      4000                  0.0250                                4.03

The largest gap the registered sweep can produce is `0.0250 - 0.0016 = 0.0234`,
so **SC-4 cannot pass on its state clause whatever the science does**. The
threshold was set without checking the arithmetic of the treatment it measures.

SC-4 is kept and will be answered as registered, and its failure is a mis-set
threshold rather than a result. **SC-7** asks the same question with the
statistic that does move: LOAD, `(L + 1) k / n_state`, the number of states
sharing the average neuron, which runs 0.25 to 4.03 across the same arms -- a
sixteenfold change. At `n_state = 4000` every neuron carries about four states,
which is the interference SC-3 is about; pairwise overlap stays small there
precisely because the crowding is spread across many pairs rather than
concentrated in any one.

This does not rescue SC-3, which is untouched and still the amendment's main claim:
fewer than 5 of 20 brains exact at the smallest area. A SC-3 failure remains
the stronger and more interesting outcome.

## Amendment 3 result (2026-09-13): the end-of-chain structure replicates; the backward claim does not, on a two-point median

Run `chain-limit-am3-20260913`, the fresh block 82..101, same cells, corrected
instrument. Artifact
`research/results/runs/sequence.autonomous-chain/chain-limit-am3-20260913/results.json`.

    FAIL  AL-1  (unchanged; the coincidence detector)
    PASS  AL-2
    FAIL  AL-3  (unchanged; ratio 2.4 again)
    PASS  AL-4  VOID, tautology
    PASS  AL-5
    PASS  AL-6  the break is at the END
    FAIL  AL-7  the break is a BACKWARD jump
    PASS  AL-8  the state area control HOLDS
    PASS  AL-9  relative arc crowding RISES with n_arc

**The limit replicates cell for cell.** exact/20 by length, the two blocks:

    n_arc    62..81                82..101
     1000    [20,  0,  0,  0]      [20,  0,  0,  0]
     2000    [20, 14,  0,  0]      [20, 14,  0,  0]
     3000    [20, 20,  5, 18]      [20, 20,  5, 18]
     4000    [20, 20, 20,  2]      [20, 20, 20,  4]

`exact_length` is `{1000: 160, 2000: 160, 3000: 256, 4000: 384}` on both, ratio
2.4 on both. AL-3 fails identically: a fourfold arc ratio buys 2.4x, sublinear.

**The seeds do reach the brains**, checked rather than assumed: the eight cells
whose per-brain vectors are byte-identical across blocks are exactly the
saturated ones (every brain at L, or the all-zero null). Every non-saturated
cell has different per-brain values -- `n3000-L384` runs
`[383, 380, 384, 383, ...]` against `[384, 383, 383, 382, ...]` -- and still
lands on 5 of 20. The replication is real.

**So the hole at n3000 is structural, not seed noise.** L = 384 gives 5 of 20
while L = 512 gives 18 of 20, on two independent seed blocks, to the brain.
Exactness is not monotone in chain length and `exact_length` cannot express
that. No explanation is offered here; it is now a fact that needs one.

**AL-6 confirms.** In every edge cell the median first error is at 0.99 to 1.00
of L, on both blocks. The chain does not degrade along its length.

**AL-7 fails, and the failure is in its threshold, not its claim.** The bar has
two clauses. "Every landing earlier than expected" HELD in every edge cell on
both blocks -- there is not one forward jump anywhere. "Median landing before
0.5 L" failed in exactly one cell, `n3000-L512` on the fresh block, where 18 of
20 brains are exact so the median is taken over TWO failures, `[2, 274]`, and
lands on 274 against a threshold of 256. A two-point median is not a
measurement of central tendency. The bar is recorded FAILED as written; the
backward-jump claim is neither confirmed nor refuted by it, and a bar stated
over so few failing brains was the wrong instrument for the claim.

## Amendment 2 result (2026-09-13): the chain does NOT care whether states collide

Run `chain-states-20260913`, block 62..81, `L = 160`, `n_arc = 3000`. Artifact
`research/results/runs/sequence.autonomous-chain/chain-states-20260913/results.json`.

    arm                exact   state overlap   load   arc overlap
    blocks-n64000      20/20          0.0000   0.25        0.0921
    random-n64000      20/20          0.0016   0.25        0.0817
    random-n32000      20/20          0.0031   0.50        0.0757
    random-n16000      20/20          0.0062   1.01        0.0675
    random-n8000       20/20          0.0126   2.01        0.0563
    random-n4000       20/20          0.0251   4.03        0.0447

    PASS  SC-1 the decoder is not the treatment
    PASS  SC-2 a roomy random code is as good as a disjoint one
    FAIL  SC-3 crowding the states breaks the chain
    FAIL  SC-4 (threshold unreachable, as recorded before the run)
    PASS  SC-5, PASS SC-6 (vacuous: nothing broke)
    FAIL  SC-7 (no effect to attribute)

**SC-3 fails, and the registration says in advance what that means:** "A fail
of SC-3 would be the stronger result: it would say the chain tolerates state
collision, that teacher-forcing is not doing the work, and that our sequence
results carry over to the papers' setting unchanged."

Every arm is exact on every brain. Across a sixteenfold change in load -- from
a quarter of a state per neuron to four states per neuron, where the code
CANNOT be disjoint -- not one brain drops a single step. The measured state
overlap matches the `k / n` arithmetic to the digit (predicted 0.0016, 0.0031,
0.0063, 0.0125, 0.0250), so the treatment was applied and measured as intended.

The arc overlap FALLS as the state area shrinks, 0.0817 to 0.0447. Crowding the
states made the arcs more distinct, not less. No mechanism is claimed for that.

### What this settles, and the three things it does not

It settles that the disjoint teacher-forced block code is not what carries this
repository's sequence results, at this operating point. That was a live worry
-- every sequence result here was measured in a world where state collision is
impossible by construction -- and it is now measured rather than assumed.

It does NOT settle:

1. **Collision at the EDGE.** This ran at a cell that is 20 of 20 exact with
   room to spare. A treatment that costs nothing where there is margin may
   still cost everything where there is none. The same sweep at `n3000-L384`
   (5 of 20) is the test that matters and has not been run.
2. **Correlated collision.** Random `k`-subsets are uncorrelated. Projection
   forms assemblies whose overlap TRACKS input similarity, which is a
   structured interference this sweep does not produce. The papers' states are
   projection-formed; these are not.
3. **Higher load.** The sweep stops at 4.03 states per neuron. Nothing here
   says where it would break, only that 4.03 is not enough.

## Amendment 4 (2026-09-13, registered before running): collision where there is NO margin

Amendment 2's null was measured at a cell that is 20 of 20 exact with room to
spare. A treatment that costs nothing where there is slack can still be
decisive where there is none, and the registration listed that as the first
thing its result does not settle. This runs the same sweep at a MARGINAL cell.

**The cell.** `n_arc = 2000, L = 256`, which gave exactly 14 of 20 exact on
both independent seed blocks of the limit grid -- high enough that a cost shows
as a drop, low enough that it is not at ceiling and can move in either
direction. It is the most stable marginal point in the grid.

Same six arms: the disjoint block code, plus random `k`-subsets in areas
64000 down to 4000. A chain of length 256 has 257 states, so a disjoint code
needs 25700 neurons: **64000 and 32000 are roomy, 16000, 8000 and 4000 force
overlap.** Load `(L + 1) k / n_state` runs 0.40, 0.80, 1.61, 3.21 and 6.42 --
reaching HALF AGAIN the crowding Amendment 2's tightest arm achieved, because
the same areas now hold a longer chain.

SC-1 to SC-7 are stated for a cell at ceiling and their thresholds are wrong
here by construction -- SC-1 demands 20 of 20 from an arm that is 14 of 20 by
design, so it would fail on the cell rather than on the science. Amendment 4
gets its own bars, and SC-1 to SC-7 are not evaluated in this mode.

- **MC-1, the cell is the marginal one it was chosen for.** The disjoint-block
  arm is between 8 and 18 of 20 exact. The grid measured 14 twice, at a state
  area of 51300; this runs at 64000, so the band is generous rather than tight.
  If MC-1 fails the cell is not the one these bars are about and MC-2 to MC-5
  are not interpretable.
- **MC-2, collision at the margin is COSTLY.** The tightest area loses at least
  5 brains against the disjoint arm. This is the directional prediction
  Amendment 2 could not test.
- **MC-3, the cost orders by crowding.** Every crowded arm is at most every
  roomy arm in exact brains. Stated as an ordering between groups rather than
  strict monotonicity among neighbours, because five counts of 20 carry
  binomial noise of about 2 and a strict chain would fail on that alone.
- **MC-4, it is the states and not the arc.** Arc overlap moves by less than
  0.05 between the roomiest and tightest random arms.
- **MC-5, a roomy random code is still as good as blocks**, within 4 brains.
  Randomness per se must not be the thing that costs.

### What each outcome means, stated in advance

**MC-2 passes:** state collision is real and Amendment 2's null was a ceiling
effect. The chain tolerates collision only while it has margin, and this
repository's sequence results DO depend on the teacher-forced code once the
chain is worked near its limit. That would reinstate state formation as a
candidate for the papers' 20-to-40 limit.

**MC-2 fails:** the stronger and more surprising result. Collision is costless
even with no margin and at load 6.42, which would say the disjoint code is not
doing the work anywhere we have looked, and that the sequence line transfers to
the papers' setting without qualification on this axis. Amendment 2's null
would then generalise rather than being a ceiling artifact.

Either way the third limit stands and is NOT addressed here: random `k`-subsets
are uncorrelated, and projection forms assemblies whose overlap tracks input
similarity. Correlated collision remains unmeasured.

## Amendment 4 result (2026-09-13): crowding costs enormously, and three of the five bars cannot show it

Run `chain-margin-20260913`, block 62..81, `n_arc = 2000, L = 256`. Artifact
`research/results/runs/sequence.autonomous-chain/chain-margin-20260913/results.json`.

    PASS  MC-1   PASS  MC-2   PASS  MC-3   FAIL  MC-4   FAIL  MC-5

    arm               load   exact/20   mean correct   of L      range
    blocks-n64000     0.40      14/20          255.5   1.00   253..256
    random-n64000     0.40       0/20          251.8   0.98   249..253
    random-n32000     0.80       0/20          220.8   0.86   137..252
    random-n16000     1.61       0/20          180.8   0.71   137..251
    random-n8000      3.21       0/20          142.0   0.55   123..175
    random-n4000      6.42       0/20          110.0   0.43    31..133

**The science: crowding the states is devastating at a marginal cell.** Mean
consecutive-correct falls 251.8 -> 220.8 -> 180.8 -> 142.0 -> 110.0 as load
runs 0.40 to 6.42 -- a monotone dose-response losing 142 of 256 steps. At the
tightest area the chain gets less than half way. **Amendment 2's null was a
ceiling effect**, exactly as MC-2 was registered to mean: the chain absorbs
state collision while it has margin and does not when it has none.

**But three of the five bars cannot carry that conclusion, and I have to say
so.** All five used `exact/20` as the outcome, and every random arm is 0 of 20
-- roomy and crowded alike.

- **MC-3 passed VACUOUSLY.** "Every crowded arm at most every roomy arm" is
  `0 <= 0`. It would pass against any data in which the random arms are all
  zero, including data with no crowding effect at all.
- **MC-2 passed for a confounded reason.** It compares the tightest RANDOM arm
  against the BLOCKS arm, so it mixes crowding with whatever randomness alone
  costs. With the roomy random arm already at 0, MC-2 would pass with zero
  crowding effect. Its PASS does not support the sentence it states. The
  dose-response above does support that sentence -- but on `mean correct`,
  which no bar reads.
- **MC-5 failed and is the reason.** Randomness per se costs: blocks 255.5 to
  roomy random 251.8, **3.7 steps of 256**, about 1.4%. That is small, and
  `exact/20` magnifies it into 14 -> 0 because exactness is all-or-nothing. So
  the two effects differ by a factor of forty (3.7 against 141.8) and the
  registered outcome statistic cannot tell them apart.
- **MC-4 failed: the arms do not isolate the state.** Arc overlap moves 0.1493
  to 0.0625 across the random arms, a gap of 0.0868 against the 0.05 bar. The
  arc changes when the states are crowded -- same direction as Amendment 2,
  now large enough to break the bar. **Attribution of the cost to state
  collision specifically is therefore NOT established**, only that crowding the
  state area costs the chain.

This is the tie-fragility of `exact@L` biting a third time in this
registration, after `exact_length` in Amendment 1 and the two-point median in
AL-7. The graded measure was retained and answered the question immediately; no
re-run was needed. Bars on this line should read `mean correct` and reserve
`exact/20` for cells where it is not saturated.

**Why unexplained (again): a roomy random code should be nearly free.** At
`n_state = 64000` the code's pairwise overlap is 0.0016 of `k` and a disjoint
assignment is possible in principle. Why the contiguous block code is worth 3.7
steps over a scattered one at the same overlap is not answered here.

### What is confounded between Amendments 2 and 4, and the run that would fix it

Amendment 2 (no cost) and Amendment 4 (large cost) differ in TWO ways at once:
the cell's margin (20/20 with room against 14/20) AND the load reached (4.03
against 6.42). Nothing here separates them.

The clean test is cheap and is NOT yet run: sweep the ROOMY cell
(`L = 160, n_arc = 3000`) down to state areas that reach load 6.42 and beyond
-- `n_state` of 2500 and 2000 give 6.44 and 8.05. If the roomy cell absorbs
load 8 without cost, margin is the variable; if it breaks near 6.4, load is.

## Amendment 5 (2026-09-13, registered before running): is it the margin or the load?

Amendments 2 and 4 disagree, and they differ in TWO ways at once. The roomy
cell (`L = 160, n_arc = 3000`, 20/20 exact with room) absorbed the collidable
code at every load it was given, topping out at 4.03. The marginal cell
(`L = 256, n_arc = 2000`, 14/20) collapsed to 0.43 of its chain length, but it
was also driven to load 6.42. Neither run says whether the variable is the
cell's MARGIN or the LOAD reached.

**This drives the roomy cell past the load that killed the marginal one.**
Same cell, same seeds, state areas 4000, 2500, 2000, 1600 and 1200, giving
load 4.03, 6.44, 8.05, 10.06 and 13.42. `random-n2500` at 6.44 is the
load-matched arm: within 0.02 of the marginal cell's tightest.

**Every bar here reads MEAN CORRECT as a fraction of the chain length, not
`exact/20`.** Amendment 4 is the reason: three of its five bars used `exact/20`,
which saturates at 0 the moment a random code is used at all, and two of them
passed without evidence because of it. This is the third time `exact@L`
tie-fragility has cost this registration a bar.

- **LM-1, the reference still recalls exactly.** The disjoint-block arm sits at
  1.00 of L. If it does not, the cell is not the one Amendment 2 measured.
- **LM-2, DECISIVE.** At the load-matched arm (6.44), the roomy cell holds
  above 0.90 of L. The marginal cell at the same load was at 0.43.
- **LM-3, the dose-response is monotone in load**, mean correct non-increasing
  as the area shrinks.
- **LM-4, the sweep has RANGE.** The tightest arm (load 13.42) falls below 0.90
  of L. Without this, LM-2 could pass because the sweep never bit at all, and a
  pass of LM-2 alongside a fail of LM-4 means exactly that and nothing more.
- **LM-5, the arc moves too.** Arc overlap falls monotonically as the area
  shrinks. This turns an effect seen unexplained in both previous runs into a
  stated prediction rather than an observation noticed after the fact.

### What each outcome means, stated in advance

**LM-2 passes and LM-4 passes:** MARGIN is the variable. The same load that
destroys a marginal chain is nearly free for one with room, so state collision
is not a fixed cost but something a chain pays out of slack it may or may not
have. The papers' 20-to-40 limit would then be about where that slack runs out,
not about collision as such.

**LM-2 fails:** LOAD is the variable and Amendment 2's null was simply never
driven hard enough. Collision has a threshold near 6, the roomy cell was tested
below it, and the margin story is unnecessary.

**LM-2 passes and LM-4 fails:** the sweep never bit and the run is
uninformative about both. It would need areas smaller than 1200.

Note that LM-2 and LM-4 are the same statistic at two loads, so they cannot
both be gamed by a threshold choice: the bar that makes LM-2 easy makes LM-4
hard.

## Amendment 5 result (2026-09-13): it is the MARGIN, not the load

Run `chain-load-20260913`, block 62..81, the roomy cell `L = 160, n_arc = 3000`.
Artifact
`research/results/runs/sequence.autonomous-chain/chain-load-20260913/results.json`.

    PASS  LM-1   PASS  LM-2   PASS  LM-3   FAIL  LM-4   PASS  LM-5

    load    ROOMY cell (L=160)    MARGINAL cell (L=256)
    0.40                     -               0.984 of L
    0.80                     -               0.862 of L
    1.61                     -               0.706 of L
    3.21                     -               0.555 of L
    4.03            1.000 of L                        -
    6.42                     -               0.430 of L
    6.44            1.000 of L                        -
    8.05            1.000 of L                        -
   10.06            1.000 of L                        -
   13.42            1.000 of L                        -

**At matched load the two cells could not differ more.** 6.44 against 6.42:
the roomy cell is perfect, the marginal cell is at 0.43 of its chain. The roomy
cell then holds at 1.000 all the way to load 13.42, **more than twice the load
that destroyed the marginal one**, with one brain of twenty losing one step.

### I am revising a pre-declared interpretation, and flagging it as such

The registration says of this exact bar combination: "LM-2 passes and LM-4
fails: the sweep never bit and the run is uninformative about both." That
reading was miscalibrated, and revising an interpretation after seeing the data
is the move that most deserves suspicion, so the reasoning is set out rather
than assumed.

LM-4 was written to guard against ONE failure: a sweep whose treatment never
reaches the organ, which would make LM-2 pass vacuously. That did not happen.
The treatment demonstrably applied -- state overlap more than tripled across
the random arms (0.0251 to 0.0831), arc overlap moved monotonically (0.0447 to
0.0346, LM-5), and at the tightest area one brain did break. What LM-4 failed
to establish is a BREAKING POINT for the roomy cell, which is a different
proposition from the treatment being inert.

**And the decisive evidence does not rest on LM-4 at all.** It is the
load-matched comparison: hold load fixed at 6.4, change only the cell, and the
outcome goes from 1.000 to 0.430 of L. LM-4 could have failed for either reason
without touching that.

The bar stands FAILED as written. The pre-declared reading of it is withdrawn,
with the reason given above, and a reader who disagrees has the numbers.

### What this settles

**Load is not the variable; margin is.** State crowding is not an independent
failure mode with a threshold near 6. A chain with room absorbs load 13.42 --
161 states packed into 1200 neurons, every neuron carrying about thirteen
states, pairwise code overlap 0.083 -- and still recalls all 160 steps. The
same load region annihilates a chain that is already near its limit.

So state collision costs NOTHING until the chain is already marginal, and then
it costs enormously. It is an amplifier of an existing limit rather than a
limit of its own.

### What "margin" is remains unmeasured, and one hypothesis is worth naming

"Margin" here is a label for "the cell had slack", not a mechanism. The two
cells differ in chain length AND arc size (160 at n_arc 3000 against 256 at
n_arc 2000), and those jointly determine whether a cell is marginal, so nothing
separates them.

The hypothesis worth testing: the mediating variable is the ARC, not the state
area. The marginal cell runs 256 arcs in 2000 arc neurons, and in BOTH crowded
runs, crowding the STATES moved the ARC overlap -- MC-4 failed on exactly that
(0.0868 against a 0.05 bar) and LM-5 predicted and confirmed the monotone fall.
If the arc is the bottleneck, state crowding would be costly precisely when the
arc has no room, which is what these three amendments look like from outside.
That is a prediction and it has not been run.

## Diagnostic (2026-09-13): the backward jump is an ARC that degrades in the last few steps

Not a registered study: eight brains, seeds 82..89, at `n_arc = 4000, L = 512`,
recording the decode margin at every step and then comparing the arc actually
driven at the break against every trained arc. No bars, no artifact. It informs
a registration.

**The decode margin does not decay. It falls off a cliff.** Top block count
minus runner-up, of `k = 100`:

    step      0    64   128   256   384   448   496   508   510   511
    seed 82 100    97    98    95    98    93    94    29     4     3
    seed 85  98   100    98    96    95   100    92     1     1     3
    seed 88  97    97   100    95    93    96    92    20     0     0

Every brain holds a margin averaging 96.2 to 96.4 out of 100 for the whole
chain, with a pre-break minimum of 17 to 47, and then collapses to 0-7 in the
final three or four steps. **This rules out margin decay** as the mechanism:
there is no downward trend to find. Whatever happens, happens at the end.

**And the arc being driven at the break is the RIGHT arc, degraded.** For every
broken brain, the arc driven at the failing step best matches -- over all 511
trained arcs -- the arc out of the state the chain was correctly at. But it
matches it at only 0.42 to 0.51, where a healthy arc would be near 1. The
landing state's own source arc matches much less (0.18 to 0.40), so the chain is
not being pulled onto a competing arc:

    seed   broke at   wanted   landed   best match   overlap   landing source
      82        509     q510     q335    q509 (right)    0.430           0.400
      83        511     q512       q6    q511 (right)    0.500           0.180
      87        508     q509       q2    q508 (right)    0.420           0.230

So the readout is reporting honestly. The arc has degraded to about half its
trained assembly, the state drive that follows is ambiguous, and the decode
flips to whichever block the noise favours -- which is why the landing state
looks arbitrary (q2, q6, q42, q139, q335) rather than systematic.

**This eliminates two of the three candidates** and leaves the question one
step further back: why is the arc weak specifically at the END of the chain?

**A mechanism worth testing, from the training loop.** `train` sweeps the table
in order, `q0 -> q1` through `q511 -> q512`, repeated `presentations` times, and
the arc is REFRACTED: bias accumulates on every winner and never decays. So
within each sweep, by the time the last transitions are presented, nearly every
arc neuron already carries fresh bias from that same sweep, and the late arcs
are recruited from whatever is least suppressed. The end of the chain would then
be weak BECAUSE it is trained last, not because it is far along.

That predicts something sharp and cheap: **shuffle the presentation order within
each sweep and the cliff should move off the end of the chain.** Not yet run.

## Amendment 6 (2026-09-13, registered before running): is the ARC what makes a cell marginal?

Amendment 5 showed margin decides whether state crowding costs anything, but
"margin" is a label rather than a mechanism, and the two cells it compared
differed in chain length AND arc size together.

**This holds the chain and the state crowding fixed at the combination that
collapsed** -- `L = 256`, random code in a 4000-neuron state area, load 6.42,
measured at 0.430 of L -- **and gives the arc more neurons**: `n_arc` of 2000,
3000, 4000 and 6000. A disjoint-block control runs at `n_arc` 2000 and 4000 in
a roomy 32000 state area. Every cell is checked against the device ceiling
first: the largest here is 128 M cell-products against the 205 M that has run
and the 384 M that refused.

All bars read mean correct as a fraction of chain length.

- **AB-1, the killing cell reproduces**, below 0.60 of L at the smallest arc.
  Measured 0.430 in Amendment 4; if it does not reproduce the rest is moot.
- **AB-2, DECISIVE.** At `n_arc = 6000` the same crowded code is above 0.90 of
  L. Enlarging only the arc rescues a chain that state crowding destroyed.
- **AB-3, the rescue is monotone in arc size.**
- **AB-4, the blocks control is above 0.95 of L at BOTH arc sizes**, so the
  rescue is specific to the crowded code rather than the arc helping everything.
- **AB-5, arc overlap falls as the arc gets room.**

**AB-2 passes:** the ARC is the bottleneck, and one variable explains all four
amendments -- state crowding is costly exactly when the arc has no room. That
would make the papers' 20-to-40 limit an arc-capacity limit, and it would say
the state area is not where the sequence limit lives at all.

**AB-2 fails:** the arc is not sufficient, "margin" is something else, and the
next variable to isolate is chain length at fixed arc size.
