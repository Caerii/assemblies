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
