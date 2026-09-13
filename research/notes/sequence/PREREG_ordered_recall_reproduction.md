# Registration: reproduce multi-step ordered recall, the mechanism the sequences paper is about

> **Status (2026-09-13): step 1 is DONE and it changed the conclusion.** The
> reference comparison this registration ordered first has been run. NEITHER
> reference implementation contains `ordered_recall`, `sequence_memorize`, a
> refractory period, or any autonomous chain recall. Both contain the
> refracted-arc construction this repository already ported and measured
> working. So the failure below is a failure of OUR OWN construction, not a
> failure to reproduce the paper. Bars OR-1 to OR-6 are suspended pending a
> decision on whether the construction should exist at all; see Amendment 1.

## Why this exists

`dabagia2025sequences` ("Computation with Sequences of Assemblies in a Model of
the Brain") is the paper behind the register's `SEQ-FSM`, `SEQ-TRANSDUCER` and
`SEQ-TM` entries, all of which carry status PROVED **from the paper** rather
than measured here. Its central mechanism is multi-step ordered recall:
`sequence_memorize` writes strong within-assembly recurrence and weaker
between-assembly bridges, and long range inhibition removes the just-fired
assembly from the competition so the bridge to the next one can win.
`mitropolsky2025simulated` builds its whole generation algorithm on the same
idea in a different form, a trigger that inhibits the role areas for one step.

This repository implements `ordered_recall` and marks the paper `partial`. The
maintained parity test is
`test_literature_parity.py::test_sequence_memorize_ordered_recall`. It asserts

    len(recalled) >= 1
    overlap(recalled[0], memorized[0]) > 0.3

`ordered_recall` fires the cue once and snapshots before its loop, so
`recalled[0]` IS the cue's own assembly. **Both assertions describe the cue and
both hold on a run that advances zero steps.** The paper's claim is about
`recalled[1]` onwards.

## Diagnostic (2026-09-13)

Sequence of L stimuli into one area, `n = 4000, k = 50, p = 0.05, beta = 0.10,
w_max = 20`; memorize with `phase_b_ratio` and `repetitions >= 3` as the parity
test's own docstring requires for the bridge to be written at all; then
`set_lri(period = 3, strength = 100)` AFTER memorizing, as the paper does, and
recall from the first stimulus.

**Steps taken after the cue: ZERO, everywhere tested.**

    varied                                              steps after the cue
    64 combinations of repetitions {3,5}, phase_b_ratio
      {0.5,0.7}, rounds_per_step {8,16}, refractory
      period {3,6}, recall rounds_per_step {1,3},
      known_assemblies {on,off}, at L = 8              0 in all 64
    L in {3, 8, 16} x repetitions {3, 10} x
      {sampled, materialized}, 5 seeds each            0 in all 60 runs

The chain leaves the cued assembly and lands on something the novelty check
does not recognise, so with `known_assemblies` supplied recall stops at length
one, and without it, it runs on producing assemblies that match nothing.

**And the cue retrieval itself is a sampler artifact.** The same comparison,
`overlap(recalled[0], memorized[0])`, five seeds:

    connectome        L=3     L=8     L=16
    sampled           0.92    0.84    0.86
    materialized      0.30    0.28    0.30

`PREREG_sampler_audit.md` voids sequence-dynamics numbers taken on the sampled
connectome. The maintained parity test does not materialize. On the engine our
own rules require, cue retrieval sits on the 0.3 threshold rather than at 0.9.

Pinned by `neural_assemblies/tests/test_ordered_recall_advances.py`: one
`xfail(strict=True)` for the advancement, and two passing tests for the sampler
gap and for the fact that the parity assertions hold with zero advancement.

## Bars for a working reproduction

To be met on the MATERIALIZED connectome, twenty seeds, before any claim that
this repository reproduces the paper's recall.

- **OR-1, it advances at all.** At `L = 8`, at least one step after the cue
  matches its memorized assembly at overlap >= 0.3, on at least 18 of 20 seeds.
- **OR-2, it advances more than one step.** Median steps after the cue >= 3 at
  `L = 8`.
- **OR-3, it advances in ORDER.** Every matched step matches the assembly at
  its own index and not a later or earlier one, on every seed.
- **OR-4, inhibition is what does it.** With `inhibition_strength = 0` and the
  same trained brain, steps after the cue is 0 on every seed. This is the
  mechanism-disabled null, and the paper's claim is exactly that recall is
  driven by inhibition.
- **OR-5, it is not the sampler.** The materialized and sampled arms agree on
  steps after the cue to within 1 on at least 18 of 20 seeds. A reproduction
  that works only on the sampled connectome is void under
  `PREREG_sampler_audit.md`.
- **OR-6, cue retrieval is real.** `overlap(recalled[0], memorized[0]) >= 0.7`
  on the materialized connectome on at least 18 of 20 seeds.

A fix that passes OR-1 and OR-2 while failing OR-4 has found some other route
to advancement and is not the paper's mechanism; it would be recorded as such
rather than counted.

## What is deliberately NOT claimed here

That the paper is wrong. The paper reports this working; the reference
implementation exists (`reference_repo` in
`research/literature/parity/configs/sequences2025.yaml` names `turing_sim.py`
and `fsm_learn.py`). The failure is ours until a like-for-like comparison
against the reference says otherwise, and running that comparison is the first
step of the fix rather than a later one.

## Consequences for the register, recorded now

Three register entries cite this paper as PROVED. That status refers to the
paper's proofs, not to a local measurement, and the register already
distinguishes those. But the sequence line's own constructions -- the refracted
arc machine, the transducer, the temporal carry -- were built on a DIFFERENT
mechanism of our own, and none of them exercises `ordered_recall`. So the
paper's recall mechanism has no working local reproduction, and until OR-1 to
OR-6 pass, no claim in this repository should describe it as reproduced.

## The order of work this implies

1. Compare against the reference implementation on identical parameters, to
   separate "our port is wrong" from "the regime is different".
2. Fix, then meet OR-1 to OR-6.
3. Only then the sequence-length limit study: the paper reports a limit of 20
   to 40 assemblies that "varies with the parameters of the Nemo model", and
   this repository has a closed form for the weight clip,
   `c* = ln(w_max max(1, kp) / base) / ln(1 + beta)`, which at
   `k p = 2.5` gives 26.5, 33.8 and 41.0 for `w_max` of 5, 10 and 20 at
   `beta = 0.10` -- the paper's band. Testing whether the limit IS the clip
   needs recall that advances, so it waits.

## Amendment 1 (2026-09-13): the reference comparison, which this registration ordered first

Both reference repositories were cloned (`mdabagia/nemo` and
`dmitropolsky/assemblies`, the two the manifest names for this paper) and
searched exhaustively.

    construction                     files in either reference
    refractory_period                0
    inhibition_strength              0
    ordered_recall                   0
    sequence_memorize                0
    steps_ago / recently-fired       0
    RefractedArea                    1
    self.bias (cumulative)           1
    FSMNetwork                       3
    any function named *recall*      0

**What the reference actually does for sequences.** `FSMNetwork` in
`.reference/mdabagia-nemo/brain.py` is

    state_area = FFArea(n_arc, n_state, ...)                      # feedforward
    arc_area   = RefractedArea([n_symbol, n_state], n_arc, ...)   # feedforward

with `forward()` driving the arc from symbol and state, then the state from the
arc. `RefractedArea` is the CUMULATIVE bias:

    update:           bias[new] += get_total_input()[new] * plasticity
    get_total_input:  super().get_total_input() - bias

which is `bias += s x raw` at every win with `s = plasticity = beta`, never
decaying. That is exactly the construction this repository ported as the hashed
refracted arc and measured at width: 2000 steps without error on 40 of 40
brains (`SEQ-EXACT-RECOVERY`), which exceeds what the paper reports.

**So three things are now true at once.**

1. This repository DOES reproduce the reference's sequence mechanism, under a
   different name, and has measured it further than the reference goes.
2. `sequence_memorize`, `ordered_recall` and the refractory period are this
   repository's OWN constructions. They are attributed to the paper in
   `test_literature_parity.py` and in `research/literature/index.json`, and
   that attribution is not supported by either reference.
3. The failure documented above is therefore a failure of our own
   construction, not a failed reproduction. It is still a failure, and the
   xfail stays, but it does not license any statement about the paper.

**Caveat on the absence.** These two repositories are the ones the manifest
names for this paper, and `mdabagia/nemo` is the site source plus the engine
rather than a full experiment suite. Absence here is strong evidence and not
proof; the paper's own autonomous-recall experiment may live somewhere these
clones do not reach. The local PDF is untracked in this worktree, so the text
has not been read directly.

### What this changes

OR-1 to OR-6 are SUSPENDED. Fixing `ordered_recall` to advance is no longer
obviously the right work, because nothing in the reference needs it and the
mechanism the papers do use already works here. The decision to make first is
whether the construction should exist, and that decision needs the paper text
rather than more simulation.

Three options, none taken yet:

- **Retire it.** Mark `sequence_memorize` and `ordered_recall` as a local
  construction that does not work, remove the paper attribution, and keep the
  refracted arc as this repository's sequence mechanism.
- **Re-attribute and keep.** If the paper's autonomous recall is real and this
  is an honest port of it, fix it against the paper's own parameters and
  restore OR-1 to OR-6.
- **Replace.** Build autonomous chain recall on the refracted arc, which works,
  rather than on the refractory period, which does not.

### The sequence-length limit is still blocked, for a clearer reason

The prize remains: the paper reports a 20 to 40 assembly limit that varies with
parameters, and `c* = ln(w_max max(1, kp) / base) / ln(1 + beta)` gives 26.5,
33.8 and 41.0 at `w_max` of 5, 10 and 20. But that limit is about a chain
recalled AUTONOMOUSLY with no external input, while the refracted arc is driven
by an incoming symbol at every step, so its 2000-step result does not speak to
it. Neither this repository nor either reference has working autonomous recall.
That is the gap the limit study needs closed, and option three above is the
route to it.
