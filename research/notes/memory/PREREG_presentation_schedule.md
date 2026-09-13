# Registration: does the presentation SCHEDULE change what a memory retains? Massed against interleaved, twenty brains

> **Status (2026-09-13): run; SR-4 and SR-5 pass, SR-1, SR-2 and SR-3 FAIL
> as registered (Result below). The quantities they aimed at moved hugely
> in the predicted direction at five of six checkpoints; the bar's own
> checkpoint-selection rule picked the sixth. Amendment 1 fixes the
> checkpoints as numbers, adds the published one-episode arm, and runs on
> fresh seeds.** Bars were fixed before the instrument existed. This registration asks the spaced-repetition question in
> the only form this substrate can answer it, and states in advance why the
> naive form is empty.

## Why the naive question is empty here, and what is left of it

Spaced repetition in humans is a claim about TIME: the same number of
repetitions retains better when spread out. This substrate has no time. Every
deadline it has is a COUNT:

- potentiation multiplies a synapse by `1 + beta` once per WIN, not per second;
- the refraction bias charges `s x raw` once per WIN;
- the weight clip arrives after `c* = ln(w_max max(1, kp) / base) / ln(1 + beta)`
  PRESENTATIONS ([[SEQ-EXACT-RECOVERY]], `PREREG_s5_cliff_anatomy.md` Addendum 5);
- a refracted assembly relocates on a period of
  `ln(w_max) / ln(1 + beta) + (1 - 1 / w_max) / beta` ROUNDS
  ([[REFRACTION-CANCELS-CONVERGENCE]], `PREREG_refraction_convergence.md`
  Amendment 2, confirmed to 1.4% on a fresh seed block).

There is no decay term anywhere in the model. An interval in which nothing is
presented therefore leaves every weight, every bias and every winner set
exactly as it found them. **Idle spacing is a no-op by construction, and this
registration asserts it rather than measuring it** (bar SR-4). Any claim that
"spacing helps" in this substrate that is not mediated by something happening
in the gap would be an instrument fault.

What is left is the real mechanism: what happens BETWEEN repetitions is other
items, and other items are interference. So the spacing question becomes a
scheduling question about ORDER at matched presentation count, and this
substrate already has a known order-dependent failure mode to aim at.

`PREREG_refraction_memory.md` records it: the Hebbian control at T = 16
collapses to `M* = 8` with pairwise overlap 15x chance, which it attributes to
hub formation, rich-get-richer, the items written FIRST becoming attractors
that later items fall into. Refraction at half beta stops exactly that and is
the whole 25x of [[REFRACTION-ANTI-MERGING]]: it prevents repeat winners from
becoming hubs, so items overlap at chance.

If that mechanism is right, then a SCHEDULE that also denies any single item a
long consecutive run of reinforcement should do some of refraction's job for
it. That is a prediction about an interaction, and an interaction is a sharper
test of a mechanism than either main effect.

## What runs

`python -m research.runner presentation-schedule --tag UNIQUE`

Engine `hashed_assembly_memory`, the adopted memory protocol's operating
point: n = 4000, k = 100, p = 0.5, beta = 0.10, w_max = 20, normalized
initialization, no synaptic scaling, stimulus size k, half-cue rank-1 recall,
the refracted arm read with the bias MASKED and the control read plain. Seeds
42 to 61, twenty brains, paired across all four arms.

**The visit primitive.** Every arm is built from one primitive so the arms
differ only in ORDER: a visit to item i is one EPISODE, `inhibit_areas([A])`
then `project({s_i: [A]}, {A: [A]})` for `r` rounds, which is the capacity
protocol's write spent in a shorter episode. An item's stimulus fiber is
created once and reused across its episodes, so stimulus-side potentiation
accumulates across a schedule instead of restarting; this differs from
`AssemblyMemory.store`, which builds a fresh stimulus fiber per call and is
therefore not re-entrant for the same item. That difference is a declared
protocol decision, not a defect of either: the capacity protocol writes each
item once and has no need to revisit.

**Total rounds per item are held fixed at 16**, the published collapse point,
in every arm. Only their arrangement varies: `E = 4` episodes of `r = 4`
rounds. The capacity protocol's own cell spends the same 16 rounds as one
episode. A difference between arms therefore cannot be a difference in how
much training an item received.

> **Correction (2026-09-13, before any study run).** This registration first
> specified a visit as a single round. The VOID smoke
> (`ps-smoke-20260913`) showed that wrong: the protocol's recurrence only
> engages from the second round of an episode, so a one-round visit inhibits
> the area, reads the stimulus alone, stores nothing, and recalls at chance
> (rank-1 0.042 at M = 8 on the control, below the 0.125 a uniform guess
> gives). The primitive is now an episode of `r >= 2` rounds, the plan
> refuses `r < 2`, and a test pins both. No study has been run under either
> version and no bar below has changed.

**The arms.** E = 4 episodes of r = 4 rounds per item, 16 rounds in total,
the window where the control is recorded as collapsing.

- `massed-control`: items in order, each given its E episodes consecutively
  (`AAAA...BBBB...`), refraction strength 0.
- `interleaved-control`: E passes over all M items, each pass giving every
  item one episode (`ABC...ABC...`), refraction strength 0.
- `massed-refracted`: massed, strength 0.5 beta, masked readout.
- `interleaved-refracted`: interleaved, strength 0.5 beta, masked readout.

Every arm runs exactly `M x E` episodes, every item exactly E of them, and
every item exactly `E x r = 16` rounds. M
checkpoints: 8, 16, 32, 64, 128, 256.

**What is retained, per brain, arm and checkpoint.** Rank-1 half-cue recall
per item (so the per-item curve against write order is kept, not only its
mean); mean pairwise overlap between stored assemblies divided by the chance
level k/n, which is the hub statistic; area fill; and the stored assembly of
every item, defined for both schedules as its winners after its LAST episode.

Smoke (`--smoke --seeds 1 2 3`) runs M up to 8 at E = 2 and is VOID.

## Bars

Priors: the control collapses at T = 16 to `M* = 8` with pairwise overlap 15x
chance; the refracted arm at the same T holds `M* = 203`
(`PREREG_refraction_memory.md`, arms CTL T16 and T16).

- **SR-1, the interaction, primary.** At the largest checkpoint where the
  massed control's mean rank-1 recall is below 0.5, the interleaved control's
  paired per-brain rank-1 recall is higher on at least 18 of 20 brains, with a
  paired mean difference whose lower 95% bound exceeds 0.05.
  PREDICTION: passes. Denying early items a consecutive run should cost them
  their hub status. This is the bar the mechanism claim lives on.
- **SR-2, the schedule does nothing once refraction is on.** On the refracted
  arms the paired difference in rank-1 recall between interleaved and massed
  has a 95% interval containing zero at every checkpoint at or below 128.
  PREDICTION: uncertain, and the point of the arm. Refraction is claimed to
  already prevent hubs; if the schedule still moves the refracted arm, then
  either it is doing something other than hub prevention, or refraction is not
  doing all of what the register credits it with.
- **SR-3, the mechanism, not just the outcome.** At the SR-1 checkpoint the
  interleaved control's hub statistic (pairwise overlap over chance) is lower
  than the massed control's on at least 18 of 20 paired brains.
  PREDICTION: passes if SR-1 passes and the mechanism is hub prevention. SR-1
  passing while SR-3 fails would mean the schedule helps for some other
  reason, and the entry would say so.
- **SR-4, idle spacing is a no-op, asserted.** A fifth arm, `idle-massed`,
  runs the massed control schedule with an elapsed but empty gap between
  visits and must return stored assemblies BIT-IDENTICAL to `massed-control`
  on every brain and every item. This is the constructed true negative: the
  substrate has no time, so a gap containing nothing must change nothing, and
  a difference here voids the run rather than being reported as a spacing
  effect.
- **SR-5, instrument.** Every arm records exactly `M x E` episodes, exactly E
  per item and exactly 16 rounds per item, and the control arms' fill and recall at M = 8 reproduce
  the massed capacity protocol's published cell at T = 16 within 10%.
  PREDICTION: passes. Failure means the visit primitive is not the protocol's
  write and voids every other bar.

A bar that fails is recorded with its numbers and the entry is amended to what
twenty brains support. If SR-1 fails, the finding is that presentation order
does not matter in this substrate at matched count, which would put the whole
of the control's collapse on the number of repetitions rather than on their
arrangement, and would be reported as such.

## Scope stated in advance

One operating point, one T, one corpus of unrelated random stimuli. Nothing
here is a model of human spaced repetition: there is no forgetting curve to
fit because there is no forgetting between presentations. The result, either
way, is about interference scheduling in a k-WTA associative memory, and the
honest headline if SR-1 passes is that what looks like a spacing effect in
this substrate is an interference effect wearing a schedule's clothes.

If SR-1 passes and SR-2 shows the refracted arm unmoved, the natural follow-up
is the one this registration does NOT run: add an explicit weight-decay term
and ask whether a genuine time-mediated spacing effect appears. That is a
change to the substrate and needs its own registration.

## Result (2026-09-13): three bars fail, and presentation ORDER is worth about an order of magnitude of capacity

Artifact
`research/results/runs/memory.presentation-schedule/presentation-schedule-20260913/results.json`
(protocol version 1, seeds 42 to 61, pinned worktree at 47276039; smoke
`ps-smoke-episodes-20260913`, VOID). Verdict FAIL.

    bar                                              verdict
    SR-1 interleaving rescues the massed control     FAIL
    SR-2 the schedule does nothing once refracted    FAIL
    SR-3 the rescue comes with the hub statistic     FAIL
    SR-4 idle spacing is a no-op, bit-identical      PASS
    SR-5 instrument, matched counts                  PASS

### Mean rank-1 half-cue recall, twenty brains

    arm                        M=8    M=16   M=32   M=64   M=128  M=256
    massed-control             0.144  0.069  0.033  0.016  0.000  0.006
    interleaved-control        1.000  1.000  1.000  1.000  0.696  0.000
    massed-refracted           1.000  1.000  1.000  0.991  0.966  0.922
    interleaved-refracted      1.000  1.000  1.000  1.000  1.000  1.000

### Hub statistic, pairwise overlap over chance

    arm                        M=8    M=16   M=32   M=64   M=128  M=256
    massed-control            18.28  25.76  33.27  36.47  38.41  39.00
    interleaved-control        1.18   1.11   1.14   1.15   1.27  40.00
    massed-refracted           0.00   0.00   0.00   0.50   0.74   0.85
    interleaved-refracted      0.00   0.00   0.00   0.49   0.71   0.84

### Why SR-1 and SR-3 failed, and what that does not excuse

Both bars are evaluated "at the largest checkpoint where the massed control's
mean rank-1 is below 0.5". The massed control is below 0.5 at EVERY
checkpoint, never rising above 0.144, so the rule selected M = 256, the single
point where the interleaved arm has collapsed too. Paired
interleaved-minus-massed on the control, with 95% intervals:

    M=8     +0.8562  [+0.8276, +0.8849]   20/20 brains higher
    M=16    +0.9313  [+0.9222, +0.9403]   20/20
    M=32    +0.9672  [+0.9639, +0.9705]   20/20
    M=64    +0.9836  [+0.9820, +0.9852]   20/20
    M=128   +0.6961  [+0.5534, +0.8388]   20/20
    M=256   -0.0063  [-0.0099, -0.0026]   0/20   <- the checkpoint the rule chose

The bars are recorded FAILED and are not amended in place. The rule was
written expecting the massed control to be healthy at small M and to collapse
somewhere inside the grid, which is what the published cell does; this massed
arm is not healthy anywhere, and the reason is in the next section.

### An instrument defect: SR-5 was implemented smaller than it was registered

SR-5 as registered has two clauses: matched counts, AND that the control arms'
fill and recall at M = 8 reproduce the massed capacity protocol's published
cell at T = 16 within 10%. Only the first was implemented, so SR-5 passed on
counts alone. The second clause would have failed and would have caught the
problem before the bars were read: the massed control recalls 0.144 at M = 8,
where the published control's ceiling is M* = 8.

The cause is the episode split itself. The published cell spends its 16 rounds
as ONE episode; this study spends them as four episodes of four rounds, and
each episode begins with an inhibit, so the assembly re-forms from scratch
four times instead of converging once. For an unrefracted area that is
harmful, because every re-formation re-enters an area whose hubs have grown.
**The massed arm here is therefore not the published protocol**, and no claim
of the form "interleaving rescues the published control" is supported. What is
supported is a comparison at matched episode structure, which is what the
study ran.

### The mechanism is hub formation, and staleness is refuted

A rival reading of the whole effect: in the massed schedule an item's stored
assembly is recorded early and the remaining items are written on top of it,
while under interleaving every item's last episode falls in the final pass, so
every stored assembly is fresh. That reading predicts that recall in the
massed control should RISE with write order. Mean rank-1 by write order, in
eighths, earliest first:

    M=32   massed-control        0.04  0.19  0.04  0.00  0.00  0.00  0.00  0.00
           interleaved-control   1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00
    M=64   massed-control        0.11  0.02  0.00  0.00  0.00  0.00  0.00  0.00
           interleaved-control   1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00

The sign is the opposite: in the massed control only the EARLIEST items retain
anything and everything later is destroyed, which is hub formation, the first
items becoming the attractors that later items fall into. The interleaved
control is flat across write order. The hub statistic agrees, sitting at
chance (1.11 to 1.27) under interleaving where massed reaches 18 to 39 times
chance. Fill is close across arms at every checkpoint (0.22 against 0.24 at
M = 8), so this is not a difference in how much of the area was used.

### SR-2 failed in the informative direction: refraction does not absorb it

    refracted, paired interleaved - massed
    M=8, 16, 32   +0.0000  exactly zero, every brain
    M=64          +0.0094  [+0.0050, +0.0137]   11/20 higher, 0/20 lower
    M=128         +0.0336  [+0.0232, +0.0440]   17/20 higher, 0/20 lower
    M=256         +0.0781                        19/20 higher, 0/20 lower

Below M = 32 the refracted arm is saturated at 1.000 and the schedule cannot
show anything. Above it the schedule still adds, with intervals excluding zero
at M = 64 and M = 128, so SR-2 fails. Refraction is not doing the whole of
what a good schedule does.

### What the run supports, stated at the scope it was measured

At matched rounds per item and matched episode structure, presentation ORDER
moves an unrefracted Hebbian memory's usable load from below 8 items to
between 64 and 128, by the same anti-hub mechanism the register credits
refraction with. The protection is brittle where refraction's is graceful: the
interleaved control holds 1.000 to M = 64, 0.696 at M = 128, and falls to
0.000 with its hub statistic jumping from 1.27 to 40.00 at M = 256, while the
refracted arms degrade smoothly to 0.922 and 1.000. Nothing here is adopted:
three bars failed and the arms that produced these numbers were chosen after
seeing them fail.

## Amendment 1 (2026-09-13, registered before running): fixed checkpoints, the published arm restored, fresh seeds

Three corrections, none of which may be read off the run above without a new
test, so all of them are tested on the FRESH block **seeds 62 to 81**.

1. **No data-dependent checkpoint selection.** The comparison checkpoints are
   fixed here as M = 32 and M = 128 and are not chosen by any rule evaluated
   on the data.
2. **The published protocol is restored as an arm.** Protocol version 2 adds
   `single-control` and `single-refracted`, one episode of 16 rounds, which is
   the capacity protocol's own write. Three schedules crossed with two rules
   gives six arms, all at 16 rounds per item. `single` against `massed`
   isolates the cost of SPLITTING a round budget into episodes; `massed`
   against `interleaved` isolates the effect of ORDER. Version 1's four arms
   conflated them.
3. **SR-5's second clause is implemented**, as PS-7.

- **PS-1, order rescues the control.** At M = 32 the interleaved control
  exceeds the massed control on 20 of 20 paired brains with a paired lower
  bound above 0.5.
- **PS-2, and it does so by not building hubs.** At M = 32 the interleaved
  control's hub statistic is below 2.0 on every brain and the massed
  control's is above 10 on every brain.
- **PS-3, the write-order signature.** At M = 32 in the massed control, mean
  rank-1 over the first eighth of items exceeds that over the last eighth on
  at least 18 of 20 brains; in the interleaved control the two are within 0.10
  on at least 18 of 20. This is the bar that separates hub formation from
  staleness, and it is stated because the version-1 run could only report it.
- **PS-4, refraction does not absorb the schedule.** At M = 128 the refracted
  paired difference interleaved minus massed has a 95% lower bound above zero.
- **PS-5, the cliff.** The interleaved control's mean rank-1 is at least 0.4
  at M = 128 and at most 0.05 at M = 256.
- **PS-6, idle spacing is still a no-op.** SR-4 carried over unchanged.
- **PS-7, the splitting cost, and the published reproduction.** The
  `single-control` arm's mean rank-1 at M = 8 is at least 0.5, and exceeds the
  `massed-control` arm's at M = 8 on at least 18 of 20 brains.
  PREDICTION: passes. If it fails, the episode primitive does not reproduce
  the protocol's write at all and every comparison in this registration is
  between two things neither of which is the protocol.

The version-1 bars SR-1, SR-2, SR-3 stay failed and are reported on every run
as superseded. A bar that fails is recorded with its numbers.
