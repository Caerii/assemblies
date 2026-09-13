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
> (`research/results/runs/memory.presentation-schedule/ps-smoke-20260913/results.json`,
> retained) showed that wrong: the protocol's recurrence only
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
`research/results/runs/memory.presentation-schedule/ps-smoke-episodes-20260913/results.json`,
VOID). Verdict FAIL.

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

## Amendment 1 result (2026-09-13): six of eight bars pass on the fresh block, and splitting and ordering do opposite things

Artifact
`research/results/runs/memory.presentation-schedule/presentation-schedule-fresh2-20260913/results.json`
(protocol version 2, six arms, seeds 62 to 81, pinned worktree at 9ba0f885).
PS-1, PS-2, PS-4, PS-5, PS-6 and PS-8 pass; PS-3 and PS-7 fail. Verdict FAIL.

    mean rank-1              M=8    M=16   M=32   M=64   M=128  M=256
    single-control           0.150  0.081  0.041  0.020  0.016  0.005
    massed-control           0.169  0.066  0.031  0.016  0.005  0.001
    interleaved-control      1.000  1.000  1.000  1.000  0.741  0.000
    single-refracted         1.000  1.000  1.000  0.763  0.452  0.198
    massed-refracted         1.000  1.000  1.000  0.990  0.964  0.925
    interleaved-refracted    1.000  1.000  1.000  1.000  1.000  1.000

    hub statistic            M=8    M=16   M=32   M=64   M=128  M=256
    single-control          35.61  36.62  37.85  38.43  39.09  39.34
    massed-control          20.17  28.67  33.31  37.25  38.47  39.16
    interleaved-control      1.04   1.11   1.14   1.16   1.25  40.00
    all three refracted      0.00   0.00   0.00   ~0.47  ~0.72  ~0.86

### The finding Amendment 1 was written to make possible

The version-1 arms could not tell splitting from ordering. With the
single-episode arm restored they separate completely, and in opposite
directions for the two rules:

    rule        M      single -> massed   (splitting)   -> interleaved  (ordering)
    control     64     0.020 -> 0.016       -0.004        -> 1.000        +0.984
    control     128    0.016 -> 0.005       -0.010        -> 0.741        +0.735
    control     256    0.005 -> 0.001       -0.005        -> 0.000        -0.001
    refracted   64     0.763 -> 0.990       +0.227        -> 1.000        +0.010
    refracted   128    0.452 -> 0.964       +0.512        -> 1.000        +0.036
    refracted   256    0.198 -> 0.925       +0.727        -> 1.000        +0.075

**For the unrefracted memory, ORDER is the whole effect and splitting is
nothing. For the refracted memory, SPLITTING is the whole effect and order
adds a little.** Both are at 16 rounds per item; no arm receives more training
than another.

The control's story is the one the registration predicted: interleaving keeps
the hub statistic at chance (1.04 to 1.25) where both massed schedules run to
20 to 39 times chance, and recall follows. PS-1 and PS-2 pass, 20 of 20 brains
each.

> **The reading below is WITHDRAWN (2026-09-13).** Amendment 3 of
> `PREREG_refraction_period_law.md` tested it directly, moving the tenure with
> refraction strength while holding capacity inside the register's plateau.
> The penalty moved the WRONG WAY: it grows as the episode becomes a smaller
> share of a tenure (+0.8016, +0.8328, +0.8641 at shares of 38.5%, 33.9% and
> 26.9%), with non-overlapping intervals at the extremes. The episode penalty
> is real and it is not the tenure. A better candidate, also post hoc, is the
> number of episodes rather than their length: each episode's first round is
> driven by the stimulus alone, so the episode count sets the anchor to
> recurrence ratio that [[CAP-ANCHOR-RATIO]] identifies.

The refracted arm's story is new and was not predicted. One long episode is
markedly worse than four short ones at load: 0.198 against 0.925 at M = 256.
The plausible mechanism, stated as a reading and not adopted, is the bias
itself. Refraction charges per win, so sixteen consecutive winning rounds
leave the assembly carrying sixteen units of bias against the neurons it just
recruited, and the stored assembly is the one selected BEFORE that charge is
paid. Four episodes re-form the item under the bias as it accumulates, so the
retained assembly is one the charged area still selects. The relocation period
measured in `PREREG_refraction_convergence.md` (Amendment 2) is about 41
rounds at this operating point, and a single 16-round episode is a substantial
fraction of one tenure, which is why this reading is worth testing rather than
asserting.

PS-4 passes: interleaving still adds on the refracted arms at M = 128 with the
paired interval excluding zero, so refraction does not absorb the schedule.
PS-5 passes: the interleaved control holds 0.741 at M = 128 and falls to 0.000
at M = 256, its hub statistic jumping 1.25 to 40.00, so its protection ends in
a cliff while refraction degrades smoothly. PS-6 passes: idle spacing remains
a no-op bit for bit.

### PS-3 FAILED: the write-order signature is a mean, not a per-brain fact

    massed first eighth      mean 0.037   range [0.00, 0.25]
    massed last eighth       mean 0.000   range [0.00, 0.00]
    interleaved first        mean 1.000
    interleaved last         mean 1.000

The interleaved clause passes 20 of 20. The massed clause fails: first beats
last on 3 of 20 brains and the two are EQUAL on 17 of 20, because at M = 32
both ends sit on the floor for most brains. The direction is right in the mean
and the effect is real in the version-1 curve at M = 32 in eighths
(0.04, 0.19, 0.04, then zeros), but a bar demanding a strict per-brain
ordering between two quantities that are both usually zero cannot pass. The
bar is kept failed. A checkpoint nearer the control's edge, or a comparison of
the first eighth against zero rather than against the last, is what would test
it, and neither may be chosen now on this data.

### PS-7 FAILED: this instrument's write is NOT the published capacity cell

    single-control at M = 8   mean 0.150   range [0.12, 0.25]   bar >= 0.5
    beats massed-control      3 of 20 brains                    bar 18 of 20

The single-episode arm is one episode of sixteen rounds, which is the capacity
protocol's own write, and it recalls 0.150 at M = 8 where the published
control's ceiling is M* = 8. So the arm does not reproduce the published cell,
and **no number in this registration may be compared with the capacity line's
published cells.** Every comparison reported above is internal: all six arms
share one instrument, so splitting and ordering are measured against each
other and not against the register.

The leading candidate cause, identified and not tested: this instrument reads
with a fixed 8 frozen completion rounds, while `AssemblyMemory.recall` uses as
many rounds as the protocol writes with, which is 16 for a T = 16 cell. A
half-cue completed for 8 rounds instead of 16 is a weaker read, and the
control, whose assemblies are hub-contaminated, has the most to lose from it.
The hub statistic at M = 8 is already 35.6 on the single arm against the
published control's 15 times chance at its collapse, which points the same
way.

## Amendment 2 (2026-09-13, registered before running): make the read the protocol's read

One change, and it is the candidate cause named above: the completion read
uses as many frozen rounds as the arm writes per item, 16, rather than a fixed
8. Everything else is held. Protocol version 3. Tested on seeds 42 to 61,
which have seen version 1's four arms but never the single-episode arm nor
this read.

- **PS-9, the instrument reproduces the protocol.** The `single-control` arm's
  mean rank-1 at M = 8 is at least 0.5, and its hub statistic at M = 8 is
  below 20 times chance.
  PREDICTION: uncertain. This is PS-7 restated against the one change. If it
  fails again, the difference is not the read and the instrument's write must
  be diffed against `AssemblyMemory.store` line by line before any further
  comparison is reported.
- **PS-10, the dissociation survives the read.** With the longer read, the
  splitting gain on the refracted arm at M = 256 (`massed` minus `single`) has
  a paired lower bound above 0.3, and the ordering gain on the control at
  M = 64 (`interleaved` minus `massed`) has a paired lower bound above 0.5.
  PREDICTION: passes. A change in how a stored assembly is READ should not
  reverse which schedule stored it better.
- **PS-11, the write-order signature, restated where it can be seen.** At
  M = 8, where the massed control is not yet at the floor, its first-eighth
  mean rank-1 exceeds its last-eighth mean on at least 16 of 20 brains.
  PREDICTION: uncertain, and PS-3's failure is the reason it is restated at a
  checkpoint chosen for having signal rather than for being deep in collapse.

PS-3 and PS-7 stay failed and are reported on every run as superseded, with
SR-1, SR-2 and SR-3.

## Amendment 2 result (2026-09-13): the read was not the cause, the instrument IS the protocol, and the write-order reading is WITHDRAWN

Artifact
`research/results/runs/memory.presentation-schedule/presentation-schedule-read16-20260913/results.json`
(protocol version 3, seeds 42 to 61, pinned worktree at 7f9ae3c2). PS-1, PS-2,
PS-4, PS-6, PS-8 and PS-10 pass; PS-3, PS-5, PS-7, PS-9 and PS-11 fail.

    mean rank-1, 16-round read   M=8    M=16   M=32   M=64   M=128  M=256
    single-control               0.175  0.091  0.044  0.020  0.017  0.002
    massed-control               0.144  0.069  0.033  0.016  0.004  0.001
    interleaved-control          1.000  1.000  1.000  1.000  0.355  0.000
    single-refracted             1.000  1.000  1.000  0.762  0.408  0.195
    massed-refracted             1.000  1.000  1.000  0.990  0.963  0.846
    interleaved-refracted        1.000  1.000  1.000  1.000  1.000  1.000

### PS-9 FAILED, and a parity test settles why: the bar's premise was wrong

Lengthening the read from 8 frozen rounds to 16 moved the single-episode
control from 0.150 to 0.175 at M = 8, nowhere near the 0.5 the bar asked for,
with its hub statistic at 35.5. The read was not the cause.

The registration said that if PS-9 failed the write must be diffed against
`AssemblyMemory.store` line by line before any further comparison is reported.
That diff is now a test,
`neural_assemblies/tests/test_presentation_schedule_parity.py`, and it shows
the write is **bit-identical**: one episode of sixteen rounds through
`ScheduledMemory.visit` returns the same sorted winners as
`AssemblyMemory.store` on every one of eight items and four brains, with
identical fill. A true negative in the same file confirms the test can fail,
by showing that splitting the budget DOES change the winners.

So the instrument is the protocol, and the fault is in what PS-7 and PS-9
compared it to. `M* = 8` was read as "recall near 0.5 at M = 8". It is not:
`ceiling_from_curve` returns the grid's FIRST point whenever NO point clears
the threshold (`research/experiments/_substrate.py`, the `if not above`
branch), so `M* = 8` on a grid that starts at 8 means the control was already
below the bar there. Recall of 0.15 to 0.18 at M = 8 is exactly what that
describes, and the third test in the parity file pins the sentinel so the
misreading cannot recur.

**The Amendment 1 caveat is therefore LIFTED.** Numbers from this protocol may
be compared with the capacity line, because the write is the same write. PS-7
and PS-9 stay recorded as failed; they failed on their premise.

### PS-3 and PS-11 FAILED, and the write-order mechanism is NOT established

PS-11 put the write-order signature at M = 8, where the massed control is not
yet on the floor. It failed, and in the OPPOSITE direction to the prediction:

    massed control, M = 8    first item 0.050    last item 0.200
                             first > last on 1 of 20 brains

At M = 8 the LAST item is the one that survives, which is recency, not
primacy. At M = 32 the version-1 curve in eighths was 0.04, 0.19, 0.04 and
then zeros, which is neither: a bump just after the start. PS-3 failed there
because both ends sit on the floor on 17 of 20 brains.

**The reading offered in the version-1 result, that the effect is hub
formation by the items written FIRST and that staleness is refuted, is
withdrawn.** It was drawn from checkpoints where the massed control is at or
near zero everywhere, which cannot support a claim about which items survive.
What survives the withdrawal is the part the hub statistic carries directly,
and that is unaffected: the massed schedules run at 20 to 39 times chance
overlap while interleaving holds 1.04 to 1.27, so interleaving demonstrably
prevents the merging. WHICH items become the attractors is not established by
this instrument, and a study that wants it needs checkpoints chosen for having
signal and a per-item curve as its primary quantity rather than a by-product.

### PS-5 FAILED under the longer read, which is itself a result

    interleaved control at M = 128    8-round read 0.741    16-round read 0.355

More completion rounds make the interleaved control WORSE at the edge of its
range, while leaving every arm at M <= 64 at 1.000 and the refracted arms
essentially unchanged. A longer frozen completion gives a half cue more
opportunity to walk into a neighbouring basin, and the interleaved control at
M = 128 is exactly where basins are closest. Reported, not barred: the cliff
in PS-5 is real in both reads, and only its position moved.

### PS-10 PASSED: the finding is robust to the read

    refracted, massed minus single at M = 256    +0.6508  [+0.6165, +0.6851]
    control, interleaved minus massed at M = 64  +0.9836  [+0.9820, +0.9852]

The dissociation between splitting and ordering survives a change in how a
stored assembly is read, which is what PS-10 was written to check. That, with
PS-1, PS-2 and PS-4, is what this registration supports:

- an unrefracted memory needs ORDER, and interleaving takes it from 0.14 to
  1.000 through M = 64 while holding the hub statistic at chance;
- a refracted memory needs SHORT EPISODES, worth +0.65 at M = 256;
- neither substitutes for the other, since interleaving still adds on the
  refracted arms and the interleaved control still falls off a cliff;
- idle spacing changes nothing, bit for bit, because the substrate has no
  decay term.

What it does not support is any account of WHICH items are lost.
