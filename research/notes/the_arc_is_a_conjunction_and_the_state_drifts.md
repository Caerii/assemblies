# The arc is a conjunction; the state assembly drifts

A1 result. Pre-registered in `PREREG_seq_a1_fsm_parity.md` (040f201, amended
twice before data in 06432df). Scripts: `seq_a1_fsm_parity.py`,
`seq_a1_step_accuracy.py`, `seq_a1_drift.py`. Enabling fixes: de7a32d (engine),
7588a44 (program).

## Bars

| arm | decided | across-state | across-symbol | max w |
|---|---|---|---|---|
| main (refraction, drive-proportional) | **5/10** | **0.000** | **0.000** | 7.40 |
| null (refraction off) | 0/10 | 0.885 | 0.999 | 20.00 |
| untrained | 0/10 | 1.000 | 1.000 | 0.00 |
| beta = 0 | 1/10 | 0.171 | 0.052 | 1.00 |
| constant-rule A/B | 0/10 | 0.900 | 1.000 | 20.00 |

P-PRE **pass** (0.000), P-CONJ **pass**, P-NULL **pass**, P-DEGEN **pass**,
P-GOLD **FAIL** (5/10 against a bar of 8/10).

## The conjunctive arc has an operating point

Across-state AND across-symbol arc overlap are both **0.000** on every seed:
33 transitions, 33 disjoint assemblies, no collapse in either direction.
Chance is k/n = 0.014, so this is below chance -- the assemblies are actively
tiled apart, not merely uncorrelated.

Task #92 concluded from a 0.90-0.99 reading that a conjunctive arc
"has no operating point". It has one. What #92 was missing was a force opposing
the more-exposed conjunct; see
`the_arc_collapses_onto_whichever_conjunct_fires_more.md`.

## Without the engine fix, this experiment would have read 0/10

The constant-increment rule -- what every engine here used before de7a32d --
is **indistinguishable from having no refraction at all** on our Brain:
0.900/1.000 and 0/10, against the ablation's 0.885/0.999 and 0/10. On the
reference a constant did have a (knife-edge, scale-bound) operating point;
here it has none at the strength every call site passes.

So the pre-fix repo could not have run this experiment at all, and the result
would have looked like an architectural failure rather than an engine defect.
That is the cost of a law implemented three times and wrong in all three.

## The failure is drift, not the transitions

Every single transition is learned perfectly: **330/330** over 10 seeds
(`seq_a1_step_accuracy.py`), digits and `end` alike. The pre-stated prediction
was 0.88-0.97 per-step accuracy if sequence failure were accumulated per-step
error; 1.000 falsifies that and selects the other pre-stated branch.

The mechanism is that a single-step test starts from the EXACT stored state
assembly, while step 2 of a sequence starts from whatever the arc recovered.
Overlap with the correct stored assembly decays along the sequence
(`seq_a1_drift.py`):

    seed 1   0.99  0.97  0.93  0.87  0.59  0.19     fails at step 6
    seed 8   1.00  0.99  1.00  0.99  0.99  0.97     decides
    mean     step 1 = 0.987  ->  final step = 0.327

Deciding seeds hold 0.824 mean overlap, failing seeds 0.685. Each step loses a
little, the loss compounds, and once overlap falls under about 0.5 the
nearest-overlap readout starts naming the wrong state.

Two channels are consistent with this and are not yet separated:

1. the state area is BELOW the regime floor -- its afferent kp from the arc is
   k*p = 14 against 3 ln 500 = 18.6, while the arc sits at 28 against 25.6 and
   is perfect. A recovery that is 99% right is what a slightly-underpowered
   k-cap looks like.
2. the arc's conjunction is maximally sharp (0.000 between assemblies), so it
   has no tolerance for an imperfect state input. Sharpness and robustness are
   in tension here, and refraction buys the first.

Nothing in these data chooses between them.

## Two honesty corrections

**5/10 overstates competence.** Seed 4 is counted as decided but diverges at
step 5 (`got '1', want '0'`) and recovers onto `accept` by luck. Only 4/10
seeds have a fully correct trajectory. A correct verdict over a wrong
trajectory is a sequence-level fake-perfect, and the verdict-only readout
cannot see it -- future runs should score the trajectory, not the last state.

**Our null is not the reference's null.** The refraction-off and constant-rule
arms both hit max weight exactly 20.00, our `w_max` clamp, which the reference
does not have. This was pre-stated as a risk and it fired. Both arms collapse
in the same direction as the reference's ablation, so the conclusion stands,
but the arms are not literally comparable.

## Which link drifts: the ARC, and the state area is cleaning up after it

`seq_a1_where_drift_starts.py` compares, at every step, what actually fired
against what an EXACT cue would have selected. Mean over 10 seeds:

    ours   arc    1.00  0.82  0.64  0.46  0.23  0.14
    ours   state  0.99  0.96  0.93  0.79  0.53  0.33
    ref    state  1.00  0.99  0.99  0.98  0.93  0.77

The arc crosses 0.9 at mean step index 1.4, the state at 3.5. The pre-stated
branch that fires is ARC SENSITIVITY: at step 2 the state assembly is still
96% correct while the arc has already fallen to 0.82. A conjunction with 0.000
between-assembly overlap has no tolerance for an imperfect input.

The second reading is the one worth keeping: **state overlap is HIGHER than arc
overlap at every step**. `arc -> state` is not lossy -- it is performing partial
CLEAN-UP, pulling 0.82 back to 0.96. The state area corrects; the arc's error
simply grows faster than the correction. That kills the "lossy recovery"
hypothesis and, with it, the regime explanation: an area doing clean-up is not
an area starved of afferents. (The reference's state area runs at the same
kp = 14 against the same 18.6 floor and decides 3/3, which said the same thing
a priori.)

The reference drifts too -- 0.93 and 0.77 on its last two steps -- but slowly
enough to keep the trajectory. Ours is 4/10 on trajectory, the reference 3/3.

So the question is no longer "which link" but "why is OUR arc so much more
sensitive than the reference's, given both are 0.000-overlap conjunctions".
That is [[kwta-amplifies-input-overlap]] territory: our k-WTA is known to
amplify rather than contract input differences, with beta as the gain.

## The arc is NOT the discrepancy: recovery has to be EXACT

`seq_a1_arc_transfer.py` perturbs the state cue by swapping m of its k neurons
for UNUSED ones and reads how far the arc assembly moves:

    input overlap  1.000  0.986  0.971  0.957  0.929  0.900  0.857  0.800
    ours  arc      1.000  0.805  0.710  0.643  0.476  0.390  0.248  0.129
    reference      1.000  0.767  0.719  0.524  0.390  0.310  0.233  0.071

The prediction was that ours amplifies and the reference does not. **It is
false, in the informative direction.** Both arcs amplify violently, and the
REFERENCE amplifies more: d(output loss)/d(input loss) is 10.33 for the
reference against 8.17 for us, and at m=3 the reference lands at 0.524 where we
land at 0.643. Our arc is the more tolerant of the two.

So arc sensitivity cannot be the discrepancy, and the previous section's
framing ("why is OUR arc so much more sensitive") was wrong. The reference wins
while being MORE sensitive, which leaves only one way it can work: its recovery
never introduces an error for the arc to amplify.

The per-step traces say exactly that. Reference seed 7 holds 1.00 at every one
of the six steps; seed 13 holds 1.00 for four. Ours begins at 0.99 -- about 69
of 70 neurons -- and compounds from there. With ~8x amplification per step, an
initial error of 1/70 = 0.014 reaches order 1 within five steps, which is the
observed collapse.

**The requirement is EXACT recovery, not good recovery.** The state area is a
discrete attractor and 69/70 is a failure, not a near-miss. This also explains
the per-seed pattern in both implementations: our seed 8 recovers exactly and
holds 0.97 to the end; reference seed 42 does NOT recover exactly and decays to
0.34 like ours.

That reframes the fix. It is not tolerance, not extra afferents in the general
sense, and not clean-up strength -- it is whatever makes the k-th winner's
margin large enough that recovery is exact every time.

## RESOLVED: honour the floor for EVERY area, and A1 passes 10/10

`seq_a1_exactness_sweep.py` sweeps `p`, which sets how many afferents an active
assembly delivers and therefore the margin at the k-th winner. The target
metric is the fraction of steps where recovery is EXACT, not mean overlap:

    p     state kp   exact steps   traj ok   decided   mean overlap
    0.20     14.0       4/100        4/10      5/10       0.812
    0.30     21.0      80/100       10/10     10/10       0.997
    0.40     28.0     100/100       10/10     10/10       1.000
    0.50     35.0     100/100       10/10     10/10       1.000

The state area's floor is 3 ln 500 = 18.6, i.e. p = 0.266. **The transition
happens exactly where the theory says it should.** Below the floor recovery is
essentially never exact (4/100) and the organ fails; above it, recovery is
exact and every seed runs the machine correctly.

Re-running the full bar set at p=0.4:

| arm | decided | across-state | across-symbol |
|---|---|---|---|
| main | **10/10** | 0.000 | 0.000 |
| null (refraction off) | 0/10 | 0.891 | 1.000 |
| untrained | 0/10 | 1.000 | 1.000 |
| beta = 0 | 0/10 | 0.309 | 0.106 |
| constant-rule A/B | 0/10 | 0.916 | 1.000 |

**P-GOLD, P-CONJ, P-NULL, P-DEGEN and P-PRE all pass.** A1 is closed: the
transition organ learns and RUNS a finite-state machine end to end, decided
from the assembly, on 10/10 seeds.

### What the earlier failure actually was

I put the ARC above its floor (kp = 28 vs 25.6) and left the STATE below its
own (14 vs 18.6), because I copied the reference's density without checking the
floor separately for each area. The reference survives marginally there -- 2 of
3 seeds recover exactly -- and our substrate does not. The regime condition is
per-AREA, and an organ is only in-regime when every area in it is.

### Two lessons worth keeping

**The mean hid an all-or-nothing mechanism.** At p=0.2 mean overlap reads a
respectable 0.812 while exactness is 4/100. Reporting the mean would have
suggested a system that mostly works and needs tuning; the truth was a system
whose attractor almost never landed. Pick the statistic the mechanism is
actually made of.

**Amplification was a red herring twice over.** The arc amplifies ~8x, the
reference ~10x, and it does not matter: with exact recovery there is no error
to amplify. Two of my predictions were falsified on the way here (per-step
error accumulation, then arc hypersensitivity), and both falsifications were
worth more than the confirmations would have been -- the first ruled out the
transitions, the second ruled out tolerance and pointed at exactness.

## How long can it run? 2000 steps, zero errors

If recovery is exactly exact there is no error to amplify, so the horizon
should be unbounded; a residual per-step error rate should instead give a
finite one. `seq_a1_horizon.py` runs one 2000-digit random string per seed and
records the index of the FIRST divergence from ground truth (one long run
yields every prefix).

    p = 0.4 (kp 28)   5/5 seeds: first error NEVER, accuracy 1.0000,
                      exact recovery on 2000/2000 steps
    p = 0.3 (kp 21)   4/5 seeds: first error NEVER, accuracy 1.0000
                      seed 1:    first error at step 759, accuracy 0.552

Comfortably above the floor the machine is a genuine unbounded-horizon
sequential computation: 10000 projections with not one wrong state. Just above
it, most seeds run forever and one falls out of the basin partway and never
returns -- note seed 1's accuracy after failure (0.552) is near chance for a
3-residue machine, so leaving the basin is terminal, not a stumble.

Two details worth keeping. Seeds 3 and 4 at p=0.3 hold accuracy 1.0000 while
recovering exactly on only ~92% of steps, so the attractor does pull a slightly
wrong state back -- exactness is sufficient but not strictly necessary, and the
basin has width. And the failure mode is a cliff rather than a decay, which is
what an amplifying map with a discrete attractor should do.

Caveat: 2000 is the limit of what was run, not a proof of unboundedness. This
is also a 3-state machine, so what is unbounded here is the STATE-TRACKING
horizon, not memory that grows with the input.

## Under constant input it is a limit cycle, closed exactly in assembly space

`seq_a1_limit_cycle.py`, 3 seeds x 10 digits, 60 steps each:

* measured period equals 3/gcd(d, 3) in **30/30** cases -- a fixed point for
  d in {0,3,6,9}, a 3-cycle otherwise;
* the orbit closes **exactly**: assembly at step t is bit-identical to step
  t+period, overlap 1.000 on every revolution, 30/30, with no decay over 20
  revolutions;
* different phases of a cycle share **0.000** overlap -- maximally separated.

So the driven map has genuine attracting periodic orbits, and they are periodic
in the full assembly state, not merely in the label readout.

The mechanism is the pairing of the two stages, and it is worth stating
plainly. The arc EXPANDS -- an ~8x amplifier that drives different
(state, symbol) pairs to 0.000-overlap assemblies. The state area QUANTIZES --
k-WTA maps a whole neighbourhood onto exactly one stored assembly in a single
step, which is superattracting, not merely contracting. Expansion gives
separation; quantization gives exactness. Together they are a clean discrete
dynamical system, and that composition is why error does not accumulate at all
inside a basin while failure outside one is a cliff.

Terminology, to be exact: this is a discrete map on a finite set, so "limit
cycle" is by analogy -- an attracting periodic orbit with a finite basin. Under
a varying input stream there is no cycle at all; the same attractors are
visited in an aperiodic, input-determined order. Which is the point: it is a
machine, not an oscillator.

## Next step (superseded -- A1 passed)

The registered rule for "P-GOLD fails while the reference passes" is that the
divergence is in OUR substrate, with the sparse sampler as prime suspect
([[sampler-is-the-whole-discrepancy]]), and the next step is the same
experiment on `numpy_exact` -- which needs refraction implemented there, the
gap C2 item 3 named. Amendment 2 additionally named the state area's regime as
the first suspect if the arc read conjunctive, which it did. Both point at the
same experiment: rerun on an exact substrate and sweep the state area's kp.
