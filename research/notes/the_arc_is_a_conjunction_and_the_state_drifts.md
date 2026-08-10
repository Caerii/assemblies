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

## Next step, pre-registered

The registered rule for "P-GOLD fails while the reference passes" is that the
divergence is in OUR substrate, with the sparse sampler as prime suspect
([[sampler-is-the-whole-discrepancy]]), and the next step is the same
experiment on `numpy_exact` -- which needs refraction implemented there, the
gap C2 item 3 named. Amendment 2 additionally named the state area's regime as
the first suspect if the arc read conjunctive, which it did. Both point at the
same experiment: rerun on an exact substrate and sweep the state area's kp.
