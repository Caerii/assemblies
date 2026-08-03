# The arc cannot hold both halves of the conjunction

Follow-on to `conjunctive_arc_measured.md`. Three pre-registered predictions,
all three false, and the reason is a cleaner result than the predictions were.

---

## What was predicted, and what happened

`research/experiments/task95_readout_and_arc.py`, 2x2 over scoring x arc, four
mood pairs x four seeds:

      scoring   arc   multi-mood   1-mood  signal/bias
     pre_kwta False      24/32        4/4         0.01
     pre_kwta  True       7/32        1/4         0.22
      winners False      24/32        4/4         0.01
      winners  True       6/32        0/4         0.30

    R1 `winners` beats `pre_kwta` at arc=off:  FALSE   24/32 -> 24/32
    R2 the arc's effect improves:              FALSE   -17 -> -18
    R3 single-mood floor is 4/4 everywhere:    FALSE   [4, 1, 4, 0]

**R1 is the one that matters, and it refutes my own diagnosis.** I had recorded
the size-confounded readout as THE blocker (#95). Swapping to the reference's
winner-to-winner sum -- which cannot depend on recruited size -- changes the
outcome by exactly nothing. The readout is confounded and the correction is
still right on faithfulness grounds, but it was not what was holding the result
back, and "the metric is confounded" did not license "the metric is the cause".

R3 was the useful pre-registration. The arc arms do not merely score worse on
multi-mood; they cannot do SINGLE mood, where there is no mood to condition on
and the arc should behave exactly like the direct order synapse it replaced.
An arm that fails the floor was never producing an interpretable multi-mood
number, and without R3 I would have spent the evening comparing two nuisance
terms.

## Why: the arc encodes the symbol and drops the state

Measured on the trained model, ARC assembly overlap:

    across STATES within one mood     0.90 - 0.98     <- should be LOW
    across MOODS for one state        0.46 - 0.78

So `A_{q,sigma}` is really `A_sigma`. With one mood that makes the cue a
constant, which is exactly why the single-mood floor collapses.

The mechanism is firing frequency. MOOD fires into ARC at every one of the three
constituents in every sentence; any particular `syn_prev` fires into ARC only
when it happens to be the preceding constituent. MOOD therefore accumulates
roughly three times the potentiation and wins the k-WTA.

## And it is a conserved budget, not a tuning problem

Freezing `MOOD -> ARC` (beta = 0) recovers the state and loses the mood. Sweeping
the plasticity between those poles, two seeds:

    MOOD->ARC beta   across-STATE   across-MOOD    sum
        0.000            0.57           1.00       1.57
        0.005            0.57           0.99       1.56
        0.010            0.62           0.95       1.57
        0.020            0.82           0.78       1.60
        0.040            0.91           0.69       1.60
        0.060            0.93           0.67       1.60

**Monotone, with no interior optimum, and the sum is flat to within 0.05.**
Every unit of mood-separation is bought with a unit of state-separation. There
is no setting of the gain at which the area holds a genuine conjunction.

That is a statement about a single shared k-WTA area with two competing inputs:
it has a fixed budget of distinguishability to allocate between them, and the
gain only decides the split. It belongs next to
[[mood-collapse-is-a-drive-ratio]], [[kp-decides-whether-beta-helps]] and
[[beta-opposes-capacity-and-depth]] -- the recurring shape where one knob has
two opposed effects and the interesting question is what changes the BUDGET
rather than the split.

Reported as an observation, not a law: two seeds, one architecture, one
(n, k, p, beta).

## What the theorem asks for that we are not giving it

Dabagia/Papadimitriou/Vempala Thm 4 does not merely require an arc area; it
requires it under conditions we violate on three axes at once:

* **p.** Their sequence experiments run at `p = 0.2`; we are at `0.05`, the
  bottom of their own Fig. 8 sweep, where they report exactly our failure mode
  (high overlap between assemblies for distinct sequence elements). Task #93.
* **beta.** "The plasticity cannot be too high", and the number of presentations
  needed grows as it falls. We are at 0.06 with 60 sentences.
* **homeostasis.** Thm 2 assumes incoming weights are renormalised EVERY round.
  We apply `norm_init` once. Since the failure here is one input out-accumulating
  the other, per-round renormalisation is not a detail -- it is the thing that
  would stop the accumulation.

Capacity is not the constraint: `n >= |Q|^2|Sigma|^2` is a few hundred neurons
and we have 1000, consistent with the collapse surviving n=1e5.

So the next experiment is #93 with the conjunction as its readout -- raise p,
lower beta, add per-round renormalisation, and ask whether the BUDGET moves,
rather than sweeping the split again.

## One correction kept even though it changed nothing

`scoring="winners"` is now the default. The port had described `pre_kwta` as an
improvement over the reference on the grounds that a winner-sum "biases the
comparison toward whichever area recruited more neurons". That reasoning is
wrong -- the reference's sum has `k_from * k_to` terms and both are always `k`.
The change is outcome-neutral here (24/32 either way) and is kept because the
stated justification for the old default was false, not because it helped.
