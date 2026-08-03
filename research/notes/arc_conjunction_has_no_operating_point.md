# The arc has no operating point, and the trade is 2:1 against

Follow-on to `conjunctive_arc_measured.md`. Three pre-registered predictions,
all three false; then the probe that produced the follow-up numbers turned out
to be reading the wrong index space, and the corrected numbers overturn this
note's own first conclusion.

> ⚠️ **RETRACTED, by me, one commit later.** This note first claimed a
> **conserved budget**: state-separation and mood-separation trading off with a
> sum "flat to within 0.05". That was measured with `set(area.winners)` --
> COMPACT ENGINE INDICES -- compared between two deepcopies of one brain. Under
> `frozen()` recruitment still runs, so the copies can hand the same compact
> index to different neurons. Re-measured on stable neuron IDs via
> `diagnostics.read_assembly`, the sum is **not** flat: it rises 1.60 -> 1.79.
> The trade-off is real; the conservation was an artifact. Every number below is
> the corrected one. Caught by `test_index_space_ratchet`, not by me.

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

**R1 refutes my own diagnosis.** I had recorded the size-confounded readout as
THE blocker (#95). Swapping to the reference's winner-to-winner sum -- which
cannot depend on recruited size -- changes the outcome by exactly nothing. The
readout is confounded, and the correction is still right on faithfulness
grounds, but "confounded" did not license "causal".

**R3 earned its place.** The arc arms do not merely score worse on multi-mood;
they cannot do SINGLE mood, where there is no mood to condition on and the arc
should behave exactly like the direct order synapse it replaced. An arm that
fails the floor was never producing an interpretable multi-mood number.

## Why: the arc encodes the symbol and drops the state

ARC assembly overlap on the trained model, **neuron IDs**:

    across STATES within one mood     0.90 - 0.99     <- should be LOW
    across MOODS for one state        0.48 - 0.97     <- should be LOW

So `A_{q,sigma}` is really `A_sigma`, and even the mood term is weak for some
states (V reads 0.97). With one mood the cue becomes a constant, which is
exactly why the single-mood floor collapses.

Mechanism is firing frequency: MOOD fires into ARC at every one of the three
constituents in every sentence, while any particular `syn_prev` fires only when
it happens to precede. MOOD accumulates roughly three times the potentiation and
takes the k-WTA.

## The trade-off, corrected

Sweeping `MOOD -> ARC` plasticity between the poles, two seeds, neuron IDs:

    MOOD->ARC beta   across-STATE   across-MOOD    sum
        0.000            0.60           1.00      1.60
        0.005            0.62           0.99      1.61
        0.010            0.68           0.97      1.66
        0.020            0.89           0.87      1.76
        0.040            0.99           0.80      1.79
        0.060            0.98           0.81      1.79

Monotone, **no interior optimum**, and the sum RISES. Going from beta=0 to
beta=0.06 spends **0.38 of state-separation to buy 0.19 of mood-separation** --
a strictly losing trade, roughly 2:1 against. By beta >= 0.04 the state is gone
entirely (0.98-0.99).

So there is no setting at which this area holds a conjunction, and it is worse
than a zero-sum split: raising the gain destroys more than it creates. The
best-looking cell (beta=0: state 0.60, mood 1.00) carries no mood information at
all, which is what makes it useless.

Reported as an observation, not a law: two seeds, one architecture, one
(n, k, p, beta).

## What the theorem asks for that we are not giving it

Dabagia/Papadimitriou/Vempala Thm 4 needs the arc under conditions we violate on
three axes at once:

* **p.** Their sequence experiments run `p = 0.2`; we are at `0.05`, the bottom
  of their own Fig. 8 sweep, where they report exactly our failure mode (high
  overlap between assemblies for distinct sequence elements). Task #93.
* **beta.** "The plasticity cannot be too high", and the presentations needed
  grow as it falls. We are at 0.06 with 60 sentences.
* **homeostasis.** Thm 2 renormalises incoming weights EVERY round; we apply
  `norm_init` once. Since the failure here is one input out-accumulating the
  other, per-round renormalisation is not a detail -- it is the thing that would
  stop the accumulation.

Capacity is not the constraint: `n >= |Q|^2|Sigma|^2` is a few hundred neurons
against our 1000, consistent with the collapse surviving n=1e5.

Next experiment is #93 with the conjunction as its readout -- raise p, lower
beta, add per-round renormalisation -- asking whether the trade RATIO moves off
2:1, rather than sweeping the split again.

## Three probe defects in one evening, all on the same object

1. Driving SYNTAX from MOOD alone: **1.000 at init** where the truth is ~0.04.
   Untrained weights all tie and the index tie-break returns identical winners.
2. Switching to `read_only()` for isolation: **1.000 again**, opposite cause --
   it freezes winners, so every mood got the same stale set.
3. Comparing `set(area.winners)` across deepcopies: compact indices, not neuron
   IDs. Produced the retracted conservation above.

(1) and (2) I found by disbelieving a round number. (3) I did not find -- the
repo's `test_index_space_ratchet` did, and by then I had already committed the
claim it invalidates. [[two-index-spaces-compact-vs-neuron-id]] is now a
repeat offender here, and this is the second time it has produced a *published*
number rather than just a wrong session reading.

Both diagnostics now go through `diagnostics.read_assembly`, run against a
deepcopy, and are verified non-perturbing: generation output is identical with
and without probing.
