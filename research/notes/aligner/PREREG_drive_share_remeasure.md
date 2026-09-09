# PREREG: re-measure the drive-share results on the fixed stimulus divisor

Registered before running. `27a5779` fixed `norm_init`'s stimulus divisor: it
was a VIEW of the live weights (`asarray` on float32), so the divisor moved
with every potentiation and the stimulus contribution `w / (w + unknown*p)`
drifted toward 1 -- cancelling 9-36% of the stimulus's LEARNED gain, growing
with training depth. Every drive-share result below was measured on that biased
engine (all predate the fix; the phon sweep is 0875b6e, 2026-08-07).

## What is re-run, and what is read

**R1 -- `parser_phon_weight_sweep.py`, verbatim** (same seeds 11/42, same grid
W in {1, 3, 6} x beta in {0.10, 0.05}). The committed table said:

    beta/W          dup     ret@6 P/A       GROUND
    0.10 / 1        0.317   0.806/0.733     0.819
    0.05 / 6        0.063   0.974/0.964     0.671

The fix RAISES the stimulus's effective learned contribution, i.e. it acts in
the same direction as raising W. Stated expectation: the un-reweighted arm
(W=1) improves, and the optimal W may fall. Possible outcomes, stated now:

  * table reproduces within seed noise -> the finding survives; the damping was
    not load-bearing for THIS result; say so and stop.
  * W=1 improves materially but W=6 still wins on dup+retrieval without
    grounding collapse -> the LEVER survives, the CALIBRATION shifts; report
    the new curve.
  * W=1 now matches W=6 -> phon_weight=6 was largely compensating for the bug;
    the memory node is rewritten and the parser default revisited.

Judged on the same three metrics the original used, with its own reading rules.
Two seeds is what the original had; conclusions stay comparative (same seeds,
same script, engine the only change).

**R2 -- `mood_drive_decomposition.py`**: the MOOD/SYN ratio is area-fiber
drive, but the learner also fires stimuli under norm_init; whether the ~4%
share moves is an empirical question. Re-run, compare the ratio.

**R3 -- Zipf/frequency** (`hebbian-mass-follows-frequency`,
`zipf-gain-must-come-from-corpus`): NOT re-run here -- corpus-scale training is
a larger unit. Recorded as still-standing-on-biased-engine, with direction:
head words received the most presentations, so their stimulus gains were damped
MOST, and the frequency effect was understated.

---

## Result (2026-08-25)

**R1 -- the finding SURVIVES; outcome (a) with a caveat on magnitudes.**
Same script, same seeds, engine the only change:

    beta/W        OLD (biased engine)                NEW (fixed divisor)
    0.10 / 1.0    dup 0.317  ret 0.806/0.733  GRD 0.819    dup 0.287  ret 0.747/0.721  GRD 0.796
    0.10 / 6.0    --                                        dup 0.115  ret 0.904/0.915  GRD 0.677
    0.05 / 1.0    --                                        dup 0.333  ret 0.828/0.772  GRD 0.742
    0.05 / 6.0    dup 0.063  ret 0.974/0.964  GRD 0.671    dup 0.086  ret 0.924/0.972  GRD 0.621

The structure is identical: duplicates fall monotonically with W in both betas
(0.287 -> 0.115 and 0.333 -> 0.086), retrieval rises (0.734 -> 0.910 and
0.800 -> 0.948), grounding declines moderately, and the script's own registered
reading fires the same clause both times -- "an INTERIOR optimum, report it
with the trade-off, not alone". W = 6 at beta = 0.05 remains the adopted cell.
Individual metrics shift by 0.02-0.08 at n=2 seeds, which does not clear seed
noise; no direction is claimed for the shifts, and GROUND being uniformly
slightly lower is noted as CONSISTENT with the fix strengthening the phon
stimulus (a stronger identity channel leaves grounding a smaller share) without
being established by it. The damping was NOT load-bearing for this result.

**R2 -- resolved ANALYTICALLY, no run.** `mood-collapse-is-a-drive-ratio` ran
on `WordOrderLearner`, which is `norm_init=False` by default (literature-
reproduction convention, confirmed in the constructor) and drives every
projection area-to-area with an EMPTY stimulus map. `_norm_scale` returns None
before the 1-D branch when norm_init is off, so the buggy path never executed.
The MOOD/SYN ~4% ratio stands because the bug was never on for it.

**R3 -- exposure CONFIRMED, re-run deferred as registered.** `zipf_grammar.py`
builds `Brain(p=P, seed=seed, synaptic_scaling=SCALING)` -- norm_init defaults
True -- and fires stimuli per Zipf frequency: the bug's exact habitat, with
head items fired most and therefore damped most. The standing claims are
recorded as biased-engine measurements whose frequency effect is UNDERSTATED;
the direction strengthens, not weakens, the qualitative conclusion. A full
re-run (112 cells, hours) is its own unit.

---

## R3 follow-up (2026-08-25): the Zipf re-run, on the fixed engine

Run from the worktree at current dev (`zipf_grammar.py`, 4 skews x 7 gains x
4 seeds, n=4000 k=50 M=64; log committed as
`zipf_rerun_fixed_divisor.log`).

**The mechanism REPRODUCES.** Uniform (s=0) holds acc = 1.000 across
g = 1.40-1.96; the skewed arms collapse at those same fixed gains (0.04-0.22)
and prefer the smallest gain in the grid, monotonically in skew:

    best in-grid gain / acc:  s=0.0 -> 1.40-1.96 / 1.000
                              s=0.7 -> 1.08 / 0.926
                              s=1.0 -> 1.08 / 0.781
                              s=1.3 -> 1.08 / 0.281

which is `zipf-gain-must-come-from-corpus`'s content: effective gain is
g^count, so under Zipf a fixed gain over-potentiates the head and the gain
must be DERIVED downward from corpus statistics.

**The numeric headline is NOT re-judged.** The script's grid has evolved since
the 2026-07-29 measurement (the committed `zipf_gain.csv` carries gains
1.03-1.96 with depth/level columns from mixed historical runs; the current
script runs 1.08-2.20 without them), so the old 0.152-vs-1.000 figure has no
same-grid counterpart. No number is carried over; the qualitative conclusion
stands re-verified on the fixed engine, and at s=1.3 even the best in-grid
gain reaches only 0.281 -- deriving the gain matters MORE at high skew, not
less.
