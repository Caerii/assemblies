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
