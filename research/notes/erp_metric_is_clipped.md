# The ERP Cohen's d is computed on a variable clipped at its own null

#102 / #32. Setting out to test whether the ERP separation survives a parse
that does not grow the parser, I found that it does -- and that the metric the
separation is scored on cannot support the claim being made with it.

## The read-only parse is safe

`calibrate_erp_thresholds` on SENTENCES, 10 seeds, paired:

    metric            parse grows [HEAD]     parse read_only        delta
    neurons recruited   1106.0 +/- 65.5        120.0 +/- 0.0      -986.0 +/- 65.5
    p600 Cohen's d       1.632 +/- 0.401       3.973 +/- 0.060     +2.342 +/- 0.398
    p600 median gap     0.0030 +/- 0.0004     0.0047 +/- 0.0004   +0.0017 +/- 0.0003
    n400 Cohen's d       2.120 +/- 1.038       2.120 +/- 1.038      IDENTICAL

The separation does not collapse; it strengthens, the recruitment drops by 986
of 1106, and the residual 120 is EXACTLY 120 on every seed -- the documented
below-`k` exemption, deterministic. On the reproducibility axis this is a clear
win: an evaluation that is not idempotent is broken regardless of what it
measures.

## But the d improvement is variance, not effect

    p600_excess(value) = max(0.0, value - baseline.p600_median)

and `ErpBaseline` is documented as "median over recent grammatical parses". The
GRAMMATICAL condition is therefore clipped against its own median **by
construction**. Measured, seed 11:

    parse grows      grammatical  p600_excess = [0.0, 0.0016, 0.0]   sd 0.00092
                     violation    p600_excess = [0.0021, 0.0037, 0.0032]
    parse read_only  grammatical  p600_excess = [0.0, 0.0002, 0.0]   sd 0.00012
                     violation    p600_excess = [0.0044, 0.0043, 0.0040]

Two of three grammatical samples are EXACTLY ZERO in both arms. Cohen's d is
`(mean_c - mean_g) / pooled_sd`, and the pooled sd is dominated by a near-
constant floor. So ANY reduction in measurement noise inflates d without the
effect being larger: 1.632 -> 3.973 while the absolute gap moved 0.0030 ->
0.0047.

**`d > 0.3` is therefore close to a vacuous test threshold.** Against a null arm
pinned at a floor, almost any nonzero violation excess clears it. Every Cohen's
d recorded in this package is a clipped-variable statistic, not an effect size.

## And the raw quantity is saturated, in BOTH arms

    parse grows      grammatical raw p600 = [0.9881, 0.9930, 0.9881]
                     violation   raw p600 = [0.9935, 0.9951, 0.9946]
    parse read_only  grammatical raw p600 = [0.9879, 0.9911, 0.9879]
                     violation   raw p600 = [0.9953, 0.9952, 0.9949]

Everything lives in [0.9879, 0.9953] -- **0.7% of the [0,1] range**. P600 is
`1 - normalized_energy`, so the role area is receiving ~1% of its normalizing
scale in every condition, grammatical included. The premise "a trained pathway
delivers HIGH energy, an untrained one LESS" is running as 0.011 against 0.005.

The separation is real and the direction is right -- and read_only widens it,
0.0047 -> 0.0062 raw. But it is a 0.6% difference between two numbers both
pinned near 1.0. That is #32's saturated half, and **the read-only parse does
not fix it.** It is a normalization problem in the metric, not a protocol one.

## A correction to my own previous commit

`ac22d4a` says: "note what it reports: P600 0.43, not the 0.99 the shipped path
gives -- the shipped protocol sits at the ceiling". That was over-read from ONE
probe on ONE word (the determiner "the", first position). Across the calibration
samples BOTH arms sit at ~0.99. The read-only parse does not move the metric off
the ceiling in general, and I should not have generalized from a single word.

## N400 is insensitive to the parse entirely

`compare_arms(strict=True)` would have raised: n400 Cohen's d is bit-identical
across all 10 seeds, and the per-sample `n400_excess` values match exactly
([0.0, 0.0265, 0.0] grammatical, [0.0237, 0.0282, 0.0338] violation) in both
arms. Whether the parse recruits makes no difference to N400 at all.

That is consistent with #28 ("N400 is saturated") and locates it further: the
N400 path does not read incremental parse state, so it cannot be contaminated by
parse growth -- and equally cannot benefit from fixing it.

## What this means for the order of work

Adopting the read-only parse is well-evidenced on reproducibility and does not
harm the separation. But its VALUE is limited while the metric is clipped and
saturated: it buys idempotence, not interpretability.

The metric comes first. Until `p600_excess` stops being scored against its own
null, and until the raw quantity uses more than 0.7% of its range, no Cohen's d
from this package should be quoted as an effect size -- including the ones in
[[erp_probe_isolation]] earlier today, which are subject to exactly the same
critique.
