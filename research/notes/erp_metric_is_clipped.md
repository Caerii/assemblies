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

## WHY the clipping is there, and what actually went wrong

The clipping is not a mistake. It was correct for a quantity that no longer
exists.

`gates.py` records the scale it was designed against:

    Grammatical null:   N400 ~ 0.09,  anchored P600 ~ 0.02-0.12
    Category violation: N400 ~ 0.35,  anchored P600 ~ 5.0 (cumulative)

P600 was UNBOUNDED and CUMULATIVE, with a **44x** separation. For that quantity,
"excess over baseline, clipped at zero" is exactly right: it is a DETECTOR
statistic. You do not care how far below the null you sit, only how far above,
and `P600_EXCESS_MARGIN = 0.152` sits sensibly between 0.12 and 5.24.

Then `adapters.py` replaced the quantity. Its module docstring says why, and the
reason was good -- post-k-WTA churn reverses sign under `norm_init`, measured at
Cohen's d = -1.94. So P600 became `1 - normalized_energy`, **bounded in [0,1]**,
and now lives at 0.989 vs 0.995.

**The quantity was replaced. The statistic, the clipping, and the constants
around it were not.** That is the root cause, and it is this repo's dominant
defect pattern wearing new clothes: one name, `p600`, meaning two things on
scales three orders of magnitude apart, with the machinery keeping the old
meaning ([[same-name-two-meanings]]).

### Consequence: the P600 detector cannot fire

    seed 11: source=empirical  p600_margin=0.0760  max observed excess=0.0064  -> 11.9x below
    seed 12: source=empirical  p600_margin=0.0760  max observed excess=0.0064  -> 11.9x below
    seed 42: source=empirical  p600_margin=0.0760  max observed excess=0.0063  -> 12.1x below

    wobbly signatures = {}   STRUCTURAL fires = 0    on every seed

And the "empirical" label is false. The tuning is

    p600_margin = max(fb.p600_excess_margin * 0.5, (g_p75 + c_p25) / 2.0)

so the hardcoded floor `0.152 * 0.5 = 0.076` always beats the data-derived
midpoint (~0.0021). The calibration reports `source="empirical"` while using a
constant tuned for the old scale. This is [[silent-no-op-dead-fibers]] at the
level of a metric: configured, wired, labelled, and unable to fire.

## How it should be designed

**1. Separate the DETECTOR from the EFFECT MEASUREMENT.** They are different
jobs and want different statistics. A detector needs a threshold and one-sided
clipping is fine. An effect measurement must never see clipped data.

**2. Score separation with a RANK statistic, on the RAW value.** Implemented as
`diagnostics.separation` -> AUC (probability of superiority), null 0.5. It is
invariant under every monotone transform, so it is unmoved by clipping, by
rescaling, and by the redefinition that actually happened here. Measured on four
encodings of one identical ordering, Cohen's d spanned 2.241 to 24.754 while AUC
was 1.000 throughout.

What the ERP contrast reads in the honest statistic:

    parse grows      p600 AUC 0.889 (span 0.0071)   n400 AUC 0.778 (span 0.0329)
    parse read_only  p600 AUC 1.000 (span 0.0081)   n400 AUC 0.778 (span 0.0329)

P600 goes from one misordered pair in nine to perfectly ordered -- a real,
interpretable improvement. And **N400 is 0.778, not the ~1.0 its Cohen's d of
1.79-2.50 implied**: nearly a quarter of pairs are misordered. With n=3 per arm
the resolution is 1/9, which is itself worth stating.

**3. Report the SPAN beside it.** AUC deliberately ignores magnitude, so it
would call a perfect ordering across 0.8% of the range perfect. Both facts
matter and neither should hide the other; `Separation.saturated` flags it.

**4. Never let a reference constant override a data-derived threshold.**
`max(reference * 0.5, data)` is the specific construct that let a rescaled
metric silently fall back to a stale number. A reference belongs in the
"not enough data" branch -- which `if len(gram) < 2: return fb` already does
correctly -- not in a max() against the data.

**5. A baseline must not be the null arm's own median.** Scoring the grammatical
condition against the grammatical median guarantees a floor. If the detector
needs a baseline, it should come from held-out samples, not from the samples
being scored.

**6. Fix the saturation itself.** Span 0.008 on a [0,1] quantity means the
normalizer is far off: the role area receives ~1% of its normalizing scale in
every condition. That is a separate defect from the statistics, still open, and
the one that would make these numbers interpretable rather than merely correct.

## A STALE BACKBONE INVERTED THE EFFECT, and that invalidates today's numbers

Found while checking the above. Seed 42, same code, same seed:

    stale disk backbone   gram [0.9939, 0.9952, 0.9935]   catv [0.9922, 0.9951, 0.9945]
                          Cohen's d = -0.26      AUC = 0.444    INVERTED
    cache cleared         gram [0.9873, 0.9934, 0.9873]   catv [0.9925, 0.9941, 0.9949]
                          Cohen's d = +1.80                     correctly signed

Ruled out as my own doing: with the cache cleared, the code before my edit gives
d = 1.800 and after it gives d = 1.781 -- float noise. The edit is innocent; the
CACHE was producing an inverted result.

WHY. `.cache/backbones` pickles a trained parser and fingerprints it on SOURCE.
But #102 says parsing MUTATES the parser, so a pickled backbone also carries its
PARSE HISTORY -- how many sentences were run through it before it was written.
Two backbones from identical source and seed are different objects if they were
saved after different amounts of evaluation, and the fingerprint cannot see it.

That is [[backbone-fingerprint-gap]] meeting [[erp_probe_isolation]]'s finding,
and it is a complete mechanism for **#80** ("parser training is NOT reproducible
across processes -- p600 spread 1.37-2.52 at one seed"). The spread is parse
history, cached.

**CONSEQUENCE FOR TODAY'S OTHER NUMBERS.** Everything measured in
[[erp_probe_isolation]] and in the read-only-parse table above ran against
cached backbones of unknown parse history. The probe-isolation delta
(p600 d 2.192 -> 1.853) and the read-only-parse delta (1.632 -> 3.973) are
therefore NOT SAFE to quote until re-run on cleared caches. The DIRECTION of the
read-only result is corroborated by the mechanism (less contamination, tighter
distributions) but the magnitudes are not established.

This also explains three test failures seen today that had nothing to do with
the change under test, and it is the second time in this session that a
"result" turned out to be a property of the measurement apparatus.

## What this means for the order of work

Adopting the read-only parse is well-evidenced on reproducibility and does not
harm the separation. But its VALUE is limited while the metric is clipped and
saturated: it buys idempotence, not interpretability.

The metric comes first. Until `p600_excess` stops being scored against its own
null, and until the raw quantity uses more than 0.7% of its range, no Cohen's d
from this package should be quoted as an effect size -- including the ones in
[[erp_probe_isolation]] earlier today, which are subject to exactly the same
critique.
