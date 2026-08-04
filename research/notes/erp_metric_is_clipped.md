# The ERP Cohen's d is computed on a variable clipped at its own null

#102 / #32. Setting out to test whether the ERP separation survives a parse
that does not grow the parser, I found that it does -- and that the metric the
separation is scored on cannot support the claim being made with it.

## The read-only parse is safe

RE-MEASURED 2026-08-04 on top of #80 and #103, so unlike the table in
[[erp_probe_isolation]] this one is confirmed rather than corrected.
`research/experiments/task102_rerun.log`. SENTENCES, 10 seeds, paired:

    metric            parse grows [HEAD]     parse read_only        delta
    neurons recruited   1107.0 +/- 71.7        120.0 +/- 0.0      -987.0 +/- 71.7
    p600 AUC            0.9167 +/- 0.0570     1.0000 +/- 0.0000   +0.0833 +/- 0.0570
    p600 span           0.0071 +/- 0.0012     0.0084 +/- 0.0014   +0.0012 +/- 0.0003
    p600 median gap     0.0028 +/- 0.0004     0.0044 +/- 0.0006   +0.0016 +/- 0.0004
    p600 Cohen's d      1.8765 +/- 0.4462     3.8433 +/- 0.2130   +1.9669 +/- 0.4503
    n400 AUC            0.9111 +/- 0.0977     0.9111 +/- 0.0977    IDENTICAL
    n400 Cohen's d      2.6534 +/- 1.0276     2.6534 +/- 1.0276    IDENTICAL

The separation does not collapse; it strengthens on EVERY axis at once -- rank,
span and absolute gap -- the recruitment drops by 987 of 1107, and the residual
120 is EXACTLY 120 on every seed (CI 0.0000), the documented below-`k` exemption,
deterministic. On the reproducibility axis this is a clear win: an evaluation
that is not idempotent is broken regardless of what it measures.

ON THE 1.0000, WHICH IS THE KIND OF NUMBER THIS REPO GETS WRONG. Zero variance
across 10 seeds is [[fake-perfect-probe-signatures]] on its face, so it was
checked against the three ways that reading fails here, and it survives all
three: the PAIRED arm reads 0.9167, not 1.000, so it is not a degenerate score
both arms achieve; `span` INCREASES rather than collapsing, so it is not a tie
being resolved by index order; and n=3 per arm means 1.000 is 9 correctly
ordered pairs per seed, 90 in total, on a statistic whose granularity is 1/9.
It is a real perfect ordering at this sample size, not an apparatus artifact --
which is a different claim from "the ordering is perfect", since 90 pairs cannot
distinguish 1.000 from 0.99.

CONSISTENCY CHECK ACROSS THE TWO EXPERIMENTS. This table's `grows` arm reads
p600 AUC 0.9167 and [[erp_probe_isolation]]'s ISOLATED arm reads 0.917 -- they
are the same configuration measured by two independent scripts (isolated probes
are now the default), and they agree. That is the cross-check that would have
caught the stale-backbone inversion had it existed at the time.

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
effect being larger: 1.877 -> 3.843, a 2.05x jump, while the absolute gap moved
0.0028 -> 0.0044 and the rank statistic moved 0.917 -> 1.000. d more than
DOUBLED on a separation that widened by 0.16% of the [0,1] range.

That ratio is the cleanest statement of the defect, and it is now measured on a
deterministic substrate: the three statistics disagree about the SIZE of one
identical improvement by an order of magnitude, and only d disagrees wildly.

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

What the ERP contrast reads in the honest statistic, now over 10 seeds rather
than the single seed this section originally quoted:

    parse grows      p600 AUC 0.917 +/- 0.057 (span 0.0071)   n400 AUC 0.911 +/- 0.098
    parse read_only  p600 AUC 1.000 +/- 0.000 (span 0.0084)   n400 AUC 0.911 +/- 0.098

P600 goes from about one misordered pair in nine to perfectly ordered -- a real,
interpretable improvement, and the one place where the read-only parse buys
something beyond idempotence. And **N400 is 0.911, not the ~1.0 its Cohen's d of
2.65 implied**: roughly one pair in eleven is misordered while d reads 2.65,
which is the same clipping inflation seen on P600. With n=3 per arm the
resolution is 1/9, which is itself worth stating -- 0.911 is the mean of ten
seeds each landing on a multiple of 1/9, not a value any single seed returned.

(The single-seed figures this section used to quote, 0.889 and 0.778, were seed
11 alone. They sit inside the intervals above. A point estimate that survives
does not retroactively become a measurement -- [[ensemble-not-realization]].)

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

> RESOLVED 2026-08-04, and the two experiments came apart. Both were re-run at
> 10 seeds on top of #80 (training bit-identical across processes) and #103
> (fork no longer clones a mutated parser), which between them are the real
> mechanism this section was groping at.
>
>   * The READ-ONLY-PARSE result HELD and strengthened -- see the table at the
>     top of this note, now the measured version.
>   * The PROBE-ISOLATION result DID NOT. Its "~15% P600 attenuation"
>     (-0.338 +/- 0.327) is now -0.034 +/- 0.581: gone, and its interval was
>     never the thing to trust.
>
> So the suspicion recorded here was correct in kind and wrong about which
> result it endangered. Worth keeping in view: the endangered number was the one
> whose interval only just excluded zero, and the robust one was the number with
> a large margin -- which is what a marginal interval on a confounded apparatus
> is supposed to look like, and is not how I read it at the time.

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
