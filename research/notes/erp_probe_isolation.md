# The ERP effects survive an isolated probe. Parsing is not idempotent.

#100 / #32. Two findings, and the second is much larger than the one I set out
to test.

## What was asked

Every ERP number here is read through a probe running under `brain.frozen()`,
which stops plasticity but NOT recruitment -- so the measurement grows the brain
while reading it. #32 has named this as half the P600 root cause since it was
filed, on suspicion. The question: under `read_only()`, does the
grammatical/violation separation survive?

## Answer: yes, with a ~15% attenuation

`calibrate_erp_thresholds` on SENTENCES, 10 seeds, paired -- each seed trains
ONE parser and forks it twice, so the arms differ only in probe context.
Intervals are t-based 95% CIs from `diagnostics.ensemble`.

    metric           frozen() [shipped]   read_only() [isolated]   delta
    p600 Cohen's d     2.192 +/- 0.252      1.853 +/- 0.371       -0.338 +/- 0.327
    n400 Cohen's d     2.500 +/- 1.418      2.380 +/- 0.992       -0.120 +/- 0.704

P600 attenuates by about 15%. The interval [-0.665, -0.011] excludes zero, but
only just. N400 does not move.

**The effect survives.** 1.853 sits far above the 0.3 threshold the tests
assert, so the separation is not an artifact of probe contamination, and #32's
probe half does not explain the P600 problem.

A NOTE ON MY OWN READING. At 5 seeds this same delta read as "no change"
(-0.193 +/- 0.728) and I wrote it up that way before adding seeds. At 10 it
excludes zero. The 5-seed answer was underpowered, not wrong-in-kind, and the
10-seed interval is still marginal -- treat the attenuation as real but small.
[[report-distributions-not-point-estimates]], demonstrated on myself inside the
same session that quotes it.

Isolation is also ~3x faster on the ERP suite (53s -> 17s), because a probe that
does not grow the brain has less to do.

Default flipped to isolated; `NEURAL_ASSEMBLIES_ISOLATED_PROBES=0` restores the
old behaviour for A/B work.

## The larger finding: PARSING THE SAME SENTENCE CHANGES THE PARSER

Isolating the probes did NOT close the strict-mode raises in this package. The
remainder traces to `runner.py -> _advance_incremental_word` -- the parse
ADVANCE, which is wrapped in its own outer `frozen()`. Strict mode reported
`DET_CORE` recruiting 30 neurons mid-parse.

Parsing one 5-word sentence of KNOWN words, on an ALREADY-TRAINED parser, four
times in a row:

    before       materialized 16462
    after pass0  materialized 17125  (+663)   connectome CHANGED
    after pass1  materialized 17192  (+ 67)   connectome CHANGED
    after pass2  materialized 17218  (+ 26)   connectome CHANGED
    after pass3  materialized 17228  (+ 10)   connectome CHANGED

The ERP values follow. First-word P600 reads 0.434, 0.988, 0.9868, 0.9865 --
a 2x jump between the first read and the second, then a drift that never
converges. No two passes agree.

**Evaluation is not idempotent.** Every reported ERP number depends on how many
times, and in what order, sentences were parsed before it. That is a strictly
larger problem than probe contamination, and isolating the probes does not fix
it -- the recruitment is in the parse, not the readout.

This is the obvious candidate for #80 ("parser training is NOT reproducible
across processes -- p600 spread 1.37-2.52 at one seed"). That spread may be read
ORDER rather than training nondeterminism. Not yet confirmed.

## And the fix is measured

Running the parse advance under `read_only()` as well (an outer block subsumes
the inner `frozen()`, so this needs no edit to test):

    pass0  grew +120   P600 0.4343
    pass1  grew +  0   P600 0.4308
    pass2  grew +  0   P600 0.4308     <- identical
    pass3  grew +  0   P600 0.4308     <- identical

Idempotent from the second read. The residual pass-0 growth is the documented
below-`k` exemption in `read_only()` warming up areas that have nothing to
select from yet.

Note what the reported value becomes: **0.43, not the 0.99 the shipped path
reports.** The shipped protocol sits at the ceiling; the idempotent one lands
mid-range. That is #32's OTHER half -- the saturated metric -- and it suggests
the saturation is itself a product of the parse growing during evaluation.

NOT ADOPTED YET, deliberately. Making the parse read-only is a change to what
parsing MEANS, not a probe fix: a word with no existing assembly could no longer
form one mid-sentence. Whether the grammatical/violation separation survives it
is unmeasured, and the numbers above are one sentence at one seed. It needs the
same paired treatment the probe question just got.

## What is legitimately `frozen()`

`binding.materialize_fiber` exists to ALLOCATE COLUMNS, which happens only as a
side effect of the target recruiting. Under isolation it would become a silent
no-op that still returns True. Plasticity-off-with-recruitment-on is exactly
what it wants, and it is marked as such.
