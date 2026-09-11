# The ERP effects survive an isolated probe. Parsing is not idempotent.

#100 / #32. Two findings, and the second is much larger than the one I set out
to test.

## 2026-09-11 protocol correction

Calibration now collects each frame once and relabels that immutable sample
set after tuning. The old `fast=False` branch collected the frames again; since
parsing recruits, it measured a later model state and made `fast` versus `full`
an undocumented training-schedule comparison. The compatibility flag remains
but cannot change observations. This removes that calibration-mode confound.
The separate repeated-parse defect remains a strict expected failure.

## What was asked

Every ERP number here is read through a probe running under `brain.frozen()`,
which stops plasticity but NOT recruitment -- so the measurement grows the brain
while reading it. #32 has named this as half the P600 root cause since it was
filed, on suspicion. The question: under `read_only()`, does the
grammatical/violation separation survive?

## Answer: yes, at no cost -- and the "~15% attenuation" was NOT REAL

RE-MEASURED 2026-08-04 on top of #80 (training now bit-identical across
processes) and #103 (fork no longer clones a mutated parser). `research/
experiments/task100_erp_rerun.log`. SENTENCES, 10 seeds, paired -- each seed
trains ONE parser and forks it twice, so the arms differ only in probe context.
Intervals are t-based 95% CIs from `diagnostics.ensemble`.

    metric           frozen() [shipped]   read_only() [isolated]   delta
    p600 AUC           0.972 +/- 0.063      0.917 +/- 0.057      -0.056 +/- 0.099
    n400 AUC           0.922 +/- 0.084      0.911 +/- 0.098      -0.011 +/- 0.059
    p600 span          0.006 +/- 0.001      0.007 +/- 0.001      +0.001 +/- 0.000  CHANGED
    p600 median gap    0.002 +/- 0.001      0.003 +/- 0.000      +0.001 +/- 0.001  CHANGED
    p600 Cohen's d     1.910 +/- 0.534      1.876 +/- 0.446      -0.034 +/- 0.581
    n400 Cohen's d     2.555 +/- 0.731      2.653 +/- 1.028      +0.098 +/- 0.915

**The effect survives isolation, and isolation costs nothing.** Both arms sit
far above the 0.5 null on the rank statistic, so the separation is not an
artifact of probe contamination and #32's probe half does not explain the P600
problem. That conclusion is unchanged.

WHAT CHANGED IS THE ATTENUATION, WHICH IS GONE. The table this section used to
carry read `2.192 -> 1.853`, delta `-0.338 +/- 0.327` -- an interval that
excluded zero by 0.011 -- and I wrote it up as "real but small". It now reads
`1.910 -> 1.876`, delta `-0.034 +/- 0.581`. Not a smaller effect: a TENTH the
size with an interval nearly twice as wide, centred on zero.

That delta was parser variation, not probe contamination. Both of its sources
are now closed, and neither was visible in the numbers at the time.

READ THE GAP ROWS, NOT THE AUC ROW, for the direction. `p600_span` and the
median gap both INCREASE under isolation and both exclude zero, so isolation
slightly WIDENS the raw separation. The AUC nominally dips, which is not
evidence against that: at 0.972 it is one granularity step (1/9, n=3 per arm)
from the ceiling and has almost nowhere to go but down.

No row read `IDENTICAL`, so both arms genuinely ran different computations --
the guard that has caught a dead pathway in this repo before.

A NOTE ON MY OWN READING, KEPT BECAUSE IT GOT WORSE. At 5 seeds this delta read
"no change" (-0.193 +/- 0.728); at 10 it excluded zero and I reported an
attenuation; on a trustworthy substrate it is zero again. The 5-seed reading was
underpowered AND the 10-seed reading was confounded, and adding seeds fixed only
the first of those. Seeds do not rescue a measurement whose apparatus varies
between arms -- [[ensemble-not-realization]] and
[[backbone-fingerprint-gap]] together, demonstrated on myself twice in two days.

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
