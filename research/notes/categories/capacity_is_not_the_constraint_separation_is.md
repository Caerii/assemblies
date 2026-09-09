# A shared target holds 96 bindings perfectly. The parser's problem is separation.

> **PARTIALLY RETRACTED 2026-08-07** — see
> `the_mean_spread_was_hiding_a_bimodal_distribution.md`. The capacity
> conclusion stands. **The spread comparison below does not.**
>
> 1. The spreads are quoted RAW across two substrates with different
>    random-pair floors (`overlap` normalises by k, so the floor is `k/n`). The
>    toy's 0.0172 sits against a floor of 0.0200 — it is **pinned at its floor**,
>    not well separated — and the parser's 0.13 against 0.0100. "8–12× more
>    overlapping" compares two different zeros.
> 2. Mean pairwise overlap is a function of the neuron DEGREE SEQUENCE alone
>    (`Σ C(d_x,2) / (C(M,2)·k)`, exact). It cannot see that the parser's
>    distribution is **bimodal** — 24% of `NOUN_CORE` assemblies are exact
>    duplicates of another word's — which is the defect it was being used to
>    argue about.
>
> The section below titled "The experiment this names" is what led to the
> two-area sweep that found no intermediate regime. That negative is now
> explained: a substrate at its floor cannot express one.

## The sweep

Source population built **once** at M=96 and never changed, so source separation
is held exactly fixed while target load varies. Bindings written incrementally,
targets snapshotted at bind time and never refreshed. Retrieval always scored
among a random **6** candidates, so chance stays 1/6 and only interference
varies.

| n | M | α = Mk/n | ret@6 | margin | tgt spread | src spread |
|---|---|---|---|---|---|---|
| 2000 | 6 | 0.120 | **1.000** | 17.56 | 0.0272 | 0.0172 |
| 2000 | 24 | 0.480 | **1.000** | 14.05 | 0.0382 | 0.0172 |
| 2000 | 96 | 1.920 | **1.000** | 6.04 | 0.0415 | 0.0172 |
| 4000 | 6 | 0.060 | **1.000** | 33.19 | 0.0072 | 0.0109 |
| 4000 | 48 | 0.480 | **1.000** | 20.65 | 0.0167 | 0.0109 |
| 4000 | 96 | 0.960 | **1.000** | 14.70 | 0.0167 | 0.0109 |

**Accuracy never breaks** — not at α = 1.92, past the α\* ≈ 1.15 the storage
regime tolerates. Reporting margin was not optional: it is the only thing that
moves, decaying ~3× from M=6 to M=96, and an accuracy-only sweep would have read
"no effect" at every point.

Margin tracks neither M nor α cleanly. At matched α = 0.48 it reads 14.05 (n=2000)
against 20.65 (n=4000); at matched M = 48, 10.06 against 20.65. Bigger areas are
simply better, by more than α accounts for.

## So capacity is not what the parser is hitting

The parser sits at **α = 0.36–0.46 with ret@6 = 0.729 / 0.721**. The toy at
α = 0.48 reads **1.000**.

Same readout, same difficulty, comparable load, and a quarter of the parser's
bindings are unretrievable while none of the toy's are. **Whatever the parser is
losing, this design does not model it** — and target load is now excluded along
with the readout instrument, the training route, `norm_init`, the index space,
and consolidation.

## What the toy has that the parser does not

The spread columns, which I nearly did not print:

| | source spread | target spread |
|---|---|---|
| toy, n=2000 | **0.0172** | 0.027–0.042 |
| toy, n=4000 | **0.0109** | 0.007–0.017 |
| parser `NOUN_CORE` | **0.13** | — |
| parser `ROLE_PATIENT` | — | **0.16** |

**The parser's assemblies are 8–12× more overlapping at the source and 4–6× more
overlapping at the target.** The toy's capacity headroom may be nothing more
than the fact that its assemblies are nearly disjoint.

This forces a correction. I earlier called source crowding "ruled out" because
`NOUN_CORE` reads 0.13, which is 13× chance and therefore *distinct*. Distinct
from chance is not the same as distinct enough — the right reference was never
the chance floor, it was a substrate that retrieves perfectly, and that one sits
at 0.017.

## The experiment this names

**Sweep source separation in the toy and find where retrieval falls to 0.73.**
If it falls as source spread approaches 0.13, the constraint is how core
assemblies form — a lexicon question, not a role-area one — and the whole
diagnosis moves upstream of binding entirely.

That is a single knob (build rounds, k/n, or explicit source mixing) against a
readout that is already built and already reads 1.000 as its control. It is the
cleanest remaining question in this arc.

## Limits

3 seeds, two area sizes, `CANDIDATES = 6`. Accuracy at ceiling everywhere means
this sweep bounds capacity from below only: it says ≥96 bindings are holdable at
this separation, not where the actual limit is. Finding that needs either much
larger M or the harder readout (rank-1 among all M), and neither is required for
the conclusion drawn here.
