# Phase A: β 0.10 → 0.05 buys the parser +0.10 retrieval, as the substrate law predicted

## The prediction, pre-registered

The substrate sweep measured overlap amplification with β as the gain
(α=0.25 → 0.497 at β=0.10, → 0.283 at β=0.05). Pre-registered: at β=0.05 the
parser's high-α overlap should **fall**, duplicates should **fall**, retrieval
should **rise or hold**.

β moved **alone**. The architecture (#125) is the other candidate and changing
both would leave neither attributable.

## Result — 3 seeds, paired same-seed

| metric | β=0.10 | β=0.05 | Δ | paired CI excludes 0 |
|---|---|---|---|---|
| core lexicon size | 71.0 | 71.0 | 0.0 | — |
| mean spread | 0.1233 | 0.0927 | −0.0306 | **yes** |
| **overlap, α ≥ 0.35** | **0.5982** | **0.4868** | **−0.1114** | **yes** |
| overlap, α < 0.35 | 0.0498 | 0.0317 | −0.0181 | **yes** |
| ret@6 `ROLE_PATIENT` | 0.8019 | 0.8833 | +0.0815 | no |
| **ret@6 `ROLE_AGENT`** | **0.7398** | **0.8426** | **+0.1028** | **yes** |
| distinct-assembly frac | 0.7418 | 0.7793 | +0.0376 | no |
| duplicated-word frac | 0.3192 | 0.3052 | −0.0141 | no |

**Not the degenerate arm.** The whole reason retrieval is in this table is that
lowering β can buy distinctness by learning *less* — beautifully separated,
useless assemblies. Retrieval went **up**, and the core lexicon size is
**identical (71) in both arms**, so the comparison is like-for-like rather than
one arm having written less.

## What moved and what did not

The win is in **graded overlap**, not in eliminating duplicates. High-α overlap
fell 0.11 and retrieval rose ~0.10, but the exact-duplicate fraction barely
moved (−0.014, CI includes zero) and `check_distinct` still reports
`PARTIAL-COLLAPSE` in both arms.

That is consistent with the earlier decomposition: exact duplicates are the
**tail**, the near-collision body is what β acts on, and at least one cluster is
already identical **at the input** — which no amount of β can separate. So β is
a real gain and not the whole fix, exactly as the two-candidate framing assumed.

## The infrastructure bug this turned up first

`beta` was **not cache-key material** — neither in `backbone_cache_filename` nor
in `ParserCache._key`. Both arms would have shared one pickled backbone, and
since warm runs do not train, the study would have measured **exactly zero** and
read as a clean negative. Fixed in `cf6cb0f` by making `params` a *required*
keyword, so a future training knob has to be spelled into the key rather than
forgotten. `p`, `rounds` and the vocabulary had the same hole.

## Recommendation, and the decision I did not make

β = 0.05 should become the parser default. I have **not** flipped it: it is a
one-line change with repo-wide blast radius — goldens, ERP thresholds and
parity tests are all calibrated at 0.10 — and that is a decision to take
deliberately, not as a side effect of an experiment.

Phase B (#125, PHON→LEX1/LEX2) should be measured against **both** β values
until the default is settled, so the architecture's effect is not read off a
moving baseline.

## Limits

3 seeds. `ROLE_PATIENT`'s +0.082 has a CI that includes zero (half-width 0.096);
only `ROLE_AGENT` is individually decisive. The α≥0.35 bin pools a wide range of
α, so the −0.111 is an average over a curve, not a point on it. Both arms still
flag partial collapse.
