# The mean spread was hiding a bimodal distribution — and the toy was pinned at its floor

## The arithmetic that reopened a closed question

The whole separation arc compared source spread RAW across two substrates. But
`overlap` normalises by `k`, so the random-pair floor is `k/n`, and the two
substrates do not share one:

| | k | n | floor k/n | observed | vs floor |
|---|---|---|---|---|---|
| toy | 40 | 2000 | 0.0200 | 0.017 | **0.85×** |
| parser `NOUN_CORE` | 30 | 3000 | 0.0100 | 0.125 | **12.5×** |

The toy is not "well separated". It is **pinned at its own floor** — its
assemblies are as unrelated as random sets can be. That is why the separation
sweep found two regimes and nothing between them: a substrate sitting on the
floor has no room to express intermediate overlap. Only one of those two numbers
was ever doing anything.

(The parser's n/k are measured off the built area. `EmergentParser.__init__`
defaults to 10000/100 and `train_parser_to_depth` overrides both — the floor
lands at 0.01 either way, but the class default is not what runs.)

## The mean could never have answered the question

Let `d_x` be how many of the M assemblies contain neuron x. Then

    mean pairwise overlap  =  Σ_x C(d_x,2) / ( C(M,2) · k )

exactly — verified numerically to 1.4e-17. **The mean is a function of the
degree sequence alone.** It cannot distinguish "every pair shares a little" from
"some pairs are the same assembly", and 12.5× the floor is a statement about
degree over-dispersion, not about which words are related.

So the statistic four experiments in this arc quoted as the headline was
structurally incapable of answering the question they were asking.

## What the distribution actually is

A degree-preserving shuffle keeps every `d_x` and every `k`, so by the identity
it must reproduce the mean — which it does, exactly, and that is the check that
the null is valid (42600/42600 swaps accepted, 20 per edge):

| | observed | shuffled |
|---|---|---|
| mean | 0.1247 | 0.1247 |
| sd | **0.2598** | 0.0567 |
| max pair | **1.0000** | 0.3533 |

**Bimodal.** Most pairs near the floor, a tail sitting at identity. And
`check_distinct` said so in one line: `PARTIAL-COLLAPSE(distinct=0.761)` — 24%
of the 71 `NOUN_CORE` assemblies are exact duplicates of another word's.

## Why four experiments missed it

`_substrate.check_distinct` carries a partial-collapse detector *precisely*
because mean overlap is nearly blind to duplicates — its own docstring records
256 items collapsing onto 58 distinct assemblies while spread still read 0.0129
against a 0.0125 floor.

Four experiments in this arc hand-rolled a local `_spread` instead of calling
it. The detector never ran. **One canonical way, again — and the hand-rolled
path is the one without the alarm.**

## The errors follow the overlap, decisively

For each failed retrieval, where does the winning impostor sit in the *source*
overlap ordering of that trial's competitors? Tie-safe probability of
superiority, null 0.500:

| area | ret@6 | impostor on ERRORS | runner-up on HITS |
|---|---|---|---|
| `ROLE_PATIENT` | 0.803 | **0.968 ± 0.018** | 0.806 ± 0.031 |
| `ROLE_AGENT` | 0.717 | **0.914 ± 0.026** | 0.782 ± 0.032 |

Confusions land on the nearest source essentially every time. Source overlap is
**causal**, not a correlate.

## But removing the duplicates does not fix it — the pre-registered falsifier

Retrieval at a fixed candidate-set size of 6, so difficulty is identical across
arms, over three pools:

| pool | `ROLE_PATIENT` | `ROLE_AGENT` |
|---|---|---|
| all words | 0.803 / 0.808 | 0.717 / 0.750 |
| **unique core only** | 0.756 / 0.778 | 0.753 / 0.789 |
| duplicated core only | **0.489 / 0.631** | **0.575 / 0.600** |

Duplicated words retrieve far worse — that half of the prediction holds. But
**unique-only lands at 0.75–0.79, not 1.000**, and for `ROLE_PATIENT` it is
*lower* than all-words. Deleting the duplicates buys nothing.

The reason is in the same output: **234 further pairs sit at overlap in
[0.5, 1.0)** without being identical. Exact duplicates are the tail of a broad
near-collision population, and filtering the tail leaves the body.

## The mechanism, and it is not the one I expected

Q1's clusters are semantically coherent, identically across seeds:

    x10  death fear life love place time way week word year   (abstract)
    x5   face foot head mouth nose                            (body parts)
    x4   baby king sister teacher                             (people)
    x2   egg meat  /  x4 bread egg food meat                  (food)

`train_lexicon` projects phon + grounding **simultaneously**, so I expected the
inputs to be identical and the substrate to be exonerated. They are not:

    71 words -> 47 distinct grounding sets -> 52-54 distinct assemblies
    only 1 of 4 (seed 11) and 1 of 5 (seed 42) duplicate clusters
    were ALREADY identical at the input

Both directions are broken at once. Phon **does** add discrimination — there are
more distinct assemblies (54) than distinct grounding sets (47). And yet words
with *different* grounding still land on *identical* assemblies.

So the substrate is losing information it was handed. The mechanism is k-WTA
turning graded input similarity into all-or-nothing output identity: two words
sharing most-but-not-all of their grounding features get literally the same top-k
because the shared features dominate the drive ranking, and the phon stimulus is
not strong enough to break the tie.

**That is a drive-ratio problem, and this repo has already characterised one.**
`mood-collapse-is-a-drive-ratio`: MOOD controlled ~4% of SYN's drive and
separation tracked the ratio, which explained five failed interventions. This is
the same shape — phon's share of core-area drive against grounding's — and it is
measurable and parameterisable rather than mysterious.

## What this retires and what it opens

**Retired:** "the parser's assemblies are 8–12× more overlapping than the toy's"
— a raw comparison across different floors, of a statistic that could not see
the actual defect. And the toy-vs-parser gap generally: the toy has no partially
overlapping inputs by construction, so it cannot exhibit this at all. That is
why nine localization attempts failed, and it is a better answer than the
stopping rule I invoked.

**Open, and now concrete:** measure phon's share of the drive in
`apply_lexicon_word`, and test whether raising it separates the clusters. That
is a parameter with a known law in this codebase, not a new hypothesis.

## Limits

2 seeds. Retrieval pools differ in size across the three arms (25/34 vs 11/12),
so the candidate-set size is matched but the pool composition is not — the
unique-vs-duplicated gap is large enough to survive that, the unique-vs-all
comparison is not. The near-collision count uses a 0.5 threshold chosen for
readability, not derived. Q2's frequency table counts the SENTENCE corpus, which
is not what trains the core lexicon, so its flat result is **not measured**
rather than null — the same wrong-corpus error that put `chases` in the ERP
frames.
