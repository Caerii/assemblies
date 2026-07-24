# Assembly recruitment vs reuse, and the capacity limit of an area

Generated from `research/experiments/capacity/`:
`lexicon_capacity.py` (controlled model), `parser_recruitment.py` (real
`EmergentParser`), `analyze.py` (aggregation). Raw data:
`results_lexicon_capacity.json` (21 cells x 5 seeds),
`results_parser_recruitment.json` (10 cells x 3 seeds). Regenerate this file
with `python -m research.experiments.capacity.analyze`; the prose lives in
`PROSE.md` and the tables are recomputed from the JSON on every run.

---

## Summary

The motivating hypothesis was that the lexical core areas of `EmergentParser`
sit near a **tiling limit**: ~60% of `NOUN_CORE` materialised at n=3000, k=30
was read as ~60 assembly-sized slots consumed by a vocabulary of the same
order, with interference expected once the vocabulary exceeds `n/k`.

**Both halves of that reading are wrong, and the truth is worse.**

**1. The 60% is not tiling.** Reproducing the TWO_WORD measurement (Table 8)
gives `NOUN_CORE` w = 1737 +/- 34 of 3000 (0.579 +/- 0.011) from **19 words**,
not ~60. Those 19 stored assemblies account for 19 x 30 = 570 neurons; the
area materialised 3.05x that. `VERB_CORE` is worse: 10 words, tiling ratio
5.70. The excess is transient winners thrown off during multi-round training
plus the plasticity-disabled pre-grow pass that `link_lexicon_topology` runs
before compiled training. `w/n` is a churn statistic, not a vocabulary
statistic.

**2. Recruitment stops far short of the tiling limit.** In the controlled
model (Table 1) recruitment ceases at V* = 17 / 22 / 33 words for
n = 1000 / 3000 / 10000 — 51% / 22% / 10% of the `n/k` slots the area
nominally has. At n=10000 the area stops recruiting while 82% of its neurons
have never fired (Table 2: observed 0.00 vs null 0.82, d = -60, p = 2e-8).
The same shutdown appears in the real parser: in every core area with more
than a handful of words the recruit fraction is 0.3-0.8 over the first
quarter of the vocabulary and **0.00** over the last three quarters, while
the random-tiling null sits at 0.88-0.97 (Table 7).

**3. The mechanism is Hebbian rich-get-richer, not exhaustion.** With
beta = 0 the recruit fraction tracks the coupon-collector null and the area
fills to w/n = 0.93-0.99. Raising beta collapses the horizon monotonically,
and with it the scaling exponent of capacity in area size (Table 4b):

| beta | 0.0 | 0.01 | 0.05 | 0.2 |
|---|---|---|---|---|
| capacity exponent `d log V* / d log n` | 0.84 | 0.51 | 0.29 | 0.14 |

Once a neuron has fired, its potentiated afferents make it win again for the
*next* word too. Incumbency compounds. Plasticity is simultaneously what
stores a word and what forecloses the next one.

**4. What breaks is discriminability, and it breaks immediately.** Pairwise
overlap between stored assemblies is already 6-45x chance at V=25 and reaches
~0.75 by V=400 at *every* n (Table 3) — the absolute collapse level does not
improve with a bigger area. Identification accuracy in the real parser falls
from 1.00 (45-word vocabulary, every core area) to 0.23-0.32 (329-word
vocabulary), and this too fails to improve with n: `NOUN_CORE` at 222 words
scores 0.57 / 0.38 / 0.31 at n = 1000 / 3000 / 10000 (Table 7). Growing the
area does not buy capacity. (The n=1000 figure is optimistic: 44 of its 71
words were unprobeable, see finding 5.)

**5. A hard wall sits behind the soft one.** At n=1000, k=30 the 329-word
vocabulary cannot be trained at all — the sparse engine raises `Remaining
size of area too small to sample k new winners`. At 222 words the same
n=1000 areas are already unprobeable for 44/71 `NOUN_CORE` words. The beta=0
control hits the wall at V = 42 +/- 4 (n=1000) and V = 353 +/- 34 (n=3000).
So there are two distinct limits: a plasticity-induced soft horizon at V*,
and a combinatorial hard floor at roughly `n/k` words.

**6. Interference runs in opposite directions depending on how the stimulus
is wired — a methodological trap.** In the controlled model, where every word
shares one `PHON -> LEX` matrix, retrieval of the *earliest* words is worst
(n=3000, V=400: 0.555 early vs 0.996 late) — classic retrograde catastrophic
forgetting. In the real parser, where each word gets its own private
`phon_<word>` stimulus matrix, the gradient **reverses**: early words
retrieve best and late words fail (n=10000, 329 words, `NOUN_CORE`:
Q1 = 0.39, Q4 = 0.10). The private matrices protect old words from being
overwritten, so failure appears as **anterograde** interference — the area
cannot acquire anything new — rather than forgetting. Reporting only the
parser numbers would have produced the false headline "no catastrophic
forgetting".

**7. Two clean nulls.**

* **Category structure does not transfer.** Words sharing 10 of 30 input
  neurons produce within-category assembly overlap indistinguishable from
  between-category overlap: |separation| <= 0.013 against overlaps of
  0.02-0.84, at every n, beta and vocabulary size (Table 4c). Several cells
  reach p < 0.05 on a separation of 0.006 — statistically detectable,
  functionally nil. The lexical area does not build a category geometry out
  of input similarity.
* **E%-WTA does not rescue capacity, because in this regime it does not form
  assemblies at all.** Both the paper rule (`window="epsilon"`) and the
  scale-invariant variant (`window="sigma"`) settle at a mean assembly size of
  1.1-1.6 neurons under a k_s=30, p=0.05 feedforward drive, and identification
  is at chance in all 6 cells (p >= 0.056, Table 5). This is the
  window-collapse failure already documented in `winner_policies.py`: with
  `mu/sigma ~ sqrt(k_s p_s)` small, the firing window shrinks onto
  `min_winners`. The prior observation of emergent sizes 17-44 must belong to
  a denser-drive regime this protocol does not reproduce. **The fixed-k vs
  E%-WTA capacity comparison is therefore not answerable from these data**;
  reporting "E%-WTA has lower capacity" would be an artefact of a degenerate
  selection window, not a property of the rule.

**8. The plastic recurrent fiber makes everything worse.** Engaging
`LEX -> LEX` with plasticity during learning lowers V* (17->14, 22->14,
33->17) and raises pairwise overlap (0.75->0.83, 0.76->0.85, 0.77->0.90) at
every n (Table 6). The classic Assembly Calculus project-until-convergence
loop, applied to a shared lexical substrate, accelerates collapse into a
small number of attractor states.

---

## What this implies for growing emergent capability

* **Do not budget area size by `n/k`.** Measured capacity grows like
  `n^0.29` at beta=0.05, not `n^1`. Going from n=3000 to n=10000 (3.3x the
  neurons, 3.3x the memory and time) bought 22 -> 33 words of recruitment
  horizon and *no* improvement in identification accuracy. Scaling `n` is the
  most expensive and least effective lever available.

* **beta is a capacity knob, not just a learning-rate knob.** It is the
  largest effect measured here: beta 0.2 -> 0.0 moves the capacity exponent
  from 0.14 to 0.84, and identification at V=400, n=10000 from 0.62 to 0.98.
  A curriculum that wants a large vocabulary in one area needs a decaying
  plasticity schedule or per-fiber gating, so that consolidating word i does
  not foreclose word i+1.

* **Recruitment needs an explicit mechanism.** Nothing in the current
  dynamics reserves fresh neurons for new items. Two candidates are already
  half-present in the codebase and were *not* engaged in these runs:
  refractory / `refracted` suppression (`Area(refractory_period=,
  refracted=)`), which penalises recently-fired neurons, and synaptic scaling
  (`Brain(synaptic_scaling=True)`). Testing whether either restores the
  beta=0 scaling exponent while keeping beta>0 learning is the obvious next
  experiment.

* **The parser is at its lexical ceiling, not near it.** At the shipped
  n=3000, `NOUN_CORE` holds 111 words from a 329-word vocabulary at 0.28
  identification accuracy with 0.74 of the area materialised. At n=1000 that
  vocabulary does not train at all. Growing past a few hundred words needs
  one of: more core areas (partitioning the lexicon), a plasticity schedule,
  or an explicit recruitment mechanism — not a bigger n.

* **Fix the measurement, not just the model.** `w/n` has been read as a
  saturation signal; it is dominated by training churn (tiling ratios of
  0.43-5.70 across cells in Tables 7-8) and is a poor proxy for how full an
  area is. Recruit fraction against the `1 - w/n` null is the honest
  instrument, and it is cheap — it needs only the stored per-word assemblies.

---

## Method notes and threats to validity

* **Shared substrate.** In the controlled model every word is a fixed
  30-subset of a 600-neuron explicit `PHON` area, and all words drive `LEX`
  through the same connectome. This was deliberate: per-word stimulus
  matrices make retrieval trivially perfect and hide interference (finding 6).
* **Instrumentation is exact, not estimated.** The numpy_sparse engine
  materialises neurons lazily and assigns compact indices in first-fire
  order, so a winner index >= `area.w` before the word is by construction a
  first-time winner. `w` is sampled before `inhibit_areas()`, because the
  `Area.winners` setter resets `.w` to `len(winners)`.
* **Probing never contaminates training.** Checkpoint retrieval runs on a
  `Brain.clone()` with plasticity disabled. (`clone()` drops
  `_explicit_engine`, and the lazy re-creation path does not re-register the
  area it is asked about, so the experiment shares the parent's explicit
  engine; `PHON` is never a projection target and carries no plastic state.
  `neural_assemblies/core/` was treated as read-only throughout.)
* **Pool exhaustion is recorded, not worked around.** Runs that hit
  `Remaining size of area too small to sample k new winners` are truncated at
  that point and reported with `V learned` (Table 4) or `EXHAUSTED`
  (`sweep_log.txt`, `parser_log.txt`). This affects the beta=0 control at
  n=1000/3000 and the n=1000 x 329-word parser cell.
* **Single parameter point per axis.** p=0.05, k=30, 6 training rounds,
  8 categories, single-pass sequential introduction. The recruitment shutdown
  is robust across n, beta, policy and mode, but the absolute V* values are
  specific to this point.
* **Vocabulary is synthetic in the controlled model** (random subsets with a
  planted category block). The parser experiment uses the real vocabulary
  presets (45 / 222 / 329 words); the two agree on the qualitative result.
* **`V*` definition.** Vocabulary size past which no 10-word window has mean
  recruit fraction >= 0.05. Taking the *last* qualifying window rather than
  the first matters for the noisy beta=0 control.
* **Statistics.** 5 seeds (controlled) / 3 seeds (parser) per cell;
  one-sample t-tests against explicit nulls via
  `research.experiments.base.ttest_vs_null`, which returns
  `significant=False` with a `degenerate` marker at zero variance — several
  ceiling-bound parser cells (identification = 1.00 in every seed) are
  reported that way rather than as p=0.

---

## Tables

### Table 1 — Capacity curve (fixed-k TopK, feedforward, beta=0.05)

`V*` is the vocabulary size at which recruitment stops (mean recruit fraction over 10 consecutive words < 0.05). `slots = n/k` is how many disjoint assemblies the area could hold if it tiled perfectly.

| n | slots = n/k | V* (recruitment horizon) | V*/slots | final w/n | w/k (slots consumed) |
|---|---|---|---|---|---|
| 1000 | 33 | 17.0 +/- 0.0 | 0.51 +/- 0.00 | 0.599 +/- 0.017 | 20 +/- 1 |
| 3000 | 100 | 22.4 +/- 1.1 | 0.22 +/- 0.01 | 0.399 +/- 0.025 | 40 +/- 3 |
| 10000 | 333 | 33.0 +/- 1.4 | 0.10 +/- 0.00 | 0.179 +/- 0.014 | 60 +/- 5 |

### Table 2 — Recruit fraction vs the random-tiling null

Observed recruit fraction per word, averaged in eighths of the 400-word vocabulary, against the coupon-collector null `1 - w/n` (what an unbiased winner-take-all would give). TopK, feedforward, beta=0.05.

| n | source | 1-50 | 51-100 | 101-150 | 151-200 | 201-250 | 251-300 | 301-350 | 351-400 |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | observed | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1000 | null | 0.46 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| 3000 | observed | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | null | 0.65 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 |
| 10000 | observed | 0.21 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10000 | null | 0.85 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 |

Test of `recruit_excess = observed - null` over the second half of the vocabulary, against H0: excess = 0.

| n | mean excess (words 201-400) | test |
|---|---|---|
| 1000 | -0.401 +/- 0.017 | p=7.7e-07 d=-23.61 * |
| 3000 | -0.601 +/- 0.025 | p=7.7e-07 d=-23.59 * |
| 10000 | -0.821 +/- 0.014 | p=1.9e-08 d=-59.62 * |

### Table 3 — What breaks past the transition

TopK, feedforward, beta=0.05. `pairwise` is the mean overlap between stored assemblies (null: chance = k/n). `ident` is the fraction of words whose retrieved assembly is nearest to their OWN stored assembly (null: 1/V). `retr early`/`retr late` are retrieval self-overlap for the first and last tenth of the vocabulary — a forgetting gradient would show early << late.

| n | V | w/n | pairwise (chance) | pairwise test | ident (chance) | ident test | retr early | retr late |
|---|---|---|---|---|---|---|---|---|
| 1000 | 25 | 0.598 +/- 0.017 | 0.315 +/- 0.012 (0.030) | p=8.1e-07 d=23.35 * | 0.688 +/- 0.077 (0.040) | p=4.7e-05 d=8.42 * | 0.423 +/- 0.045 | 0.997 +/- 0.007 |
| 1000 | 50 | 0.599 +/- 0.017 | 0.470 +/- 0.009 (0.030) | p=4.5e-08 d=48.04 * | 0.528 +/- 0.018 (0.020) | p=3.7e-07 d=28.40 * | 0.345 +/- 0.030 | 0.996 +/- 0.006 |
| 1000 | 100 | 0.599 +/- 0.017 | 0.604 +/- 0.010 (0.030) | p=2e-08 d=58.52 * | 0.636 +/- 0.045 (0.010) | p=6.4e-06 d=13.89 * | 0.391 +/- 0.025 | 0.995 +/- 0.003 |
| 1000 | 200 | 0.599 +/- 0.017 | 0.692 +/- 0.009 (0.030) | p=9.8e-09 d=70.33 * | 0.707 +/- 0.037 (0.005) | p=1.8e-06 d=18.97 * | 0.512 +/- 0.018 | 0.995 +/- 0.002 |
| 1000 | 300 | 0.599 +/- 0.017 | 0.727 +/- 0.010 (0.030) | p=1.1e-08 d=67.77 * | 0.708 +/- 0.066 (0.003) | p=1.9e-05 d=10.63 * | 0.602 +/- 0.020 | 0.996 +/- 0.003 |
| 1000 | 400 | 0.599 +/- 0.017 | 0.748 +/- 0.011 (0.030) | p=1.4e-08 d=64.01 * | 0.710 +/- 0.060 (0.003) | p=1.2e-05 d=11.81 * | 0.663 +/- 0.013 | 0.997 +/- 0.001 |
| 3000 | 25 | 0.398 +/- 0.024 | 0.184 +/- 0.031 (0.010) | p=0.00023 d=5.63 * | 0.640 +/- 0.075 (0.040) | p=5.7e-05 d=8.02 * | 0.347 +/- 0.036 | 0.993 +/- 0.009 |
| 3000 | 50 | 0.399 +/- 0.025 | 0.378 +/- 0.032 (0.010) | p=1.4e-05 d=11.48 * | 0.480 +/- 0.065 (0.020) | p=9.2e-05 d=7.10 * | 0.196 +/- 0.035 | 0.996 +/- 0.004 |
| 3000 | 100 | 0.399 +/- 0.025 | 0.556 +/- 0.024 (0.010) | p=9.3e-07 d=22.52 * | 0.574 +/- 0.075 (0.010) | p=7.2e-05 d=7.55 * | 0.234 +/- 0.031 | 0.995 +/- 0.003 |
| 3000 | 200 | 0.399 +/- 0.025 | 0.682 +/- 0.019 (0.010) | p=1.7e-07 d=34.63 * | 0.612 +/- 0.060 (0.005) | p=2.3e-05 d=10.04 * | 0.354 +/- 0.033 | 0.995 +/- 0.001 |
| 3000 | 300 | 0.399 +/- 0.025 | 0.732 +/- 0.021 (0.010) | p=1.6e-07 d=35.10 * | 0.591 +/- 0.098 (0.003) | p=0.00018 d=6.00 * | 0.473 +/- 0.033 | 0.995 +/- 0.001 |
| 3000 | 400 | 0.399 +/- 0.025 | 0.759 +/- 0.020 (0.010) | p=1.3e-07 d=36.75 * | 0.603 +/- 0.120 (0.003) | p=0.00037 d=5.00 * | 0.555 +/- 0.026 | 0.996 +/- 0.002 |
| 10000 | 25 | 0.174 +/- 0.012 | 0.136 +/- 0.036 (0.003) | p=0.0012 d=3.71 * | 0.712 +/- 0.118 (0.040) | p=0.00022 d=5.70 * | 0.343 +/- 0.035 | 0.900 +/- 0.081 |
| 10000 | 50 | 0.179 +/- 0.014 | 0.334 +/- 0.045 (0.003) | p=7.8e-05 d=7.40 * | 0.404 +/- 0.030 (0.020) | p=8.5e-06 d=12.94 * | 0.128 +/- 0.020 | 0.999 +/- 0.003 |
| 10000 | 100 | 0.179 +/- 0.014 | 0.536 +/- 0.041 (0.003) | p=8.3e-06 d=13.02 * | 0.556 +/- 0.030 (0.010) | p=2.3e-06 d=17.90 * | 0.193 +/- 0.031 | 0.999 +/- 0.002 |
| 10000 | 200 | 0.179 +/- 0.014 | 0.680 +/- 0.035 (0.003) | p=1.7e-06 d=19.45 * | 0.636 +/- 0.043 (0.005) | p=5.2e-06 d=14.65 * | 0.299 +/- 0.042 | 0.996 +/- 0.001 |
| 10000 | 300 | 0.179 +/- 0.014 | 0.733 +/- 0.032 (0.003) | p=8.5e-07 d=23.07 * | 0.631 +/- 0.056 (0.003) | p=1.5e-05 d=11.17 * | 0.410 +/- 0.048 | 0.996 +/- 0.002 |
| 10000 | 400 | 0.179 +/- 0.014 | 0.765 +/- 0.028 (0.003) | p=4.6e-07 d=26.81 * | 0.628 +/- 0.041 (0.003) | p=4.3e-06 d=15.36 * | 0.507 +/- 0.042 | 0.998 +/- 0.002 |

### Table 4 — Plasticity strength is what stops recruitment (TopK, feedforward)

beta=0 is the random-tiling control: no Hebbian bias toward neurons that already fired. `V learned` is how many words the run got through before the area ran out of never-fired neurons and training aborted; consequence metrics are quoted at `V meas`, the last checkpoint every seed of that cell reached.

| n | beta | V learned | V meas | V* | final w/n | pairwise overlap | ident acc | retr early |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.0 | 42 +/- 4 | 25 | 29 +/- 5 | 0.967 +/- 0.001 | 0.078 +/- 0.004 | 0.984 +/- 0.022 | 0.400 +/- 0.059 |
| 1000 | 0.01 | 400 +/- 0 | 400 | 39 +/- 3 | 0.943 +/- 0.009 | 0.567 +/- 0.012 | 0.359 +/- 0.033 | 0.305 +/- 0.023 |
| 1000 | 0.05 | 400 +/- 0 | 400 | 17 +/- 0 | 0.599 +/- 0.017 | 0.748 +/- 0.011 | 0.710 +/- 0.060 | 0.663 +/- 0.013 |
| 1000 | 0.2 | 400 +/- 0 | 400 | 14 +/- 0 | 0.297 +/- 0.030 | 0.761 +/- 0.019 | 0.740 +/- 0.065 | 0.844 +/- 0.011 |
| 3000 | 0.0 | 353 +/- 34 | 300 | 70 +/- 6 | 0.990 +/- 0.000 | 0.039 +/- 0.005 | 0.968 +/- 0.021 | 0.232 +/- 0.067 |
| 3000 | 0.01 | 400 +/- 0 | 400 | 57 +/- 9 | 0.759 +/- 0.036 | 0.512 +/- 0.034 | 0.262 +/- 0.035 | 0.175 +/- 0.018 |
| 3000 | 0.05 | 400 +/- 0 | 400 | 22 +/- 1 | 0.399 +/- 0.025 | 0.759 +/- 0.020 | 0.603 +/- 0.120 | 0.555 +/- 0.026 |
| 3000 | 0.2 | 400 +/- 0 | 400 | 16 +/- 1 | 0.147 +/- 0.013 | 0.814 +/- 0.012 | 0.656 +/- 0.059 | 0.860 +/- 0.009 |
| 10000 | 0.0 | 400 +/- 0 | 400 | 197 +/- 58 | 0.929 +/- 0.012 | 0.019 +/- 0.001 | 0.978 +/- 0.006 | 0.228 +/- 0.013 |
| 10000 | 0.01 | 400 +/- 0 | 400 | 125 +/- 10 | 0.527 +/- 0.029 | 0.449 +/- 0.032 | 0.257 +/- 0.030 | 0.145 +/- 0.012 |
| 10000 | 0.05 | 400 +/- 0 | 400 | 33 +/- 1 | 0.179 +/- 0.014 | 0.765 +/- 0.028 | 0.628 +/- 0.041 | 0.507 +/- 0.042 |
| 10000 | 0.2 | 400 +/- 0 | 400 | 19 +/- 1 | 0.067 +/- 0.009 | 0.836 +/- 0.019 | 0.620 +/- 0.024 | 0.864 +/- 0.024 |

### Table 4b — How capacity scales with area size

Least-squares slope of `log V*` against `log n` over n = 1000 / 3000 / 10000 (TopK, feedforward). An area whose capacity were simply its number of assembly-sized slots would give an exponent of 1.

| beta | V* @ n=1000 | V* @ n=3000 | V* @ n=10000 | scaling exponent |
|---|---|---|---|---|
| 0.0 | 29 +/- 5 | 70 +/- 6 | 197 +/- 58 | 0.84 |
| 0.01 | 39 +/- 3 | 57 +/- 9 | 125 +/- 10 | 0.51 |
| 0.05 | 17 +/- 0 | 22 +/- 1 | 33 +/- 1 | 0.29 |
| 0.2 | 14 +/- 0 | 16 +/- 1 | 19 +/- 1 | 0.14 |

### Table 4c — Category structure in the input does not survive into the assemblies

Words in the same category share 10 of their 30 PHON input neurons; words in different categories share only what the random draw gives them. If that structure were carried into LEX, within-category assembly overlap would exceed between-category overlap. TopK, feedforward, at the last checkpoint of each cell. H0: separation = 0.

| n | beta | V | within-cat overlap | between-cat overlap | separation | test |
|---|---|---|---|---|---|---|
| 1000 | 0.0 | 25 | 0.083 +/- 0.010 | 0.077 +/- 0.005 | 0.0063 +/- 0.0121 | p=0.31 d=0.52 ns |
| 1000 | 0.01 | 400 | 0.564 +/- 0.012 | 0.567 +/- 0.012 | -0.0030 +/- 0.0023 | p=0.041 d=-1.33 * |
| 1000 | 0.05 | 400 | 0.753 +/- 0.012 | 0.748 +/- 0.011 | 0.0054 +/- 0.0032 | p=0.019 d=1.71 * |
| 1000 | 0.2 | 400 | 0.762 +/- 0.020 | 0.761 +/- 0.018 | 0.0003 +/- 0.0043 | p=0.89 d=0.06 ns |
| 3000 | 0.0 | 300 | 0.050 +/- 0.005 | 0.037 +/- 0.005 | 0.0122 +/- 0.0008 | p=4.5e-06 d=15.19 * |
| 3000 | 0.01 | 400 | 0.509 +/- 0.034 | 0.512 +/- 0.034 | -0.0025 +/- 0.0019 | p=0.04 d=-1.34 * |
| 3000 | 0.05 | 400 | 0.765 +/- 0.021 | 0.758 +/- 0.020 | 0.0064 +/- 0.0007 | p=3.9e-05 d=8.80 * |
| 3000 | 0.2 | 400 | 0.813 +/- 0.011 | 0.814 +/- 0.012 | -0.0004 +/- 0.0024 | p=0.71 d=-0.18 ns |
| 10000 | 0.0 | 400 | 0.025 +/- 0.002 | 0.018 +/- 0.001 | 0.0070 +/- 0.0022 | p=0.0021 d=3.16 * |
| 10000 | 0.01 | 400 | 0.444 +/- 0.028 | 0.449 +/- 0.033 | -0.0046 +/- 0.0051 | p=0.11 d=-0.92 ns |
| 10000 | 0.05 | 400 | 0.766 +/- 0.029 | 0.764 +/- 0.028 | 0.0016 +/- 0.0029 | p=0.29 d=0.54 ns |
| 10000 | 0.2 | 400 | 0.836 +/- 0.021 | 0.835 +/- 0.019 | 0.0005 +/- 0.0024 | p=0.64 d=0.23 ns |

### Table 5 — Fixed-k TopK vs E%-WTA (feedforward, beta=0.05, V=400)

`size` is the emergent assembly size averaged over the whole vocabulary; TopK pins it at k=30 by construction.

| n | policy | assembly size | V* | final w/n | pairwise overlap (chance) | ident acc (chance) | ident test |
|---|---|---|---|---|---|---|---|
| 1000 | epsilon | 1.2 +/- 0.4 | 100 +/- 161 | 0.180 +/- 0.072 | 0.825 +/- 0.103 (0.030) | 0.007 +/- 0.004 (0.003) | p=0.099 d=0.96 ns |
| 1000 | sigma | 1.5 +/- 0.5 | 13 +/- 1 | 0.429 +/- 0.061 | 0.727 +/- 0.160 (0.030) | 0.004 +/- 0.001 (0.003) | p=0.18 d=0.73 ns |
| 1000 | topk | 30.0 +/- 0.0 | 17 +/- 0 | 0.599 +/- 0.017 | 0.748 +/- 0.011 (0.030) | 0.710 +/- 0.060 (0.003) | p=1.2e-05 d=11.81 * |
| 3000 | epsilon | 1.1 +/- 0.0 | 58 +/- 31 | 0.126 +/- 0.029 | 0.893 +/- 0.026 (0.010) | 0.006 +/- 0.003 (0.003) | p=0.056 d=1.19 ns |
| 3000 | sigma | 1.5 +/- 0.3 | 15 +/- 4 | 0.265 +/- 0.070 | 0.726 +/- 0.124 (0.010) | 0.004 +/- 0.003 (0.003) | p=0.24 d=0.61 ns |
| 3000 | topk | 30.0 +/- 0.0 | 22 +/- 1 | 0.399 +/- 0.025 | 0.759 +/- 0.020 (0.010) | 0.603 +/- 0.120 (0.003) | p=0.00037 d=5.00 * |
| 10000 | epsilon | 1.2 +/- 0.1 | 230 +/- 115 | 0.085 +/- 0.010 | 0.809 +/- 0.041 (0.003) | 0.008 +/- 0.007 (0.003) | p=0.13 d=0.86 ns |
| 10000 | sigma | 1.6 +/- 0.4 | 16 +/- 2 | 0.104 +/- 0.003 | 0.869 +/- 0.054 (0.003) | 0.005 +/- 0.003 (0.003) | p=0.14 d=0.82 ns |
| 10000 | topk | 30.0 +/- 0.0 | 33 +/- 1 | 0.179 +/- 0.014 | 0.765 +/- 0.028 (0.003) | 0.628 +/- 0.041 (0.003) | p=4.3e-06 d=15.36 * |

### Table 6 — Control: engaging the plastic recurrent fiber (TopK, beta=0.05, V=400)

| n | mode | V* | final w/n | pairwise overlap (chance) | ident acc (chance) | ident test |
|---|---|---|---|---|---|---|
| 1000 | ff | 17 +/- 0 | 0.599 +/- 0.017 | 0.748 +/- 0.011 (0.030) | 0.710 +/- 0.060 (0.003) | p=1.2e-05 d=11.81 * |
| 1000 | rec | 14 +/- 1 | 0.523 +/- 0.031 | 0.832 +/- 0.040 (0.030) | 0.581 +/- 0.087 (0.003) | p=0.00012 d=6.67 * |
| 3000 | ff | 22 +/- 1 | 0.399 +/- 0.025 | 0.759 +/- 0.020 (0.010) | 0.603 +/- 0.120 (0.003) | p=0.00037 d=5.00 * |
| 3000 | rec | 14 +/- 1 | 0.300 +/- 0.013 | 0.853 +/- 0.030 (0.010) | 0.597 +/- 0.055 (0.003) | p=1.7e-05 d=10.89 * |
| 10000 | ff | 33 +/- 1 | 0.179 +/- 0.014 | 0.765 +/- 0.028 (0.003) | 0.628 +/- 0.041 (0.003) | p=4.3e-06 d=15.36 * |
| 10000 | rec | 17 +/- 1 | 0.130 +/- 0.008 | 0.895 +/- 0.050 (0.003) | 0.372 +/- 0.197 (0.003) | p=0.014 d=1.88 * |

### Table 7 — Real EmergentParser core lexical areas

Whole vocabulary trained into the core areas via `train_lexicon()`. `tiling` is `w / (words * k)`: 1.0 means the area is tiled with perfectly disjoint assemblies, below 1.0 means words share neurons, above 1.0 means multi-round training materialised transient winners that no stored assembly kept.

| n | vocab | area | words | w/n | tiling | recruit Q1 | recruit Q4 | null Q4 | pairwise (chance) | ident (chance) | retr Q1 | retr Q4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 45 | ADJ_CORE | 5 +/- 0 | 0.213 +/- 0.003 | 1.42 +/- 0.02 | 0.83 +/- 0.00 | 0.88 +/- 0.04 | 0.89 +/- 0.00 | 0.049 +/- 0.002 (0.030) | 1.00 +/- 0.00 (0.200) | 0.89 +/- 0.03 | 0.89 +/- 0.07 |
| 1000 | 45 | DET_CORE | 10 +/- 0 | 0.403 +/- 0.002 | 1.34 +/- 0.01 | 0.98 +/- 0.01 | 0.82 +/- 0.07 | 0.77 +/- 0.00 | 0.026 +/- 0.005 (0.030) | 1.00 +/- 0.00 (0.100) | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| 1000 | 45 | NOUN_CORE | 10 +/- 0 | 0.374 +/- 0.008 | 1.25 +/- 0.03 | 0.68 +/- 0.06 | 0.51 +/- 0.06 | 0.83 +/- 0.01 | 0.128 +/- 0.009 (0.030) | 1.00 +/- 0.00 (0.100) | 0.81 +/- 0.03 | 0.66 +/- 0.01 |
| 1000 | 45 | VERB_CORE | 8 +/- 0 | 0.317 +/- 0.012 | 1.32 +/- 0.05 | 0.98 +/- 0.02 | 0.72 +/- 0.11 | 0.82 +/- 0.00 | 0.037 +/- 0.003 (0.030) | 1.00 +/- 0.00 (0.125) | 0.97 +/- 0.00 | 0.83 +/- 0.02 |
| 1000 | 222 | ADJ_CORE | 37 +/- 0 | 0.716 +/- 0.019 | 0.64 +/- 0.02 | 0.41 +/- 0.02 | 0.09 +/- 0.00 | 0.82 +/- 0.00 | 0.682 +/- 0.006 (0.030) | 0.30 +/- 0.00 (0.027) | 0.51 +/- 0.04 | 0.26 +/- 0.05 |
| 1000 | 222 | DET_CORE | 11 +/- 0 | 0.430 +/- 0.005 | 1.30 +/- 0.01 | 0.97 +/- 0.02 | 0.76 +/- 0.04 | 0.75 +/- 0.01 | 0.027 +/- 0.004 (0.030) | 1.00 +/- 0.00 (0.091) | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| 1000 | 222 | NOUN_CORE | 71 +/- 0 | 0.909 +/- 0.006 | 0.43 +/- 0.00 | 0.39 +/- 0.02 | 0.00 +/- 0.00 | 0.73 +/- 0.01 | 0.212 +/- 0.003 (0.030) | 0.57 +/- 0.02 (0.014) | 0.52 +/- 0.08 | 0.35 +/- 0.04 |
| 1000 | 222 | VERB_CORE | 57 +/- 0 | 0.883 +/- 0.018 | 0.52 +/- 0.01 | 0.69 +/- 0.01 | 0.00 +/- 0.00 | 0.67 +/- 0.00 | 0.384 +/- 0.010 (0.030) | 0.67 +/- 0.03 (0.018) | 0.88 +/- 0.02 | 0.24 +/- 0.03 |
| 3000 | 45 | ADJ_CORE | 5 +/- 0 | 0.080 +/- 0.002 | 1.60 +/- 0.04 | 0.82 +/- 0.03 | 0.97 +/- 0.03 | 0.96 +/- 0.00 | 0.044 +/- 0.011 (0.010) | 1.00 +/- 0.00 (0.200) | 0.86 +/- 0.04 | 0.99 +/- 0.02 |
| 3000 | 45 | DET_CORE | 10 +/- 0 | 0.164 +/- 0.005 | 1.64 +/- 0.05 | 0.98 +/- 0.02 | 0.91 +/- 0.03 | 0.92 +/- 0.00 | 0.015 +/- 0.001 (0.010) | 1.00 +/- 0.00 (0.100) | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| 3000 | 45 | NOUN_CORE | 10 +/- 0 | 0.154 +/- 0.001 | 1.54 +/- 0.01 | 0.80 +/- 0.02 | 0.61 +/- 0.06 | 0.93 +/- 0.00 | 0.077 +/- 0.009 (0.010) | 1.00 +/- 0.00 (0.100) | 0.85 +/- 0.04 | 0.65 +/- 0.03 |
| 3000 | 45 | VERB_CORE | 8 +/- 0 | 0.127 +/- 0.002 | 1.59 +/- 0.03 | 0.99 +/- 0.01 | 0.85 +/- 0.02 | 0.94 +/- 0.00 | 0.015 +/- 0.001 (0.010) | 1.00 +/- 0.00 (0.125) | 0.98 +/- 0.02 | 0.89 +/- 0.03 |
| 3000 | 222 | ADJ_CORE | 37 +/- 0 | 0.356 +/- 0.005 | 0.96 +/- 0.01 | 0.42 +/- 0.02 | 0.11 +/- 0.00 | 0.94 +/- 0.00 | 0.688 +/- 0.008 (0.010) | 0.25 +/- 0.03 (0.027) | 0.51 +/- 0.02 | 0.29 +/- 0.04 |
| 3000 | 222 | DET_CORE | 11 +/- 0 | 0.179 +/- 0.006 | 1.62 +/- 0.05 | 0.98 +/- 0.00 | 0.89 +/- 0.02 | 0.91 +/- 0.00 | 0.014 +/- 0.001 (0.010) | 1.00 +/- 0.00 (0.091) | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| 3000 | 222 | NOUN_CORE | 71 +/- 0 | 0.565 +/- 0.006 | 0.80 +/- 0.01 | 0.41 +/- 0.01 | 0.00 +/- 0.00 | 0.90 +/- 0.00 | 0.194 +/- 0.010 (0.010) | 0.38 +/- 0.02 (0.014) | 0.53 +/- 0.02 | 0.22 +/- 0.02 |
| 3000 | 222 | VERB_CORE | 57 +/- 0 | 0.503 +/- 0.012 | 0.88 +/- 0.02 | 0.74 +/- 0.01 | 0.00 +/- 0.00 | 0.88 +/- 0.00 | 0.368 +/- 0.017 (0.010) | 0.63 +/- 0.05 (0.018) | 0.78 +/- 0.03 | 0.22 +/- 0.03 |
| 3000 | 329 | ADJ_CORE | 56 +/- 0 | 0.438 +/- 0.006 | 0.78 +/- 0.01 | 0.31 +/- 0.01 | 0.00 +/- 0.00 | 0.94 +/- 0.00 | 0.725 +/- 0.006 (0.010) | 0.27 +/- 0.03 (0.018) | 0.44 +/- 0.01 | 0.20 +/- 0.01 |
| 3000 | 329 | DET_CORE | 12 +/- 0 | 0.195 +/- 0.003 | 1.63 +/- 0.02 | 1.00 +/- 0.01 | 0.85 +/- 0.03 | 0.91 +/- 0.00 | 0.014 +/- 0.002 (0.010) | 1.00 +/- 0.00 (0.083) | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| 3000 | 329 | NOUN_CORE | 111 +/- 0 | 0.739 +/- 0.001 | 0.67 +/- 0.00 | 0.33 +/- 0.01 | 0.03 +/- 0.00 | 0.90 +/- 0.00 | 0.166 +/- 0.004 (0.010) | 0.28 +/- 0.04 (0.009) | 0.44 +/- 0.01 | 0.22 +/- 0.00 |
| 3000 | 329 | VERB_CORE | 86 +/- 0 | 0.639 +/- 0.006 | 0.74 +/- 0.01 | 0.54 +/- 0.01 | 0.00 +/- 0.00 | 0.88 +/- 0.00 | 0.396 +/- 0.005 (0.010) | 0.50 +/- 0.02 (0.012) | 0.60 +/- 0.01 | 0.23 +/- 0.01 |
| 10000 | 45 | ADJ_CORE | 5 +/- 0 | 0.025 +/- 0.001 | 1.64 +/- 0.06 | 0.90 +/- 0.02 | 0.98 +/- 0.02 | 0.99 +/- 0.00 | 0.027 +/- 0.007 (0.003) | 1.00 +/- 0.00 (0.200) | 0.91 +/- 0.01 | 0.97 +/- 0.00 |
| 10000 | 45 | DET_CORE | 10 +/- 0 | 0.052 +/- 0.001 | 1.73 +/- 0.03 | 0.99 +/- 0.01 | 0.97 +/- 0.01 | 0.97 +/- 0.00 | 0.003 +/- 0.002 (0.003) | 1.00 +/- 0.00 (0.100) | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| 10000 | 45 | NOUN_CORE | 10 +/- 0 | 0.049 +/- 0.000 | 1.63 +/- 0.00 | 0.77 +/- 0.04 | 0.61 +/- 0.01 | 0.98 +/- 0.00 | 0.074 +/- 0.003 (0.003) | 1.00 +/- 0.00 (0.100) | 0.84 +/- 0.03 | 0.66 +/- 0.04 |
| 10000 | 45 | VERB_CORE | 8 +/- 0 | 0.040 +/- 0.001 | 1.68 +/- 0.03 | 1.00 +/- 0.00 | 0.88 +/- 0.02 | 0.98 +/- 0.00 | 0.012 +/- 0.003 (0.003) | 1.00 +/- 0.00 (0.125) | 0.97 +/- 0.02 | 0.91 +/- 0.03 |
| 10000 | 222 | ADJ_CORE | 37 +/- 0 | 0.134 +/- 0.001 | 1.21 +/- 0.01 | 0.42 +/- 0.02 | 0.11 +/- 0.00 | 0.98 +/- 0.00 | 0.687 +/- 0.009 (0.003) | 0.30 +/- 0.05 (0.027) | 0.45 +/- 0.01 | 0.17 +/- 0.02 |
| 10000 | 222 | DET_CORE | 11 +/- 0 | 0.057 +/- 0.001 | 1.72 +/- 0.04 | 1.00 +/- 0.01 | 0.99 +/- 0.02 | 0.97 +/- 0.00 | 0.002 +/- 0.001 (0.003) | 1.00 +/- 0.00 (0.091) | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| 10000 | 222 | NOUN_CORE | 71 +/- 0 | 0.241 +/- 0.003 | 1.13 +/- 0.02 | 0.41 +/- 0.01 | 0.00 +/- 0.00 | 0.97 +/- 0.00 | 0.191 +/- 0.002 (0.003) | 0.31 +/- 0.04 (0.014) | 0.46 +/- 0.01 | 0.07 +/- 0.02 |
| 10000 | 222 | VERB_CORE | 57 +/- 0 | 0.213 +/- 0.002 | 1.24 +/- 0.01 | 0.76 +/- 0.02 | 0.00 +/- 0.00 | 0.96 +/- 0.00 | 0.360 +/- 0.020 (0.003) | 0.47 +/- 0.04 (0.018) | 0.76 +/- 0.03 | 0.10 +/- 0.02 |
| 10000 | 329 | ADJ_CORE | 56 +/- 0 | 0.184 +/- 0.001 | 1.10 +/- 0.01 | 0.32 +/- 0.00 | 0.00 +/- 0.00 | 0.98 +/- 0.00 | 0.711 +/- 0.010 (0.003) | 0.24 +/- 0.07 (0.018) | 0.36 +/- 0.01 | 0.07 +/- 0.01 |
| 10000 | 329 | DET_CORE | 12 +/- 0 | 0.063 +/- 0.001 | 1.76 +/- 0.03 | 1.00 +/- 0.00 | 0.98 +/- 0.00 | 0.97 +/- 0.00 | 0.003 +/- 0.001 (0.003) | 1.00 +/- 0.00 (0.083) | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| 10000 | 329 | NOUN_CORE | 111 +/- 0 | 0.341 +/- 0.001 | 1.02 +/- 0.00 | 0.35 +/- 0.01 | 0.03 +/- 0.00 | 0.97 +/- 0.00 | 0.159 +/- 0.003 (0.003) | 0.23 +/- 0.01 (0.009) | 0.39 +/- 0.01 | 0.10 +/- 0.01 |
| 10000 | 329 | VERB_CORE | 86 +/- 0 | 0.286 +/- 0.005 | 1.11 +/- 0.02 | 0.56 +/- 0.01 | 0.00 +/- 0.00 | 0.96 +/- 0.00 | 0.429 +/- 0.005 (0.003) | 0.32 +/- 0.03 (0.012) | 0.57 +/- 0.00 | 0.08 +/- 0.02 |

Identification accuracy against its null (chance = 1/words), per cell:

| n | vocab | area | ident acc | test |
|---|---|---|---|---|
| 1000 | 45 | ADJ_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 1000 | 45 | DET_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 1000 | 45 | NOUN_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 1000 | 45 | VERB_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 1000 | 222 | ADJ_CORE | 0.30 +/- 0.00 | degenerate:zero_variance |
| 1000 | 222 | DET_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 1000 | 222 | NOUN_CORE | 0.57 +/- 0.02 | p=0.00062 d=23.24 * |
| 1000 | 222 | VERB_CORE | 0.67 +/- 0.03 | p=0.00075 d=21.06 * |
| 3000 | 45 | ADJ_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 3000 | 45 | DET_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 3000 | 45 | NOUN_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 3000 | 45 | VERB_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 3000 | 222 | ADJ_CORE | 0.25 +/- 0.03 | p=0.0063 d=7.22 * |
| 3000 | 222 | DET_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 3000 | 222 | NOUN_CORE | 0.38 +/- 0.02 | p=0.0012 d=16.80 * |
| 3000 | 222 | VERB_CORE | 0.63 +/- 0.05 | p=0.0019 d=13.23 * |
| 3000 | 329 | ADJ_CORE | 0.27 +/- 0.03 | p=0.0038 d=9.38 * |
| 3000 | 329 | DET_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 3000 | 329 | NOUN_CORE | 0.28 +/- 0.04 | p=0.0059 d=7.50 * |
| 3000 | 329 | VERB_CORE | 0.50 +/- 0.02 | p=0.00076 d=21.00 * |
| 10000 | 45 | ADJ_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 10000 | 45 | DET_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 10000 | 45 | NOUN_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 10000 | 45 | VERB_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 10000 | 222 | ADJ_CORE | 0.30 +/- 0.05 | p=0.0099 d=5.77 * |
| 10000 | 222 | DET_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 10000 | 222 | NOUN_CORE | 0.31 +/- 0.04 | p=0.0068 d=6.98 * |
| 10000 | 222 | VERB_CORE | 0.47 +/- 0.04 | p=0.002 d=13.00 * |
| 10000 | 329 | ADJ_CORE | 0.24 +/- 0.07 | p=0.03 d=3.26 * |
| 10000 | 329 | DET_CORE | 1.00 +/- 0.00 | degenerate:zero_variance |
| 10000 | 329 | NOUN_CORE | 0.23 +/- 0.01 | p=0.0013 d=15.93 * |
| 10000 | 329 | VERB_CORE | 0.32 +/- 0.03 | p=0.0039 d=9.24 * |

Cells where retrieval itself could not run because the area had fewer than k never-fired neurons left (n, vocab, area, failed probes): (1000, 222, 'NOUN_CORE', 44), (1000, 222, 'VERB_CORE', 20), (1000, 222, 'NOUN_CORE', 40), (1000, 222, 'VERB_CORE', 21), (1000, 222, 'NOUN_CORE', 44), (1000, 222, 'VERB_CORE', 18)

### Table 8 — The motivating measurement, reproduced (TWO_WORD curriculum, n=3000, k=30)

| area | words | w | w/n | words*k/n | tiling | pairwise (chance) | ident (chance) |
|---|---|---|---|---|---|---|---|
| NOUN_CORE | 19 +/- 0 | 1737 +/- 34 | 0.579 +/- 0.011 | 0.19 +/- 0.00 | 3.05 +/- 0.06 | 0.163 +/- 0.008 (0.010) | 0.72 +/- 0.06 (0.053) |
| VERB_CORE | 10 +/- 0 | 1709 +/- 16 | 0.570 +/- 0.005 | 0.10 +/- 0.00 | 5.70 +/- 0.05 | 0.122 +/- 0.005 (0.010) | 1.00 +/- 0.00 (0.100) |
