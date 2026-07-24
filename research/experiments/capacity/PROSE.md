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

