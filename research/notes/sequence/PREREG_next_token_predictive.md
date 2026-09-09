# Pre-registration: next-token prediction with a distinct prediction site

**Written 2026-08-02, BEFORE implementing or running anything.** Task #87.
Nothing in this document may be edited after the first result is seen; results
go in a separate section appended at the end, and any deviation from the plan
gets recorded there as a deviation rather than silently applied.

Why pre-register this one specifically: the seed-to-seed sd on this task
(0.0180) is comparable to the entire effect being chased, and I have already
once acted on a 1.68-sd point estimate as though it were a finding. The
failure mode here is not being wrong, it is being *retrospectively right* --
tuning `k`, `beta`, `rounds` or the readout until a number clears a bar and
then reporting the bar as met.

---

## Background, already measured (not part of this study)

On the test grammar, 1042 predictions, optimal predictors:

| predictor | MRR |
| --- | ---: |
| chance (uniform over 50) | 0.0900 |
| unigram (word frequency, no context) | 0.1362 |
| **bigram-optimal (current word only)** | **0.2454** |
| prefix-optimal (sparse estimate, overfit) | 0.1732 |
| current model (18 seeds) | 0.0904 ± 0.0083 |

Within a class the grammar is uniform and carries no information, so no
predictor can approach 1.0. **0.2454 is the honest ceiling** for anything
using current-word information only.

## The architecture under test

    stim[w]  -> LEX     word representation
    gstim[w] -> PRED    grounding signature for w, independent of LEX
    LEX      -> PRED    the learned transition

Train on adjacent pair (a, b): drive LEX with `stim[a]` and PRED with
`gstim[b]` in the SAME `project()` call, so they are co-active and Hebbian
learns `a -> b` on the LEX->PRED fiber.

Read out after word a: drive LEX with `stim[a]`, project LEX->PRED with no
grounding and no plasticity, rank words by overlap of PRED against each stored
grounding signature.

The grounding pathway is load-bearing: without it PRED's per-word assemblies
would themselves be defined by LEX->PRED, and that one fiber would have to
carry both `w -> PRED(w)` and `w -> PRED(next(w))`.

## Parameters, FIXED NOW

Chosen before any run, and not to be adjusted afterwards:

| | value | why this value, decided in advance |
| --- | --- | --- |
| n (LEX, PRED) | 10000 | large enough that 50 words do not saturate |
| k | 200 | with p=0.05 gives **k*p = 10** |
| p | 0.05 | |
| beta | 0.10 | |
| train rounds / pair | 3 | |
| corpus | 200 sentences | |
| seeds | 10 (42..51) | |

`k*p = 10` is the one parameter choice that is a real decision rather than a
convention, so the reasoning is recorded: drive is Binomial(total_k, p), and at
low `k*p` the top-k boundary falls inside a large tied band, so a large
fraction of every assembly is decided by index convention rather than by
drive. Measured earlier: 62% tie-decided at `k*p = 1`, 40% at 2.25, 12% at
15.9. Running this study at `k*p ~ 1` would mean measuring allocation order.
**If I later change k or p, that invalidates this pre-registration.**

## Statistic and decision rule, FIXED NOW

* Statistic: mean MRR over 10 seeds, 95% CI from the t distribution.
* A hypothesis counts as SUPPORTED only if the relevant **confidence bound**
  clears the threshold -- never the point estimate.
* All four arms below run on the same 10 seeds, paired.

---

## Hypotheses

### H1 (primary) — the architecture can represent the transition

The three-pathway model **exceeds the unigram baseline**: lower 95% bound
> 0.1362.

*Rationale:* unigram uses no context at all. A model that has genuinely
learned `a -> b` must beat it, or it has not learned a transition.
*Falsified if:* upper bound ≤ 0.1362.
*Prediction:* mean in **0.15 – 0.22**.

### H2 — it will NOT reach the bigram optimum

Lower bound < 0.2454.

*Rationale:* a k-WTA overlap readout is lossy, 50 grounding signatures in one
PRED area interfere, and ~12% of each assembly is still tie-decided at
k*p = 10. Claiming the optimum would more likely indicate a leak than a
success.
*Falsified if:* lower bound ≥ 0.2454 — which I would treat as **evidence of a
defect**, and would then check for the readout seeing `gstim` directly.

### H3 (mechanism) — the grounding pathway is what does the work

Ablation: define PRED's per-word signatures via LEX->PRED itself instead of a
separate grounding stimulus, changing nothing else. This arm falls back to
within noise of chance (upper bound < 0.1362).

*Rationale:* the identity and transition mappings then compete on one fiber.
*Falsified if:* the ablation matches the full model — which would mean the
grounding pathway is NOT the mechanism and my explanation of the old failure
is wrong.

### H4 (null control) — beta = 0 must sit at chance

With plasticity off, upper bound < 0.1362 and CI covering 0.0900.

*Rationale:* nothing is learned, so anything above chance is the readout
leaking structure rather than the model predicting. This is the arm that
catches a "fake perfect" result.
*Falsified if:* beta=0 beats chance — in which case **H1 is void regardless of
its own result**, because the readout is scoring something unlearned.

### H5 (secondary, exploratory) — interference scales with vocabulary

At fixed n and k, MRR degrades monotonically as vocabulary grows 25 -> 50 ->
100. Labelled exploratory: no decision rule, reported as a curve.

---

## Committed in advance

1. **No parameter changes after seeing results.** If the result is negative it
   is reported negative. A tuned rerun becomes a NEW pre-registration.
2. **No dropping seeds**, no swapping the statistic, no moving to top-3 or
   class accuracy because MRR disappointed. Additional metrics may be reported
   as clearly-labelled exploratory.
3. **H4 is checked first.** If the null control beats chance, the study stops
   and becomes a bug hunt.
4. Report **all four arms** whatever they show, including H1 failing.
5. The comparison is against **0.1362 (unigram)**, not 0.0900 (chance).
   Beating chance is the floor here and would not be evidence of anything.

## Results

*(empty — to be appended after the runs; nothing above may change)*

**Run 2026-08-02, 10 seeds (42..51), paired.** Baselines recomputed on this
generator: chance 0.0900, unigram 0.1178, bigram-optimal 0.2338.

| arm | MRR (mean ± 95% CI) | range |
| --- | ---: | --- |
| **H1 three-pathway, beta=0.10** | **0.2074 ± 0.0126** | 0.1758 – 0.2394 |
| H3 ablation (grounding removed) | 0.0815 ± 0.0074 | 0.0665 – 0.0931 |
| H4 null (beta = 0) | 0.0954 ± 0.0093 | 0.0779 – 0.1182 |
| *(old single-area architecture)* | *0.0904 ± 0.0083* | *0.0551 – 0.1220* |

**H1 SUPPORTED.** Lower bound 0.1948 > 0.1362. Pre-registered prediction was
"mean in 0.15 – 0.22"; observed 0.2074.

**H2 SUPPORTED.** Upper bound 0.2200 < 0.2338, so it does not claim the
bigram optimum — no leak indicated.

**H3 SUPPORTED.** Removing ONLY the grounding pathway drops to 0.0815, upper
bound 0.0889, i.e. chance. The grounding pathway is the mechanism, as
theorised: without it the identity and transition mappings compete on one
fiber and neither survives.

**H4 SUPPORTED.** CI [0.0861, 0.1047] covers chance 0.0900.

The model captures **(0.2074 − 0.0900) / (0.2338 − 0.0900) = 81.6%** of the
span from chance to the bigram optimum, against **0.3%** for the old
architecture.

## Deviation from the plan (one, recorded)

The readout originally broke ties with `sorted()`, which falls through to
vocabulary order (DET, ADJ, NOUN, VERB, PREP) — and that order correlates with
which classes actually follow. Overlap is quantised to multiples of 1/k and
was frequently **0 for all 50 candidates**, so ranking was decided almost
entirely by that tie-break. The H4 null control scored **0.1191** this way,
i.e. the unigram baseline, with nothing learned.

Fixed by breaking ties with a seeded random key, after which H4 fell to
chance. This was a harness defect found BY the null control before any
hypothesis was tested, not a change made after seeing H1 — which is what H4
was placed first to guarantee. No parameter was altered.
