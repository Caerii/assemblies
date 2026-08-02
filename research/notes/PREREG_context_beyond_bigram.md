# Pre-registration II: can a CONTEXT area carry information beyond one symbol?

**Written 2026-08-02, BEFORE implementing or running.** Follows
`PREREG_next_token_predictive.md`, whose result stands: a three-pathway model
(LEX / grounded PRED / learned LEX->PRED) reached **0.2074 ± 0.0126**, i.e.
81.6% of the span from chance to the bigram optimum.

**That model is a bigram learner by construction** -- LEX holds only the
current word, so nothing about the earlier prefix can reach PRED. This study
asks the question that actually matters for language: can assembly calculus
hold context across more than one symbol?

## Architecture under test

    stim[w]   -> LEX                       current word
    gstim[w]  -> PRED                      grounding signature (unchanged)
    LEX       -> CONTEXT, CONTEXT -> CONTEXT   accumulating prefix trace
    LEX + CONTEXT -> PRED                  prediction from word AND history

CONTEXT is reset between SENTENCES and NOT between words, so within a sentence
it carries a trace of everything seen so far. Training is otherwise unchanged:
drive LEX with `a` and PRED with `gstim[b]` co-actively.

## Parameters, FIXED NOW

Inherited unchanged from study I: n=10000, k=200, p=0.05 (k*p=10), beta=0.10,
3 train rounds/pair, 200 training sentences, seeds 42..51. CONTEXT gets the
same n and k as the other areas. **Changing any of these invalidates this
pre-registration**; a tuned rerun is a new one.

Reference values (same generator): chance **0.0900**, unigram **0.1178**,
bigram-optimal **0.2338**, no-context model **0.2074 ± 0.0126**.

## Hypotheses, with honest predictions

### H1 — CONTEXT helps at all

CONTEXT arm > no-context arm: lower bound of the PAIRED per-seed difference
> 0.
*Prediction:* **marginal, and I expect it to fail or barely clear.**

### H2 (the real question) — information beyond the bigram

CONTEXT arm exceeds the bigram optimum: lower bound > 0.2338.
*Prediction:* **I predict this FAILS.** Beating it requires CONTEXT to hold
prefix information distinctly, and the repo's standing result on shared helper
areas ([[mood-chain-collapse-mechanism]]) is that Hebbian k-WTA merges
distinct chains into one attractor within ~20 sentences. I expect the same
mechanism here.

### H3 (diagnostic, runs regardless) — does CONTEXT collapse?

Measure mean pairwise overlap of CONTEXT assemblies across DIFFERENT prefixes
at the same sentence position, after training.
*Prediction:* **high overlap, > 0.5** -- i.e. CONTEXT has collapsed toward a
single attractor and is carrying little prefix information. If H2 fails, this
is the explanation; if H3 shows LOW overlap while H2 still fails, my
explanation is wrong and the bottleneck is the PRED readout instead.

### H4 (null control) — beta = 0 at chance

Runs FIRST, as before. Upper bound < 0.1178.
*If it beats chance the study stops and becomes a bug hunt.*

## Committed in advance

1. H4 first. No parameter changes after seeing results.
2. Report all arms including failures. A negative H2 is a real result: it
   would say AC needs an explicit mechanism for history, not just a recurrent
   area, and would point at the fiber-gating primitive the repo lacks.
3. H3 is reported whatever H2 does -- it is the mechanism, not a consolation.

## Results

*(empty — appended after the runs; nothing above may change)*

**Run 2026-08-02, 10 seeds (42..51), paired.**

| arm | MRR (mean ± 95% CI) |
| --- | ---: |
| no-context (study I, reproduced exactly) | 0.2074 ± 0.0126 |
| **CONTEXT arm** | **0.1046 ± 0.0128** |
| paired difference | **−0.1028 ± 0.0161** |
| bigram optimum (H2 bar) | 0.2338 |
| H4 null (beta = 0) | ~0.086, at chance |

**H1 FALSIFIED, decisively and in the wrong direction.** CONTEXT does not
merely fail to help -- it HALVES performance. The paired difference excludes
zero by a wide margin.

**H2 FALSIFIED.** 0.1046 is nowhere near 0.2338. No information beyond the
bigram was captured; in fact the bigram information already present was
destroyed.

**H3 CONFIRMED — and it is the explanation.** Mean pairwise overlap of CONTEXT
assemblies across DIFFERENT prefixes at the same position: **0.7566 ± 0.0958**,
against a pre-registered threshold of 0.5. CONTEXT collapsed toward a single
attractor, so it carries almost no prefix information. Worse, being nearly
constant, it injects the SAME drive into PRED on every prediction, swamping
the informative LEX->PRED signal -- which is why the arm lands below the
no-context model rather than beside it.

Predictions recorded in advance were "H1 marginal or fails", "H2 fails",
"H3 overlap > 0.5". All three correct.

## What this reproduces

This is [[mood-chain-collapse-mechanism]] and
[[recurrence-is-the-collapse-channel]] appearing in a third setting: a SHARED
area with SELF-RECURRENCE, trained by Hebbian k-WTA, merges its distinct
states into one attractor. The standing repo finding is that self-recurrence
during TRAINING is the collapse channel, and that feed-forward builds show no
comparable ceiling.

So the negative result is not "AC cannot hold context". It is "a recurrent
helper area cannot hold context under Hebbian k-WTA", which is a much more
specific and more actionable claim -- and it predicts the fix.

## Next experiment (new pre-registration required)

Close CONTEXT's self-recurrence during training and keep everything else
fixed. If the collapse is the recurrence, overlap should fall well below
0.7566 and the arm should recover to at least the no-context 0.2074. The
repo already has the fiber-gating primitive (task #45) needed to open and
close that fiber on a schedule rather than leaving it always-on.
