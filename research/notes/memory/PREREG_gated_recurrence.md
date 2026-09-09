# Pre-registration IV: can CONTEXT hold history if its recurrence cannot learn?

**Written 2026-08-02, BEFORE running.** Follows `PREREG_context_beyond_bigram.md`
(study II) and `PREREG_context_recurrence_off.md` (study III). Task #87.

## What the first three studies established

| study | result |
| --- | --- |
| I | three-pathway model reaches **0.2074 ± 0.0126**, 81.6% of the chance→bigram span; ablation confirms the grounding pathway is the mechanism |
| II | adding a recurrent CONTEXT area **halves** it to **0.1046 ± 0.0128**; CONTEXT assemblies across different prefixes overlap **0.7566** — collapsed to one attractor |
| III | closing CONTEXT→CONTEXT during TRAINING drops overlap to 0.1755, so the collapse is training-time. H2 was never tested: the arm meant to test it was a **dead fiber** |

The open question is the one study III could not ask: **is it the recurrence
itself that collapses CONTEXT, or the PLASTICITY on that recurrence?**

## Why this needs per-fiber beta and not gating

Established today and pinned by
`test_self_recurrence_is_implied_by_any_open_input_fiber`:
`InhibitionState.project_map` derives `X -> X` from the state of the fibers
**into** `X`, never from `fiber_states[X][X]` — faithful to
`.reference/dmitropolsky-assemblies/parser.py:413`. So in NEMO an area that is
receiving is necessarily self-sustaining. Gating can only switch CONTEXT off
entirely, which is study III's arm, not this one.

Therefore the only expressible form of "live but non-plastic recurrence" is a
per-fiber beta: `brain.update_plasticity("CONTEXT", "CONTEXT", 0.0)`.
`Area.update_beta_by_area` does NOT work — it writes a dict the engine does not
read, and using it is what made study III's arm identical to its control
(task #88, now fixed to raise).

## Arms, all paired on the same 10 seeds (42..51)

| arm | CONTEXT→CONTEXT | beta on it | reference value |
| --- | --- | --- | ---: |
| **A** no-context (study I) | absent | — | 0.2074 ± 0.0126 |
| **B** recurrent, plastic (study II) | open | 0.10 | 0.1046 ± 0.0128 |
| **C** recurrent, frozen — **the new arm** | open | **0.0** | ? |

## Parameters, FIXED NOW

Inherited unchanged from studies I–II: n=10000, k=200, p=0.05 (k·p=10),
beta=0.10 elsewhere, 3 train rounds/pair, 200 training sentences, CONTEXT reset
per SENTENCE. **Changing any of these invalidates this pre-registration.**

Reference baselines on this generator: chance **0.0900**, unigram **0.1178**,
bigram-optimal **0.2338**.

## Hypotheses

### H0 (gate, runs FIRST) — arm C is a real arm

Two checks before any hypothesis is read:
1. `engine.get_beta("CONTEXT", "CONTEXT") == 0.0` — the override arrived.
2. The CONTEXT→CONTEXT fiber carries **nonzero drive** during training, and arm
   C is **not bit-identical** to arm B on any seed (`compare_arms` enforces this).

*If either fails, arm C is a dead fiber and the study stops.* This gate exists
because study III's H2 failed exactly here and was almost reported as a
negative result.

### H1 — the collapse is plasticity, not recurrence

Arm C's CONTEXT overlap across different prefixes is **below 0.5** (arm B: 0.7566).
*Prediction:* **supported.** Hebbian potentiation on a self-fiber is what drives
distinct states into a shared attractor; a frozen random self-projection has no
mechanism to merge them.

### H2 — arm C recovers from arm B

Paired per-seed difference C − B has lower bound > 0.
*Prediction:* **supported**, and largely as a consequence of H1: a
non-collapsed CONTEXT stops injecting the same swamping drive into PRED.

### H3 (the real question) — does frozen recurrence ADD anything?

Paired per-seed difference C − A has lower bound > 0.
*Prediction:* **I predict this FAILS.** A frozen self-fiber is a fixed random
projection: it can keep CONTEXT's states apart, but nothing ever writes the
prefix→prediction mapping onto it, and LEX→PRED already carries all the bigram
information. I expect arm C to land at **parity with arm A**, near 0.2074.

If H3 fails while H1 and H2 hold, the conclusion is sharp and not a
disappointment: **preventing collapse is not the same as storing history.** AC
would need a mechanism that WRITES prefix structure without merging it —
plausibly separate areas time-shared by gating, as
`.reference/.../recursive_parser.py` does for constituents, rather than one
shared recurrent area at any beta.

### H4 — arm C does not exceed the bigram optimum

Upper bound < 0.2338.
*Prediction:* supported. Exceeding it would mean genuine supra-bigram
information and I would first check the readout for a leak.

## Committed in advance

1. H0 first; if it fails the study stops and becomes a bug hunt.
2. No parameter changes after seeing results. A tuned rerun is a NEW
   pre-registration.
3. All arms reported, including H3 failing — which is the outcome I expect.
4. `beta_rec` is swept at **0.0 only**. Trying 0.01, 0.03, … after seeing 0.0
   would be tuning; if a sweep is wanted it gets its own pre-registration.
5. Every number is an ensemble over the 10 seeds with a confidence bound, via
   `diagnostics.compare_arms` / `paired_delta`. Point estimates decide nothing.

## Standing caveat on all four studies

MRR here ranks by PARTIAL OVERLAP, which `ENGINE.md` records as exactly the
class of readout the candidate sampler distorts (next-token MRR: 0.1159 exact
vs 0.0744 sampled). The PAIRED comparisons should survive, since every arm runs
on the same sampler; the absolute "81.6% of span" figure is the fragile one.
Task #85 (compute the drive instead of sampling it) is what settles this, and
study I should be re-run on it.

## Results

**Run 2026-08-02, 10 seeds (42..51), paired. Nothing above was changed.**

**H0 PASSED.** `get_beta("CONTEXT","CONTEXT") == 0.0` in the engine, and
`compare_arms` accepted all three arms — so arm C is real this time, unlike
study III's.

| arm | MRR (mean ± 95% CI) |
| --- | ---: |
| A no-context | 0.2074 ± 0.0126 |
| B ctx recurrent, plastic | 0.1046 ± 0.0128 |
| **C ctx recurrent, frozen (β_rec = 0)** | **0.1152 ± 0.0121** |

A and B reproduce studies I and II to four decimals, which is the check that the
harness did not drift.

| paired difference | value | pre-registered bar |
| --- | ---: | --- |
| H2  C − B | **+0.0106 ± 0.0142** | lower bound > 0 |
| H3  C − A | **−0.0922 ± 0.0148** | lower bound > 0 |

| CONTEXT overlap across different prefixes | value |
| --- | ---: |
| B plastic | 0.7566 ± 0.0958 |
| **C frozen** | **0.1756 ± 0.0200** |

### H1 SUPPORTED, decisively

Freezing the self-fiber takes prefix-overlap from **0.7566 to 0.1756**, far
below the pre-registered 0.5. **The collapse is plasticity ON the recurrence,
not the recurrence itself.** Prediction correct.

### H2 FALSIFIED — and I predicted it would be supported

C − B is +0.0106 with a CI of (−0.0146, +0.0431), which covers zero. A **4.3×
improvement in distinctness bought no measurable prediction.** I predicted this
would follow from H1. It did not.

### H3 FALSIFIED, as predicted — but worse than predicted

I predicted arm C would land at *parity* with no-context (~0.2074). It lands at
0.1152, i.e. **0.0922 BELOW it**, with the whole CI negative. Direction right,
magnitude wrong: a frozen recurrent CONTEXT still actively hurts.

### H4 SUPPORTED trivially

0.1152 is nowhere near 0.2338.

## The finding: distinctness is not information

This is a clean dissociation, and it is more informative than H2 succeeding
would have been. CONTEXT can be made to hold **distinct** states per prefix —
that is now demonstrated — and those distinct states carry **no usable
information** about the next word.

Both failing arms have a nameable defect, and they are opposite:

* **B (plastic)** over-generalises. Every prefix maps to the same attractor, so
  CONTEXT injects a constant into PRED and swamps the informative LEX→PRED
  signal.
* **C (frozen)** under-generalises. A frozen random self-fiber is a **hash of
  the prefix**: maximally distinct, with no similarity structure at all, so
  nothing transfers between related prefixes.

Neither is what context requires, which is *graded* similarity — close states
for similar prefixes, distant for different ones.

### Why C cannot learn, measured (exploratory, not pre-registered)

Repetition counts on the training corpus, 200 sentences, 1056 prediction sites:

| prefix length | distinct | mean repetitions | seen exactly once |
| --- | ---: | ---: | ---: |
| 1 (unigram) | 50 | 21.12 | 0% |
| 2 (bigram) | 431 | 2.45 | 44% |
| **full sentence prefix** | **743** | **1.42** | **90%** |

CONTEXT accumulates the FULL prefix, and 90% of those are seen exactly once.
Hebbian potentiation cannot accumulate anything from a state that occurs once,
so **arm C is unlearnable by construction at this corpus size** — no substrate
would learn a full-prefix→next-word map from 743 prefixes seen 1.42 times each.
The bigram pathway works precisely because its contexts repeat 2.45× and the
unigram 21×.

## The caveat that decides how much this means

Graded similarity is exactly the structure the candidate sampler discards.
`ENGINE.md`: keying fixes the diagonal, not the off-diagonal — "similar inputs,
similar drives" is gone at the draw, and the measured cost on this very task is
0.1159 exact vs 0.0744 sampled.

So arm C is handicapped in precisely the way the engine is known to handicap
partial-overlap structure, and **this experiment cannot distinguish "AC cannot
represent graded context" from "the sampler removed the mechanism that
would."** The repo has separately demonstrated graded composition
([[vp-composition-structured-and-productive]]), which is evidence for the
second reading.

**Task #85 is therefore no longer only a soundness chore — it is the blocker on
the scientific question.** Study IV should be re-run on exact drive before its
negative half is treated as a statement about the assembly calculus.

## What is safe to conclude now

1. A recurrent shared area collapses under Hebbian k-WTA, and the collapse
   channel is **plasticity on the self-fiber** (H1, 4.3× effect).
2. Removing that plasticity **does not** produce a usable context
   representation (H2, H3).
3. The reason is structural, not a tuning failure: β_rec is a single scalar
   choosing between over- and under-generalisation, and **the useful regime may
   be empty at this corpus size regardless of β**, because the states it would
   have to learn from do not repeat.

That third point is what the next study must test, and it is a claim about the
*window* rather than a point — see the follow-up pre-registration.
