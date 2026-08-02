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

*(empty — appended after the runs; nothing above may change)*
