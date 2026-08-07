# The parser's regime is not reachable from a two-area model — stopping here

> **SUPERSEDED 2026-08-07, and the negative below is now EXPLAINED** — see
> `the_mean_spread_was_hiding_a_bimodal_distribution.md`.
>
> The measurements here are correct. The conclusion drawn from them — "the
> behaviour is not derivable from the minimal model, so stop" — was premature
> by one piece of arithmetic. The toy's spread of 0.017 sits against a
> random-pair floor of `k/n` = 0.0200: it is **pinned at its floor**, and a
> substrate at its floor has no room to express intermediate overlap. That is
> why the knobs produced only "separated or collapsed" and nothing between.
>
> The parser's 0.13 is 12.5× ITS floor, and its distribution is bimodal in a way
> the mean cannot represent. The toy's words are independent random stimuli, so
> they share no input features and there is nothing to harden — which is
> precisely the mechanism the parser exhibits. **The model was wrong for the
> question, not the question unanswerable.**

## The stopping rule, invoked

Before running this I wrote: *"#123 is the last localization. If it confirms
separation, we stop diagnosing and start fixing. If it disconfirms, I'd stop the
chase entirely and report the honest state."*

It disconfirmed. This note is the honest state.

## Two knobs, neither reaches the parser

**Shared drive** — every word fires a shared stimulus alongside its private one,
sized from 0 to 40:

| shared k | src spread | ret@6 | margin |
|---|---|---|---|
| 0 | 0.0171 | 1.000 | 13.4 |
| 2 | 0.0149 | 1.000 | 13.8 |
| **4** | **0.4963** | **0.297** | 1.08 |
| 40 | 0.9261 | 0.168 | 1.07 |

**A phase transition, not a dial.** Between shared k=2 and k=4 the area jumps
from near-perfect separation to near-collapse. There is no intermediate regime.

**Build rounds** — the convergence hypothesis, which the shared-drive knob was
deliberately designed not to touch so it would not be quietly assumed:

| rounds | src spread | ret@6 |
|---|---|---|
| 1 | 0.0133 | 1.000 |
| 3 | 0.0150 | 1.000 |
| 24 | 0.0172 | 1.000 |

**Flat.** One build round produces spread 0.013 and retrieval 1.000, the same as
twenty-four. Under-convergence does not produce intermediate overlap either —
which also retires the neat story that the parser's 0.27 freshness and 0.13
spread were one thing.

## What that means

The toy substrate has **two regimes and nothing between them**: separated
(spread ~0.015, retrieval 1.000) or collapsed (spread ~0.4–0.9, retrieval at
chance). The parser sits at **spread 0.13, retrieval 0.73** — genuinely
intermediate, and not reproducible by either manipulation.

So the parser's assemblies live somewhere a two-area model does not go. Whatever
puts them there is a property of the **full system** — eight core areas, a
recurrent `CONTEXT` loop, fiber gating, a multi-stage curriculum, 517 stimuli
with shared grounding, an area that grows throughout training — and not of the
`SRC → DST` primitive.

**That is a result about the method, not only about this question.** Reductive
localization worked for eight mechanisms and has now hit its limit: the
behaviour of interest is not derivable from the minimal model, so no further
minimal-model experiment can explain it.

## What is actually established

Solid, after nine attempts:

- **Role binding works** — retrieval 12–19× chance on the parser.
- **The instrument was wrong** — `input_drive` measures how much; binding
  encodes where. Pinned by a test.
- **Excluded as explanations for the parser's weakness**: the readout
  instrument, the training route, `norm_init`, the index space, consolidation,
  target capacity, packing density, shared drive, under-convergence.
- **The parser retrieves 0.73 at 6 candidates and 0.375 at 36**; a clean
  substrate retrieves 1.000 at both, out to α = 1.92.

Not established, and now looking hard to establish reductively: **why.**

## What I would do instead of a tenth hypothesis

Two options, and neither is more localization.

1. **Measure the parser directly.** Ablate parts of the full system — the
   `CONTEXT` recurrence, the gating circuit, the multi-stage schedule — and see
   which one moves spread off 0.13. Expensive, and it works on the system that
   actually exhibits the phenomenon rather than on a model that does not.
2. **Accept 0.73 and design around it.** A downstream reader that tolerates a
   0.73 retrieval rate is a different engineering problem from one that assumes
   1.000, and it does not require knowing why.

The second is cheaper and the first is the science. Neither should start before
someone decides which question is being answered — and my recommendation is to
make that call deliberately rather than by running one more experiment.

## The methodological note worth keeping

The first version of this sweep **printed a confirmation**. Its acceptance test
was "spread ≥ 0.13 and retrieval ≤ the parser's", and a fully collapsed area
satisfied both at chance. A degenerate arm passed the criterion — the third time
this session — and the guard I had written watched only for the knob
UNDERSHOOTING, never for it overshooting past the regime of interest.

The corrected form requires landing in a **window** around the target and
retrieval staying **above chance**, and it correctly refuses both tables.
