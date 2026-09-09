# k-WTA amplifies input overlap — the law, and it predicts the parser

## What I got wrong, and why it matters

I proposed reserving a "private fraction" of k per word so that
`max overlap ≤ 1 − private_fraction` by construction. That was a hand-rolled
restatement of **COLT 2022, Theorem 4 (Multiple Assemblies)**:

> the overlap in the core sets A\* and B\* will preserve the overlap of the
> stimulus classes, so that `|A* ∩ B*| ≤ αk`

Overlap preservation is **proved**, not designed. So the parser isn't missing a
feature — the question is whether the guarantee holds in our implementation.

Second correction, larger: I earlier concluded the parser's regime was "not
reachable from a two-area model" and invoked a stopping rule. **That was wrong,
and for a specific reason — every toy ran at α = 0**, independent random
stimuli, which is the one column of the table below where nothing happens. The
minimal model reproduces the parser perfectly well once its inputs share
features.

## β₀ is vacuous everywhere, including the papers' own parameters

Theorem 1 requires `β ≥ β₀ = (1/r²)·[√(2−r²)·√(2ln(n/k)) + √6] / [√(kp) + √(2ln(n/k))]`
and bounds support by `|A*| ≤ k / (1 − exp(−(β/β₀)²))`.

| regime | β | β₀ | β/β₀ | support bound |
|---|---|---|---|---|
| parser `NOUN_CORE` | 0.10 | 1.669 | 0.060 | ≤ 279k |
| our toy | 0.10 | 1.613 | 0.062 | ≤ 261k |
| **acquisition paper** | **0.06** | **1.349** | **0.044** | **≤ 506k** |

The bound is vacuous at every β anyone runs, **the papers' included**. So "we are
below β₀" is *not* evidence that our β is wrong — these are asymptotic sufficient
conditions with unoptimised constants. Whether preservation survives below β₀ is
an empirical question, and that is what was measured.

## The law, on the bare substrate, with the paper's own stimulus model

`numpy_exact`, n=10000, k=100, p=0.05, r=0.9, two classes with `|S_A ∩ S_B| = αk`.
Theorem 3 (Recall) predicts ≥ 0.989 and **measured 0.998–1.000 in every cell**, so
the assemblies really form; support 1.00–1.56k, so nothing is collapsed.

| β | α=0.10 | α=0.25 | α=0.50 | α=0.75 |
|---|---|---|---|---|
| **0.05** | **0.100** | **0.283** | 0.697 | 1.023 |
| 0.10 | 0.170 | 0.497 | 0.897 | 1.023 |
| 0.50 | 0.683 | 0.977 | 1.000 | 1.000 |
| 1.00 | 0.683 | 0.977 | 1.000 | 1.000 |
| 1.50 | 0.680 | 0.977 | 1.000 | 1.000 |
| 2.00 | 0.680 | 0.977 | 1.000 | 1.000 |
| 4.00 | 0.673 | 0.977 | 1.000 | 1.000 |

**k-WTA amplifies overlap, and β is the gain.** The bound held in 9/35 cells —
all of them at low α or low β.

Three things worth stating precisely:

1. **Going *above* β₀ does not help.** β = 1.5, 2.0, 4.0 are at and beyond
   β₀ = 1.349 and are identical to β = 0.5. The theorem's own hypothesis regime
   does not restore preservation here. Testing only below β₀ would have let
   "higher β is worse" ship without ever entering the regime that would falsify
   it.
2. **The curve saturates at β ≈ 0.5** and never moves again — consistent with the
   weight cap, not with plasticity continuing to bite.
3. **Low β is best.** At β = 0.05 preservation holds through α = 0.25.

The mechanism is plain once seen: A is built first, so the shared sensory
neurons already carry potentiated synapses onto A\*, and they pull B onto it.
β sets how hard they pull.

## It predicts the parser

Grounding features per word, α as Dice, against measured assembly overlap
(2485 pairs, both seeds):

| α bin | pairs | mean α | mean overlap | frac identical |
|---|---|---|---|---|
| 0 | 2093 | 0.000 | 0.044 | 0.0% |
| [0.15,0.35) | 59 | 0.320 | 0.332 | 0.0% |
| [0.35,0.65) | 224 | 0.447 | 0.551 | **17.0%** |
| [0.65,0.95) | 63 | 0.783 | 0.653 | **17.5%** |
| [0.95,1.0] | 46 | 1.000 | 0.718 | **28–39%** |

Monotone, and the identical-assembly fraction rises with α exactly as the law
says it should.

**The pre-named confound is excluded.** Words are trained sequentially, so
training order could have produced the same pattern. Rank correlation of overlap
with **α is +0.399 / +0.575**; with **training-order gap it is −0.080 / −0.063**.
Order is not the driver.

Two honest qualifications:

- The parser is **milder than the bare law** (0.33 vs 0.497 at α≈0.32; 0.55 vs
  0.897 at α≈0.45). It has phon drive that the substrate sweep does not, plus a
  different n/k. Directionally confirmed, quantitatively not a fit.
- **α = 0 pairs still overlap 0.044 = 4.4× the k/n floor.** Shared-area crowding
  at M = 71 is a second, additive effect that α does not explain.

## The design rule this yields

For a lexicon that scales, the constructional constraint is now a number:

- **β ≈ 0.05**, not 0.10 — preservation holds to α = 0.25 at 0.05 and fails
  already at α = 0.10 when β = 0.10.
- **any two words may share at most ~25% of their grounding features.**

The parser currently violates both: β = 0.10, and 16% of word pairs have α > 0
with 392 pairs above it, 333 of them at α ≥ 0.35.

## What this does and does not settle

**Settled:** the collision is a substrate property, reproducible in two areas,
governed by α and β, and it predicts the production system. The earlier
"not derivable from the minimal model" conclusion is retracted.

**Not settled:** whether the fix is (a) lower β, (b) a featural code with lower
α, or (c) the papers' architecture — PHON→LEX1/LEX2 with semantics in separate
areas, which routes word identity through a pathway that is unique per word by
construction. These are not exclusive and (c) is #91.

## Also checked

E%-WTA (Hoff 2026) **is** available on `NumpyExactEngine` — #94 was genuinely
closed, and the engine's `add_area` supports `winner_policy`. But at our sparse
parameters the paper's ε-window yields **|A| = 1** and the repo's σ-window
yields 6, against k = 40. That is the sparse-regime collapse the policy
docstring predicts and the paper's own Fig. 4 reports as making AC operations
"unfeasible". Available is not the same as usable here.

## Limits

3 seeds on the substrate sweep, 2 on the parser. `r = 0.9, q = 0` — the outside
noise term is untested. α as Dice for unequal feature-set sizes reduces to the
paper's α only when sizes match. The substrate sweep uses a sensory area with
pinned winners as the stimulus source; recall at 0.998–1.000 says plasticity on
that fiber works, but it is not the paper's literal "sensory column".
