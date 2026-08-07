# Phase B: the semantic drive share is the lever — duplicates 0.32 → 0.06, retrieval 0.77 → 0.97

## The mechanism, and why this knob

`apply_lexicon_word` fires phon **plus one stimulus per grounding feature**,
simultaneously, all of size k. So word identity was **1 of (1+F) equal drivers**
— measured share 0.20–0.33 for F = 2..4 — and two words sharing all their
features had input overlap `2F/(2+2F) = 0.667` at F = 2, which the substrate law
maps to assembly overlap ≈1.0. That is the duplicate clusters.

`phon_weight` (W) makes phon contribute `W·k`, so identity's share is `W/(W+F)`
and the worst pair becomes `2F/(2W+2F)`.

This is the papers' change *in the relevant sense*: Mitropolsky 2025 delivers
semantics from a **bounded set of areas**, we deliver it as an **unbounded set of
stimuli**. Raising W re-weights that pathway rather than restructuring it — and
if re-weighting does nothing, restructuring probably wouldn't either.

## Result — 2 seeds, both β

| β | W | share | distinct | dup | spread | retP | retA | **GROUND** |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 1 | 0.25 | 0.746 | 0.317 | 0.1231 | 0.806 | 0.733 | 0.819 |
| 0.10 | 3 | 0.50 | 0.866 | 0.176 | 0.0741 | 0.906 | 0.879 | 0.731 |
| 0.10 | 6 | 0.67 | 0.930 | 0.120 | 0.0597 | 0.940 | 0.910 | 0.744 |
| 0.05 | 1 | 0.25 | 0.775 | 0.303 | 0.0906 | 0.872 | 0.844 | 0.783 |
| 0.05 | 3 | 0.50 | 0.944 | 0.085 | 0.0548 | 0.953 | 0.931 | 0.679 |
| **0.05** | **6** | **0.67** | **0.958** | **0.063** | **0.0420** | **0.974** | **0.964** | **0.671** |

Against the original default (β=0.10, W=1):

- **exact duplicates 0.317 → 0.063** — a 5× reduction, and the thing β alone
  could not move (Phase A: −0.014, CI including zero)
- **role retrieval 0.806/0.733 → 0.974/0.964**
- **grounding-only retrieval 0.819 → 0.671** — the cost

## It is not the degenerate arm, and that was the pre-registered risk

`GROUND` fires **only** the word's grounding stimuli and asks whether semantics
alone still finds the word among 6 candidates — the pathway a novel word with
known features needs. Chance is 0.167.

It falls, but to **0.671 — four times chance**, not to the floor. So the
representation has *not* become phon-only. The cost is also **not monotone**: at
β=0.10 it goes 0.819 → 0.731 → 0.744, so it is essentially a one-step price paid
between W=1 and W=3 and flat thereafter, while the benefit keeps accruing.

Both directions had to be in the same table. Distinctness bought by making the
representation phon-only would score beautifully on every other column here, and
this project has shipped that shape before.

## One column that is NOT like-for-like

`hi_ov` (mean overlap among pairs with α_eff ≥ 0.35) reads 0.667 → 0.375 → 0.362
at β=0.10. **Do not read that as a 2× improvement.** α_eff has W in its
denominator, so raising W moves pairs *out of the bin*: the sample is 110 → 102
→ **7** pairs. It is measuring "how bad are the pairs still in the danger zone",
over a shrinking and differently-composed set.

The W-independent columns — distinct, dup, spread, retP, retA, GROUND — are the
ones that carry the result.

## What this settles

The drive share **is** the lever. The collision was not a capacity limit, not a
separation limit, and not a plasticity limit: it was word identity being
outvoted by however many grounding features a word happened to have.

β and W are complementary and roughly additive — β acts on the graded
near-collision body, W acts on the duplicate tail — which is what the two-phase
design assumed and now confirms.

## What I have not done

Neither default is flipped. `phon_weight = 1.0` and `beta = 0.1` remain
production; both changes are opt-in parameters. Flipping them moves goldens, ERP
thresholds and parity tests, and that is a deliberate decision with a
re-baselining cost attached, not an experimental side effect.

The paper-faithful version — pooling grounding into a bounded number of
area-sized inputs rather than re-weighting one stimulus — is still untested. The
re-weighting result makes it worth doing, since it says the mechanism is real;
it does not say the re-weighting is the best implementation of it.

## Limits

2 seeds, no confidence intervals in the table above (Phase A's paired analysis
is the model for that and was not repeated here). `GROUND` drives
grounding→core directly rather than through a live parse, so it is an upper
bound on what generalisation actually sees. W was sampled at 1/3/6 only, so the
interior optimum between 3 and 6 is not resolved — and the two are within
0.02 on retrieval and 0.01 on GROUND at β=0.05, so 3 may be the better buy.
