# The margin lever only pays below the margin (#52 recipe propagation)

**Registered:** `research/experiments/role_recipe_2x2.py` (01f8b9e history:
registration 00cfc8c, sequential extension 01f8b9e)
**Artifacts:** `role_recipe_2x2_results.json` (30 + 10 cells),
`role_recipe_2x2.log`, `role_recipe_2x2_ext.log`
**Figures:** `research/figures/fig_52_{interaction,regressions,exposure}.{png,pdf}`

## Question

The #151 morphology arc closed with `morph_beta_gain=4` as the at-scale
recipe: at kp = 1.5 the per-episode Hebbian boost is too small to clear
the extreme-value margin, and gain 4 buys the crossover. Role binding
trains through the SAME operating point (k=30, p=0.05) with T=2 rounds.
Does the recipe's gain axis propagate to roles?

The k*p-law arithmetic that motivated the 2x2: per-episode boost
(1.05)^2 = 1.10 (gain 1) vs (1.2)^2 = 1.44 (gain 4), against a demand of
sqrt(2 ln(n/k)/kp) = 2.48 at n = 3e3 and 3.29 at n = 1e5; compounding
crossovers near E~9/12 (gain 1) vs E~3/4 (gain 4).

## Verdict: NO. Gain stays 1.0 for roles — per the pre-stated rule.

The decision rule was: "role_bind_gain=4 becomes the documented at-scale
recipe for roles ONLY if the P-SCALE interaction shows with nothing
regressing." It does not show. All arms, 10 seeds each, mean ± CI95:

| arm            | parse         | retrieval        | occupant gap    |
|----------------|---------------|------------------|-----------------|
| n=3e3,  gain 1 | 1.000 ± 0.000 | 0.990 ± 0.007    | 0.979 ± 0.008   |
| n=3e3,  gain 4 | 1.000 ± 0.000 | 0.933 ± 0.011    | 0.895 ± 0.033   |
| n=1e5,  gain 1 | 1.000 ± 0.000 | **1.000 ± 0.000**| 0.990 ± 0.007   |
| n=1e5,  gain 4 | 1.000 ± 0.000 | 0.993 ± 0.008    | 0.983 ± 0.013   |

Paired (gain4 − gain1) retrieval delta: **−0.057 ± 0.012** at n=3e3,
**−0.007 ± 0.008** at n=1e5 (extended to 10 seeds per the pre-stated
sequential plan; the 5-seed read was −0.006 ± 0.015, within CI of zero).

## Predictions, scored

- **P-BASE: PASS.** Gain 1 at n=3e3 reproduces the standing exam (parse
  1.000 both voices, every seed).
- **P-GAIN@3e3: PASS on its registered claim** — parse accuracy is
  completely gain-insensitive (1.000 in all 20 cells; the exam words are
  far above any margin). The allowed side-channel ("retrieval may move
  in its low-exposure tail") happened, but INVERTED: retrieval fell, and
  not in the low-exposure tail (below).
- **P-SCALE: the ordering held, the premise failed.** The delta is
  indeed less negative at n=1e5 (−0.007 vs −0.057), but not through the
  registered mechanism. The premise was that gain-1's low-exposure tail
  would FALL at n=1e5 (margin demand 2.48 → 3.29) and gain would restore
  it. Instead gain 1 reads a flat 1.000 ± 0.000 at n=1e5: the ~33x
  collision-load reduction dominates the margin-demand rise. With the
  baseline at ceiling there is nothing for gain to rescue — the
  interaction shrank because the gain-4 PENALTY shrank, not because a
  gain-4 benefit appeared.

## The mechanism is frequency swamping, not margin

Where the gain-4 errors live at n=3e3 (final data, 10 seeds; fig_52_exposure):

- Errors concentrate at the TOP of the exposure distribution, not the
  bottom: words at exposure 12–18 drop to 0.55–0.68 under gain 4 while
  exposure-2 words stay ~0.98.
- The damage is ROLE_AGENT-specific: `child` (E=18) fails 9/10 seeds,
  `mom` (E=12) 9/10, `man` (E=12) 7/10 — the animate nouns that dominate
  agent slots. ROLE_AGENT draws from a small animate pool, so it is
  exactly where per-word Hebbian mass concentrates; gain 4 multiplies
  that mass until high-frequency assemblies absorb their neighbors.
  This is the frequency-swamping law from the morphology arc
  (E1–E4) reappearing in role space.
- A within-word refutation of the margin account: `father` (E=4) is the
  one word plausibly UNDER the margin — the account says gain should
  rescue it. Gain deepens its failure, 4/10 → 9/10 seeds.

So the same lever is margin-buying in one regime and crowding in
another, and the regime variable is the target area's mass
concentration, not n or kp alone: morphology's value areas needed the
boost because label-free feature evidence is DIFFUSE; ROLE_AGENT does
not because agent evidence is CONCENTRATED on few, frequent words.
`k*p decides whether beta helps` gets a sharper scoping: at fixed kp,
the SIGN of gain's effect follows the target's exposure concentration.

## Scale is free (the second finding)

Gain aside: role retrieval at n=1e5 is PERFECT (1.000 ± 0.000, 10
seeds, 72 words/seed) with parse at ceiling — role binding gets
BETTER with scale at fixed k, because collision load falls faster than
the margin demand rises. Combined with #151's morphology result, every
measured level of the pipeline now improves or holds under n-scaling on
the recipe. No knob needs retuning per scale on this axis.

## Infrastructure this unit forced (and what it caught)

`role_bind_gain` had to be a parser property writing the ENGINE's
per-fiber beta store, because `role_lexicons` has five writers and the
curriculum path does not call `train_roles` — a bracket there would
have been a dormant selector on the exact path every standing exam
uses. The liveness test (`test_role_bind_gain.py`, trains through
`CurriculumTrainer`) then caught `_set_global_beta` erasing per-fiber
overlays at every stage boundary: two writers of one store, last-writer
wins. `EmergentParser.set_base_beta` is now THE ONE writer of the
global plasticity rate (base write + overlay re-application); the
trainer delegates. The no-gain path is byte-identical to the pre-change
engine over a full curriculum train; full suite 1351/0.

## Production recipe (roles addendum)

- `role_bind_gain = 1.0` (measured: gain crowds at small n, buys
  nothing at 1e5; adopt-only-if bar not met)
- everything else per `production_configuration.md`; the gain REMAINS
  correct for morphology value areas (#151), where evidence is diffuse.
