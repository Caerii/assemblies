# Registration: refraction in the vendored reference arc, twenty seeds

> **Status (2026-09-12): registered, not yet run.** The bars below were
> fixed before the twenty-seed run. The only prior data are the three-seed
> legacy log (`research/results/logs/seq_arc_refraction_reference.log`,
> seeds 42, 7, 13) whose numbers the register entries `ARC-CONJUNCT-EXPOSURE`
> and `REFRACTION-PROPORTIONAL` currently cite. That log has no run record,
> no seed identities in its artifact and no source archive; this
> registration exists to replace it with a retained runner artifact whose
> mechanism-disabled arms the register can check by seed.

## Why

Two register entries rest on one script and one log:

- `ARC-CONJUNCT-EXPOSURE` cites the ablation (bias never accumulates): the
  reference arc collapses onto the symbol-blind degeneracy and the task
  fails.
- `REFRACTION-PROPORTIONAL` cites the constant-versus-proportional sweep:
  a constant increment has an operating point at fifteen presentations that
  moves when the drive scale moves, while the drive-proportional rule does
  not.

Both are legitimate mechanism ablations, and both are held as prose. The
rule every adopted result meets is a retained treatment/control vector that
still moves. This run supplies it.

## What runs

`python -m research.runner arc-refraction-reference --tag UNIQUE`

- Engine `reference_nemo_numpy`, the vendored `reference/nemo_numpy` FSM
  network as-is (no `Brain`), with its declared model profile
  (`describe_nemo_numpy_reference`): dense stream-addressed Bernoulli
  matrices, all neurons candidates, float64, no normalization, unbounded
  multiplicative potentiation, the reference's own top-k.
- Sizes as in the legacy script: symbol 1000, state 500, arc 5000, cap 70,
  density 0.2, plasticity 0.1. Chance arc overlap k/n = 0.014.
- Seeds 42 to 61 (twenty). Each seed draws its own network and its own
  symbol and state assemblies.
- Arms, each trained from scratch per seed:
  - `proportional` at 5, 15 and 30 presentations (the reference rule:
    `bias[winner] += drive[winner] * plasticity`).
  - `off` at 15 presentations (the ablation: `FFArea.update` in place of
    `RefractedArea.update`, so the bias stays at zero; manipulation-checked).
  - `constant` at strengths 1, 3, 10, 30 and 100, each at 5, 15 and 30
    presentations (this repository's engine rule at the time:
    `bias[winner] += strength`).
- Per seed and arm the run retains: across-state arc overlap (mean pairwise
  overlap of the arc for one symbol across the three states), across-symbol
  arc overlap (mean pairwise overlap across the ten digits at one state),
  and whether the machine decides both strings (positive accepted, negative
  rejected) as an integer.

Smoke (`--smoke --seeds 42 43 44`) runs two presentations on three arms and
is VOID.

## Bars

Written against the three-seed priors (proportional 3/3 with overlaps
0.000/0.000; off 0/3 with across-symbol 0.989; constant 10 alone decides at
15 presentations and fails at 30, where constant 30 decides).

- **AR-1, golden.** `proportional` at 15 presentations decides both strings
  on at least 18 of 20 seeds, and its mean across-state and across-symbol
  overlaps are each at most 0.05.
- **AR-2, ablation (for `ARC-CONJUNCT-EXPOSURE`).** `off` at 15
  presentations has across-symbol overlap at least 0.5 on every seed and
  decides on at most 2 of 20; across-state overlap stays at most 0.2 in the
  mean. The retained sensitivity check is across-symbol overlap, `off`
  against `proportional`, all seeds greater by at least 0.5.
- **AR-3, a constant operating point exists.** At 15 presentations at least
  one constant strength decides on at least 18 of 20 seeds.
- **AR-4, the operating point moves (for `REFRACTION-PROPORTIONAL`).** Let
  s* be the constant strength with the most decided seeds at 15
  presentations. At 30 presentations s* decides on at most 2 of 20 seeds,
  while `proportional` decides on at least 18 of 20 at 30 presentations.
  The retained sensitivity check for the register is chosen after the data
  among the retained per-seed quantities at 30 presentations (decided,
  across-state, across-symbol), and must move on every seed; the choice is
  recorded here with the result. It is a sensitivity, not a bar.

A bar that fails is recorded with its numbers; the register entries are
then amended to say what the twenty-seed run supports, not left on the
three-seed log.

Not claimed: anything about this repository's own engines. The constant
rule is measured on the reference substrate to ask whether it is an
acceptable substitute; the engines' actual rule has since become
drive-proportional (`REFRACTION-PROPORTIONAL`, `implemented_by`).

## Smoke (2026-09-12, VOID)

API check only, three seeds, two presentations, three arms:
`research/results/runs/sequence.arc-refraction-reference/arc-ref-smoke-20260912/results.json`.
Its numbers are VOID by construction.

## Result (2026-09-12, twenty seeds, all four bars PASS)

Artifact: `research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json`
(engine `reference_nemo_numpy`, seeds 42 to 61, run from the pinned commit
e0d89e1a in a worktree reserved for studies; source archive and run record
retained). Chance arc overlap k/n = 0.014.

| arm | across-state | across-symbol | decided |
|---|---|---|---|
| proportional, 5 | 0.000 | 0.000 | 3/20 |
| proportional, 15 | 0.000 | 0.000 | 18/20 |
| proportional, 30 | 0.000 | 0.000 | 20/20 |
| off, 15 | 0.098 | 0.988 | 0/20 |
| constant 1 (5 / 15 / 30) | 0.042 / 0.034 / 0.034 | 0.945 / 0.963 / 0.966 | 0 / 0 / 0 |
| constant 3 | 0.102 / 0.113 / 0.110 | 0.643 / 0.873 / 0.880 | 0 / 0 / 0 |
| constant 10 | 0.116 / 0.001 / 0.131 | 0.046 / 0.000 / 0.369 | 0 / 20 / 0 |
| constant 30 | 0.167 / 0.104 / 0.013 | 0.050 / 0.070 / 0.006 | 0 / 0 / 20 |
| constant 100 | 0.167 / 0.410 / 0.419 | 0.050 / 0.407 / 0.237 | 2 / 0 / 0 |

- **AR-1 PASS.** Proportional at 15 decides 18 of 20 (at the bar), both
  overlaps 0.000. At 5 presentations it decides 3 of 20 (the three-seed log
  had 0 of 3); at 30 it decides all 20.
- **AR-2 PASS.** Refraction off: across-symbol overlap 0.977 to 0.997 on
  every seed (mean 0.988), across-state 0.098, decided 0 of 20. The arc
  collapses onto the symbol-blind degeneracy on every brain. Retained
  sensitivity for `ARC-CONJUNCT-EXPOSURE`: across-symbol overlap, `off-p15`
  against `proportional-p15`, every seed greater by at least 0.5 (the
  smallest retained effect is 0.977).
- **AR-3 PASS.** Constant 10 decides 20 of 20 at 15 presentations with
  overlaps 0.001 and 0.000: at that scale the constant increment matches the
  proportional rule's operating point.
- **AR-4 PASS.** s* = 10 decides 0 of 20 at 30 presentations (across-symbol
  0.369: the arc has begun to collapse onto the symbol-blind degeneracy as
  potentiation outruns the fixed increment), while proportional decides 20
  of 20. Constant 30 decides 0 of 20 at 15 and 20 of 20 at 30: the operating
  point moves with the drive scale, exactly as the three-seed log showed.
  Retained sensitivity for `REFRACTION-PROPORTIONAL`, chosen after the data
  among the retained per-seed quantities at 30 presentations: `decided`,
  `proportional-p30` against `constant-s10-p30`, every seed greater by 1
  (all twenty flip). The across-symbol contrast at 30 presentations also
  moves on every seed (0.000 against 0.369 in the mean) but with a smaller
  minimum effect; `decided` is the quantity the claim is about.

What the run does not say: nothing about this repository's engines, whose
refraction rule is now drive-proportional; and the constant rule's failure
at 30 presentations is a scale mismatch, not a proof that no constant
schedule could work (constant 30 does, at that presentation count).
