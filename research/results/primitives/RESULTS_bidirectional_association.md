# Bidirectional association

Script: `research/experiments/primitives/test_bidirectional_association.py`

Result artifact: `bidirectional_association_20260206_173748.json`

Date: 2026-02-06

Implementation note: this result used the historical `src.core.brain.Brain`
path with weight saturation, corrected winner remapping, and stimulus
plasticity.

## Question

Does training an `A -> B` association also create useful `B -> A` recall, or
does reverse recall require explicit reverse training?

## Protocol

The experiment trains assemblies in areas A and B, then compares three
association protocols:

- unidirectional `A -> B`
- simultaneous bidirectional `A -> B` and `B -> A`
- sequential bidirectional training

Forward recall activates A and projects to B. Reverse recall activates B and
projects to A.

Main parameters:

- `n=1000`
- `k=100`
- `p=0.05`
- `beta=0.10`
- `w_max=20.0`
- `establish_rounds=30`
- `assoc_rounds=30`
- `test_rounds=15`
- `N_SEEDS=10`

The null overlap is `k/n = 0.100`.

## Main results

### Unidirectional training

| direction | recovery | SEM | Cohen's d |
|-----------|----------|-----|-----------|
| forward `A -> B` | 0.992 | 0.002 | 141.0 |
| reverse `B -> A` | 0.097 | 0.009 | -0.1 |

Training only `A -> B` gives strong forward recall and chance-level reverse
recall. Co-activation alone does not train the unspecified reverse fiber.

### Explicit bidirectional training

| training method | forward `A -> B` | reverse `B -> A` |
|-----------------|------------------|------------------|
| simultaneous | 0.991 | 0.992 |
| sequential | 0.998 | 0.995 |

Both bidirectional protocols produce strong recall in both directions.

### Forward recall with and without reverse training

| condition | forward recovery | SEM |
|-----------|------------------|-----|
| `A -> B` only | 0.992 | 0.002 |
| simultaneous bidirectional | 0.991 | 0.003 |

The paired test reports no significant degradation of forward recall when the
reverse direction is also trained (`p=0.758`).

### Sparse scaling

| n | k | k^2 p | forward | reverse |
|---|---|-------|---------|---------|
| 500 | 22 | 24 | 0.727 | 0.709 |
| 1000 | 31 | 48 | 0.752 | 0.800 |
| 2000 | 44 | 97 | 0.900 | 0.891 |
| 5000 | 70 | 245 | 0.970 | 0.967 |

Forward and reverse recall track each other closely when both directions are
trained.

## Interpretation

Association is directional in this implementation because plasticity applies
along the projection fibers named in the projection map. If the map trains
`A -> B`, the `B -> A` fiber is not strengthened merely because A and B are
co-active.

The result supports a bounded engineering claim: symmetric recall requires
explicit bidirectional training or an implementation that couples the two
directions by design.

## Limits

- The result is about this projection-map and plasticity implementation.
- It does not claim that biological reciprocal connectivity works exactly this
  way.
- The hippocampal analogy is motivation for future experiments, not direct
  validation.

## Relationship to other results

| Experiment | Tests | Representative finding |
|------------|-------|------------------------|
| Association | one-hop `A -> B` recovery | about `0.994` |
| Bidirectional association | forward and reverse recall | reverse is chance without explicit reverse training |
| Association chain | multi-hop feedforward recall | each hop is trained directionally |
| Recurrent loops | closed directional loop | each edge is a trained one-way association |
