# Multi-pattern loop switching

Script: `research/experiments/stability/test_loop_switching.py`

Result artifact: `loop_switching_20260206_191220.json`

Date: 2026-02-06

Implementation note: this result used the historical `src.core.brain.Brain`
path with weight saturation, corrected winner remapping, stimulus plasticity,
and an optional homeostatic-normalization variant.

## Question

Can the same trained recurrent loop maintain and switch between multiple
patterns?

## Protocol

Each pattern has assemblies in three loop areas. The experiment trains loop
edges for one or more patterns, kick-starts a pattern, removes the stimulus,
and measures autonomous circulation. In the switching condition, it then
kick-starts another pattern and measures overlap with both patterns.

Main parameters:

- `n=1000`
- `k=100`
- `p=0.05`
- `beta=0.10`
- `w_max=20.0`
- `establish_rounds=30`
- `assoc_rounds=30`
- `kick_rounds=15`
- `autonomous_rounds=20`
- `N_SEEDS=10`

## Main results

### Single-pattern baseline

| condition | final overlap | SEM | Cohen's d |
|-----------|---------------|-----|-----------|
| one pattern | 0.996 | 0.001 | 398.1 |

A single trained pattern circulates with high overlap under these parameters.

### Two-pattern switching

| phase | overlap with A | overlap with B |
|-------|----------------|----------------|
| after kick A and autonomous run | 0.874 | not measured |
| after kick B and autonomous run | 0.708 | 0.376 |

Training a second pattern already degrades pattern A. After the second kick,
the loop does not cleanly replace A with B. The state remains a mixture:
pattern A stays partially active while pattern B rises above chance but far
below the single-pattern baseline.

### Pattern capacity

| trained patterns | mean switching quality | SEM | Cohen's d |
|------------------|------------------------|-----|-----------|
| 1 | 0.994 | 0.002 | 138.8 |
| 2 | 0.606 | 0.012 | 13.0 |
| 3 | 0.458 | 0.012 | 9.1 |
| 5 | 0.332 | 0.012 | 6.1 |

The shared loop degrades sharply as more patterns are trained. At these
parameters, it behaves like a good single-pattern maintenance circuit and a
poor multi-pattern switching circuit.

### Homeostatic normalization

The tested Turrigiano-style normalization reduced some training interference
but made switching worse. In this setup, preserving each source neuron's
outgoing weight budget made the active pattern harder to replace.

## Interpretation

The failure mode is pattern blending. Multiple trained patterns share the same
areas and cross-area connectomes. Without a clean external anchor during the
autonomous phase, the loop feeds its own mixed state forward and reinforces the
mixture.

The result suggests that multi-pattern switching needs an additional mechanism,
such as active inhibition, gating, context separation, or independent loops.

## Limits

- This result does not prove a general working-memory capacity limit.
- The "single-item" interpretation is about this shared-loop protocol.
- Other architectures might support multi-item maintenance through separated
  loops, area partitions, context gates, or stronger inhibition.
- The biological discussion should be treated as motivation for follow-up
  experiments, not as direct neural validation.

## Relationship to other results

| Experiment | Substrate | Items | Pattern |
|------------|-----------|-------|---------|
| Association interference | shared `A -> B` connectome | 1-10 pairs | graceful degradation with stimulus support |
| Loop switching | shared 3-area loop | 1-5 patterns | steep degradation without stimulus support |
| Recurrent loops | shared 3-area loop | 1 pattern | high-overlap autonomous maintenance |

The key variable is not just storage load. It is whether a stimulus or other
external signal anchors the target state during recall.
