# Noise robustness

Script: `research/experiments/stability/test_noise_robustness.py`

Result artifact: `noise_robustness_20260206_155522.json`

Date: 2026-02-06

Implementation note: this result used the historical `src.core.brain.Brain`
path with weight saturation, corrected winner remapping, and stimulus
plasticity.

## Question

How well can trained assemblies recover after winner corruption?

## Protocol

The experiment corrupts a fraction of an assembly's winners by replacing them
with random non-winners, then tests three recovery paths:

1. stimulus-driven recovery
2. autonomous self-projection
3. association-based recovery from a clean source assembly

All assemblies are trained with 30 rounds of stim+self projection. Association
uses 30 rounds of co-stimulation.

## Dense setting: `n=1000`, `k=100`

At `k/n=0.10`, stimulus-driven recovery returns `1.000 +/- 0.000` at every
tested noise level, including full winner replacement. This mainly shows that
the learned stimulus pathway dominates the corrupted state when the stimulus is
available.

Autonomous self-projection also returns about `1.000` at every tested noise
level. The interpretation is more subtle: at this density, even full random
replacement is likely to include enough trained neurons by chance to restart
the trained self-connectome cascade.

Association-based recovery from a clean source assembly gives about
`0.984-0.997` recovery across noise levels. This shows that the learned
cross-area pathway can reconstruct the target when the source remains intact.

## Sparse scaling: `k = sqrt(n)`

| n | k | k/n | noise 0.3 | noise 0.5 | noise 0.7 | noise 1.0 |
|---|---|-----|-----------|-----------|-----------|-----------|
| 200 | 14 | 0.070 | 0.800 | 0.664 | 0.536 | 0.529 |
| 500 | 22 | 0.044 | 0.923 | 0.914 | 0.855 | 0.877 |
| 1000 | 31 | 0.031 | 0.945 | 0.971 | 0.945 | 0.958 |
| 2000 | 44 | 0.022 | 0.986 | 0.998 | 0.993 | 0.995 |

The smaller network is vulnerable under severe corruption. Recovery improves
strongly by `n=500` and is near ceiling by `n=2000` in this protocol.

## Interpretation

The dense result is a ceiling-effect result, not a final noise-robustness
claim. It shows that the training protocol creates strong recovery paths, but
the dense setting is too easy to separate mechanisms.

The sparse sweep is more informative. It shows where corruption begins to
matter and gives a bounded empirical transition region for this protocol.

## Limits

- Stimulus-driven recovery is not evidence for autonomous attractor recovery.
- Dense autonomous recovery is partly explained by chance overlap with the
  trained assembly.
- The sparse sweep is small and needs denser parameter coverage.
- The result does not yet justify a broad claim that assemblies are
  error-correcting codes in general.

## Relationship to attractor dynamics

The attractor-dynamics result also found high recovery at `n=1000`, `k=100`.
This experiment adds stimulus-driven recovery, association-based recovery, and
a sparse scaling sweep that exposes a more meaningful transition.
