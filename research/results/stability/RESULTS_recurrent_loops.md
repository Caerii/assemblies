# Recurrent loop dynamics

Script: `research/experiments/stability/test_recurrent_loops.py`

Result artifact: `recurrent_loops_20260206_171955.json`

Date: 2026-02-06

Implementation note: this result used the historical `src.core.brain.Brain`
path with weight saturation, corrected winner remapping, and stimulus
plasticity.

## Question

Can a trained multi-area loop maintain assemblies after the initial stimulus is
removed?

## Protocol

The experiment trains loops of areas `X0 -> X1 -> ... -> X(N-1) -> X0`.

Training has four phases:

1. Train each stimulus-to-area pathway for 30 rounds.
2. Train each adjacent loop edge by co-stimulation for 30 rounds.
3. Kick-start the loop by stimulating `X0` while running the loop projections.
4. Remove all stimuli and run the loop autonomously.

Main parameters:

- `n=1000`
- `k=100`
- `p=0.05`
- `beta=0.10`
- `w_max=20.0`
- `establish_rounds=30`
- `assoc_rounds=30`
- `kick_rounds=15`
- `test_rounds=30`
- `N_SEEDS=10`

The null overlap is `k/n = 0.100`.

## Main results

### Loop size at dense sparsity

| loop size | post-kick | final at t=30 | SEM | Cohen's d |
|-----------|-----------|---------------|-----|-----------|
| 3 areas | 0.997 | 0.995 | 0.001 | 198.8 |
| 4 areas | 0.994 | 0.991 | 0.002 | 182.3 |
| 5 areas | 0.995 | 0.994 | 0.001 | 197.7 |
| 6 areas | 0.995 | 0.994 | 0.001 | 242.3 |

In this regime, 3- to 6-area loops preserve high overlap for 30 autonomous
rounds. The experiment does not show that loop size is irrelevant in general;
it shows that these loop sizes do not separate at these parameters.

### Fifty autonomous rounds

For the 3-area loop, mean overlap stayed at `0.991 +/- 0.002` from round 1
through round 50. The exponential decay fit reported slope `0.000000` over the
measured window.

The careful claim is: no measurable decay was observed over 50 rounds in this
tested regime.

### Scaling with `k = sqrt(n)`

| n | k | k^2 p | post-kick | final autonomous | decay |
|---|---|-------|-----------|------------------|-------|
| 200 | 14 | ~10 | 0.590 | 0.326 | -0.264 |
| 500 | 22 | ~24 | 0.723 | 0.488 | -0.235 |
| 1000 | 31 | ~48 | 0.833 | 0.587 | -0.246 |
| 2000 | 44 | ~97 | 0.922 | 0.861 | -0.061 |
| 5000 | 70 | ~245 | 0.980 | 0.967 | -0.013 |

Autonomous persistence improves as `k^2 p` rises. The transition appears
between the weaker regimes below `k^2 p ~ 100` and the stronger regimes above
that range, but this result should be treated as empirical rather than a
derived threshold.

## Interpretation

The trained loop can maintain a high-overlap state after stimulus removal when
the per-hop signal is strong enough. The likely mechanism is straightforward:
trained cross-area weights favor the next area's trained assembly, so each hop
reconstructs the next state and feeds the cycle.

This supports the claim that the implementation can express stable autonomous
multi-area recurrence in the tested regimes.

## Limits

- The result covers finite test windows, not indefinite persistence.
- The largest loop tested here has 6 areas.
- The result uses a particular training protocol and weight cap.
- The biological interpretation is suggestive, not a direct model of working
  memory.
- The threshold language needs a derivation or denser parameter sweep before it
  can become a stronger theory claim.

## Relationship to other results

| Experiment | Topology | Stimulus during test | Representative result |
|------------|----------|----------------------|-----------------------|
| Association | `A -> B` | present | about `0.994` recovery |
| Association chain | feedforward chain | present at source | about `0.993` at 5 hops |
| Attractor dynamics | `A -> A` | removed | about `0.990` at t=50 |
| Recurrent loops | closed multi-area loop | removed | about `0.991` at t=50 |

The recurrent-loop result sits between single-area recurrence and feedforward
chains: it tests whether trained cross-area recurrence can preserve a state
without continued stimulus support.
