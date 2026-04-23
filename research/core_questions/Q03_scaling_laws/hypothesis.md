# Q03 Hypothesis: Scaling Laws

## Central Question

How do convergence time and autonomous persistence change with network size in
the tested `k = sqrt(n)` regime?

## Working Hypothesis

In the current scaling experiment, convergence time should decrease with
network size in a way that is broadly consistent with logarithmic scaling,
while persistence should remain moderate because `k/n` shrinks as the network
grows.

## Scope

- This question is empirical first, theoretical second.
- The current evidence is about the documented `k = sqrt(n)` regime.
- It should not be generalized to all sparsity schedules or all attractor
  settings.

## Predictions

1. Convergence time should trend downward with increasing `n`.
2. Persistence should not simply increase with `n`; sparsity matters.
3. Scaling behavior should be interpretable together with the nearby phase
   boundary and attractor-basin results, not in isolation.
