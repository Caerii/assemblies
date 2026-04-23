# Conclusions

## Defensible Conclusion

In the documented `k = sqrt(n)` regime, the repo has empirical evidence that
convergence time decreases with network size and is broadly consistent with a
logarithmic trend, while persistence remains moderate rather than monotonically
improving.

## Current Boundaries

- The evidence is empirical and somewhat noisy.
- The repo does not yet have a theory derivation for this scaling law.
- This should be framed as "consistent with O(log n) in the tested regime,"
  not as a universal law.

## Next Steps

- derive one theory result that predicts the observed trend
- connect Q03 more explicitly to the phase-boundary work in Q02
- rerun the scaling study with more seeds around the unstable regimes
