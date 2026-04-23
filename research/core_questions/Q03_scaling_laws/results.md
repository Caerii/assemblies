# Results

From `RESULTS_scaling_laws.md`:

- Tested sizes: `n = 100, 200, 500, 1000, 2000, 5000`
- Sparsity schedule: `k = floor(sqrt(n))`
- Persistence stayed in a moderate band, roughly `0.37 - 0.66`
- Convergence time trended downward with `n`

Empirical fit:

- `T = -17.49 * log10(n) + 78.68`
- `R^2 = 0.601`
- `p = 0.070`

Important nuance:

- The fit is suggestive rather than cleanly decisive.
- The `n=2000` condition is an outlier and weakens the regression.
- Persistence does not improve monotonically with `n` because `k/n`
  simultaneously becomes sparser.
