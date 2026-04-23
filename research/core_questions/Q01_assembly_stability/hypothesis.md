# Q01 Hypothesis: Assembly Stability

## Central Question

Under the recurrent stim+self training protocol used in this repo, do sparse
assemblies settle into stable autonomous patterns that remain well above chance
after the external stimulus is removed?

## Working Hypothesis

In the tested explicit-area regimes, Hebbian strengthening of both the
stimulus-to-area pathway and the area self-connectome creates assemblies that
persist under autonomous recurrence. The strength of that persistence depends on
training exposure, sparsity, and whether the question is about a single
assembly or a competitive multi-assembly setting.

## Scope

- This is a question about the current implementation and its tested regimes.
- It is not a universal theorem covering every Assembly Calculus parameter
  regime.
- It does not assume multi-assembly competition has already been solved.

## Predictions

1. Stim+self training should outperform stim-only training on autonomous
   persistence.
2. Persistence should remain significantly above the random-overlap null
   in the tested recurrent settings.
3. Heavy single-assembly training should yield stronger persistence than the
   lighter fixed-training projection regime.
