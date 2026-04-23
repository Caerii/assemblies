# Results

## Projection Regime

From `RESULTS_projection.md`:

- At `n=1000, k=100`, stim+self training reached persistence
  `0.709 +/- 0.032`.
- The stim-only control stayed near chance at `0.113 +/- 0.007`
  for null `k/n = 0.100`.
- Cross-area recovery after learned A->B projection was `1.000 +/- 0.000`
  at all tested sizes (`n = 500, 1000, 2000`).

Interpretation: self-connectome training matters. The stimulus path alone is
not enough to maintain an attractor once the stimulus is removed.

## Single-Assembly Attractor Regime

From `RESULTS_attractor_dynamics.md`:

- With heavier single-assembly training, persistence rises to the
  `0.97 - 1.00` range in many tested conditions.
- The transition to near-perfect persistence tracks `beta * train_rounds`.
- In the isolated single-assembly setting, even severe perturbations can be
  recovered after autonomous recurrence.

## Combined Reading

These two result sets are not contradictory:

- `test_projection.py` is a lighter, fixed-training projection study with
  corrected indexing and more moderate persistence.
- `test_attractor_dynamics.py` is a stronger single-assembly attractor study
  with heavier training and more favorable persistence.
