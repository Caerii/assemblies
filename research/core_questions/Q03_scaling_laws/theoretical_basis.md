# Theoretical Basis

Assembly Calculus theory often studies sparse assemblies with `k` much smaller
than `n`, and several repo experiments use `k = sqrt(n)` as a convenient sparse
regime.

This question asks for a scaling law, but the repo currently has:

- empirical measurements of convergence and persistence
- no finished derivation explaining the observed exponent or the `n=2000`
  outlier behavior

So the theoretical status is:

- Assembly Calculus motivates why scaling questions are natural.
- The repo has an empirical trend.
- The actual derivation remains open and belongs with Q02 / theory work.
