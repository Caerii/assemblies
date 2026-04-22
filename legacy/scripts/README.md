# Archived Scripts

This directory contains historical scripts that used to live at the repository
root under `scripts/`.

## Layout

- `simulations/`
  Standalone overlap, Turing-style, and toy projection experiments from the
  pre-package layout.
- `tooling/`
  Historical build helpers that predate the current `cpp/` and test-based CUDA
  workflow.
- `visualization/`
  Visualization helpers retained for the archived image-learning stack.

## Current equivalents

- Prefer `neural_assemblies.simulation` for supported simulation runners.
- Prefer `benchmarks/` and `tests/performance/` for active profiling and CUDA
  checks.
- `legacy.root_modules.image_learner` still depends on
  `legacy.scripts.visualization.animator`.
