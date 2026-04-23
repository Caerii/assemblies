# Experiments

## Primary Experiments

1. `research/experiments/primitives/test_projection.py`
   - compares stim+self against stim-only training
   - measures autonomous persistence and cross-area recovery

2. `research/experiments/stability/test_attractor_dynamics.py`
   - varies training exposure, noise level, network size, and `w_max`
   - focuses on single-assembly autonomous recurrence

## Primary Artifacts

- `research/results/primitives/RESULTS_projection.md`
- `research/results/primitives/projection_20260206_144109.json`
- `research/results/stability/RESULTS_attractor_dynamics.md`
- `research/results/stability/attractor_dynamics_20260206_121741.json`

## Measured Quantities

- persistence under autonomous recurrence
- recovery vs random-overlap null
- cross-area recovery fidelity
- dependence on `beta * train_rounds`, `w_max`, and network size
