# Experiments

## Primary Experiment

`research/experiments/stability/test_scaling_laws.py`

Protocol:

1. establish an assembly with stimulus drive
2. train with stim+self recurrence up to convergence or a training ceiling
3. remove the stimulus
4. measure autonomous persistence against the trained assembly

## Primary Artifacts

- `research/results/stability/RESULTS_scaling_laws.md`
- `research/results/stability/scaling_laws_20260206_155822.json`

## Related Context

- `research/results/stability/RESULTS_phase_diagram.md`
- `research/results/stability/RESULTS_attractor_dynamics.md`

Those related artifacts help interpret why persistence does not simply grow
monotonically with `n`.
