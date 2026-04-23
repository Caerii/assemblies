# Compute

The `neural_assemblies.compute` package contains reusable mathematical
primitives used by the runtime and higher-level experiments.

## Main Components

| Component | File | Role |
|-----------|------|------|
| `StatisticalEngine` | `statistics.py` | Sampling and statistical helpers. |
| `NeuralComputationEngine` | `neural_computation.py` | Input aggregation and activation math. |
| `WinnerSelector` | `winner_selection.py` | Winner selection and remapping logic. |
| `TopKPolicy` | `winner_policies.py` | Fixed-size competition rule. |
| `ThresholdPolicy` | `winner_policies.py` | Absolute-threshold competition rule. |
| `RelativeThresholdPolicy` | `winner_policies.py` | Variable-size competition rule based on relative input strength. |
| `PlasticityEngine` | `plasticity.py` | Hebbian update logic. |
| `SparseSimulationEngine` | `sparse_simulation.py` | Sparse projection helpers. |
| `ExplicitProjectionEngine` | `explicit_projection.py` | Dense projection helpers. |

## Example

```python
import numpy as np

from neural_assemblies.compute import TopKPolicy, WinnerSelector

selector = WinnerSelector(np.random.default_rng(0))
winners = selector.select_with_policy([0.1, 0.9, 0.5, 0.9], TopKPolicy(k=2))
print(winners)
```

The policy layer is the current extension seam for richer competition rules.
It does not mean every engine path already uses every policy by default.

## See Also

- [../core/README.md](../core/README.md)
- [../../docs/api.md](../../docs/api.md)
