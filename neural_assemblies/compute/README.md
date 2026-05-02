# Compute

`neural_assemblies.compute` holds the reusable math behind the runtime:
sampling, aggregation, competition, and Hebbian updates.

## Objects

| Object | File | Use it for |
|--------|------|------------|
| `StatisticalEngine` | `statistics.py` | Sampling and statistical helpers. |
| `NeuralComputationEngine` | `neural_computation.py` | Input aggregation and activation math. |
| `WinnerSelector` | `winner_selection.py` | Winner selection and remapping. |
| `TopKPolicy` | `winner_policies.py` | Fixed-size competition. |
| `ThresholdPolicy` | `winner_policies.py` | Absolute-threshold competition with a `k` cap. |
| `RelativeThresholdPolicy` | `winner_policies.py` | Competition based on a fraction of the strongest input. |
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

Winner policies give new competition rules a clear API. They do not mean every
engine already runs every policy by default.

## See Also

- [Core](../core/README.md)
- [API guide](../../docs/api.md)
