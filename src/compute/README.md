# Compute — Math and primitives

The **compute** module holds the mathematical primitives used by the core engines: statistics (e.g. Bernoulli sampling), plasticity (Hebbian), winner selection (top-k), and projection logic (sparse and explicit).

## What’s here

| Component | File | Role |
|-----------|------|------|
| **StatisticalEngine** | `statistics.py` | Sampling (e.g. binomial), normalization |
| **NeuralComputationEngine** | `neural_computation.py` | Input aggregation, activation |
| **WinnerSelector** | `winner_selection.py` | Top-k selection (winners per area) |
| **PlasticityEngine** | `plasticity.py` | Hebbian weight updates (co-activity → strengthen) |
| **SparseSimulationEngine** | `sparse_simulation.py` | Sparse projection path (used by numpy_sparse) |
| **ExplicitProjectionEngine** | `explicit_projection.py` | Dense matrix projection (used by numpy_explicit) |
| **ImageActivationEngine** | `image_activation.py` | Image → assembly activation for vision experiments |

You typically don’t call these directly; `Brain` and the engines in `src/core` use them. Use this module when you need to reuse or extend the same math (e.g. custom engine or analysis).

## Quick use

```python
from src.compute import WinnerSelector, PlasticityEngine, StatisticalEngine

# Example: top-k from a score vector
selector = WinnerSelector()
winners = selector.select(inputs, k=100)

# Hebbian update (conceptually)
plasticity = PlasticityEngine()
plasticity.update(connectome, pre_winners, post_winners, beta=0.05)
```

## See also

- [src/core](core/README.md) — Brain and engines that use these primitives.
- [ARCHITECTURE.md](../../ARCHITECTURE.md) — Where compute fits in the stack.
