# Core

The `neural_assemblies.core` package implements the main runtime substrate:
brains, areas, stimuli, connectomes, and compute-engine dispatch.

## Main Components

| Component | File | Role |
|-----------|------|------|
| `Brain` | `brain.py` | Top-level orchestrator for areas, stimuli, routing, and projection cycles. |
| `Area` | `area.py` | Area parameters, winner history, and activation state. |
| `Stimulus` | `stimulus.py` | External fixed input representation. |
| `Connectome` | `connectome.py` | Connectivity and learned weights. |
| `ComputeEngine` | `engine.py` | Abstract engine interface used by `Brain`. |
| `backend.py` | `backend.py` | Array backend and auto-engine heuristic helpers. |

## Engine Names

Known engine names in the current codebase:

- `numpy_sparse`
- `numpy_explicit`
- `cuda_implicit`
- `cupy_sparse`
- `torch_sparse`

`engine="auto"` currently prefers `torch_sparse` only when `n_hint` is large
and PyTorch CUDA is available; otherwise it falls back to `numpy_sparse`.

## Example

```python
from neural_assemblies.core.brain import Brain

b = Brain(p=0.05, engine="numpy_sparse", seed=0)
b.add_stimulus("stim", size=100)
b.add_area("A", n=10_000, k=100, beta=0.05)
b.project({"stim": ["A"]}, {})
```

## See Also

- [../../docs/architecture.md](../../docs/architecture.md)
- [../../docs/api.md](../../docs/api.md)
