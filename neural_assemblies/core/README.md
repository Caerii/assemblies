# Core

`neural_assemblies.core` contains the runtime objects that every higher layer
builds on: brains, areas, stimuli, connectomes, and compute engines.

## Objects

| Object | File | Use it for |
|--------|------|------------|
| `Brain` | `brain.py` | Building areas, adding stimuli, routing projections, and choosing an engine. |
| `Area` | `area.py` | Area size, sparsity, plasticity, winners, and history. |
| `Stimulus` | `stimulus.py` | Fixed external inputs. |
| `Connectome` | `connectome.py` | Connectivity and learned weights. |
| `ComputeEngine` | `engine.py` | Engine interface behind `Brain.project(...)`. |
| `backend.py` | `backend.py` | Backend detection and engine selection helpers. |

## Engines

Known engine names:

- `numpy_sparse`
- `numpy_explicit`
- `cuda_implicit`
- `cupy_sparse`
- `torch_sparse`

`engine="auto"` chooses `torch_sparse` only when `n_hint` is large and PyTorch
CUDA is available. Otherwise it uses `numpy_sparse`.

## Example

```python
from neural_assemblies.core.brain import Brain

b = Brain(p=0.05, engine="numpy_sparse", seed=0)
b.add_stimulus("stim", size=100)
b.add_area("A", n=10_000, k=100, beta=0.05)
b.project({"stim": ["A"]}, {})
```

## See Also

- [Architecture](../../docs/architecture.md)
- [API guide](../../docs/api.md)
