# Core — Brain, areas, engines

The **core** module implements the main assembly-calculus runtime: `Brain`, areas, stimuli, connectomes, and the **ComputeEngine** abstraction so you can run the same logic on CPU or GPU.

## What’s here

| Component | File | Role |
|-----------|------|------|
| **Brain** | `brain.py` | Top-level orchestrator: areas, stimuli, `project()`, Hebbian, recording |
| **Area** | `area.py` | One brain area: `n` neurons, `k` winners, optional self-connectome |
| **Stimulus** | `stimulus.py` | External input (fixed set of active neurons) |
| **Connectome** | `connectome.py` | Sparse connectivity and weights between areas / stimulus → area |
| **ComputeEngine** | `engine.py` | ABC and registry; `project_into()`, Hebbian, plasticity |
| **Engines** | `numpy_engine.py`, `cuda_engine.py` | `numpy_sparse`, `numpy_explicit`, `cuda_implicit` |
| **Backend** | `backend.py` | NumPy/CuPy switching (`get_xp()`, `set_backend()`) |
| **Kernels** | `kernels/` | CUDA RawKernels for projection, Hebbian, batched ops |

## Quick use

```python
from src.core.brain import Brain

b = Brain(p=0.05, engine="auto")  # or "numpy_sparse" / "cuda_implicit"
b.add_stimulus("stim", size=100)
b.add_area("A", n=10000, k=100, beta=0.05)
b.project({"stim": ["A"]}, {})
for _ in range(9):
    b.project({}, {"A": ["A"]})
```

## Engines

- **`numpy_sparse`** — CPU; sparse / statistical; scales to large `n`. Default when no GPU.
- **`numpy_explicit`** — CPU; explicit matrices; faithful for small `n`.
- **`cuda_implicit`** — GPU; hash-based connectivity; ~40× speedup at large `n` (when CuPy is available).

`Brain(..., engine="auto")` picks `cuda_implicit` if CuPy is present, else `numpy_sparse`.

## See also

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — Brain vs NEMO, engine dispatch, GPU layout.
- [DOCUMENTATION.md](../../DOCUMENTATION.md) — Full module and API guide.
