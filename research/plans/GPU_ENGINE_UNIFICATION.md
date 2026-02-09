# GPU Engine Unification Plan

## Problem Statement

The codebase has three separate CUDA implementations that share the same
mathematical foundations (hash-based implicit connectivity, Hebbian
plasticity, top-k winner selection) but use incompatible APIs:

1. **`src/core/cuda_engine.py::CudaImplicitEngine`** (750 lines) — Wraps
   CUDA kernels into the `ComputeEngine` ABC. Used by `Brain` when
   `engine="cuda_implicit"`.

2. **`src/nemo/core/brain.py`** — Direct CUDA kernel calls with torch/cupy.
   Independent Brain class with per-area state management. Not a
   ComputeEngine subclass.

3. **`src/nemo/language/emergent/cuda_backend.py::CUDAProjector`** — C++ DLL
   wrapper with Python fallback. Used by the emergent learner subsystem.

Meanwhile, the CPU engine (`NumpySparseEngine`, 900+ lines) has the richest
feature set and passes 285+ tests.

## Capability Matrix

| Feature | NumpySparseEngine | CudaImplicitEngine | nemo/core/brain | nemo/emergent/cuda_backend |
|---------|:-:|:-:|:-:|:-:|
| **ComputeEngine ABC** | Yes | Yes | No | No |
| **Connectivity model** | Compact [0..w-1], explicit weights | Full [0..n-1], hash + COO | torch-based Area | DLL ctypes |
| **Memory scaling** | O(w^2) per pair | O(learned) per pair | O(n*k) per area | O(learned) |
| **LRI (refractory)** | Yes | **No** | No | No |
| **Refracted mode** | Yes | **No** | No | No |
| **Weight normalization** | Yes | **No** | No | No |
| **project_into** | Yes | Yes | Custom API | Custom API |
| **project_rounds** | Sequential fallback | GPU-resident tight loop | N/A | N/A |
| **project_into_batch** | Sequential fallback | 2-D grid batching | N/A | N/A |
| **fix_assembly** | Yes | Yes | Implicit | No |
| **reset_area_connections** | Yes (reset weights) | Yes (reset COO) | No | No |
| **Deterministic mode** | Yes (legacy RNG) | Yes (hash seed) | N/A | N/A |
| **Test coverage** | 285 tests | Basic registration | Separate suite | Separate suite |

## Key Architectural Differences

### 1. Neuron Addressing

- **NumpySparse**: Compact indices `[0, 1, ..., w-1]` where `w` = number of
  ever-fired neurons. New neurons are allocated from a pool on first firing.
  `get_num_ever_fired()` returns the actual `w`.

- **CudaImplicit**: All `n` neurons are directly addressable from the start.
  Hash function `fnv1a(seed, src, dst)` determines baseline connectivity
  on-the-fly. `get_num_ever_fired()` returns `n` (all addressable).

**Implication**: Assembly snapshots from NumpySparse use compact indices;
from CudaImplicit they use neuron IDs directly. Cross-engine comparison
requires index translation.

### 2. Connectivity Storage

- **NumpySparse**: Explicit 1-D (stimulus) and 2-D (area) weight arrays
  that grow as new neurons fire. Bernoulli(p) sampling at expansion time.
  Total memory grows as `O(sum of w_src * w_tgt)`.

- **CudaImplicit**: Baseline connectivity computed on-the-fly via hash.
  Only *learned deltas* stored in COO format per (src, tgt) pair:
  `(src_idx[], dst_idx[], delta[])`. Hash table for collision resolution.
  Total memory: `O(total learned connections * 12 bytes)`.

### 3. Hebbian Plasticity

- **NumpySparse**: `w[active_src, active_tgt] *= (1 + beta)`, clamped at
  `w_max`. Direct matrix mutation.

- **CudaImplicit**: `hebbian_update_kernel` inserts/updates entries in the
  COO hash table. New connections created on first co-activation.

**Compatibility**: Both implement `w *= (1 + beta)` semantics. The
difference is storage format, not learning rule.

## Minimum Viable Unification Plan

### Phase A: Bring CudaImplicitEngine to Feature Parity

**Goal**: Make CudaImplicitEngine pass the same test suite as NumpySparseEngine.

| Feature | Implementation | Effort |
|---------|---------------|--------|
| **LRI** | Add `_refractory_history` (circular buffer on GPU, length = refractory_period). In `project_into`, subtract decay-weighted penalty from activations for recently-fired neurons. | Medium |
| **Refracted mode** | Add `_cumulative_bias` array (float32, length n) on GPU. In `project_into`, subtract bias. After winner selection, increment bias for winners. | Low |
| **Weight normalization** | Add kernel: for each column, compute sum and divide. Or do on CPU via `cp.sum(axis=0)`. | Low |
| **clear_refractory** | Reset circular buffer to empty. | Trivial |
| **set_lri / set_refracted** | State management on `_CudaAreaState`. | Trivial |

**LRI kernel pseudocode**:
```cuda
// After computing all_inputs (activations), before top-k:
for step in range(refractory_period):
    for neuron_idx in refractory_buffer[step]:
        decay = 1.0 - step / refractory_period
        all_inputs[neuron_idx] -= inhibition_strength * decay
```

On GPU, this is a scatter-subtract on the activation array. The refractory
buffer is a circular array of winner sets (each set is K uint32 indices).

**Test plan**: Run existing tests with `--engine cuda_implicit`:
- `test_lri.py` — requires LRI
- `test_sequences.py` — requires LRI for ordered_recall
- `test_refracted.py` — requires refracted mode
- `test_fsm.py` — requires refracted mode
- All others should pass already

### Phase B: Adapter for nemo/core/brain.py

**Goal**: Allow NEMO's Brain to use CudaImplicitEngine under the hood.

Create `NemoBrainAdapter(ComputeEngine)` that wraps NEMO's torch-based
Brain into the ComputeEngine interface:

```python
class NemoBrainAdapter(ComputeEngine):
    def __init__(self, nemo_brain):
        self._nemo = nemo_brain

    def project_into(self, target, from_stimuli, from_areas, plasticity):
        # Translate to NEMO's projection API
        # Map between NEMO's Area enum and string names
```

This is a compatibility layer, not a rewrite. The NEMO Brain continues
to use its own CUDA kernels internally.

### Phase C: Retire nemo/emergent/cuda_backend.py

The `CUDAProjector` is a legacy DLL wrapper with a Python fallback. Once
CudaImplicitEngine has feature parity:

1. Point `CUDAProjector` at `CudaImplicitEngine` via a thin shim
2. Keep the Python fallback path for systems without CUDA
3. Remove the C++ DLL dependency

## CPU-Only Features (Phase A Blockers)

These features exist only in NumpySparseEngine and need GPU kernels:

### LRI (Long-Range Inhibition)

**CPU implementation** (numpy_engine.py:280-291):
```python
for steps_ago_idx, winner_set in enumerate(reversed(refractory_history)):
    steps_ago = steps_ago_idx + 1
    decay = 1.0 - (steps_ago - 1) / refractory_period
    penalty = inhibition_strength * decay
    for cidx in winner_set:
        all_inputs[cidx] -= penalty
```

**GPU implementation**: Store refractory history as a 2-D array
`[refractory_period, K]` on device. Each row holds the K winner indices
from that step. After computing activations, run a scatter-subtract
kernel over all rows with appropriate decay weights.

### Refracted Mode

**CPU** (numpy_engine.py:293-298): Subtract `_cumulative_bias[:end]` from
`all_inputs[:end]`, then increment bias for winners.

**GPU**: `_cumulative_bias` is a float32 array of length n on device.
Subtract and update are simple vectorized operations (`cp.subtract`,
`cp.add` with scatter).

## Test Plan for GPU Parity

### 1. Parametrize Engine Selection

Add to `conftest.py`:
```python
def pytest_addoption(parser):
    parser.addoption("--engine", default="numpy_sparse",
                     choices=["numpy_sparse", "cuda_implicit"])

@pytest.fixture
def engine_name(request):
    return request.config.getoption("--engine")
```

Update `_make_brain()` helpers in test files to use `engine_name` fixture.

### 2. Run Full Suite on GPU

```bash
# CPU (baseline)
uv run python -m pytest src/tests/ -v --engine numpy_sparse

# GPU (target)
uv run python -m pytest src/tests/ -v --engine cuda_implicit
```

### 3. Expected Failures Before Phase A

Tests that will fail on CudaImplicitEngine before LRI/refracted are added:
- `test_lri.py` (all tests)
- `test_sequences.py::TestOrderedRecall` (requires LRI)
- `test_refracted.py` (all tests)
- `test_fsm.py` (requires refracted mode)
- `test_pfa.py` (uses RandomChoiceArea with refracted)
- `test_sequence_recall_sweep.py` (requires LRI)

Tests that should pass already:
- `test_assembly.py`
- `test_brain.py`
- `test_connectome.py`
- `test_core_operations.py`
- `test_readout.py`
- `test_noise_robustness.py` (no LRI needed)
- `test_autonomous_recurrence.py` (no LRI needed)

## Summary

| Phase | What | Tests Gained |
|-------|------|-------------|
| A | LRI + refracted + normalization on GPU | ~50 tests |
| B | NEMO Brain adapter | NEMO test suite passes via ComputeEngine |
| C | Retire cuda_backend DLL | Simplification, not new tests |

Phase A is the critical path. Phases B and C are optional cleanup that
can happen independently.
