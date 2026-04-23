# Architecture

## Brain Implementations

There are three Brain implementations, each serving a different purpose:

### 1. Root `brain.py` — Backward-Compatible Shim

A thin re-export of `neural_assemblies.core.brain.Brain` and `neural_assemblies.core.area.Area`. Old scripts
that `import brain` continue to work. The original 804-line standalone
implementation has been replaced by this 5-line shim.

### 2. `neural_assemblies/core/brain.py` — Production Orchestrator

The primary Brain implementation. Brain is a pure orchestrator that delegates
all computation to a ComputeEngine:

- `Brain`, `Area`, `Stimulus`, `Connectome` — each in its own file under `neural_assemblies/core/`
- Engine dispatch via `create_engine()`: numpy_sparse, numpy_explicit, cuda_implicit, or torch_sparse
- Mathematical primitives in `neural_assemblies/compute/` for statistics, plasticity, winner selection
- ~500 lines: routing, area/stimulus management, save_winners/save_size recording

### 3. `neural_assemblies/nemo/core/` — NEMO Experiments

Experimental NEMO-related brains and learners for language-learning work:

- Excitatory and inhibitory neuron populations
- Learners aimed at grounded categories, role structure, and word order
- Paired with `neural_assemblies/nemo/language/` for higher-level training and generation utilities

## ComputeEngine Architecture

Brain delegates all projection math to a `ComputeEngine` (ABC in `neural_assemblies/core/engine.py`):

```
Brain.project()
  |
  v
ComputeEngine.project_into(area, from_stimuli, from_areas)
  |
  +-- NumpySparseEngine   (CPU, statistical sparse simulation, scales to large n)
  +-- NumpyExplicitEngine  (CPU, dense matrices, faithful for small n)
  +-- CudaImplicitEngine   (GPU, hash-based implicit connectivity)
  +-- TorchSparseEngine    (GPU, CSR sparse connectivity; heuristic best engine at large n)
```

**Engine selection**: `Brain(p=0.05, engine="auto", n_hint=...)` uses a simple
heuristic. With large `n_hint` and PyTorch CUDA available it prefers
`torch_sparse`; otherwise it falls back to `numpy_sparse`. Explicit engine
selection is still recommended for reproducible benchmarking.

## GPU Acceleration

GPU code lives alongside the domain it accelerates — not in a single `gpu/` folder:

| Component | Location | Backend | Purpose |
|-----------|----------|---------|---------|
| CUDA kernels | `neural_assemblies/core/kernels/` | CuPy RawKernel | Hash-based projection, Hebbian update, batched ops |
| CUDA engine | `neural_assemblies/core/cuda_engine.py` | CuPy + PyTorch | ComputeEngine impl using implicit connectivity |
| Backend switch | `neural_assemblies/core/backend.py` | NumPy/CuPy | Seamless array library switching |
| NEMO brain | `neural_assemblies/nemo/core/` | CuPy + PyTorch | Experimental GPU-oriented language-learning brains |
| NEMO CUDA FFI | `neural_assemblies/nemo/language/emergent/cuda_backend.py` | ctypes + C++ | FFI to compiled CUDA kernels |

The `CudaImplicitEngine` uses hash-based connectivity: O(learned_connections) memory
instead of O(n^2). At n=100,000 this means ~25 MB vs ~40 GB.

## Module Dependency Graph

```
Root compatibility layer
  |
  brain.py  <-- 5-line shim re-exporting neural_assemblies.core.brain
  brain_util.py  <-- shim -> legacy/root_modules/brain_util.py
  |
  +-- learner.py       (shim -> legacy/root_modules/learner.py)
  +-- parser.py        (shim -> legacy/root_modules/parser.py)
  +-- simulations.py   (shim -> legacy/root_modules/simulations.py)
  +-- image_learner.py (shim -> legacy/root_modules/image_learner.py)
  +-- recursive_parser.py (shim -> legacy/root_modules/recursive_parser.py)

legacy/
  root_modules/      (historical repo-root implementations)
  scripts/           (archived standalone scripts and helpers)
  artifacts/         (tracked image-learning GIFs and outputs)
  experiments/       (archived top-level experiment notes)
  matlab/            (MATLAB prototypes)

neural_assemblies package (pip install neural-assemblies or pip install -e .)
  |
  neural_assemblies/__init__.py  -->  neural_assemblies.core, neural_assemblies.compute, neural_assemblies.constants, neural_assemblies.utils
  |
  neural_assemblies.core/
  |   brain.py, area.py, stimulus.py, connectome.py
  |   engine.py (ComputeEngine ABC + registry)
  |   numpy_engine.py (NumpySparse + NumpyExplicit)
  |   cuda_engine.py (CudaImplicit)
  |   backend.py (NumPy/CuPy switching)
  |   kernels/ (implicit.py, batched.py, v2.py)
  |
  neural_assemblies.compute/
  |   statistics.py, plasticity.py, winner_selection.py,
  |   neural_computation.py, explicit_projection.py,
  |   image_activation.py, sparse_simulation.py
  |
  neural_assemblies.simulation/   (uses neural_assemblies.core.Brain with engine="numpy_sparse")
  |   projection_simulator, merge_simulator, pattern_completion,
  |   association_simulator, density_simulator, turing_simulations,
  |   advanced_simulations
  |
  neural_assemblies.language/     (grammar rules, parser, language areas)
  |
  neural_assemblies.lexicon/      (standalone word/curriculum system)
  |   lexicon_manager.py, curriculum/, data/, statistics/
  |   gpu_assembly_learner.py, gpu_language_learner.py
  |
  neural_assemblies.nemo/         (NEMO-related experimental systems)
  |   core/   (brain, area, kernel)
  |   language/ (learner, generator, emergent/)
  |
  neural_assemblies.constants/   (DEFAULT_P, DEFAULT_BETA, DEFAULT_W_MAX)
  neural_assemblies.utils/       (math utilities)
  neural_assemblies.tests/       (package test suite)

research/
  experiments/  (stability, primitives)
  nemo/         (standalone NEMO runner scripts)
  results/      (experiment outputs)
  plans/        (research plans)

cpp/
  cuda_kernels/  (raw CUDA .cu files)
  build_scripts/ (compilation scripts)
```

## How the Pieces Fit Together

**Assembly Calculus primitives** (`neural_assemblies.core.Brain` + `ComputeEngine`) provide
the foundation: `project()` creates and reinforces assemblies via Hebbian
plasticity. Every higher-level capability is built by composing `project()` calls.

**Simulations** (`neural_assemblies.simulation`) study the dynamics: how assemblies stabilize,
how patterns complete from partial activation, how two stimuli merge.

**Language** (`parser.py`, `learner.py`, `neural_assemblies.language`) uses assembly operations
to parse and learn sentences. The parser creates assemblies for words and projects
them through syntactic areas.

**NEMO** (`neural_assemblies.nemo`) contains experimental language-learning
systems that aim to learn word order and syntactic structure from data using
assembly-style and GPU-accelerated components.

**Lexicon** (`neural_assemblies.lexicon`) provides the vocabulary infrastructure — 5000+ words
with frequency statistics, semantic features, and curriculum-based learning
progressions.

## Key Design Decisions

1. **Engine-based dispatch**: Brain is a pure orchestrator. All computation goes
   through `ComputeEngine.project_into()`. This allows swapping CPU/GPU backends
   without changing Brain logic.

2. **Root brain.py is a shim**: Old scripts get `neural_assemblies.core.brain.Brain` via the
   5-line re-export. No duplicate implementation to maintain.

3. **GPU code is distributed by domain**: CUDA kernels live in `neural_assemblies/core/kernels/`
   (next to the engine that uses them), NEMO GPU code lives in `assemblies/nemo/`.
   No central `gpu/` folder — GPU is an implementation detail, not a domain.

4. **NEMO is semi-independent experimental code**: `neural_assemblies/nemo/`
   contains its own brains and learners for language experiments, but should be
   read as research-oriented package code rather than the same stability level
   as the core engine/runtime.

5. **GPU acceleration is optional**: All GPU code is behind try/except guards.
   The framework works on CPU-only machines with no code changes.
