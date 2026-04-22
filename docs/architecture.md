# Architecture

## Brain Implementations

There are three Brain implementations, each serving a different purpose:

### 1. Root `brain.py` — Backward-Compatible Shim

A thin re-export of `assemblies.core.brain.Brain` and `assemblies.core.area.Area`. Old scripts
that `import brain` continue to work. The original 804-line standalone
implementation has been replaced by this 5-line shim.

### 2. `neural_assemblies/core/brain.py` — Production Orchestrator

The primary Brain implementation. Brain is a pure orchestrator that delegates
all computation to a ComputeEngine:

- `Brain`, `Area`, `Stimulus`, `Connectome` — each in its own file under `neural_assemblies/core/`
- Engine dispatch via `create_engine()`: numpy_sparse (default CPU), numpy_explicit, or cuda_implicit
- Mathematical primitives in `assemblies/compute/` for statistics, plasticity, winner selection
- ~500 lines: routing, area/stimulus management, save_winners/save_size recording

### 3. `assemblies/nemo/core/` — NEMO v2

A GPU-accelerated brain for language learning experiments:

- Excitatory and inhibitory neuron populations
- Learned (not hardcoded) grammar and word order
- Paired with `assemblies/nemo/language/` for `LanguageLearner` and `SentenceGenerator`

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
  +-- CudaImplicitEngine   (GPU, hash-based implicit connectivity, 40x speedup)
```

**Engine selection**: `Brain(p=0.05, engine="auto")` picks the best available.
CuPy present → `cuda_implicit`. Otherwise → `numpy_sparse`.

## GPU Acceleration

GPU code lives alongside the domain it accelerates — not in a single `gpu/` folder:

| Component | Location | Backend | Purpose |
|-----------|----------|---------|---------|
| CUDA kernels | `neural_assemblies/core/kernels/` | CuPy RawKernel | Hash-based projection, Hebbian update, batched ops |
| CUDA engine | `neural_assemblies/core/cuda_engine.py` | CuPy + PyTorch | ComputeEngine impl using implicit connectivity |
| Backend switch | `neural_assemblies/core/backend.py` | NumPy/CuPy | Seamless array library switching |
| NEMO brain | `assemblies/nemo/core/` | CuPy + PyTorch | Independent GPU brain for language experiments |
| NEMO CUDA FFI | `assemblies/nemo/language/emergent/cuda_backend.py` | ctypes + C++ | FFI to compiled CUDA kernels |

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
  artifacts/         (tracked image-learning GIFs and outputs)
  experiments/       (archived top-level experiment notes)
  matlab/            (MATLAB prototypes)

assemblies/ package (pip install neural-assemblies or pip install -e .)
  |
  neural_assemblies/__init__.py  -->  neural_assemblies.core, neural_assemblies.compute, assemblies.constants, assemblies.utils
  |
  assemblies.core/
  |   brain.py, area.py, stimulus.py, connectome.py
  |   engine.py (ComputeEngine ABC + registry)
  |   numpy_engine.py (NumpySparse + NumpyExplicit)
  |   cuda_engine.py (CudaImplicit)
  |   backend.py (NumPy/CuPy switching)
  |   kernels/ (implicit.py, batched.py, v2.py)
  |
  assemblies.compute/
  |   statistics.py, plasticity.py, winner_selection.py,
  |   neural_computation.py, explicit_projection.py,
  |   image_activation.py, sparse_simulation.py
  |
  assemblies.simulation/   (uses assemblies.core.Brain with engine="numpy_sparse")
  |   projection_simulator, merge_simulator, pattern_completion,
  |   association_simulator, density_simulator, turing_simulations,
  |   advanced_simulations
  |
  assemblies.language/     (grammar rules, parser, language areas)
  |
  assemblies.lexicon/      (standalone word/curriculum system)
  |   lexicon_manager.py, curriculum/, data/, statistics/
  |   gpu_assembly_learner.py, gpu_language_learner.py
  |
  assemblies.nemo/         (NEMO v2 — self-contained GPU system)
  |   core/   (brain, area, kernel)
  |   language/ (learner, generator, emergent/)
  |
  assemblies.constants/   (DEFAULT_P, DEFAULT_BETA, DEFAULT_W_MAX)
  assemblies.utils/       (math utilities)
  assemblies.tests/      (17 test files)

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

**Assembly Calculus primitives** (`assemblies.core.Brain` + `ComputeEngine`) provide
the foundation: `project()` creates and reinforces assemblies via Hebbian
plasticity. Every higher-level capability is built by composing `project()` calls.

**Simulations** (`assemblies.simulation`) study the dynamics: how assemblies stabilize,
how patterns complete from partial activation, how two stimuli merge.

**Language** (`parser.py`, `learner.py`, `assemblies.language`) uses assembly operations
to parse and learn sentences. The parser creates assemblies for words and projects
them through syntactic areas.

**NEMO** (`assemblies.nemo`) takes a different approach: rather than hardcoding grammar,
it learns word order and syntactic structure from data using a GPU-accelerated
brain with inhibitory neurons.

**Lexicon** (`assemblies.lexicon`) provides the vocabulary infrastructure — 5000+ words
with frequency statistics, semantic features, and curriculum-based learning
progressions.

## Key Design Decisions

1. **Engine-based dispatch**: Brain is a pure orchestrator. All computation goes
   through `ComputeEngine.project_into()`. This allows swapping CPU/GPU backends
   without changing Brain logic.

2. **Root brain.py is a shim**: Old scripts get `assemblies.core.brain.Brain` via the
   5-line re-export. No duplicate implementation to maintain.

3. **GPU code is distributed by domain**: CUDA kernels live in `neural_assemblies/core/kernels/`
   (next to the engine that uses them), NEMO GPU code lives in `assemblies/nemo/`.
   No central `gpu/` folder — GPU is an implementation detail, not a domain.

4. **NEMO is self-contained**: `assemblies/nemo/` has its own Brain implementation
   independent of assemblies.core. It can be used without any other module.

5. **GPU acceleration is optional**: All GPU code is behind try/except guards.
   The framework works on CPU-only machines with no code changes.
