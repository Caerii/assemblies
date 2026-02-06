# Architecture

## Brain Implementations

There are three Brain implementations, each serving a different purpose:

### 1. Root `brain.py` — Classic API

The original implementation matching the Papadimitriou et al. paper. All root-level scripts (`learner.py`, `parser.py`, `simulations.py`, `image_learner.py`) use this directly.

- `Brain`: Manages areas, stimuli, connectomes, and the `project()` operation
- `Area`: A brain region with `n` neurons, `k` winners, plasticity parameter `beta`
- Connectomes are dense numpy matrices of shape `(n_src, n_dst)` initialized with random Bernoulli(p) weights

### 2. `src/core/brain.py` — Modular API

A refactored version that separates concerns into distinct classes:

- `Brain`, `Area`, `Stimulus`, `Connectome` — each in its own file under `src/core/`
- Uses mathematical primitives from `src/math_primitives/` for winner selection, plasticity, etc.
- Designed for extension and composition with the `src/` package system

### 3. `src/nemo/core/` — NEMO v2

A GPU-accelerated brain for language learning experiments:

- Excitatory and inhibitory neuron populations
- Learned (not hardcoded) grammar and word order
- Paired with `src/nemo/language/` for `LanguageLearner` and `SentenceGenerator`

## Module Dependency Graph

```
Root scripts (brain.py users)
  |
  brain.py  <-- authoritative Brain/Area
  brain_util.py  <-- overlap, save/load utilities
  |
  +-- learner.py       (word acquisition experiments)
  +-- parser.py        (sentence parsing)
  +-- simulations.py   (simulation harness)
  +-- image_learner.py (CIFAR-10 via assemblies)

src/ package (pip install -e .)
  |
  src/__init__.py  -->  src.core, src.math_primitives, src.gpu, src.constants, src.utils
  |
  src.core/
  |   brain.py, area.py, stimulus.py, connectome.py, brain_cpp.py
  |
  src.math_primitives/
  |   statistics.py, plasticity.py, winner_selection.py,
  |   neural_computation.py, explicit_projection.py,
  |   hyperdimensional.py, image_activation.py, sparse_simulation.py
  |
  src.simulation/          (uses src.core.Brain OR root brain.Brain)
  |   projection_simulator, merge_simulator, pattern_completion,
  |   association_simulator, density_simulator, turing_simulations,
  |   advanced_simulations, plotting_utils
  |
  src.language/            (grammar rules, parser, language areas)
  |
  src.gpu/                 (CuPy + PyTorch brain backends)
  |   cupy_brain.py, torch_brain.py, custom_kernels.py,
  |   gpu_utils.py, performance.py
  |
  src.lexicon/             (standalone word/curriculum system + NEMO runners)
  |   lexicon_manager.py, curriculum/, data/, statistics/
  |   nemo_full_system.py, nemo_hierarchical.py, nemo_ultra.py
  |   gpu_language_learner.py, assembly_language_learner.py
  |
  src.nemo/               (NEMO v2 — self-contained)
  |   core/   (brain, area, kernel)
  |   language/ (learner, generator, emergent/)
  |
  src.text_generation/    (assembly-based text generation)
      assembly_text_generator.py, grammatical_assembly_brain.py

cpp/
  cuda_kernels/  (raw CUDA .cu files for GPU acceleration)
  build_scripts/ (compilation scripts)
```

## How the Pieces Fit Together

**Assembly Calculus primitives** (`brain.py` or `src.core`) provide the foundation: `project()` creates and reinforces assemblies via Hebbian plasticity. Every higher-level capability is built by composing `project()` calls.

**Simulations** (`src.simulation`) study the dynamics: how assemblies stabilize, how patterns complete from partial activation, how two stimuli merge.

**Language** (`parser.py`, `learner.py`, `src.language`) uses assembly operations to parse and learn sentences. The parser creates assemblies for words and projects them through syntactic areas. The learner trains word-concept associations through repeated exposure.

**NEMO** (`src.nemo`) takes a different approach: rather than hardcoding grammar, it learns word order and syntactic structure from data using a GPU-accelerated brain with inhibitory neurons.

**Lexicon** (`src.lexicon`) provides the vocabulary infrastructure — 5000+ words with frequency statistics, semantic features, and curriculum-based learning progressions that feed into the NEMO and learner systems.

**Image classification** (`image_learner.py`) demonstrates that assemblies can classify visual inputs (CIFAR-10) by projecting pixel features through a hierarchy of brain areas to a class-labeled output area.

## Key Design Decisions

1. **Two Brain implementations coexist**: Root `brain.py` is the stable, paper-matching reference. `src.core.brain` is the modular refactor. Both are maintained because root scripts depend on the classic API and `src/` modules may use either via try/except imports.

2. **src.simulation uses dual imports**: Each simulator tries `from src.core.brain import Brain` first, falls back to `import brain`. This allows simulators to work both as part of the installed package and as standalone scripts.

3. **NEMO is self-contained**: `src/nemo/` has its own Brain implementation independent of both root and src.core. It can be used without any other module.

4. **GPU acceleration is optional**: All GPU code is behind try/except guards. The framework works on CPU-only machines with no code changes.
