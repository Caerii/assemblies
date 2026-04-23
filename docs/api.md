# API Guide

This document describes the current package-facing and compatibility-facing API
for the repository as it exists today.

For statement strength, read [scientific_status.md](scientific_status.md).
For package versus legacy boundaries, read
[supported_surfaces.md](supported_surfaces.md).

## Primary Entry Points

Top-level imports exposed by `neural_assemblies` are intended as convenience
re-exports over the package submodules:

```python
from neural_assemblies import Brain, create_engine
from neural_assemblies import project, merge, overlap
```

For larger codebases, direct submodule imports are usually clearer:

```python
from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import (
    Assembly,
    Sequence,
    project,
    merge,
    sequence_memorize,
    ordered_recall,
)
```

## Supported Surfaces

There are three distinct surfaces:

1. `neural_assemblies/*`
   This is the primary supported package API.
2. Repo-root compatibility shims such as `brain.py`, `parser.py`, and
   `simulations.py`
   These exist for historical checkout workflows.
3. Archived and research-only material under `legacy/` and `research/`
   These are important, but they are not the default package contract.

## Core Runtime

The `neural_assemblies.core` package provides the runtime substrate.

Main components:

| Object | Location | Role |
|--------|----------|------|
| `Brain` | `core/brain.py` | Orchestrates areas, stimuli, routing, and projection cycles. |
| `Area` | `core/area.py` | Holds area parameters such as `n`, `k`, `beta`, and winner history. |
| `Stimulus` | `core/stimulus.py` | Represents external fixed inputs. |
| `Connectome` | `core/connectome.py` | Connectivity and learned weights between sources and targets. |
| `ComputeEngine` | `core/engine.py` | Abstract engine interface used by `Brain`. |
| `create_engine()` | `core/engine.py` | Engine factory and registration seam. |

Common usage:

```python
from neural_assemblies.core.brain import Brain

b = Brain(p=0.05, engine="numpy_sparse", seed=0)
b.add_stimulus("stim", size=100)
b.add_area("A", n=10_000, k=100, beta=0.05)
b.project({"stim": ["A"]}, {})
```

### Engine Selection

Known engine names in the current codebase:

- `numpy_sparse`
- `numpy_explicit`
- `cuda_implicit`
- `cupy_sparse`
- `torch_sparse`

`engine="auto"` currently delegates to `detect_best_engine()` in
`neural_assemblies.core.backend`.

Current heuristic:

- if `n_hint >= 1_000_000` and PyTorch CUDA is available, prefer
  `torch_sparse`
- otherwise prefer `numpy_sparse`

This is a practical heuristic, not a universal optimality claim.

## Compute Primitives

The `neural_assemblies.compute` package contains reusable mathematical and
selection primitives used by the runtime.

Main exports:

| Object | Role |
|--------|------|
| `StatisticalEngine` | Sampling and statistical helpers. |
| `NeuralComputationEngine` | Input aggregation and activation math. |
| `WinnerSelector` | Winner-selection logic used by engine paths. |
| `TopKPolicy` | Fixed-size competition rule. |
| `ThresholdPolicy` | Absolute-threshold competition rule with `k` cap. |
| `RelativeThresholdPolicy` | Variable-size competition rule based on fraction of max input. |
| `PlasticityEngine` | Hebbian update logic. |

Example:

```python
import numpy as np

from neural_assemblies.compute import TopKPolicy, WinnerSelector

selector = WinnerSelector(np.random.default_rng(0))
winners = selector.select_with_policy(
    [0.2, 0.8, 0.5, 0.8],
    TopKPolicy(k=2),
)
```

The policy layer exists so competition semantics can evolve without forcing the
engines to hard-code one inhibition model forever.

## Assembly Calculus

The `neural_assemblies.assembly_calculus` package provides first-class
operations and structured computation built on top of `Brain.project()`.

### Data Types

| Object | Role |
|--------|------|
| `Assembly` | Snapshot of winners in one area. |
| `Sequence` | Ordered list of `Assembly` snapshots. |
| `Lexicon` | Mapping from tokens to assembly snapshots. |
| `Transition` | Typed state transition primitive for automata helpers. |
| `TransitionMap` | Validated container for deterministic or probabilistic transitions. |

### Core Operations

| Function | Purpose |
|----------|---------|
| `project` | Stimulus-to-area assembly formation. |
| `reciprocal_project` | Area-to-area copying. |
| `associate` | Link assemblies through shared activation. |
| `merge` | Build conjunctive assemblies. |
| `pattern_complete` | Recover from partial input. |
| `separate` | Compare distinctiveness of formed assemblies. |
| `sequence_memorize` | Learn ordered sequences. |
| `ordered_recall` | Replay a sequence with LRI support. |
| `overlap` | Measure overlap between assemblies. |
| `chance_overlap` | Baseline random overlap expectation. |

### Structured Computation

| Object | Purpose |
|--------|---------|
| `FSMNetwork` | Deterministic finite-state computation helper. |
| `PFANetwork` | Probabilistic finite automaton helper. |
| `RandomChoiceArea` | Stochastic choice primitive. |
| `FiberCircuit` | Declarative projection gating and control. |

### Language-Like Higher Layers

| Object | Purpose |
|--------|---------|
| `NemoParser` | Composed parser surface built on assembly operations. |
| `EmergentParser` | Larger learned/emergent parsing system. |
| `build_next_token_model`, `train_on_corpus`, `predict_next_token`, `score_corpus` | Next-token style utilities built on assembly sequences and overlap. |

## Simulation Package

The `neural_assemblies.simulation` package contains runnable experiment helpers
and plotting utilities.

Common exports:

- `project_sim`, `project_beta_sim`, `assembly_only_sim`
- `association_sim`, `association_grand_sim`
- `merge_sim`, `merge_beta_sim`
- `pattern_com`, `pattern_com_repeated`
- `density`, `density_sim`
- `fixed_assembly_recip_proj`, `fixed_assembly_merge`, `separate`
- `larger_k`, `turing_erase`

The Turing-style simulation helpers should be read as exploratory simulation
tools, not as a package-level proof artifact.

## Rule-Based Language Package

The `neural_assemblies.language` package is the rule-based parsing surface.

Main exports:

- `ParserBrain`
- `EnglishParserBrain`
- `RussianParserBrain`
- `LEXEME_DICT`
- `RUSSIAN_LEXEME_DICT`
- `ReadoutMethod`
- `fixed_map_readout`
- `fiber_readout`
- `ParserDebugger`
- `parse(...)`

Use this package when you want explicit grammar rules. Use
`neural_assemblies.nemo` when you want the experimental learned-language
surfaces instead.

## Lexicon Package

The `neural_assemblies.lexicon` package provides structured vocabulary and
curriculum support.

Main exports:

- `LexiconManager`
- `Word`
- `WordCategory`
- `WordStatistics`

The surrounding modules include curriculum data, GPU learners, and
assembly-based learners used by the broader language experiments.

## NEMO Package

The `neural_assemblies.nemo` package contains experimental language-learning
systems that are distinct from the core runtime.

Important point: package tests cover narrow synthetic behaviors here, but broad
curriculum and language-acquisition claims should be tied to specific research
artifacts, not inferred from package installation alone.

Main imports exposed by `neural_assemblies.nemo.language` include:

- `LanguageLearner`
- `SentenceGenerator`
- `Curriculum`
- `CurriculumLearner`
- `NemoLanguageLearner`
- `NemoBrain`
- `NemoParams`
- `IntegratedNemoTrainer`

## Legacy Compatibility

Repo-root files such as `brain.py`, `parser.py`, and `simulations.py` are now
thin compatibility shims. The historical implementations they point to live
under `legacy/root_modules/`.

Archived scripts and artifacts live under:

- `legacy/scripts/`
- `legacy/artifacts/`
- `legacy/experiments/`
- `legacy/matlab/`

Use the package modules for active work. Use the archived surfaces only when
you need to reproduce or inspect old experiments.

## Recommended Commands

```bash
# Default package gate
uv run pytest neural_assemblies/tests -q

# Docs and examples smoke coverage
uv run pytest neural_assemblies/tests/test_docs_examples_smoke.py -q

# Legacy compatibility checks
uv run pytest tests/test_legacy_root_shims.py tests/test_legacy_archived_layout.py -q
```
