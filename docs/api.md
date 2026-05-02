# API Guide

Use `neural_assemblies` for maintained package code. Use root imports only when
you are running old checkout-oriented scripts.

For scientific claim strength, read
[scientific_status.md](scientific_status.md). For code ownership boundaries,
read [supported_surfaces.md](supported_surfaces.md).

## Primary Imports

Convenience imports:

```python
from neural_assemblies import Brain, create_engine
from neural_assemblies import project, merge, overlap
```

Explicit imports are clearer in larger code:

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

## Core Runtime

```python
from neural_assemblies.core.brain import Brain

b = Brain(p=0.05, engine="numpy_sparse", seed=0)
b.add_stimulus("stim", size=100)
b.add_area("A", n=10_000, k=100, beta=0.05)
b.project({"stim": ["A"]}, {})
```

Main objects:

| Object | Location | Role |
|--------|----------|------|
| `Brain` | `core/brain.py` | Areas, stimuli, routing, projection cycles, and engine delegation. |
| `Area` | `core/area.py` | Area parameters and winner history. |
| `Stimulus` | `core/stimulus.py` | Fixed external inputs. |
| `Connectome` | `core/connectome.py` | Connectivity and learned weights. |
| `ComputeEngine` | `core/engine.py` | Engine interface used by `Brain`. |
| `create_engine()` | `core/engine.py` | Engine factory and registry entry point. |

## Engines

Engine names accepted by the package include:

- `numpy_sparse`
- `numpy_explicit`
- `cuda_implicit`
- `cupy_sparse`
- `torch_sparse`

`engine="auto"` calls `detect_best_engine()`:

- prefer `torch_sparse` when `n_hint >= 1_000_000` and PyTorch CUDA is
  available
- otherwise use `numpy_sparse`

Treat this as a default, not as a performance claim.

## Compute Primitives

```python
import numpy as np

from neural_assemblies.compute import TopKPolicy, WinnerSelector

selector = WinnerSelector(np.random.default_rng(0))
winners = selector.select_with_policy(
    [0.2, 0.8, 0.5, 0.8],
    TopKPolicy(k=2),
)
```

Important exports:

| Object | Role |
|--------|------|
| `StatisticalEngine` | Sampling and statistical helpers. |
| `NeuralComputationEngine` | Input aggregation and activation math. |
| `WinnerSelector` | Winner-selection logic. |
| `TopKPolicy` | Fixed-size competition. |
| `ThresholdPolicy` | Absolute-threshold competition with a `k` cap. |
| `RelativeThresholdPolicy` | Variable-size competition based on max-input fraction. |
| `PlasticityEngine` | Hebbian update logic. |

## Assembly Calculus

Data types:

| Object | Role |
|--------|------|
| `Assembly` | Snapshot of winners in one area. |
| `Sequence` | Ordered list of assemblies. |
| `Lexicon` | Mapping from tokens to assemblies. |
| `Transition` | Typed state transition. |
| `TransitionMap` | Validated deterministic or probabilistic transitions. |

Operations:

| Function | Purpose |
|----------|---------|
| `project` | Form an assembly from a stimulus or upstream area. |
| `reciprocal_project` | Copy an assembly between areas with reciprocal support. |
| `associate` | Link assemblies through shared activation. |
| `merge` | Build a conjunctive assembly. |
| `pattern_complete` | Recover from partial input. |
| `separate` | Measure distinctiveness. |
| `sequence_memorize` | Learn an ordered sequence. |
| `ordered_recall` | Replay a sequence with LRI support. |
| `overlap` | Measure assembly overlap. |
| `chance_overlap` | Compute random-overlap baseline. |

Structured helpers:

| Object | Purpose |
|--------|---------|
| `FSMNetwork` | Deterministic finite-state helper. |
| `PFANetwork` | Probabilistic finite automaton helper. |
| `RandomChoiceArea` | Stochastic choice primitive. |
| `FiberCircuit` | Declarative projection gating and control. |

## Simulation Helpers

`neural_assemblies.simulation` contains runnable helpers for projection,
association, merge, pattern completion, density, and Turing-style simulations.

Common imports include:

- `project_sim`, `project_beta_sim`, `assembly_only_sim`
- `association_sim`, `association_grand_sim`
- `merge_sim`, `merge_beta_sim`
- `pattern_com`, `pattern_com_repeated`
- `density`, `density_sim`
- `fixed_assembly_recip_proj`, `fixed_assembly_merge`, `separate`
- `larger_k`, `turing_erase`

The Turing-style helpers are exploratory simulation tools. The theoretical
claims belong to the sequence-computation papers.

## Language

Rule-based parsing:

```python
from neural_assemblies.language import parse

parse("cats chase mice", language="English")
```

Main exports include `ParserBrain`, `EnglishParserBrain`,
`RussianParserBrain`, `ReadoutMethod`, `fixed_map_readout`, `fiber_readout`,
`ParserDebugger`, and `parse(...)`.

Learned-language experiments live under `neural_assemblies.nemo` and the
emergent parser. Tests cover narrow synthetic behaviors; broad acquisition
claims need research artifacts.

## Lexicon

`neural_assemblies.lexicon` provides vocabulary and curriculum support:

- `LexiconManager`
- `Word`
- `WordCategory`
- `WordStatistics`

The surrounding modules include curriculum data, GPU learners, and
assembly-based learners used by language experiments.

## NEMO

`neural_assemblies.nemo` contains experimental language-learning systems.

Common imports from `neural_assemblies.nemo.language` include:

- `LanguageLearner`
- `SentenceGenerator`
- `Curriculum`
- `CurriculumLearner`
- `NemoLanguageLearner`
- `NemoBrain`
- `NemoParams`
- `IntegratedNemoTrainer`

## Compatibility Imports

Root files such as `brain.py`, `parser.py`, and `simulations.py` remain as
compatibility shims. The historical implementations live under
`legacy/root_modules/`.

Prefer package imports for new code.

## Useful Commands

```bash
uv run pytest neural_assemblies/tests -q
uv run pytest neural_assemblies/tests/test_docs_examples_smoke.py -q
uv run pytest tests/test_legacy_root_shims.py tests/test_legacy_archived_layout.py -q
```
