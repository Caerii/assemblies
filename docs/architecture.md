# Architecture

Assemblies has one maintained package and several supporting work areas.

The package code lives under `neural_assemblies/`. Root files are compatibility
shims. `research/`, `tests/performance/`, `cpp/`, and `legacy/` support
experiments, accelerator work, and historical inspection.

For scientific claim strength, read
[scientific_status.md](scientific_status.md).

## Runtime Layers

```text
Brain
  -> core runtime and engine registry
  -> compute primitives
  -> assembly_calculus operations
  -> language, lexicon, nemo, and simulation helpers
```

## Core Runtime

`neural_assemblies.core` owns the runtime objects:

| Object | Role |
|--------|------|
| `Brain` | Areas, stimuli, projection routing, LRI controls, and engine delegation. |
| `Area` | Area parameters, winners, and history. |
| `Stimulus` | Fixed external input. |
| `Connectome` | Connectivity and learned weights. |
| `ComputeEngine` | Engine interface used by `Brain`. |

`Brain` orchestrates the simulation. Engines perform the low-level math.

## Compute Layer

`neural_assemblies.compute` contains reusable primitives:

- input aggregation
- statistical helpers
- Hebbian plasticity
- winner selection
- projection helper functions

Competition rules now have explicit policy objects:

- `TopKPolicy`
- `ThresholdPolicy`
- `RelativeThresholdPolicy`

The runtime still defaults to fixed top-k behavior in most paths. The policy
objects give richer inhibition models a place to grow without rewriting every
engine first.

## Assembly Calculus Layer

`neural_assemblies.assembly_calculus` turns projection cycles into named
operations and structured computations:

- `project`, `associate`, `merge`
- pattern completion and separation
- sequence memorization and ordered recall
- Long-Range Inhibition support
- `FiberCircuit`
- FSM and PFA helpers
- `Transition` and `TransitionMap`
- `NemoParser` and `EmergentParser`

This layer is where raw activity becomes reusable computational structure.

## Engines

Known engine names:

| Engine | Location | Role |
|--------|----------|------|
| `numpy_sparse` | `neural_assemblies/core/numpy_engine/` | Default CPU path for normal package use. |
| `numpy_explicit` | `neural_assemblies/core/numpy_engine/` | Dense explicit simulation for smaller areas. |
| `cuda_implicit` | `neural_assemblies/core/cuda_engine.py` | CuPy-based implicit GPU path. |
| `cupy_sparse` | `neural_assemblies/core/cupy_engine.py` | Optional CuPy sparse path. |
| `torch_sparse` | `neural_assemblies/core/torch_engine/` | Optional PyTorch CUDA sparse path. |

`Brain(..., engine="auto", n_hint=...)` uses
`neural_assemblies.core.backend.detect_best_engine()`:

- choose `torch_sparse` when `n_hint >= 1_000_000` and PyTorch CUDA is
  available
- otherwise choose `numpy_sparse`

That heuristic is a convenience, not a benchmark result.

## Automata Helpers

FSM and PFA code use typed transitions:

- `Transition`
- `TransitionMap`

This matters for correctness. String-like accidental inputs are rejected, and
probabilistic transitions are validated before automata helpers use them.

## Language Code

The repo has two language directions:

- `neural_assemblies.language` implements explicit English/Russian grammar
  parsing and readout utilities.
- `neural_assemblies.nemo` and the emergent parser explore learned category,
  role, and word-order behavior.

The second group is more experimental. Read it with the research docs and tests
in view.

## Compatibility And Archive

Root files such as `brain.py`, `parser.py`, and `simulations.py` remain for old
checkout workflows. They route to package code or archived implementations.

Historical material lives under `legacy/`:

- `legacy/root_modules/`
- `legacy/scripts/`
- `legacy/artifacts/`
- `legacy/experiments/`
- `legacy/matlab/`

New runtime behavior should go into `neural_assemblies/`, not into root shims
or archived scripts.

## Research And Accelerator Work

- `research/` tracks questions, experiments, results, and claims.
- `tests/performance/` checks optional hardware-sensitive paths.
- `cpp/` contains lower-level accelerator and kernel work.

These areas matter, but they are not part of the ordinary package test gate.

## Design Rules

1. Put reusable runtime behavior in the package.
2. Keep root files thin.
3. Keep research claims tied to experiments and results.
4. Add validation before expanding experimental abstractions.
5. Use targeted tests for code boundaries that other modules depend on.
