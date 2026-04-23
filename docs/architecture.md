# Architecture

This document describes the current software architecture of the repository.

It is about code organization and runtime boundaries, not about proving the
underlying science. For scientific claim boundaries, see
[scientific_status.md](scientific_status.md).

## Supported Surfaces

The repository has three practical surfaces:

1. Installable package: `neural_assemblies/*`
2. Repo-root compatibility shims for old checkout workflows
3. Research, performance, and archived legacy material

Only the installable package is the default release-quality contract.

## Runtime Stack

The main package layers look like this:

```text
Brain / package entry points
  -> core runtime and engine registry
  -> compute primitives
  -> assembly_calculus operations
  -> language / lexicon / nemo higher layers
  -> simulation helpers
```

### 1. Core Runtime

`neural_assemblies.core` owns:

- `Brain`
- `Area`
- `Stimulus`
- `Connectome`
- `ComputeEngine`
- engine registration and selection

`Brain` is intentionally an orchestrator. It owns topology, routing, and high
level projection flow, while engine implementations own the low-level math.

### 2. Compute Layer

`neural_assemblies.compute` provides reusable primitives such as:

- statistical helpers
- input aggregation
- Hebbian plasticity
- winner selection
- explicit projection helpers

This separation matters because it lets the package reuse the same mathematical
building blocks across different engines and experiments.

### 3. Assembly Calculus Layer

`neural_assemblies.assembly_calculus` provides named operations and structured
computation on top of the runtime:

- `project`, `associate`, `merge`
- sequence memorization and ordered recall
- `FiberCircuit`
- FSM and PFA helpers
- typed transitions via `Transition` and `TransitionMap`
- parser-like higher layers such as `NemoParser` and `EmergentParser`

This layer is where the package moves from raw projection cycles to reusable
computational idioms.

## Brain Implementations

### Root `brain.py`

The repo-root `brain.py` file is now a compatibility shim that re-exports the
package `Brain` and `Area`. Old checkout-oriented scripts can still `import
brain`, but the maintained implementation lives in the package.

### `neural_assemblies.core.brain`

This is the primary runtime implementation.

Responsibilities:

- manage areas and stimuli
- manage projection routing
- delegate computation to a `ComputeEngine`
- expose runtime controls such as LRI and inhibition-related settings

### `neural_assemblies.nemo.core`

This is a separate experimental NEMO-oriented surface for language-learning
work. It should be read as research-oriented package code rather than the same
stability level as the core runtime.

## Engine Architecture

Known engine names in the current codebase:

| Engine | Location | Role |
|--------|----------|------|
| `numpy_sparse` | `neural_assemblies/core/numpy_engine/` | Default CPU path for most package use. |
| `numpy_explicit` | `neural_assemblies/core/numpy_engine/` | Dense explicit simulation for smaller areas. |
| `cuda_implicit` | `neural_assemblies/core/cuda_engine.py` | CuPy-based hash / implicit GPU path. |
| `cupy_sparse` | `neural_assemblies/core/cupy_engine.py` | Optional CuPy sparse path. |
| `torch_sparse` | `neural_assemblies/core/torch_engine/` | Optional PyTorch CUDA sparse path. |

### Auto Selection

`Brain(..., engine="auto", n_hint=...)` currently uses a simple heuristic in
`neural_assemblies.core.backend.detect_best_engine()`:

- if `n_hint >= 1_000_000` and PyTorch CUDA is available, choose
  `torch_sparse`
- otherwise choose `numpy_sparse`

That is a practical default, not a universal performance guarantee.

## Competition and Winner Policies

Historically the runtime centered on fixed top-k competition. The package now
has a cleaner abstraction seam for competition rules in
`neural_assemblies.compute`.

Current winner-policy objects:

- `TopKPolicy`
- `ThresholdPolicy`
- `RelativeThresholdPolicy`

`WinnerSelector.select_with_policy(...)` makes these policies explicit without
forcing the rest of the runtime to claim that a richer inhibition model is
already fully integrated everywhere.

## Typed Transitions and Automata

The FSM and PFA helpers are now built around typed transitions:

- `Transition`
- `TransitionMap`

This is an engineering cleanup with scientific consequences: automata-style
experiments now have a clearer formal boundary and input validation instead of
loosely typed tuples everywhere.

## Language Surfaces

There are two distinct language directions in the repo:

### Rule-Based Parsing

`neural_assemblies.language` contains explicit grammar-rule-based parsing for
English and Russian.

### Experimental Learned Language

`neural_assemblies.nemo` and the emergent parser code aim at learning category,
role, and word-order structure from exposure. These surfaces are more
experimental and should be read alongside the research docs.

## Legacy and Archive Layout

The top level of the repo is intentionally cleaner than it used to be.

Historical material now lives under `legacy/`:

- `legacy/root_modules/` for old repo-root implementations
- `legacy/scripts/` for historical standalone scripts
- `legacy/artifacts/` for tracked image-learning outputs
- `legacy/experiments/` for older notes
- `legacy/matlab/` for MATLAB prototypes

The root files that remain are compatibility shims, not full duplicate
implementations.

## Research and Performance Workflows

The package is not the whole repository.

- `research/` contains experiment registries, indexed claims, and curated core
  questions.
- `tests/performance/` contains optional environment-sensitive accelerator
  checks.
- `cpp/` contains lower-level accelerator and kernel work.

These are important, but they are opt-in surfaces with stricter assumptions
than the default package gate.

## Design Principles

1. Keep the installable package as the primary truth surface.
2. Keep root compatibility without letting root scripts define the package.
3. Separate package-backed claims from research claims.
4. Add abstraction seams before adding new biological or world-model features.
5. Prefer explicit validators and targeted tests over broad undocumented
   assumptions.
