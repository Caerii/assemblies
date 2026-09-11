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
| `numpy_sparse` | `neural_assemblies/core/numpy_engine/` | Default CPU path; recurrence is sampled until the target is materialized and is guarded by `SampledRecurrencePolicy`. |
| `numpy_explicit` | `neural_assemblies/core/numpy_engine/` | Dense explicit simulation for smaller areas. |
| `numpy_exact` | `neural_assemblies/core/numpy_engine/` | Fixed content-addressed graph with exact all-neuron drive and sparse learned deviations. |
| `cuda_implicit` | `neural_assemblies/core/cuda_engine.py` | CuPy-based implicit GPU path. |
| `cupy_sparse` | `neural_assemblies/core/cupy_engine.py` | Optional CuPy sparse path. |
| `torch_sparse` | `neural_assemblies/core/torch_engine/` | Optional PyTorch CUDA sparse path. |

`Brain(..., engine="auto", n_hint=...)` uses
`neural_assemblies.core.backend.detect_best_engine()`:

- choose `torch_sparse` when `n_hint >= 1_000_000` and PyTorch CUDA is
  available
- otherwise choose `numpy_sparse`

That heuristic is a convenience, not a benchmark result.

Hardware names do not define a scientific model. `Brain.model_semantics` records
the primary path's connectome realization, candidate domain, stimulus drive,
default tie rule, arithmetic, normalization, and plasticity as one immutable
object. Passing the object back through `Brain(model_semantics=...)` turns it into
an admission check: a backend or environment change that implements a different
profile raises before model topology exists. Area-local competition rules and
operation schedules remain protocol objects and are not hidden in this profile.

### The hashed substrate

`neural_assemblies/core/torch_engine/` also holds a second GPU path that
does not go through `Brain`. It runs many independent brains in one launch
and regenerates each connectome from a hash inside the kernel. Research at
twenty to a hundred brains per cell runs here. Its gates compare drives
against the numpy engine, since two implementations can select the same
winners from different drives.

| Unit | File | Role |
|------|------|------|
| `HashedArea` | `_hashed.py` | One area: winners, refraction bias, the k-WTA (`topk_select`), an optional per-brain convergence gate. |
| `DenseOrganFiber`, `PresentFiber`, `AreaFiber`, `StimulusFiber` | `_hashed.py` | Afferent fibers, one per density regime: dense int16 counts above ~10% density, present-only lists below. |
| `AssemblyMemory` | `_memory.py` | The refracted associative memory as a unit: store, gated write, masked half-cue recall. |
| `HashedArcCore` | `_arc_core.py` | The refracted arc-and-state core shared by the two sequence organs. |
| `HashedArcFSM`, `HashedTransducer` | `_hashed_fsm.py`, `_hashed_transducer.py` | The assigned-state machine and the induced-state transducer at width. |
| `ScheduledAligner` | `_scheduled_aligner.py` | The cross-situational word learner, whole schedule in one launch. |
| kernels | `_fused_cuda.py` | Presence hashing, drive, selection, write-back, and the persistent present-only kernel. |

These organs expose an immutable `organ_semantics` derived before CUDA
allocation. It composes the hashed substrate with state-code and schedule
semantics, including the stimulus law, tie jitter, normalization/scaling,
refraction, convergence gating, teacher forcing, frozen inference, successor
horizon/gain and prediction rule.
Passing a recorded profile back to the constructor makes it an admission check.
Schema-7 evidence wraps one or more named profiles in `execution_semantics`, so
paired arms cannot silently share a label while implementing different models.
The aligner has two distinct fiber families and therefore uses
`AlignerSemantics`; schema 8 adds an `alignment` execution kind without pretending
those anchor and cross-fiber laws are one organ substrate.

A unit's numbers are used only after it passes two gates: a drive replay
against the numpy engine to a relative 5e-6 with refraction included, and
identity across width, meaning a brain in a launch equals the same brain
run alone. The design notes are `research/notes/DESIGN_*.md`;
[../research/notes/README.md](../research/notes/README.md) lists them.

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

The historical shims, `brain.py` included, live in `legacy/root_shims/`
and work with that directory on `PYTHONPATH`; the repository root holds no
Python module.

Historical material lives under `legacy/`:

- `legacy/root_modules/` (the old implementations) and
  `legacy/root_shims/` (their re-exports)
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
