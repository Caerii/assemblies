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
    AssemblyTrace,
    Sequence,
    project,
    merge,
    sequence_memorize,
    ordered_recall,
)
from neural_assemblies.assembly_calculus.tracing import (
    merge_trace,
    project_trace,
    projection_sweep,
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

Registration requires unique, nonempty names across both areas and stimuli.
Registering an existing name raises `ValueError`; it does not reset or resize a
population. Area dimensions require integer `0 < k <= n <= 2**32`; stimulus sizes
require integer `0 <= size <= 2**32`. Boolean counts are rejected. A zero-sized
stimulus is allowed as a null input. These logical limits do not guarantee that
an allocation fits in memory. See the source-linked
[registration contract](../neural_assemblies/ir/VERIFICATION.md#contract-stimulus-registration).

`add_explicit_area` uses the brain-wide connection probability. Its legacy
`custom_inner_p`, `custom_out_p` and `custom_in_p` overrides raise
`NotImplementedError` when supplied; they previously had no effect. See the
[probability contract](../neural_assemblies/ir/VERIFICATION.md#contract-explicit-probability)
for the affected prototype and the missing heterogeneous-connectivity semantics.

Use `set_competition_policy(area_name, policy)` to change an existing area's
winner selection, or pass `None` to restore top-k. The executing engine accepts
the change before the Area descriptor updates; dense areas with multiple slots
reject a custom policy. See the
[runtime policy contract](../neural_assemblies/ir/VERIFICATION.md#contract-runtime-policy).

`set_input_noise(area_name, std)` uses the same finite, nonnegative standard
deviation contract as registration. Zero disables noise; nonzero noise requires
backend support and is rejected for explicit dense areas. The setting is published
only after the executing engine accepts it. See the
[input-noise contract](../neural_assemblies/ir/VERIFICATION.md#contract-input-noise).

Competition policies validate their parameters when constructed. Winner counts
are nonnegative integers; real parameters must be finite. The supported tie rule
is `value_then_index`, and E%-WTA windows are `epsilon` or `sigma`. Invalid strings
raise instead of selecting a fallback. See the
[policy-value contract](../neural_assemblies/ir/VERIFICATION.md#contract-policy-values).

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
| `AssemblyTrace` | Ordered snapshots and per-round metrics from a traced operation. |
| `Sequence` | Ordered list of assemblies. |
| `Lexicon` | Mapping from tokens to assemblies. |
| `Transition` | Typed state transition. |
| `TransitionMap` | Validated deterministic or probabilistic transitions. |

Operations:

| Function | Purpose |
|----------|---------|
| `project` | Form an assembly from a stimulus or upstream area. |
| `project_trace` | Form an assembly while recording each projection round. |
| `reciprocal_project` | Copy an assembly between areas with reciprocal support. |
| `reciprocal_project_trace` | Trace source-area projection into a target area. |
| `associate` | Link assemblies through shared activation. |
| `associate_trace` | Trace the three phases of association. |
| `merge` | Build a conjunctive assembly. |
| `merge_trace` | Build a conjunctive assembly while recording target dynamics. |
| `pattern_complete_trace` | Damage a current assembly and trace recurrent recovery. |
| `ordered_recall_trace` | Replay a sequence with LRI while recording accepted recall steps. |
| `source_response_traces` | Probe a merged target from each source and compare responses. |
| `projection_sweep` | Run small independent projection-trace parameter sweeps. |
| `lri_recall_sweep` | Run small independent LRI recall parameter sweeps. |
| `pattern_complete` | Recover from partial input. |
| `separate` | Measure distinctiveness. |
| `snapshot_area` | Snapshot current winners in an area using comparable neuron IDs. |
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

## Visualization

`neural_assemblies.viz` provides small Matplotlib-based helpers for notebooks
and diagnostics. The implementation is split into grid, flow, overlap, and
trace-specific modules, while these public imports remain stable:

```python
from neural_assemblies.viz import animate_assembly_trace, plot_assemblies
```

Use these helpers to make package state inspectable:

- `plot_projection_flow` draws the shape of a two-source projection/merge
  example.
- `plot_assembly` and `plot_assemblies` show winner sets on square neuron
  grids.
- `animate_assembly_trace` animates round-by-round winner turnover from an
  `AssemblyTrace`.
- `plot_trace_metrics` plots consecutive overlap and new-winner counts.
- `plot_winner_turnover` shows which winner IDs persist or rotate over a
  trace.
- `plot_response_overlap` compares same-area responses with a reference
  assembly and chance overlap.
- `plot_overlap_matrix` shows pairwise assembly overlap.
- `plot_recall_trace` compares recalled assemblies against known references.
- `plot_parameter_heatmap` labels compact sweep matrices.

These helpers are teaching and debugging tools. Use `research/` artifacts for
scientific evidence.

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

The historical imports (`import brain`, `import parser`, `import
simulations`, and the rest) work with `legacy/root_shims/` on `PYTHONPATH`.
`brain` routes to the package; the others to the archived implementations
under `legacy/root_modules/`. The repository root holds no Python module.

Prefer package imports for new code.

## Useful Commands

```bash
uv run pytest neural_assemblies/tests -q
uv run pytest neural_assemblies/tests/test_docs_examples_smoke.py -q
uv run pytest tests/test_legacy_root_shims.py tests/test_legacy_archived_layout.py -q
```


## Classification cues and evidence

`parser.classify_word_evidence(word, grounding, cue_mode="grounding_only")`
queries registered grounding features without adding the word's phonological
stimulus. The other modes are `phon_only` and `combined` (the unchanged default).
Use the returned `source`, `scores`, `cue_mode`, and `cues` together: an overlap is
not a calibrated probability, and registered phonological input is not necessarily
learned input. Queries remain read-only. See the
[cue contract](../neural_assemblies/ir/VERIFICATION.md#contract-classification-cues).


`classify_word_cached` retains its fast path for the word's stored grounding.
Explicitly different grounding runs uncached inference and does not replace the
word's cached category. Use `classify_word_evidence` for an explicit neural cue
mode and retained score/cue provenance.


### Explicit experimental PFA choice

A deterministic `PFANetwork` builds no coin. Branching requires an immutable
`SeedMixtureChoice`; the FSM's population and learning settings do not configure
its coin implicitly. This is an **uncalibrated seed-mixture experiment**: transition
weights set initial seed fractions, not measured outcome probabilities. Successors
are selected symbolically from coin labels, not decoded from a learned transition
network. See [the source-linked contract](../neural_assemblies/ir/VERIFICATION.md#contract-pfa-choice).

```python
from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import PFANetwork, SeedMixtureChoice

brain = Brain(p=0.05, seed=1, engine="numpy_sparse")
choice = SeedMixtureChoice(n=2000, k=200, beta=3.0,
                          rounds_train=10, fires=2, rounds=10, mode="k_split")
machine = PFANetwork(
    brain, states=["q0", "q1"], symbols=["a"],
    transitions=[("q0", "a", "q0", 0.25), ("q0", "a", "q1", 0.75)],
    initial_state="q0", n=500, k=20, beta=0.1, rounds=3, choice=choice,
)
state = machine.step("a", seed=701)
```

The coin materializes its own population during construction. FSM state encoding
still uses the specified NumPy engine; this example is API usage, not sequence
validation. `CoinFlipModel` accepts the same `choice` and builds separate PFA and
sampling coins; its input-noise option applies only to the latter. Serialize the
configuration with `dataclasses.asdict(choice)` when recording a protocol.
Legacy coin goldens remain historical artifacts. They are not automatically
reinterpreted as measurements of this construction.


### ERP context initialization and isolated reads

`ErpProtocol(context_reset="construction")` retains the existing sentence-building
reset. `context_reset="activity"` clears context activity while preserving its
neuron identities, and the runner records this choice in `erp_protocol`.
Activity reset alone does not prohibit recruitment or restore the rest of the brain.

For an already initialized parser, use a separate `parser.brain.probe()` scope
around each `run_incremental_erp_probes(..., protocol=ErpProtocol(context_reset="activity"))`
call to restore the same starting brain state. Initialize outside that scope:
read-only observations do not recruit missing populations. `read_only()` alone
allows activity evolution, so sequential reads need not agree. This compositional
pattern passed the existing sentence fixture; it does not validate ERP
calibration/discrimination or promise isolation of all Python-side parser state.


### Cyclic word-problem controls

Use `neural_assemblies.programs.word_problems.cyclic_group(order, generators)` to
configure an additive cyclic benchmark, for example `cyclic_group(30, (7, 11))`.
Generators are normalized modulo the order and keep their alphabet order. A set
that only generates a proper subgroup raises instead of silently creating an easier
benchmark. Existing `cyclic_group_60()` and `cyclic_group_120()` wrappers retain
their state labels and transition tables. This constructs symbolic ground truth;
it does not establish that a neural network learns the resulting word problem.
