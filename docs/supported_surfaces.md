# Maintained Code Boundaries

The repo has package code, compatibility code, research code, and archived
history. Treat them differently.

## Package Code

`neural_assemblies/` is the maintained package.

Use this for new reusable work:

```python
from neural_assemblies.core import Brain
from neural_assemblies.assembly_calculus import project, merge
```

Package facts:

- PyPI project name: `neural-assemblies`
- import name: `neural_assemblies`
- package test command: `uv run pytest neural_assemblies/tests -q`

## Compatibility Shims

The historical shims (`brain.py`, `parser.py`, `simulations.py`,
`learner.py`, `image_learner.py`, `recursive_parser.py`, `brain_util.py`)
live in `legacy/root_shims/` and work with that directory on `PYTHONPATH`.
`brain` routes to the package; the others to the archived implementations.

They should stay thin. Their job is to route old imports to archived
implementations, not to grow new behavior. New maintained code must not import
them. A known exception remains in the checkout-oriented
`neural_assemblies/text_generation/robust_grammatical_brain.py` prototype: it imports
`brain` and requests unimplemented per-area connection probabilities for its CORE
areas. The maintained Brain rejects those overrides. This prototype is not a
supported runnable model through that shim; porting it requires implementing and
validating its intended connectivity, not deleting the overrides. See the
[contract](../neural_assemblies/ir/VERIFICATION.md#contract-explicit-probability).

## Research Code

`research/` holds experiments, result artifacts, plans, indexed claims, and
curated questions. It is where unfinished science belongs.

Research code can be rougher than package code, but it should still be
traceable: a result should point back to an experiment, and a claim should point
back to evidence.

## Optional Accelerator Work

`tests/performance/` and `cpp/` contain hardware-sensitive checks, CUDA/C++
work, and low-level accelerator experiments.

These paths matter, but they are not part of the default package test gate.

## Archive

`legacy/` stores old root modules, standalone scripts, image-learning
artifacts, experiment notes, and MATLAB prototypes.

Archived code is allowed to be historically useful. It should not quietly
define the behavior of the maintained package.

## Rule For New Work

Put reusable runtime behavior in `neural_assemblies/`.

Put experiments in `research/`.

Put historical material in `legacy/`.

Keep root-level code limited to compatibility and project metadata.


## Markov experiment boundary

`programs.ArcMarkovNetwork` and `ArcMarkovProtocol` expose an explicitly configured
experiment composing the existing arc readout with the shared seed-mixture selector.
`MarkovChainModel` is its trace-frequency wrapper. This surface is tested for neural
readout, initialized nulls and probe isolation; it is not a calibrated Markov sampler
or an implementation of the historical alternating-area architecture.

`NemoMarkovPFA` and `AlternatingMarkovNetwork` are retired and raise before brain
mutation. Their old numerical artifacts are not relabelled as the new protocol.
`SoftmaxContextCoin` is also retired with an explicit migration error. See the
[contract and scope](../neural_assemblies/ir/VERIFICATION.md#contract-arc-markov)
and [construction example](api.md#explicit-arc-markov-experiment).


## Context readout experiment boundary

`assembly_calculus.ContextAttractorChoice` requires `ContextChoiceProtocol` and a
construction-only `AttractorConfig`. It teaches assigned disjoint context codes
against clamped output attractors, then observes without learning. Each read exposes
both overlaps and an optional label (ties have no label), with independent context
and recurrence controls and reproducible native noise. Positive noise requires a
backend supporting it; `numpy_exact` refuses it before allocation.

This is a new experiment, not the old softmax law or a calibrated probability
sampler. Endpoint diagnostics establish sensitivity to noise, not a robustness
range. See the [contract](../neural_assemblies/ir/VERIFICATION.md#contract-context-choice)
and [example](api.md#context-conditioned-attractor-observation).
