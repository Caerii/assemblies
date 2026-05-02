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

Root files such as `brain.py`, `parser.py`, `simulations.py`, `learner.py`,
`image_learner.py`, `recursive_parser.py`, and `brain_util.py` remain for old
checkout workflows.

They should stay thin. Their job is to route old imports to maintained package
code or archived implementations, not to grow new behavior.

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
