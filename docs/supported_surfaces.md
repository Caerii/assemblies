# Supported Surfaces

The repository currently has three different usage surfaces. They are not equal.

## 1. Installable package

Primary supported API:

```python
from neural_assemblies.core import Brain
from neural_assemblies.assembly_calculus import project, merge
```

- Distributed on PyPI as `assemblies`
- Imported in code as `neural_assemblies`
- Default quality gate: `uv run pytest -q`

## 2. Repo-root legacy modules

Historical checkout workflows:

- `brain.py`
- `simulations.py`
- `learner.py`
- `parser.py`
- `image_learner.py`

These remain useful for older scripts and exploratory experiments, but they are
not the primary library contract and should not define package release quality.

## 3. Research and performance workflows

Optional and more specialized:

- `research/experiments/*`
- `tests/performance/*`
- `cpp/*`

These are important for scientific progress and accelerator work, but they are
opt-in workflows with stricter environment assumptions.

## Practical rule

If a new feature is intended for downstream reuse, it should land under
`neural_assemblies/*` first. Repo-root scripts and ad hoc research harnesses
should depend on the package, not define it.
