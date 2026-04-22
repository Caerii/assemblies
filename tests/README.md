# Test Matrix

This repository currently has three distinct test surfaces:

- `neural_assemblies/tests/`
  The installable package test suite. This is the default supported contributor workflow.
- `tests/`
  Legacy repo-root compatibility and performance checks. These rely on historical scripts and optional local tooling.
- `research/experiments/`
  Research validation code and experiment harnesses. These are not part of the default package-quality gate.

## Recommended commands

```bash
# Default library confidence gate
uv run pytest -q

# Explicit package suite
uv run pytest neural_assemblies/tests -q

# Legacy / performance surface
uv run pytest tests -q

# Research validation surface
uv run pytest research/experiments -q
```

## Why the split exists

The package contract is `pip install assemblies` with `neural_assemblies` imports.
The legacy repo-root modules (`brain.py`, `simulations.py`, `learner.py`, `parser.py`)
remain useful for historical experiments, but they are not the default installable-library
surface and should not define whether the package is shippable.
