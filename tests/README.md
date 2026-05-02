# Test Matrix

The repo has three test groups.

## Package tests

```bash
uv run pytest neural_assemblies/tests -q
```

These tests define normal package confidence for `pip install
neural-assemblies` and `import neural_assemblies`.

## Legacy and performance tests

```bash
uv run pytest tests -q
```

These tests cover root compatibility shims, archived workflows, and optional
accelerator checks. Some tests depend on local CUDA/C++ tooling and may skip on
ordinary machines.

## Research experiment tests

```bash
uv run pytest research/experiments -q
```

These tests validate research harnesses and experiment code. They are useful
for scientific work, but they are not the package release gate.

## Default

The pytest configuration defaults to the package suite. Use explicit paths when
you want legacy, performance, or research coverage.
