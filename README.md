# Assemblies

Sparse neural assembly calculus runtime and research workspace for projection,
association, merge, sequence memory, language experiments, and optional GPU
execution.

The repository currently has two roles:

- an installable Python package, published as `neural-assemblies` and imported
  as `neural_assemblies`
- a research workspace built around that package

This repo is maintained by Alif Jakir. It grew out of assembly-calculus and
language-organ work around MIT's Projects in the Science of Intelligence
course and later extensions of that line of research, including collaboration
with Daniel Mitropolsky. The package is intentionally narrower than the full
research agenda.

If you care about statement strength, read
[docs/scientific_status.md](docs/scientific_status.md).
If you care about package versus legacy boundaries, read
[docs/supported_surfaces.md](docs/supported_surfaces.md).
If you want the longer project story and research motivation, read
[docs/project_context.md](docs/project_context.md).

## What This Repo Is

- A package-tested implementation of core assembly-calculus operations.
- A home for experimental language surfaces, including rule-based parsing,
  NEMO-related modules, and the emergent parser.
- A place to run optional GPU and accelerator paths without making them the
  default package contract.
- A research tree that tracks experiments, claims, and curated scientific
  questions separately from the installable API.

## What This Repo Is Not

- Not a deep-learning framework with backprop or attention.
- Not a full biological brain model.
- Not a package-level proof artifact for Turing completeness.
- Not a claim that every historical experiment at the repo root is part of the
  primary supported API.

## Current Status

- Package-backed: projection, reciprocal projection, association, merge,
  pattern completion, sequence memorization, ordered recall, Long-Range
  Inhibition (LRI), FSM/PFA helpers, and core runtime behavior.
- Research-oriented: broad language-acquisition claims, robust CIFAR-scale
  category formation, embodied multimodal grounding, and world-model style
  extensions.
- Legacy: old repo-root experiments and scripts are archived under `legacy/`.
  The root modules that remain are compatibility shims.

## Install

```bash
# Install the published package
pip install neural-assemblies

# Development install from a checkout
uv sync

# Optional GPU extras
uv sync --group gpu
```

The primary import surface is always `neural_assemblies`, including after
`pip install neural-assemblies`.

## Quick Start

```python
from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import merge, overlap, project

b = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
b.add_stimulus("s1", 80)
b.add_stimulus("s2", 80)
b.add_area("A1", n=5000, k=80, beta=0.08)
b.add_area("A2", n=5000, k=80, beta=0.08)
b.add_area("B", n=5000, k=80, beta=0.08)

a1 = project(b, "s1", "A1", rounds=8)
a2 = project(b, "s2", "A2", rounds=8)
merged = merge(b, "A1", "A2", "B", rounds=5)

print("Overlap before merge:", overlap(a1, a2))
print("Merged assembly size:", len(merged))
```

From a checkout you can also run the packaged example directly:

```bash
uv run python examples/01_basic_assembly_calculus.py
```

## Supported Surfaces

The three main surfaces are:

1. Installable package: `neural_assemblies/*`
2. Repo-root compatibility shims: `brain.py`, `parser.py`, `simulations.py`,
   `learner.py`, `image_learner.py`, `recursive_parser.py`, `brain_util.py`
3. Research and performance workflows: `research/*`, `tests/performance/*`,
   `cpp/*`

Only the first surface is the default package contract.

## Documentation

Project-wide guides:

- [docs/README.md](docs/README.md)
- [docs/api.md](docs/api.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/project_context.md](docs/project_context.md)
- [docs/scientific_status.md](docs/scientific_status.md)
- [docs/supported_surfaces.md](docs/supported_surfaces.md)

Section READMEs:

- [neural_assemblies/core/README.md](neural_assemblies/core/README.md)
- [neural_assemblies/compute/README.md](neural_assemblies/compute/README.md)
- [neural_assemblies/simulation/README.md](neural_assemblies/simulation/README.md)
- [neural_assemblies/language/README.md](neural_assemblies/language/README.md)
- [neural_assemblies/lexicon/README.md](neural_assemblies/lexicon/README.md)
- [neural_assemblies/nemo/README.md](neural_assemblies/nemo/README.md)

Research workflow:

- [research/README.md](research/README.md)
- [research/claims/index.json](research/claims/index.json)
- [research/core_questions/index.json](research/core_questions/index.json)

## Tests

Recommended package gate:

```bash
uv run pytest neural_assemblies/tests -q
```

Focused checks that matter for current repo structure:

```bash
# Legacy compatibility and archive layout
uv run pytest tests/test_legacy_root_shims.py tests/test_legacy_archived_layout.py -q

# Optional CUDA / accelerator checks
uv run pytest tests/performance/test_cuda_env.py -q
uv run pytest tests/performance/test_cuda.py::test_cuda_availability tests/performance/test_cuda.py::test_cpp_availability -q
```

Research tree validators:

```bash
uv run python research/experiments/infrastructure/validate_registry.py
uv run python research/claims/validate_index.py
uv run python research/core_questions/validate_index.py
```

## Repository Layout

```text
.
|-- neural_assemblies/        # Installable package
|-- docs/                     # Package and architecture docs
|-- examples/                 # Small runnable examples
|-- research/                 # Experiments, claims, curated questions
|-- legacy/                   # Archived root modules, scripts, artifacts
|-- tests/                    # Legacy compatibility and optional perf tests
|-- cpp/                      # Accelerator kernels and build tooling
|-- brain.py                  # Root compatibility shim
|-- parser.py                 # Root compatibility shim
|-- simulations.py            # Root compatibility shim
`-- pyproject.toml            # Package metadata
```

## Citation

If you build on this code or the underlying ideas, cite the foundational work:

- Papadimitriou et al. (2020), *Brain Computation by Assemblies of Neurons*
- Dabagia et al. (2024/2025), *Computation with Sequences of Assemblies in a
  Model of the Brain*
- Mitropolsky and Papadimitriou (2023, 2025) on the language organ and
  simulated language acquisition

The package is an implementation and experiment platform. Literature claims
should still be cited to the papers, not inferred from package installation.

## Contributing

See [docs/contributing.md](docs/contributing.md) for contributor workflow and
[docs/packaging.md](docs/packaging.md) for release steps.

## License

MIT. See [LICENSE](LICENSE).
