# Project Context

This page keeps the project history and motivation out of the root README while
still preserving them in the repository.

It is not the strongest evidence source for capability claims. For that, use
[scientific_status.md](scientific_status.md), the tests, and the indexed
research artifacts under `research/`.

## Provenance

Alif Jakir maintains Assemblies. He began this implementation work after
encountering the assembly calculus and the language-organ model in MIT's
Projects in the Science of Intelligence course, then continued it through
research extensions and collaboration with Daniel Mitropolsky (MIT Poggio Lab).

The aim is practical: make assembly-calculus ideas runnable enough to inspect,
modify, test, and use in new experiments.

## Why This Repository Exists

The package makes assembly-calculus mechanisms concrete in code. It exposes
runtime objects, compute engines, assembly operations, sequence memory,
language-oriented modules, and simulation helpers.

The repository around the package keeps the broader research record:

- experiments that are not package guarantees
- older prototypes and artifacts that still matter historically
- accelerator work that depends on local hardware
- question, result, and claim tracking for ongoing science

Keeping those layers together makes the work easier to inspect. Keeping them
separate in the docs prevents the package from claiming more than it has
earned.

## Historical Notes

### 2024: Visual discrimination experiments

One early question was whether assembly-calculus and NEMO-style mechanisms
could move beyond the usual language and toy demonstrations into visual
discrimination.

That work included CIFAR-oriented experiments. The lesson is deliberately
qualified:

- the experiments showed feasibility in principle
- they did not establish robust per-category formation at package quality

That is why the docs describe the image-learning work as research history,
not as a supported feature.

### Rewrites and cleanup

The repo did not begin as a clean package. It accumulated through research
iterations, exploratory scripts, and repeated rewrites. The structure
separates that history into:

- `neural_assemblies/` for the installable package
- `research/` for active scientific work
- `legacy/` for archived root modules, scripts, artifacts, and MATLAB code

### 2025: Scaling and accelerator work

Scaling work, including CUDA-oriented engineering, is part of the project's
trajectory. The present claim is still bounded: accelerator paths exist and are
tested where the environment supports them, but speedups depend on hardware,
problem size, engine choice, and CUDA/PyTorch/CuPy configuration.

## Research Bet

The bet behind the repo is that assemblies may be a useful computational
substrate in their own right:

- sparse intermediate structure instead of dense hidden states everywhere
- Hebbian and local updates instead of end-to-end backprop as the only path
- explicit operations such as projection, association, merge, and recall
- interpretable units that can be inspected, reused, and composed

That is a research program, not a solved result. The package gives us machinery
for testing it.

## Audience

The repo is mainly for readers who want to work close to the mechanism:

- computational neuroscience students and researchers
- neuro-inspired ML researchers
- people comparing sparse assembly models with deep-learning systems
- contributors who want runnable experiments rather than only paper diagrams

It is less useful if you want a general ML framework, a hosted service, or a
drop-in replacement for transformer tooling.

## AI Assistance

This repository has been developed with AI assistance. That means AI helped
with coding, organization, refactoring, and documentation drafts.

It does not mean authorship or scientific judgment is vague. The research
direction, curation, and responsibility for claims remain human, led here by
Alif Jakir in the context described above.

## How To Read This Page

Use this page for history and intent.

Use these sources for claims:

- package behavior: `neural_assemblies/tests/`
- scientific boundaries: [scientific_status.md](scientific_status.md)
- maintained code boundaries: [supported_surfaces.md](supported_surfaces.md)
- research evidence: [../research/claims/index.json](../research/claims/index.json)
- curated questions: [../research/core_questions/index.json](../research/core_questions/index.json)
