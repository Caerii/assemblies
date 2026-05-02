# Scientific Status

This page is the claim-strength map for the repository.

The rule is simple: package docs may say what the package implements and tests.
Research docs may say what an experiment measured. Literature claims belong to
the papers that prove or argue them.

## Package Claims

These statements are supported by code and tests under `neural_assemblies/`.

| Area | What the package can defend |
|------|-----------------------------|
| Core operations | Projection, reciprocal projection, association, merge, pattern completion, and separation are implemented and exercised by package tests. |
| Sequences and inhibition | Sequence memorization, ordered recall, Long-Range Inhibition, refracted dynamics, and related recall behavior are implemented and test-covered. |
| Automata helpers | FSM and PFA utilities exist, now backed by typed transitions and validation. They support finite-state experiments; they are not a full Turing-machine proof. |
| Engines | CPU engines are package-tested. Optional GPU paths have parity or smoke tests where the environment supports them. |
| Language modules | NEMO and emergent-parser tests cover narrow behaviors such as word-category separation, role binding, and word-order structure in controlled settings. |

Useful test files include `test_assembly_calculus.py`, `test_brain.py`,
`test_engine_parity.py`, `test_sequences.py`, `test_lri.py`, `test_fsm.py`,
`test_pfa.py`, `test_nemo_patterns_core.py`, and `test_emergent_parser.py`.

## Qualified Claims

These statements may be true in particular runs or environments, but they need
parameters, hardware, or result links.

- GPU speedups depend on hardware, CUDA stack, problem size, and engine path.
- `engine="auto"` is a heuristic based on `n_hint` and available backends.
- Biological plausibility means local learning, sparse competition, and
  assembly-style computation. It does not mean a full biological brain model.
- Language-learning results should name the exact task, corpus, curriculum, or
  synthetic setting being measured.

## Research Claims

These belong in `research/` until they are narrowed and defended:

- robust CIFAR-scale category formation
- broad language-acquisition claims
- embodied multimodal grounding
- world-model style action conditioning
- neuromorphic or large-scale accelerator claims

The research inventory lives in:

- [../research/claims/index.json](../research/claims/index.json)
- [../research/core_questions/index.json](../research/core_questions/index.json)
- [../research/experiments/](../research/experiments/)
- [../research/results/](../research/results/)

## Literature Claims

The assembly calculus, language-organ model, and sequence-computation results
come from papers. Cite those papers directly when making theoretical claims.

In particular, do not present this package as the proof of Turing completeness.
The package implements related sequence, inhibition, FSM, PFA, and simulation
utilities. The theoretical result belongs to the relevant sequence-computation
literature.

## Practical Rule

Before adding a strong sentence to the docs, ask what backs it:

- code behavior: link to tests or API docs
- measured result: link to experiment, result, or claim index entry
- theory: cite the paper
- future direction: mark it as research work
