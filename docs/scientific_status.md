# Scientific Status

This document separates **package-backed claims** from **research results,
historical artifacts, and aspirational directions**.

It exists so the public contract of `neural-assemblies` stays scientifically
honest: strong statements in the package docs should map either to code and
tests in this repo, or be explicitly labeled as research hypotheses or prior
literature.

## Package-Backed Capabilities

These are the main claims the installable package can currently defend with
code and tests in `neural_assemblies/tests/`.

- **Core assembly-calculus primitives**: projection, reciprocal projection,
  association, merge, pattern completion, and separation are implemented and
  exercised in unit/integration tests such as `test_assembly_calculus.py`,
  `test_brain.py`, and `test_engine_parity.py`.
- **Sequence and inhibition mechanisms**: sequence memorization, ordered recall,
  Long-Range Inhibition (LRI), refracted dynamics, and related recall behavior
  are implemented and test-covered in files such as `test_sequences.py`,
  `test_lri.py`, and `test_refracted.py`.
- **Finite-state utilities**: FSM and PFA helpers exist as assembly-based
  computational utilities. They support finite-state computation experiments and
  are covered by dedicated tests such as `test_fsm.py`, `test_pfa.py`, and
  `test_computation_value.py`, but they should not be conflated with a full
  Turing-machine construction.
- **Engine behavior and parity**: CPU engines are package-tested; optional GPU
  engines have parity tests and smoke tests where hardware is available (for
  example `test_backend.py`, `test_torch_parity.py`, and `test_cuda_kernels.py`).
- **NEMO-related package behaviors**: the package includes tests for narrow,
  controlled language-learning behaviors such as grounded word-category
  separation, role binding, and word-order inference / sequence structure
  (for example `test_nemo_patterns_core.py` and `test_emergent_parser.py`).

## Heuristic Or Benchmark-Dependent Claims

These are claims that may be true in some environments or experiments, but
should not be presented as universal package guarantees.

- **GPU speedups** are real targets of the engine/tooling work, but exact
  multipliers depend on hardware, CUDA stack, problem size, and the specific
  operation being benchmarked.
- **Auto engine selection** uses a simple heuristic (`n_hint` plus GPU
  availability). It is a practical default, not a proof that one engine is
  always optimal.
- **Biological plausibility** in this repo refers to local learning rules,
  sparse competition, inhibition motifs, and assembly-style computation. It is
  not a claim that the package is a full biological brain model.

## Research-Only Or Aspirational Areas

These areas remain active research topics, historical experiments, or design
directions rather than settled package claims.

- **Full Turing-completeness as a package result**: the literature motivates
  sequence/LRI-based computational expressiveness, and this repo implements
  several related primitives, but the package does not currently present a full
  Turing-machine construction as a validated software artifact. When referring
  to Turing-completeness, cite the Dabagia et al. sequence paper rather than
  treating the package itself as the proof.
- **Robust CIFAR-scale category formation**: the repo contains historical image
  learning work and tracked artifacts, but stable large-scale category
  formation is not a finished package guarantee.
- **Embodied multimodal grounding**: perception-to-assembly encoders,
  cross-turn discourse memory, and richer action-conditioned world-model style
  extensions are research directions, not package-backed features.
- **Broad language-acquisition claims**: the repo contains NEMO and emergent
  parser systems plus substantial research code, but large statements about
  full language acquisition should be tied to specific experiments and results,
  not inferred from package installation alone.

## Practical Rule

If a statement is:

- about **what the package can do today**, it should point to code and tests.
- about **measured research behavior**, it should point to a specific
  experiment/result.
- about **future capability**, it should be labeled exploratory or aspirational.
