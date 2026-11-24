# R1s — Execute exact native NumPy projection method in isolation

**Research branch:** `research/multimodal-continual-assemblies`  
**Status:** isolated actual-source arithmetic exercised; full native `Brain` execution **still blocked**  
**Date:** 2026-11-23

## Why this is a substantive step

The ChatGPT container lacks the native `neural_assemblies` package and
cannot resolve `github.com` or PyPI to clone/install the repo.
The GitHub connector can read the actual repository, but cannot mount a
full Python checkout into this container. The native package contains
hundreds of modules, so pretending a handful of research files constitute
a full checkout would invalidate scientific claims.

However, the numerical kernel of the native explicit backend can be
tested **independently of the rest of Brain**. From the current branch,
the exact `NumpyExplicitEngine.project_into` method in
`neural_assemblies/core/numpy_engine/_explicit.py` was extracted.

Upstream Git blob SHA: `58a2c0929974752fc7e5ae92c69ea67097c02976`.
The method body is 4,893 characters, followed by one blank separator
newline in the upstream class. Its FNV-1a 64-bit fingerprint of the
method text (excluding that separator) is `3368f708de449626`.
The local extracted body matched that text fingerprint exactly.

The new `test_native_explicit_component.py` in this branch parses the
upstream file's AST and compiles **the actual checked-out method**
without reimplementing its numerical body. Its surrounding interface
is an intentionally controlled test double, not the full native engine.

## Executed local numeric checks

With CPython, NumPy, and pytest, 19 checks passed locally:

- Six deterministic connectome seeds: 0, 1, 2, 17, 42, 111;
- Three plasticity strengths: beta=0, beta=0.1, beta=0.5;
- k=6 winners out of n=48, Bernoulli initial connection density 0.3;
- The selected teacher target must win, for three supervised rounds;
- Independent `verify_exact_hebbian_write` checks exact float32
  multiplication/clipping on active source-to-target edges and rejects
  unrelated matrix changes;
- Two subsequent nonplastic projections preserve learned weights;
- An additional clipping check exercises ceiling=1.2.

The local source fixture preserved the method body from the current Git
source with an exact text fingerprint. A second local run extracted the
method by its class name through the same AST mechanism, also 19 passed.

**Evidence limitation:** This runs repository-authored numerical method
code with fake surrounding services: input validation, area registration,
connectome creation, competition, and `Brain.read_only` are **not**
exercised. It validates component arithmetic only, NOT full native
execution, native R1i matched controls, native AB/BA composition,
or a measurable zero-shot compositional effect.

## How to run the source-owning version

In an actual clean checkout of this branch:

```sh
python -m pytest -q research/multimodal/test_native_explicit_component.py
python -m research.multimodal.native_execution_preflight \
  --output /tmp/r1s-native-preflight.json
python -m research.multimodal.run_local_gates \
  --output /tmp/r1s-complete-native-gates
```

The first test can run with only NumPy/pytest and the source tree;
the latter two require the complete native package and dependencies.

## Remaining hard blocker and next acceptance criteria

A true end-to-end result requires access to the full checked-out
`Caerii/assemblies` repository as an executable filesystem.
The native preflight must pass for the current branch, then all R1i
teacher/no-plasticity/shuffled-teacher arms must meet the exact synaptic
law. Only then execute `native_affine_chain` to measure AB vs BA
with no composed-target supervision and a predeclared baseline.

A GitHub API source read is not an executable checkout. This document
exists to keep that distinction explicit while capturing real,
reproducible component-level arithmetic evidence.
