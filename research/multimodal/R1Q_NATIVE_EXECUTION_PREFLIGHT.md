# R1q — Exact-checkout native execution preflight

**Date:** 2026-11-23  
**Branch:** `research/multimodal-continual-assemblies`  
**Status:** preflight and runner validated on a local research-code mirror; **native numerical execution blocked** because the complete `neural_assemblies` checkout was unavailable.

## Why this gate is necessary

The earlier R1b–R1p experiments and their test doubles are not a substitute for running the exact current native `Brain`. A published PyPI wheel may be months behind the branch, and installing it silently would yield results from a *different simulator*. The gate therefore checks all of the following before running native experiments:

1. Current working directory is the root of an actual Git checkout.
2. Current branch equals `research/multimodal-continual-assemblies`.
3. Scientific source under `research/multimodal` and `neural_assemblies` is clean by default.
4. `inspect.getfile(Brain)` resolves **exactly** to this checkout's `neural_assemblies/core/brain.py`, not site-packages.
5. The actual class exposes `external_drive` on `Brain.project` and required observation/engine APIs.
6. The numeric smoke creates two native models with matching seed and graph: `beta=.1` versus `beta=0`; uses teacher-drive on a k-of-n target, measures eligible synaptic changes, then verifies two unprompted reads leave every explicit area-to-area matrix byte-for-byte unchanged.
7. The run output identifies Git HEAD, package source SHA-256, Python/NumPy versions and whether the worktree was dirty.

Passing those checks establishes **API integration, teacher clamping, eligible synaptic potentiation and read immutability in this smoke**. It would NOT establish neural compositional generalization, native/Rust parity, or acceleration.

## Fail-closed runner integration

`run_local_gates.py` now includes `native-preflight` *before* native R1b–R1o experiments. If it fails, native experiments are recorded as `blocked`, with `blocked_by=native-preflight`, and **not executed**. Independent synthetic evidence replay still runs.

Running a native gate by name without the preflight is also blocked. CLI `--allow-dirty` exists only for local diagnosis and records the dirty flag; a clean checkout is the default.

## Real local validation performed

A local mirror of the R1o/R1p research files plus R1q and exact current runner source was executed:

```sh
PYTHONPATH=/mnt/data/r1q_native_gate python -m pytest -q \
  /mnt/data/r1q_native_gate/research/multimodal
```

**58 tests passed**, including R1o synthetic evidence replay, R1p composition read integrity, R1q stale-package/branch/dirty-checkout rejection and the runner preflight dependency tests.

A real **fail-closed gate invocation** in the available sandbox produced:

```text
protocol-tests         passed
native-preflight       failed  (not a Git checkout; native package unavailable)
r1o-native-composition blocked (by native-preflight)
r1o-evidence           passed
overall                 failed
exit code               1
```

This is the correct outcome. Do not relabel it a passing native experiment. It demonstrates that the system preserves evidence without manufacturing results under missing prerequisites.

The exact Git blob SHA-1 of the tested `native_execution_preflight.py` was `3ed488de5010c6f77dfa38269c5fdd75943bbfbf`, and `test_native_execution_preflight.py` was `c67347c529237edc41f8540f27e44e7273d82c6b`. The tested runner source was `b797f235d51c0db1f45c01cf6fe823aee85773cf`; its tests `54bc3f3b4985e81555937059a1ff842eaa266f4a`.

## How to cross the remaining native boundary

From a fresh, clean checkout of this precise branch on a machine with GitHub access and the project dependencies:

```sh
git clone https://github.com/Caerii/assemblies.git
cd assemblies
git switch research/multimodal-continual-assemblies
uv sync --no-group gpu
uv run python -m research.multimodal.native_execution_preflight \
  --output /tmp/r1q-native-preflight.json
uv run python -m research.multimodal.run_local_gates \
  --output /tmp/assemblies-native-r1q
```

These commands must be executed **in that checkout** (not against a stale PyPI wheel). Archive `/tmp/r1q-native-preflight.json`, `/tmp/assemblies-native-r1q/run.json`, stdout/stderr, and all failures. Then inspect the R1h teacher/no-plasticity contrast and R1o AB-versus-BA accuracy separately: a successful API smoke is necessary, not sufficient, for a learning claim.

## Outstanding risks

- The numeric smoke itself has **not run against current native `Brain`**. Any backend API incompatibility it detects is future work, not a previously passed test.
- `full_explicit_fingerprint` hashes the explicit engine's area-to-area matrices; it does not prove that all other mutable simulator state is preserved.
- The R1o composition adapter assumes an *index-aligned virtual codebook* when it injects role-area winner IDs into `LEX`. This representational bridge requires its own mechanistic validation before claiming biologically grounded cross-area composition.
- No compiled Rust kernels or performance measurements were produced in this milestone.
