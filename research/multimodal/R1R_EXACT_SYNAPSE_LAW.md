# R1r — Exact native Hebbian write law and full-graph certification

**Branch:** `research/multimodal-continual-assemblies`  
**Date:** 2026-11-23  
**Status:** NumPy oracle and mock-level tests executed; native `Brain` numeric execution pending.

## Objective

Close the distinction between "**some** native weights changed" and
"**precisely the correct** synapses changed according to the declared
assembly-calculus learning law."

The previous R1q preflight merely required the teacher to select its
intended target and at least one active synapse to change under
nonzero plasticity. That could pass a model that mutated unrelated
fibers, used the wrong beta, learned off-target edges or failed to clip.

## Audited native numerical contract

The current `neural_assemblies/core/numpy_engine/_explicit.py`
`NumpyExplicitEngine.project_into` multiplies each coactive synapse by
`1+beta` for actual source winners and actual target winners, with
`float32` arithmetic and optional `w_max` clipping. External teacher
drive contributes to input **before** winner selection. A teacher
signal therefore does not independently guarantee a valid write.

The R1r independent oracle `exact_synapse_law.py`:

1. Requires all nine directed dense area-to-area fibers for the
   R1h LEX/SUBJ/OBJ experiment to exist in a single explicit engine.
2. Takes a complete pretraining snapshot of those fibers.
3. Records the **actual target winners on every presentation**, not
   merely the intended teacher target or the last-round winners.
4. Applies expected in-place float32 multiplicative strengthening and
   clipping *only* to `LEX -> taught_role` coactive entries for each round.
5. Requires exact equality of the full before/after graph with the
   predicted graph, including off-target, recurrent and reverse paths.
6. Records the graph digests, number of changed edges, presentation count,
   teacher overlap by round, and categorical name of the exact law.
7. Rejects mismatched dtype, nonfinite values, missing connections,
   mixed engine ownership, invalid neuron IDs or unexpected mutations.

This checker is intentionally quadratic in each area's neuron count
and makes full copies. It is intended for correctness validation at
modest `n`, not for production fast inference.

## Integration

- `native_teacher_drive.py` calls this checker after every supervised
  training item, for both beta=0 and beta>0 arms, and preserves the exact
  result in `clamp_diagnostics`.
- `native_execution_preflight.py` now explicitly requires that
  diagnostic to exist, match the named law, report three presentation
  rounds, and prove the expected zero/nonzero change classification.
- `test_native_teacher_drive.py` was extended so a fake shared explicit
  engine must show the exact oracle, correct changed-edge count and
  per-presentation teacher-target overlap for beta=0 and beta=.1.
- `test_r1r_preflight_evidence.py` injects explicitly FAKE adapters
  to verify missing and incorrect numerical evidence blocks preflight.

The only source changes to the native engine itself are **none**:
the gate is a separate experimental oracle, not a change to its physics.

## Evidence executed locally

A local workspace reconstructed from the R1q research ZIP plus the
new oracle and tests was executed with CPython 3.13 / NumPy / pytest.

```sh
PYTHONPATH=. pytest -q research/multimodal/test_exact_synapse_law.py
# 15 passed

PYTHONPATH=. pytest -q research/multimodal/test_exact_synapse_law.py \
  research/multimodal/test_r1r_preflight_evidence.py \
  research/multimodal/test_native_execution_preflight.py
# 25 passed

PYTHONPATH=. pytest -q research/multimodal/test_*.py
# 77 passed
```

The first 15 tests include exact two-round potentiation, beta=0,
float32 saturation at `w_max`, distinct wrong-fiber/off-target
mutations, absent reverse fibers, mixed engine ownership and bad IDs.

**Critical boundary:** This 77-test workspace does not contain the
current complete native `neural_assemblies` source. The
`NativeTeacherDriven` integration and native preflight are committed
but NOT numerically validated against a complete checkout in this
sandbox. The GitHub branch contains more tests than the local R1q ZIP,
so 77 is NOT the full branch test count.

The GitHub blob SHAs for the locally verified R1r files were checked
against `git hash-object` on the local copies:

- `exact_synapse_law.py`: `4ac3b0e08e60d08a271816acf8854a7cd9b3d10d`
- `test_exact_synapse_law.py`: `0c9cfb0e2ca8e7df08882b32cd7d7feae9f41c36`
- `test_r1r_preflight_evidence.py`: `ea5e038ef57e846d428ae4a140dcaf79b5cc7a07`

No measured native retention, zero-shot composition, Rust parity or
FPGA throughput claims follow from this result.

## Next decisive run

With an actual complete checkout of the named branch and dependencies:

```sh
git switch research/multimodal-continual-assemblies
uv sync --no-group gpu
uv run python -m pytest -q research/multimodal
uv run python -m research.multimodal.native_execution_preflight \
  --output /tmp/r1r-exact-native-preflight.json
uv run python -m research.multimodal.run_local_gates \
  --output /tmp/r1r-native-all-gates
```

**Required acceptance:** The native engine must produce an exact-law
certificate for each R1h write, change no unexpected directed edges,
preserve the full graph across each no-teacher read, and pass the matched
beta=0/shuffled-teacher controls. Only then measure native AB-vs-BA
composition and compare against the learned affine-rule baseline.

On failure retain seed, commit SHA, all per-step winner sets, the
expected and actual edge mismatch, and the full runner's stdout/stderr.
No GitHub Actions dependency.
