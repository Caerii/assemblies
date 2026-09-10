# Verification of the isolated migration

Worktree: `assemblies-astra-audit-20260909`, based on
`3334876cf57991d27b1651c343f4640b4bcffaa6`. Tests used the existing Python
environment with editable-install finders removed and this worktree first on
`sys.path`. No dependencies were installed and no source files were edited in
the main checkout.

The final combined targeted CPU run passed **122 tests**, skipped **2**, and
emitted the expected sampled-recurrence warning. It covered:

- shared research contracts, runner, capacity configuration and cell identity;
- paired study orchestration and training fingerprints;
- public index boundaries and the controlled CPU teaching investigation;
- golden comparison, register rendering/citations and both methodology ratchets;
- example execution and the missing-real-data MNIST golden path.

The skips were optional Nemo dependencies and unavailable real MNIST data.
The earlier core/model compatibility run passed 83 tests (one deselected), covering
areas, brains, compact-index caches, the exact-engine ladder and examples.
Ruff passed on every changed/new Python file. `git diff --check` passed.

The new pull-request workflow has not run on GitHub. The full package suite,
CUDA compilation, hardware parity, historical scientific verdict recalculation,
and preregistered study runs were not performed. These CPU checks establish
instrument behavior and selected compatibility, not scientific adoption.

The CUDA prerequisite checker found the installed toolkit and Visual Studio
developer script. In the direct-interpreter shell, compiler and ninja were absent
from PATH. The checker did not import torch, compile, or probe the device.

The structural coverage manifest covers the baseline's 2,501 tracked files.
It does not claim line-by-line semantic review of the roughly 309,000 source lines.
`evidence-summary.json` and `evidence-review.tsv` retain the literal-reference
audit's unresolved work without treating candidate orphans as proven dead files.

See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for remaining migration and scientific
gates. The run records remain UNJUDGED or VOID; this migration adopts no new
assembly-calculus result.

## 2026-09-10: observation state and IR proof obligations

The new probe regressions first produced five failures: firing-count/history
leaks, refractory-state leakage, cold-population materialization, and a missing
exact-drive denominator. The targeted probe/link suite now passes 20 tests,
including explicit NumPy, nested restoration and retained buffer references.

A broader run passed 151 tests, xfailed one, and had three setup errors in
`test_erp_probe_is_alive.py`. Its ERP path calls `_reset_area_activity`, which
zeros CONTEXT's population and clears neuron mappings during a read. The stricter
cold-population gate rejects that path. This is an unresolved caller migration,
not a green full-suite result. Do not suppress it or automatically train inside
a probe. First specify whether CONTEXT is a disposable construction or a trained
population being observed, then give that choice its own protocol/control.
The initial broad collection also exposed `test_explicit_projection.py` importing
the removed root `brain` module; it was excluded from the subsequent run, not
represented as passing.

Lean 4.31.0 (installed) builds the generic schedule/refinement kernel. Local
`lake env leanchecker AssemblyIR` succeeds. Printed theorem dependencies are
empty or `propext`, with no `sorryAx`. The library builds with warnings as errors.
This establishes conditional mathematical proof rules only: no backend simulation
or Python-to-Lean extraction theorem has been implemented. The IR verification
boundary records that gap. CI now includes the source-link/probe checks and a
separate Lean job; GitHub execution remains unverified.

The operations module loses 257 lines of repeated/contradictory API prose and
historical narrative. This is a documentation-maintenance reduction, not a claim
that execution became faster or that 257 lines of duplicate algorithms vanished.
The activity repair adds explicit state declarations and stronger tests.

The final selected CPU gate (all workflow-listed tests plus operation conformance)
passed **164 tests**, skipped **1**, with expected sampled-engine warnings. This
selected pass does not include or resolve the ERP setup failures above.

## Follow-up: CONTEXT ownership in ERP probes

The three ERP setup errors above are resolved. The outer parser still constructs
its disposable context; the nested prefix probe now explicitly preserves the
existing population and ID mapping. Population/cursor resets reject read-only
scopes before mutation, including the bridge-reset route. One duplicated reset
helper and an unused category-map computation were removed.

The combined ERP/protocol/probe/specification/ratchet/register run passed **69**
tests with **1 existing xfail** (VP liveness). The narrower ERP run passed eight
and xfailed the same one. These runs overlap; do not add the counts. Protocol
outputs now include resolved settings and `existing-context-v1`. Historical
N400 numerical equivalence is neither assumed nor demonstrated; this is a
measurement-protocol correction. No GPU study, model adoption or merge occurred.

## Protocol IR schema unification

The shared wire corpus exposes nine acceptance disagreements in the old Python
validator. Both Python and Rust now consume the canonical Draft 2020-12 schema.
Rust validates on construction/deserialization and preserves the entire decoded
object instead of discarding metadata or introducing absent null properties.
Large integer identities are preserved. Nonfinite/non-JSON Python values and
existing output paths are rejected before writing.

The final focused Python run passed 46 tests (Julia deselected in that run);
the shared Rust corpus passed, including round-trip checks. `cargo package
-p assembly-ir --allow-dirty` successfully compiled the packaged crate, not just
the workspace version. A built Python wheel was inspected and contains the
canonical schema and shared corpus. These checks do not certify execution
parity between the languages or formal refinement of a backend.

Rust IR ownership moved to `neural_assemblies/ir/Cargo.toml` and `rust/lib.rs`,
remaining in the `crates/` workspace. This avoids an external schema path that
would break Cargo packaging, without creating a maintained schema copy.
The isolated worktree now has its own `.venv` from `uv sync --group dev`; no main
checkout environment was changed. The new runtime dependency is jsonschema;
existing uv lock resolutions otherwise remain unchanged. No GPU job was run.

The final workflow-listed CPU checks plus the cross-language runner tests passed
**211 tests**, with **2 skips**. Rust's single shared-corpus test checks 24 cases;
it passed. Both final distribution artifacts contain schema/corpus bytes identical
to the canonical files. Ruff and diff checks passed.

The initial wider run exposed two index-ratchet failures because a wheel build
left generated `build/lib` source copies. Both ratchets now share Git-based source
discovery (tracked plus nonignored new Python files), preserving tracked ignored
source while excluding generated copies. The dedicated source-inventory test and
both ratchets passed (12 tests); no baseline counts were raised. The final 211-test
run includes this repair and still runs with build artifacts present.


## Explicit operation recurrence follow-up

The P3 control on the previous commit failed for `ops.project(recurrent=False)`
with global recurrence and initialization normalization both enabled (20 passed,
1 failed). The operation now passes no self-edge in its False branch. All eight
flag combinations inspect backend calls and recurrent potentiation; this is a
schedule/learning regression, not a statistical study or a recovery measurement.
The focused operation/specification/example checks passed 30 tests, with 1 skip.
The complete workflow-listed CPU gate passed 214 tests, with 1 skip and the
expected sampled-connectome warning. Ruff and `git diff --check` passed.

The IR verification document links this obligation to the shared proof workflow.
No new Lean theorem or executable compiler lowering is claimed by this follow-up.
No GPU gates or historical numerical replays were run. The changed combination
needs a new protocol revision for newly measured evidence; old artifacts are
unchanged. Direct legacy `Brain.project_rounds` callers retain their old schedule.
