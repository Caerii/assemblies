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
