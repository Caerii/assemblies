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


## Shared multi-round execution and unused-fiber observation

Source baseline: `9cf2c3b`. The initial multi-round contract suite produced
19 failures and 1 pass: mismatched histories, ignored inhibition/clamps,
backend-dependent target selection, accepted empty schedules, and invalid counts.
The helper now validates/resolves one target then uses ordinary Brain projection
for each round. Legacy self-edge selection remains documented and unchanged.
Repeated engine calls now have one base implementation; copied Torch/CUDA/CuPy
loops are removed. Raw engine calls still do not apply Brain controls.

The expanded tests found stale backend winners after a public source was cleared
(2 failures before repair), and unused sampled fibers created inside read-only
(4 eager/deferred and stimulus/no-stimulus failures before repair). Both ordinary
and multi-round paths now propagate empty source activity. Eager and deferred
sampled-fiber construction honor no-recruitment. The observation tests compare
later learning against an unobserved brain and retain a frozen-plasticity
construction control.

Validation:

- Focused operation/probe tests: 67 passed before adding raw engine-loop cases.
- Final workflow-listed CPU gate: 262 passed, 1 skipped, expected sampled warning.
- Broader consolidation, inhibition and training-performance run: 92 passed,
  1 xfailed, 3 failed. All three failures reproduced after restoring all four
  then-changed executable methods from `9cf2c3b` in a separate Python process.
  This is a controlled method-level baseline, not a full historical checkout run.
- The three open failures are `test_parse_incremental_uses_category_cache` and
  `test_build_context_incremental_light` (cold ROLE_AGENT read-only probe), and
  `test_context_ring_reduces_expand_during_bridges` (compact index outside a
  mapping of length 176). No failure was skipped or reclassified as passing.
- Ruff and diff checks passed. Modified GPU modules compile syntactically;
  no GPU suite, extension build, historical numerical replay or performance
  benchmark was run. Unused GPU-module imports were removed after the CPU gate;
  that cleanup passed Ruff and Python compilation.

Inspection also found Torch's zero-drive fiber-repair branch without the
corresponding read-only guard. It remains an explicit hardware regression task;
this follow-up must not be cited as proof of universal backend probe isolation.
The unchanged generic Lean theorems are not backend preservation certificates.


## Role availability and context population preservation

Baseline `9d58bdd` had three verified parser failures. Two attempted role readout
before a sampled role population existed. The shared `probe_target_ready` predicate
now drives both readout availability and the low-level cold-probe guard. Missing
populations return None labels with `unavailable_areas`, without construction.
Readiness does not assert informative drive or successful recall. Recursive
parsing now retains inner-clause diagnostics alongside its role results.

The remaining failure came from `_reset_context_for_bridge`: requested ring
capacity replaced the actual population count while retaining a shorter neuron
ID mapping. The preserving path now retains the backend count and IDs. Three
new controls (requested capacity 0, 1, and 80 around a real population of 20)
failed before this repair and pass afterward, including a subsequent projection
and neuron-ID snapshot. Destructive reset remains separate and rejected in probes.

The same broader consolidation/inhibition/training-performance suite that had
three failures now passed 95 tests with 1 existing xfail. The focused role and
full trained reconstruction suite passed 16 tests, including positive margins,
voice invariance and event separation. These are software regressions, not new
adopted multi-seed research results. The original final CPU workflow run had
269 passes, 1 skip and two stale-ratchet failures: incremental.py's ambiguous
`.w` allowance was 7 while actual usage fell to 5. The allowance was lowered,
not waived or raised.

Further verified defect: uncached `CategoryClassificationMixin.classify_word`
resets fibers and projects with learning enabled. A small local diagnostic
(n=1000, k=20, seed=13, rounds=3) changed the queried stimulus weight sum from
2660.16748046875 to 2994.654296875. Classification precedes role reconstruction's
read-only scope; therefore the entire method is not yet isolated. The code and
semantic card now state that limit explicitly. Classifier observation is the
next repair target, followed by the already-open GPU probe/parity gates.

The readiness fixture explicitly sets norm_init=False for numpy_explicit;
Brain's default norm_init=True is not accepted by that backend's constructor.
That existing configuration mismatch remains part of the model/backend API work.

Final workflow rerun after tightening the ratchet: **271 passed, 1 skipped**.
Ruff and diff checks passed. No GPU jobs or historical evidence reruns were performed.

## Classification isolation and retained generalization regression

Classification now runs its explicit stimulus-only schedule inside `read_only`;
it neither resets core fibers nor inherits the legacy recurrence switch. Eight
controls cover direct/cached queries, inhibition, exceptions, ordering, grounding
availability, cold populations, and subsequent learning. The initial five controls
failed before the repair and passed afterward.

The broader classification/training/selected-holdout run returned **59 passed,
1 failed, 18 deselected, 1 xfailed**. The retained failure is
`TestGeneralizationMetrics.test_holdout_verb_classifies_via_grounding`: seed 74's
heldout `finds` returns PREP instead of VERB. Replacing only the classifier method
with the version at `c578f20` makes that test pass. This is a method-level baseline,
not a full historical-checkout replay.

The four-cell diagnostic recorded in the classification semantic card separates
training with old/new classification from observing with old/new classification.
The isolated observer still returns VERB on the legacy-trained brain. PREP's
population is 1965 there, versus 136 on the isolated-trained brain before any
legacy query. Hidden query-time recruitment therefore changed the trained
instrument too. This is a single-seed software diagnostic, not an adopted result
or justification for tuning a readout. The expected heldout label remains VERB;
the failing test is neither weakened nor marked xfail. Explicit population
preparation and typed score provenance remain open.

## Midspiral integration: checked domain kernel

`formal/AssemblyIR/Domain.lean` packages transition, invariant, admissibility,
its executable decision, and local preservation proof. Accepted execution is
proved equivalent to an admissible schedule in the existing interpreter, and
preserves an initially true invariant. Allocation controls prove acceptance of
`[1, 2]` and rejection of `[2, 2]` under capacity three; unchecked execution of
the latter violates the invariant. These are kernel controls, not neural proofs.

`lake build` and `lake env leanchecker AssemblyIR.Domain` passed on the pinned
Lean 4.31.0 toolchain. New theorem dependencies are either empty or `propext`;
no admitted proof or new axiom was introduced. The IR contract names the remaining
shared-program translation, source/proof identity, drift review and differential
boundary obligations. No Python/CUDA verification lowering, Dafny installation,
GPU gate or historical A1/capacity numerical replay is claimed.

The combined workflow CPU contract gate plus trained reconstruction suite passed
**289 tests, 1 skipped**, with three expected sampled-engine warnings. This
does not include or supersede the separate heldout generalization failure above.
The earlier two process outputs were unavailable after continuation; this count
comes from a fresh completed run saved in `.cache/midspiral-contract-gate.log`.
After extending source-link validation to Lean module documentation, the focused
specification-link/research-contract suite passed 25 tests. The final link audit
resolved 19 links with no errors. Ruff and `git diff --check` passed.

## Classification evidence provenance

`ClassificationEvidence` validates and retains source-specific score domains;
`classify_word_evidence` owns dispatch, with the legacy tuple exposed as a view.
Bootstrap fusion and decomposition now consume the typed result. Fallback
distributional scores retain their values and source, and cannot be credited
as correct neural readout. The erroneous weak-readout diagnostic expression is
replaced by an explicit maximum with a default.

Three new controls failed with the `c43a988` inference module loaded in an
isolated Python process and pass with the repaired module. This is a module
comparison, not a historical study replay. Two additional controls exposed AUX
leaking into POS scores in grounded/low-confidence distributional branches;
the high-confidence ungrounded branch was already correct. All now use the
existing function-subcategory-to-core mapping before score accumulation.

Focused evidence/observation/source-link checks: **29 passed**. The complete
workflow-listed CPU contract gate: **295 passed, 1 skipped**. Ruff passed for
the changed runtime and test modules. No fusion weights or neural readout formula
were tuned to the heldout fixture. This change does not close the outstanding
historical replay, GPU or concrete backend-proof gates.

The trained bootstrap/performance run initially returned **61 passed, 1 failed,
1 xfailed**. The failing SENTENCES-depth fork lacked `recurrent_projection`.
Replacing classification/distributional methods and the inference module with
their `c43a988` versions reproduced the same AttributeError. The independent
grounded-verb check was also rerun and still fails with PREP instead of VERB.

## Clone preservation

Tracing the fork failure found two hand-maintained copy implementations. Brain
clone omitted recurrence, normalization, mixed-connectome RNG and diagnostics,
and discarded its secondary engine. Sparse-engine clone reconstructed default
state, including deferred scaling=False even when the source had it enabled.
Four constructed controls failed before repair. Both clone methods now use
graph-preserving deepcopy, retaining internal aliases and mutable independence.
This removes the parallel lists of fields that had to track each new feature.

Focused clone/inhibition controls passed 23 tests; the added mixed explicit/
sampled case preserves both engines and their next projection. The final
workflow-listed CPU contract gate passed **300 tests, 1 skipped**, with two
sampled-engine warnings. Runtime/test Ruff and diff checks passed. No speedup
or GPU clone preservation is claimed; clone performance remains unbenchmarked.
Parser-level shallow copies and selective lexicon sharing still need their own
ownership review.

The subsequent clone + formerly failing SENTENCES-bootstrap + checkpoint-fork
suite passed **18 tests** in 171.06 seconds. This verifies the fork AttributeError
is repaired and the existing bootstrap floor passes; it does not supersede the
separate grounded-verb failure. No test expectations were changed.

## Parser graph ownership and calibration publication

Parser forks previously used a shallow copy plus selective field lists. Ordinary
forks shared lexicons, and both wobbly settings shared bootstrap categories,
function metadata and nested exposure logs. Twelve controls initially failed
(three already-isolated wobbly lexicon controls passed), also exposing suppressed
snapshot errors and a cache fallback to live state. Forking now deep-copies the
whole parser graph, retaining internal aliases without sharing mutable state
with the source. The legacy post-copy sentence preparation remains explicit in
the contract; it still resets CONTEXT construction counts/IDs on the copy.

The pristine snapshot helper raises with its original exception cause, and
`ParserCache.fork` refuses a missing snapshot. Calibration previously modified
the live cache object and left the pristine snapshot uncalibrated. Four controls
failed on contaminated inputs, partial mutation during a failing calibration,
failure to snapshot, and mismatched live/pristine calibration. Calibration now
works on an isolated pristine copy and publishes both objects after success.
This is failure isolation in the existing single-threaded cache, not a new
thread-safety guarantee.

Focused parser-fork/source-link checks: **26 passed**. Runtime/test Ruff passed.
No GPU work, scientific rerun, or modification of the heldout-verb expected
label was performed. That separate neural-population/readout regression remains
outside the ownership repair.

Final workflow-listed CPU contract gate: **319 passed, 1 skipped**. The trained
checkpoint/shared-cache isolation/bootstrap suite passed **14 tests**. A separate
new real calibration-to-cache-fork check passed, confirming empirical thresholds
reach the fork as an independent copy. No metric threshold assertions were
weakened. Runtime/test Ruff and `git diff --check` passed.

## Resolved cache requests and calibration variants

Five controls failed before the repair: engine/environment changes reused live
entries, disk reuse ignored fast-training mode, default/explicit holdouts were
inconsistently keyed, and fast/full calibration shared one Boolean state. Engine
resolution now occurs before lookup and the same request values reach training.
Disk entries must carry matching request metadata; an unmatched entry retrains.
Calibration variants derive independently from a retained uncalibrated snapshot,
so switching modes does not retrain the backbone or reuse the wrong thresholds.

Holdout resolution now has one helper for the cache, trainer and dialogue path:
None selects defaults and an empty collection stays empty. The dialogue helper's
empty-set control also failed before repair. The earlier publication test now
selects the calibrated entry explicitly because the cache retains the separate
training entry; its matching live/pristine assertions remain intact.

Focused cache/fork checks passed 27 tests; the workflow CPU contract gate passed
**327 tests, 1 skipped** before the final environment-digest follow-up. The real
cache calibration integration explicitly requests numpy_sparse and checks the
constructed backend. These are software contracts, not new generalization data.

The trained checkpoint/cache-isolation/bootstrap integration suite passed
**15 tests**. The final environment signature stores SHA-256 digests of exact
values, preserving case distinctions without persisting raw values. After that
follow-up, cache/fork/source-link checks passed **35 tests**, including disk reuse
and mode separation. Final Ruff and diff checks passed. Full target/dependency
identity, scientific calibration validity, and the heldout-verb regression remain
open; no GPU jobs or historical numerical replays were run.

## Checkpoint storage failure and publication

Storage controls exposed EOF/unsupported-protocol exceptions escaping cache load,
failed serialization/replacement leaving temporary files, and both concurrent
writers opening the same temporary pathname. Each writer now owns a unique file
in the destination directory, flushes the completed serialization, closes it,
then replaces the destination. Its temporary file is cleaned on failure.
Truncated/incompatible trusted local pickle caches produce misses.

Focused storage/cache/source-link checks initially passed 22 tests. The first
full gate returned 333 passed, 1 failed, 1 skipped: Windows denied one concurrent
replacement with WinError 5. A synthetic transient-sharing control reproduced
the missing retry. Windows errors 5/32/33 now receive bounded backoff (six
attempts, at most 310 ms waiting); permanent denial still propagates with the
old file preserved. Storage/cache checks then passed 17 tests. Twenty actual
synchronized two-writer iterations passed; an earlier attempt to repeat node IDs
through pytest collected only one test and is not counted as the stress run.

The real trained checkpoint roundtrip and cache-calibration integration both
passed. These checks concern local storage behavior, not scientific evidence
adoption, pickle security, crash durability across all filesystems, or GPU parity.
Final workflow-listed CPU contract rerun: **336 passed, 1 skipped**. The grounded
verb/readout regression and broader IR/backend obligations remain open.

### Mixed index boundary and checked IR entry (2026-09-10)

Mixed sparse-to-explicit drive now validates compact positions against the
materialized mapping, translates through the shared index helper, and validates
stable IDs against population and fiber rows. Invalid indices no longer fall
back to identity or disappear. Area assignment validates before uint32 conversion
and mutation, preventing fractional truncation and overflow. Fifteen controls
include wrong-space trap rows and failed-assignment state preservation. GPU array
validation/performance remains untested; mutable buffers, `.w` ownership and the
reinforcement write contract remain open.

The legacy explicit-projection test used a removed top-level import and a
potentially empty private-winner loop. It now uses the package API, asserts a
nonempty selected cap, checks the complete expected weight matrix and runs a
beta-zero control. The CPU workflow includes both boundary test files.

After reading both Midspiral articles, the existing IR integration was extended
with `Domain.checkedExecute`: check the initial invariant, then every intermediate
precondition using the existing interpreter. Lean proves exact acceptance and
invariant preservation without an initial-state premise from the caller. Controls
include an invalid initial state with an empty schedule that the lower-level
interpreter accepts. `lake build` and `lake env leanchecker AssemblyIR.Domain`
passed; new theorem axioms contain only `propext`. This is a reusable formal
kernel, not a Python/CUDA translation proof or an installed LemmaScript bridge.

Final workflow-listed CPU gate: **355 passed, 1 skipped**, including the two
ratchets and register-rendering checks. Separate IR/explicit/index integration:
**24 passed, 1 skipped**. Ruff and diff checks passed. This is not the full package
suite; the previously identified grounded-verb/readout regression remains open.
GPU gates and historical numerical migration replays have not been performed.

### Restricted executable projection IR (2026-09-10)

Added `ExplicitRound`, an immutable, strictly decoded instruction for a standalone
NumPy explicit CPU engine. It supports one target, ordered distinct area sources,
explicit plasticity and optional additive drive. It validates relevant state and
rejects unsupported profiles/features before dispatching to the existing kernel;
it does not duplicate winner selection or Hebbian arithmetic. The historical
projection payload is not automatically executable.

Twenty-two new cases cover hand-computed caps and full weight updates, disabled
learning, dead-fiber sensitivity, sequential recurrence, ties/clipping, copied
configuration, malformed wire/state rejection without mutation, and direct-kernel
agreement. Focused IR/wire/source checks: 60 passed. Final workflow-listed CPU
gate: 377 passed, 1 skipped, including source-link, methodology/index ratchets and
register checks. Ruff and diff checks passed.

Limits: no Lean numerical simulation proof, Rust execution consumer, Brain-facade
lowering, GPU gate, or historical evidence replay. Direct legacy engine calls
still bypass this stricter boundary. Multi-step rollback and absence of floating
overflow are not guaranteed. The earlier grounded-verb regression remains open.
This is the first restricted execution lowering, not completed IR unification.

### Shared explicit inputs and Brain drive forwarding (2026-09-10)

The direct-engine controls initially returned 19 failed, 4 passed: wrong-shaped
drive was ignored, clamped targets skipped validation, and winner assignment
could truncate fractions or accept invalid/duplicate IDs. Validation now belongs
to the engine, including rechecking mutable winners before projection. The IR
adapter delegates these checks instead of maintaining a second implementation.
Selected IDs are checked before learning; silent filtering in Hebbian updates
was replaced with validated indexing.

Four further controls exposed Brain dropping explicit drive on its primary dense
engine and on batched targets, plus ignored unsupported/unscheduled requests.
Brain now forwards drive to either dense engine, excludes driven targets from the
batch interface that cannot carry drive, and rejects unsupported/unscheduled
requests. The duplicated dispatch branches were collapsed into one call.

Focused explicit/IR/mixed-drive/schedule suite: 126 passed. The earlier full gate
before the Brain forwarding fix passed 400 tests with 1 skipped. Final workflow-listed CPU gate: **404 passed, 1 skipped**. Ruff and diff
checks passed. These CPU checks do not prove numerical overflow
safety, all-state rollback, GPU behavior, or the Lean/backend relation. The known
grounded-verb regression and historical evidence replays remain open.

### Brain execution of the explicit IR (2026-09-10)

`ExplicitRound.execute_on_brain` shares profile validation with standalone
execution and lowers into ordinary Brain projection. It checks descriptor/engine
ownership and unsupported controls, scopes and restores the learning flag, and
returns detached winners. Source synchronization, histories, counts and activation
remain owned by Brain. Primary dense and auxiliary dense arrangements are tested.

The initial adapter run passed 14 cases and failed both external-only cases,
exposing absent scheduling support. Brain now schedules explicitly driven areas
without inventing source edges, saves their winner histories, and rejects unknown
or inhibited targets. The extended focused run passed 113 tests.

The workflow-listed CPU run returned 425 passed, 1 failed, 1 skipped. The failure
was the index-space ratchet on a new test's raw set comparison of Area.winners.
The assertion now uses `diagnostics.read_assembly`; the baseline was not relaxed.
The final targeted integration/index-ratchet/source-link run passed 37 tests,
including all 22 new Brain IR cases. No runtime code changed after the full run.
Ruff and diff checks passed. This is not reported as a second full-suite rerun.

The lowering is tested, not formally certified. No GPU gate, mixed-engine IR,
whole-program rollback, or historical evidence replay is claimed. The known
grounded-verb regression and the broader semantic unification remain open.

### Raw winner injection and source synchronization (2026-09-10)

The initial cross-engine controls returned 51 failed, 9 passed. Brain cast raw
winner buffers before validating them and could mutate the first injected area
before rejecting a later input. Sampled/exact setters differed from explicit
setter validation. `validated_indices` now has an explicit uniqueness option;
Area and all three NumPy setters use it before mutation. Brain validates the
complete raw injection and route, or complete source-sync map, before applying
winner updates. The typed-space/materialization-watermark obligation remains
open; numerical range validity alone cannot establish index ownership.

Focused injection/IR/explicit checks: 109 passed. CPU workflow gate before adding
the exact-engine ladder: 486 passed, 1 skipped. Broader exact/materialization/index
integration first returned 4 failed, 67 passed, 1 xfailed. Its parity fixture
seeded only Brain's private engine; public-source synchronization then cleared it.
The fixture now seeds the public Area. Numerical drive and multi-round parity
assertions are unchanged. Rerun: 71 passed, 1 xfailed. The existing xfail is retained.
The exact-engine ladder is now included in CPU CI. No second combined run after
that workflow-list addition is claimed. Ruff and diff checks passed.

These CPU controls do not certify GPU setters, sampled index ownership, full
failure rollback or historical scientific evidence. The grounded-verb regression,
formal backend bridges and complete experiment migration remain open.

### Supervised reinforcement coordinates and mutation extent (2026-09-10)

Initial controls: 17 failed, 3 passed. Dense reinforcement indexed sampled source
positions as stable rows, accepted malformed posts when learning was disabled,
and clipped unrelated matrix entries. Source coordinate resolution is now shared
with mixed drive. Stable post IDs and beta/clip/storage are validated. Zero-to-one
seeding remains explicit supervised edge creation; beta-zero/global-disable/fiber
mask nulls do not seed. Only the selected block is updated and clipped. Compact,
virtual and GPU storage is rejected until its writable coordinate contract exists.

The patch teacher had an additional coordinate bug: it copied stable snapshot IDs
back into compact winners. After correcting a test import, the constructed sampled
case reproduced that failure; removing the redundant source rewrite fixed it.
Unbounded arithmetic overflow is also rejected without committing the block.

Focused reinforcement/mixed-drive/IR checks: 40 passed, 1 skipped before the final
three controls and teacher repair. Caller/mixed-drive/reinforcement integration:
47 passed, 1 skipped. The MNIST golden skip does not establish data-based accuracy;
these are software checks, not a replay of adopted scientific evidence.
Ruff and diff checks passed. Final workflow-listed CPU gate: **555 passed,
1 skipped**. GPU gates, historical replays, the grounded-verb regression and
formal backend proofs remain open.

### Shared fiber learning scopes (2026-09-10)

Inspection found Brain's fiber masks were honored by reinforcement but not
ordinary projection. The initial control run returned 3 failed, 2 passed: both
dense arrangements learned on the masked fiber and the sampled backend silently
ignored its unsupported request. ComputeEngine now owns a nested, exception-safe
suppression scope with explicit backend capability. Dense area and stimulus
plasticity loops consume the predicate without removing drive. Brain scopes
only active masked routes before dispatch. Unsupported learning-enabled backend
requests raise; globally frozen Brain calls need no per-fiber scope.

Additional controls exposed two API bypasses (4 failed, 7 passed): reinforcement
ignored the new scope and the IR promised learning while the scope suppressed it.
Reinforcement now checks scoped/global engine controls too; IR validation rejects
contradictory learning intent. Final focused checks: 107 passed. Controls include
full-matrix preservation, retained drive, learning on another fiber, re-enabling,
nested scope restoration, exception cleanup and stimulus masks.

Ruff and diff checks passed. Only dense NumPy projection opts into the new
capability. Sampled, exact and GPU implementations remain work to do; this scope
does not promise recruitment isolation, normalization preservation on those
backends, thread safety, or a formal proof. Historical replays and the known
grounded-verb regression remain open.

Final workflow-listed CPU gate: **566 passed, 1 skipped**.

### Exact-engine fiber learning masks (2026-09-10)

`numpy_exact` now implements the shared scoped learning predicate in its single
plasticity function, covering ordinary and fixed-target projection, area fibers
and stimulus fibers. Drive evaluation, beta, normalization and stored learned
weights are unchanged; only new potentiation counts/outer products are blocked.

All six new cases initially failed because this backend correctly rejected the
unsupported mask. After implementation, the exact ladder plus dense mask checks
passed 63 tests. Additional constructed beta-zero controls show why temporarily
zeroing beta would be incorrect: it removes previously learned activation. The
six focused cases pass with those controls, at both normalization settings and
with fixed/unfixed targets. Re-enabling restores learning, and an unmasked fiber
continues learning while the masked effective matrix stays unchanged.

Ruff and diff checks passed. Sampled/GPU mask support and formal backend proofs
remain open. Fixed-target behavior remains engine-specific; this change does not
claim dense and exact engines have identical clamp semantics or certify a
historical scientific result.

Final workflow-listed CPU gate: **572 passed, 1 skipped**.
