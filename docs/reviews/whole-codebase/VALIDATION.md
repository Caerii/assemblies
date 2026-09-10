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

### Sampled-engine learning masks and held-target synchronization (2026-09-10)

The first seven controls failed: six reached unsupported masks, and the held-target
case exposed a Brain synchronization defect before it could test learning. Brain
synced only source winners, so fixing a public target could use an empty or stale
engine cap. Held targets now join the validated synchronization set. The shared
sampled learning function also checks the global engine learning flag, which the
fixed-target caller did not check.

`numpy_sparse` now filters masked sources at its shared Hebbian/scaling boundary,
with a separate predicate at explicit-source bootstrap. Deferred scaling retains
masked pending work instead of applying or discarding it, then processes it after
the scope ends. Recruitment and area-level refraction remain separate effects;
a masked stimulus still recruits, so a mask is not a read-only probe.

Focused masks/scaling/round-schedule checks: 76 passed. Controls cover ordinary,
fixed, compiled and explicit-bootstrap paths; full masked-fiber equality on a
materialized population; another fiber changing; scaling enabled/disabled;
deferred queue retention; recruitment; and replacing a stale fixed-target cap.
The unsupported-capability control now explicitly disables the capability on its
test engine rather than assuming the sampled implementation remains unsupported.
Ruff and diff checks passed.

GPU-native masks, formal backend proofs and historical evidence replays remain
open. No sampled sequence result is adopted from these tests. The broader
unification and known grounded-verb/readout regression remain unfinished.

Final workflow-listed CPU gate: **583 passed, 1 skipped**.

### Lean learning-mask frame contract (2026-09-10)

Added `formal/AssemblyIR/Learning.lean`, imported by the normal Lean build. The
pure model separates per-fiber learned values from activity and reuses the existing
schedule interpreter. It proves blocked-value preservation, exact permitted
writes, preservation of proposed activity, the frame across arbitrary finite
schedules, and monotonic protection under nested mask union. Concrete controls
show why bypassing suppression, zeroing input weights, and freezing all activity
are different behaviors.

`lake build` and `lake env leanchecker AssemblyIR.Learning` passed. Printed theorem
dependencies contain only `propext`, with no axioms for the activity projection.
Source-link and runtime mask checks: 36 passed. Ruff and diff checks passed. No
runtime implementation changed, so the complete CPU suite was not repeated.

The source-linked IR documentation now distinguishes this proved abstract frame
from unproved concrete backend simulation. In particular, physical sampled-fiber
growth is not an invariant learned value and cannot be silently substituted into
the theorem. Exception restoration remains a runtime-tested property, not a theorem
of this pure kernel. GPU gates, historical replays, complete IR lowering and the
known grounded-verb regression remain open.


## Runner specification and build provenance (2026-09-10)

The runner source inventory previously omitted Lean source/toolchain manifests,
IR JSON contracts, CUDA/C headers and workflow configuration. Constructed
mutation tests reproduced 9 failures (20 existing/control passes), including a
completed artifact after a Lean file changed during measurement.

`source-inputs-v2` now fingerprints these inputs, records the policy name and
includes it in the digest. The negative now requires a retained failure record
and no completed observations; emitting results JSON leaves identity unchanged.
The source links to the explicit inventory contract in `research/README.md`.
This is repository-content provenance, not environment/binary identity or proof
of backend refinement. Historical artifacts retain their old inventory meaning.

Runner checks: 29 passed. Complete workflow CPU contract gate: 594 passed,
1 skipped, 2 expected sampled-engine warnings (79.68 s), recorded locally in
`.cache/runner-source-gate.log`. Ruff and `git diff --check` passed. No GPU run,
historical replay, full package suite, or scientific adoption is claimed. Runtime
environment capture, remaining runner migrations and the grounded-verb regression
remain open.


## Shared environment identity (2026-09-10)

The training cache omitted `NEURAL_ASSEMBLIES_*`, despite that namespace controlling
Rust seeding and stable-candidate behavior. The runner recorded no environment
identity and accepted observations after repository switches changed. Initial
checks reproduced 11 failures, with 39 passes.

`core/environment.py` now supplies exact-value SHA-256 fingerprints for all three
repository namespaces to both consumers. The cache preserves its explicit location
and separately keyed calibration exclusions. New run schema 2 requires the shared
policy/digest record, checks it after measurement, and retains failure on change.
The evidence validator rejects malformed records and continues to read historical
schema 1 without inventing missing environment identity.

Acceptance exercises actual memory and disk cache misses after backend switches,
cache exclusions, raw-value omission, failed-run retention, unrelated-variable
noninterference, malformed evidence and historical compatibility. The shared helper
lives within the cache's recursively fingerprinted core package; a coverage check
ensures changes to its policy also invalidate cached training sources.

CPU workflow gate: 609 passed, 1 skipped, 2 expected sampled-engine warnings,
85.54 seconds (`.cache/environment-gate.log`). After relocating the helper into
core and adding the source-coverage check, runner/cache/training-fingerprint checks
passed 61 tests with one expected sampled-engine warning. Ruff and diff checks
passed. No numerical kernel changed and no GPU gate or historical replay ran.

This is current-process repository-environment equality, not a resolved model
configuration, installed-binary identity, or a reconstruction recipe. Imported
modules can retain earlier settings; external CUDA/thread settings and transient
changes reverted between checks are outside this contract. Those gaps and the
known grounded-verb regression remain open.


## Shared numerical preflight before IR lowering (2026-09-10)

`ExplicitRound.validate` previously accepted malformed drive dimensions,
float32-overflow drives and corrupted mutable winner buffers. Execution eventually
rejected them, but Brain lowering had already synchronized public source caps into
the engine. Eight constructed failures reproduced this gap (44 passes): deliberately
unequal facade/engine caps changed despite rejected instructions on both primary
and auxiliary dense engines.

The explicit engine's existing numerical checks are now extracted into
`validate_projection_inputs`, reused by direct execution and IR validation. No
winner-selection or learning arithmetic is duplicated or changed. The source-linked
input and Brain-lowering contracts distinguish this preflight from execution-time
rollback and whole-program failure atomicity.

Focused input/IR checks: 79 passed. Complete workflow CPU contract gate: 618 passed,
1 skipped, 2 expected sampled-engine warnings in 66.26 seconds, with local log
`.cache/ir-preflight-gate.log`. Ruff and diff checks passed. No GPU gate, registered
scientific replay, complete package pass, or Lean backend-simulation proof is claimed.
Resolved model semantics and the known grounded-verb regression remain open.


## Supplied-engine identity consistency (2026-09-10)

Brain previously ignored conflicts between its p/seed/clip arguments and an
already-constructed engine. Auxiliary dense engines and mixed fibers could then
use a different identity from the primary engine. After correcting a test fixture
that assumed every engine had `clone`, all nine intended mismatch controls
reproduced failure across the three NumPy engines (12 passes).

ComputeEngine now supplies a nonmutating `validate_brain_identity` boundary,
called before Brain adopts or changes a supplied engine. Missing identity also
raises. Matching cases exercise both finite and absent clips and the identity of
auxiliary engines. The source-linked contract recommends one shared parameter
mapping rather than two copies of the same configuration values.

Torch now retains the constructor seed for the identity check. Two existing
dense-drive GPU test callers explicitly pass the engine's identity and normalization
setting to Brain; no numerical assertions were relaxed. Four pre-existing unused
imports/assignment lint findings in that touched test file were removed while
preserving its projection call. GPU execution remains unverified and requires
Claude's CUDA gates before merge.

Focused checks: 25 passed. CPU workflow gate: 634 passed, 1 skipped, two expected
sampled-engine warnings in 90.68 seconds (`.cache/engine-identity-gate.log`). Ruff
and diff checks passed. This is an intentional fail-fast compatibility change for
mismatched preconstructed-engine calls. It does not resolve normalization/scaling,
other model semantics, post-construction mutation or prepopulated-engine adoption.
Historical replay, complete package validation and the grounded-verb regression
remain open.


## Immutable shared homeostasis configuration (2026-09-10)

Brain retained mutable caller scopes while NumPy/Torch froze their own copies,
and string scopes became character sets. Supplied engines could disagree with
Brain on normalization, scope or deferral while Brain's recurrence schedule used
its own values. Twenty-one controls reproduced these discrepancies (25 passes).

The public, lazily exported `HomeostasisConfig` now owns boolean validation,
canonical immutable named scopes and the requirement that deferred scaling have
an enabled scope. Brain, sampled NumPy and Torch consume it; exact NumPy also
uses its normalization validation. Supplied-engine adoption compares canonical
homeostasis before changing the engine. `as_kwargs` supplies one configuration
to both constructors. Backend capability restrictions remain in force.

Controls cover malformed settings, frozen/copy semantics, equivalent scope
spellings, matching and conflicting supplied engines, empty scope equivalence,
and public export/type-checker consistency. Existing arithmetic, refraction,
scaling and learning-mask controls remain unchanged. No GPU execution is claimed.

CPU workflow gate plus homeostasis/scoped-scaling suites: 670 passed, 3 skipped,
3 expected sampled-engine warnings in 77.32 seconds, logged in
`.cache/homeostasis-config-gate.log`. After adding the lazy public export and its
documentation, 68 export/boundary/specification checks passed. Ruff and diff checks
passed. The immutable object does not freeze subsequent direct assignments to
legacy engine fields or establish cross-backend arithmetic equivalence. Remaining
model switches, GPU gates, historical replays and the grounded-verb regression
remain open.


## Shared homeostasis wire schema and Rust bridge (2026-09-10)

HomeostasisConfig now round-trips through a strict `homeostasis-v1` document,
validated with the packaged JSON schema. Python schema validation and Rust schema
compilation reuse the protocol bridge's machinery; both languages consume one
19-case configuration corpus. Rust exposes an opaque validated document rather
than claiming a numerical implementation.

The wire rejects missing/unknown fields, wrong versions, nonboolean flags,
malformed/empty/duplicate scopes and inactive deferral. It accepts unordered unique
scopes and serializes them sorted. Python constructor conveniences remain distinct:
empty collections normalize to False there, but have no second spelling on the
wire. No field rules are duplicated in Rust/Python parsing code.

Twenty new wire/integration checks initially failed because this API was absent
(31 existing passes). After implementation, focused wire and runtime configuration
checks passed 100 tests. The additional runner control reconstructs the recorded
configuration before creating a Brain and compares its effective configuration
with the reserved parameters.

Final CPU workflow gate: 679 passed, 1 skipped, two expected sampled-engine
warnings in 86.52 seconds (`.cache/homeostasis-wire-gate.log`). Rust `cargo test
--locked --manifest-path crates/Cargo.toml -p assembly-ir` passed both corpus tests
and doc-test processing. Rustfmt, Ruff and diff checks passed. No GPU run,
registered replay, full-package verdict or Lean configuration/backend simulation
proof is claimed. The broader IR/compiler and research migrations remain open.


## Refraction compatibility at mutation boundaries (2026-09-10)

Direct sampled-engine calls bypassed Brain's refraction/scaling guard; explicit
normalization could rescale refracted weights. Brain also published refraction
before a dense engine rejected it and routed auxiliary-area changes to the primary
engine. After correcting fixtures for CSR assignment and replacement-style
normalization, six intended negative controls failed with 50 passes.

The existing shared guard now protects sampled/Torch refraction setters, explicit
normalization and scaling writes. Brain dispatches refraction and normalization
to the owning engine and publishes refraction only after backend acceptance.
Deferred sampled scaling checks current target state and retains a rejected fiber's
pending work. A legal control disables refraction and completes normalization;
no arithmetic rule was replaced.

Focused boundary/deferred checks: 68 passed. CPU workflow plus homeostasis suite:
695 passed, 3 skipped, three expected sampled-engine warnings in 68.42 seconds
(`.cache/refraction-boundary-gate.log`). Ruff and diff checks passed. Torch code
has the guard but its CUDA execution is unverified. Whole-queue rollback, arbitrary
legacy-field mutation, historical replays, backend proofs and the grounded-verb
regression remain open.


## Runtime LRI and history-control ownership (2026-09-10)

Five controls reproduced silent unsupported LRI and history resets dispatched to
the primary mirror instead of the auxiliary area's owner (57 passes). Brain now
routes LRI/reset controls through ownership and publishes LRI fields only after
backend acceptance. ComputeEngine rejects nondefault unsupported LRI; `(0, 0)`
remains the supported disabled request. A sampled-engine control exercises actual
parameter changes and clearing populated history.

Focused checks: 62 passed. CPU workflow gate: 693 passed, 1 skipped, two expected
sampled-engine warnings in 83.10 seconds (`.cache/area-controls-gate.log`). Ruff
and diff checks passed. The preceding turn's 695 count included extra homeostasis
tests; this run used the workflow's own list. No GPU execution, historical replay,
full-package verdict or backend proof is claimed. Numeric LRI validation and
arbitrary backend-failure rollback remain outside this contract.


## Shared LRI numeric validation (2026-09-10)

Forty-one controls reproduced invalid values being accepted or rejected only after
parameter/history mutation, and a NumPy integer failing deque construction
(62 existing passes). The shared `validate_lri_parameters` now runs before Area
construction, supported engine registration/update, Brain update and the base
unsupported-backend check. It canonicalizes nonboolean integral periods and finite
nonnegative real strengths to Python scalars. Periods must fit the platform deque
length; invalid inputs preserve existing history identity/content, parameter values,
registrations and RNG state.

Focused initial post-fix checks passed 103 tests; four additional unsupported-backend
numeric cases are included in the final CPU workflow gate: 738 passed, 1 skipped,
two expected sampled-engine warnings in 71.19 seconds (`.cache/lri-numeric-gate.log`).
All six existing LRI behavior tests also passed, with expected sampled-engine
warnings. Those tests are regression checks, not adopted sequence measurements.
Ruff and diff checks passed. Torch calls the validator but CUDA execution remains
unverified. Arbitrary allocation-failure rollback, direct legacy-field mutation,
historical replays, the grounded-verb regression and wider model/IR unification
remain open.


## Refraction registration preflight (2026-09-10)

Five controls reproduced rejected refraction leaving populations/wiring registered
or an explicit area accepting a mechanism its dense owner did not implement.
A sixth check exposed the absent capability declaration (107 existing passes).
Brain now checks the declared owner capability and shared scaling incompatibility
before descriptor construction, RNG use, registration or wiring. The explicit
owner's capability is checked without constructing an auxiliary engine.

Controls compare registrations, public connectivity maps and all relevant NumPy
RNG states, then retry the same name successfully. A positive control confirms
that the supported sampled engine accumulates refraction bias. Focused checks:
113 passed. CPU workflow gate: 744 passed, 1 skipped, two expected sampled-engine
warnings in 65.73 seconds (`.cache/refraction-registration-gate.log`). Ruff and
diff checks passed. Torch/CUDA capability declarations still require CUDA gates.
This is not general add_area transactionality: other options, duplicate names,
allocation failures and incorrect custom-backend declarations remain open, along
with the broader model/IR, research migrations and grounded-verb regression.


## Shared area identity/dimension preflight (2026-09-10)

All 54 new registration controls initially failed: invalid names/dimensions were
accepted or left state behind, duplicate names replaced populations, and NumPy
integer dimensions remained inconsistent scalar types. A single pure
`core.registration.validate_area_registration` now runs before Brain, standalone
Area and supported backend registration paths mutate state or consume RNG draws.
It enforces unique nonempty area names, nonboolean integer dimensions, positive
cap/population ordering and the shared uint32 neuron-ID limit.

Controls cover Brain/direct registration on all three NumPy engines, preserve
population and descriptor identities and random streams, and check canonical
accepted NumPy counts. Two additional checks cover standalone Area and the logical
population limit without large allocations. The new registration suite is in CI.
Focused registration/IR checks: 80 passed. Final CPU workflow gate: 800 passed,
1 skipped, two expected sampled-engine warnings in 92.39 seconds
(`.cache/registration-gate.log`). Ruff and diff checks passed.

This intentionally rejects duplicate-as-reset callers; no numerical kernel or
scientific threshold changed. GPU execution, backend-specific option preflight,
stimulus namespace rules, general transactionality, historical replays and broader
model/compiler unification remain open.


## Explicit winner-policy forwarding (2026-09-10)

Three controls reproduced auxiliary registration dropping the descriptor's winner
policy in both initial and lazy paths (58 passes). A shared registration helper
now forwards dimensions, beta, LRI settings, slots, policy and input-noise setting
to the owner. The controls distinguish threshold selection (one or zero winners)
from the previous unconditional two-winner fallback and compare primary/auxiliary
dense behavior. Selection arithmetic remains in the existing engine.

Focused registration/IR/input checks: 114 passed. CPU workflow gate: 805 passed,
1 skipped, two expected sampled-engine warnings in 91.79 seconds
(`.cache/area-policy-gate.log`). Ruff and diff checks passed. This is an intentional
behavior correction for callers whose policy was previously ignored, not adoption
of a scientific result. Complete option-combination validation, general registration
transactionality, GPU gates and historical replays remain open.


## Slot layout and policy compatibility (2026-09-10)

Fourteen controls reproduced ignored primary-dense slot counts, accepted malformed
layouts and multi-slot execution bypassing custom policies (62 passes). Area and
dense-engine registration now share slot validation, also used by the standalone
selector. Counts are integral; partitions cover the complete population; custom
policies with multiple slots raise because their composition is not implemented.
Brain forwards supported slots to primary dense owners and rejects unsupported
primary backends before registration.

Controls distinguish best-slot selection from global top-k on the same drive,
exercise every dense entry path and reject a layout that previously discarded a
high-drive trailing neuron. Focused checks: 103 passed. CPU workflow gate:
821 passed, 1 skipped, two expected sampled-engine warnings in 75.72 seconds
(`.cache/slot-contract-gate.log`). Ruff and diff checks passed. No new numerical
selection algorithm, GPU verification, historical replay or backend proof is
claimed. General option transactionality and direct legacy-field mutation remain
open with the broader unification work.


## Stimulus registration and source namespace (2026-09-10)

Fifty-four controls reproduced duplicate stimulus replacement, invalid sizes,
ambiguous area/stimulus names and uncanonicalized NumPy sizes (77 passes).
Shared registration preflight now rejects duplicate/conflicting names and invalid
sizes before wiring or RNG consumption on Brain and all three NumPy engines;
Torch calls the same validator but remains unexecuted here. Standalone Stimulus
construction validates too. Zero-sized stimuli remain valid null inputs.

The namespace restriction follows the implementation: source learning rates and
connection-probability overrides use names without a node-kind discriminator.
Controls preserve a customized source rate and the original stimulus object on
rejected replacement, and exercise both orders of cross-kind name collision.
The source-linked specification and API guide record this compatibility change.

Focused checks: 131 passed. CPU workflow gate: 875 passed, 1 skipped, two expected
sampled-engine warnings in 108.23 seconds (`.cache/stimulus-contract-gate.log`).
Ruff and diff checks passed. No scientific result, historical replay or GPU
conformance is adopted. General allocation rollback and direct legacy dictionary
mutation remain outside this preflight contract.


## Explicit-area probability overrides (2026-09-10)

Six controls reproduced `add_explicit_area` accepting and ignoring each of
`custom_inner_p`, `custom_out_p`, and `custom_in_p`, including zero-valued requests
(132 passes). The wrapper now rejects non-None overrides before registration,
allocation or RNG consumption and names the unsupported options. Default dense
construction is checked at p=1 against all-one weights. No heterogeneous dense
connectivity implementation is claimed.

The sole package caller found requesting these overrides is the checkout-oriented
`text_generation/robust_grammatical_brain.py` prototype. Its `brain` import resolves
through `legacy/root_shims/brain.py` to the maintained implementation. Its requested
CORE connectivity was previously ignored; the same configuration now fails. The
API, supported-surface guide and source-linked specification record that limitation.
Porting it requires a connection policy with incoming/outgoing precedence and
future-fiber semantics, not removing the parameters to restore a green run.

Focused checks: 138 passed. CPU workflow gate: 882 passed, 1 skipped, two expected
sampled-engine warnings in 74.48 seconds (`.cache/custom-probability-gate.log`).
Ruff and diff checks passed. The prototype itself was not executed, and neither
GPU conformance nor scientific evidence was established by these software checks.


## Runtime competition-policy ownership (2026-09-10)

The baseline controls exposed three existing behavioral failures: an auxiliary
dense area's threshold policy remained top-k, and both Brain dense registration
paths allowed a runtime policy to bypass the slot restriction. Four additional
failures identified the absent direct backend setter (142 passes). Brain now
calls the executing owner's explicit method before publishing the descriptor.
NumPy and Torch implementations own their storage updates; the default engine
method rejects unsupported runtime policy changes. Dense updates reuse the
registration slot/policy validator.

Controls distinguish one threshold winner from two top-k winners, reset the
policy to None, cover direct NumPy setters, and preserve state on rejected slot
combinations. The initial full gate caught a raw winner-index comparison in the
new test (1 failed, 892 passed, 1 skipped); the test now reads stable neuron IDs
through diagnostics.read_assembly. No ratchet baseline was changed.

Focused checks before the readout correction: 149 passed. Final CPU workflow gate:
893 passed, 1 skipped, two expected sampled-engine warnings in 84.06 seconds
(`.cache/runtime-policy-gate-final.log`). Ruff and diff checks passed. GPU execution,
policy-object parameter validation, input-noise ownership and direct legacy-field
mutation remain open; these controls do not adopt a scientific result.


## Input-noise configuration and ownership (2026-09-10)

The initial controls failed in 35 cases (149 passes): invalid runtime values,
unsupported owner configurations, absent direct setters, and noncanonical values.
A shared validator now requires a finite, nonnegative, nonboolean real standard
deviation and canonicalizes it to float. Area construction and NumPy/Torch
registration use it. Brain checks noise capability before registration and routes
runtime changes to the executing owner before publishing its descriptor. Sparse
NumPy and Torch implement the setter; unsupported engines accept only zero.

Additional controls check invalid registration on every NumPy entry path and
selection changes on otherwise identical materialized all-connected brains with
noise enabled versus disabled. The first execution fixture attempted unsupported
sparse external drive and was corrected to use stimulus input. Focused checks:
203 passed. No statistical noise-law result is claimed from this seeded control.

The first CPU gate had one diagnostic mismatch (946 passed, 1 skipped): the new
early error omitted input_noise_std. It now names the owner and parameter. Final
CPU workflow gate: 947 passed, 1 skipped, two expected sampled-engine warnings in
69.47 seconds (`.cache/input-noise-gate-final.log`). Ruff and diff checks passed.
GPU execution, arbitrary-scale arithmetic, policy-parameter validation and direct
legacy-field mutation remain unverified or open.


## Immutable competition-policy values (2026-09-10)

Twenty initial control failures (8 passes) exposed unchecked constructor values,
invalid gamma constants and uncanonicalized NumPy scalars. Policies now validate
count, real, fraction and mode fields during frozen-dataclass construction using
shared helpers. Misspelled E%-WTA windows previously selected epsilon silently;
arbitrary tie strings selected an unnamed fallback sort. Both now raise. Gamma
constants have an explicit domain before division. Signed finite thresholds, zero
winner counts and zero-delay/full-delay boundaries remain supported.

Relative and E%-WTA selection branches no longer duplicate policy-parameter
validation. Policy construction owns that contract. Existing selection arithmetic
and E%-WTA population-dependent floors/caps are unchanged. Legacy pickle state or
objects created by bypassing constructors do not acquire this guarantee.

Focused policy/registration suites: 231 passed. Standalone winner-selection
regressions: 11 passed. The policy suite is now in the CPU workflow gate, whose
final result is 975 passed, 1 skipped, two expected sampled-engine warnings in
69.83 seconds (`.cache/policy-values-gate.log`). Ruff and diff checks passed.
These are software checks, not scientific adoption or GPU/backend parity proof.


## Competition configuration IR (2026-09-10)

All four policy types now have explicit versioned documents under competition-v1.
Python exports/reconstructs policies and Rust exposes an opaque validated document,
both using the packaged competition.schema.json and shared acceptance corpus.
Continuous settings must fit finite binary64; count fields retain arbitrary integer
precision and reject floating encodings. Cross-field minimum/maximum ordering is
validated by both readers, including adjacent counts above u64 range. Unknown
fields, missing defaults and misspelled modes are rejected.

The shared corpus contains 20 cases. Python round-trip selection controls cover
all policy classes; a runner integration consumes the recorded policy for each
fixture seed and obtains one threshold winner instead of default top-k's two.
This is configuration transport and reconstruction, not a Rust execution backend,
a Lean refinement proof, or adoption of a scientific result.

Focused policy/runner checks: 72 passed. Cargo format and locked assembly-ir tests
passed (three corpus tests, including existing protocol and homeostasis corpora).
CPU workflow gate: 978 passed, 1 skipped, two expected sampled-engine warnings in
74.82 seconds (`.cache/competition-wire-gate.log`). Ruff and diff checks passed.
Historical artifact migration, general compiler integration and GPU gates remain
open. The negative cases validate the new boundary; no pre-existing API failure
count is claimed for this newly introduced document format.


## Broader CPU audit and phonological registration (2026-09-10)

At aabff18 the wider package invocation was:
`uv run pytest neural_assemblies/tests -q -m "not slow and not gpu" -n 2 --dist loadfile --maxfail=8`.
It stopped after 135.61 seconds with 4 failed, 475 passed, 8 skipped, 1 xfailed
and 4 errors (`.cache/package-cpu-audit.log`). This was an early-stopped audit,
not complete package coverage. Two failures/errors came from repeated phonological
registration; three were classification failures; three were collection errors.

`add_phon_stimulus` now reuses a same-sized existing stimulus and rejects a changed
size, preserving Brain's strict duplicate registration contract. This keeps corpus
vocabulary discovery from replacing connections or source learning rates. Two
focused controls initially failed (reuse and mismatch diagnostic); both pass after
repair. The formerly failing corpus-word and scaled-vocabulary integration checks
also pass: 4 checks total, 27.24 seconds. The new focused suite is in CPU CI.

Collection repairs: image activation imports the maintained Brain; the scheduled
GPU aligner uses importorskip before importing its optional backend and declares
its GPU marker. The isolated Matplotlib install lacked animation.py; reinstalling
the locked 3.10.8 package offline repaired it, without dependency-file changes.
The three affected files then yielded 10 passed, 2 skipped in 2.72 seconds.

Complete non-slow/non-GPU test_emergent_parser.py: 149 passed, 3 failed in 270.10
seconds (`.cache/parser-broad-audit.log`). The held-out bird and finds both classify
as ADV, and generalization is 1/3 against the fixture's 0.66 bar. These tests were
not weakened or skipped. Readout/representation diagnosis remains required.

CPU workflow gate: 980 passed, 1 skipped, two expected sampled-engine warnings in
86.32 seconds (`.cache/phon-registration-gate.log`). Changed/new test files pass
Ruff. The parser core has 30 existing F401 findings on both committed and working
versions; no blanket clean-lint claim is made. Diff checks pass. The migration plan
now consolidates implemented boundaries and remaining acceptance work. Full-package
completion, GPU gates and scientific adoption remain unproven.


## Classification cue diagnosis and explicit query modes (2026-09-10)

A single-seed software diagnosis reproduced the parser fixture at n=10000, k=100,
p=.05, beta=.1, seed=42, rounds=10, holding out bird/finds/small. The fixture
training environment used EMERGENT_FAST_TRAINING=1, EMERGENT_ERP_FAST=1 and
TRAIN_PROGRESS=0. This is not preregistered research or an adopted accuracy result.

| Word | Combined | Phon only | Grounding only |
|---|---|---|---|
| bird | ADV | ADV | NOUN |
| finds | ADV | ADV | VERB |
| small | ADJ | ADV | ADJ |

Grounding-only target-core overlaps were .27, .40 and .42; combined target-core
overlaps were .03, .03 and .24. Recruited populations also differed: NOUN 2782,
VERB 2348, ADJ 1159 and ADV 817. These counts identify a score-comparability concern;
no chance correction or causal conclusion about population size is established.
The contrast shows that registered, untrained phon input can obscure the feature
readout in this fixture. It does not prove a general decay or generalization law.

The first diagnostic expressed grounding-only by querying an unregistered word.
The new keyword-only cue_mode parameter represents the contrast directly:
combined (unchanged default), phon_only, grounding_only. Evidence retains the mode
and immutable resolved cue names. The word-only category cache is not reused for
explicit variants. Four initial new-interface controls failed; after implementation,
26 observation/evidence checks passed, then five provenance validation controls
were added. All modes trace their actual schedule and preserve neural state.

A direct rerun through the new API reproduced the table and exited successfully
(`.cache/classification-cue-audit.json`). To reproduce, construct/train the fixture
with the parameters above and call classify_word_evidence(word,
parser.word_grounding[word], cue_mode=mode) for each listed word and mode.
The default combined-cue classification failures have not been fixed or hidden.

CPU workflow gate: 989 passed, 1 skipped, two expected sampled-engine warnings in
87.86 seconds (`.cache/classification-cues-gate.log`). Ruff and diff checks passed.
Protocol choice, cross-population score calibration, GPU verification and full
package completion remain open; no training rule or test expectation was changed.


## Explicit grounding and word-cache identity (2026-09-10)

Three controls reproduced category, bootstrap and distributional shortcut maps
returning an old word-only answer for explicitly different grounding (20 passes).
classify_word_cached now sends alternate context through existing bootstrapped
inference before these shortcuts and does not write its result into word-only
caches. No explicit context, or context equal to the stored GroundingContext,
retains the original fast path. No fusion formula or neural selection rule changed.

A trained-parser control matches uncached inference and preserves neural state;
the existing cached-classification performance test still passes. Focused checks:
37 passed in 11.04 seconds. CPU workflow gate: 994 passed, 1 skipped, two expected
sampled-engine warnings in 76.73 seconds
(`.cache/classification-cache-context-gate.log`). Ruff and diff checks passed.
General invalidation after training or in-place stored-context mutation remains
open. This does not resolve the default combined-cue holdout failures or establish
new scientific performance.


## Second broad audit and initial recruitment identity (2026-09-10)

The second non-slow/non-GPU package audit used two loadfile workers and
--maxfail=12 at 90fad60. It stopped with 14 failed, 1317 passed, 73 skipped,
4 xfailed and 1 xpassed in 312.69 seconds (`.cache/package-cpu-audit-second.log`).
Two extra failures came from tests already in flight after the limit. It had no
collection errors but was still an incomplete package audit.

Failure groups: three known combined-cue classifier checks; eight coin/PFA/parity
paths rejecting invalid compact indices; two rule-parser paths rejecting duplicate
stable IDs; one stability test expecting a cold population to yield a vacuous
perfect result instead of the current probe error. Legacy coin code explicitly
preserves invalid old seed/index behavior; those goldens must not be restored by
weakening index validation. The cold-probe expectation requires contract review.

The duplicate-ID failure traced to dense-source initialization of sparse areas in
both NumPy and Torch. Selected stable neuron IDs were stored, but the cursor merely
advanced into an unrelated random pool. Later recruitment could select those IDs
again and skip other neurons. Both backends now share reserve_initial_neuron_ids:
selected IDs become the reserved prefix; unselected IDs retain their original order.
Validation consumes no RNG and does not mutate inputs on failure.

Two controls failed before repair (15 passes), including a 20-neuron materialized
population with only 18 distinct IDs. After repair, the focused index suite plus
center-embedding check passed (18 tests); expanded reservation controls plus the
formerly failing TACL F1 smoke passed (23 tests). These smoke passes are not
adoption of syntax/sequence scientific evidence on the sampled engine.

CPU workflow gate: 1001 passed, 1 skipped, two expected sampled-engine warnings in
71.36 seconds (`.cache/initial-recruitment-gate.log`). Ruff and diff checks passed.
Torch uses the shared helper but was not executed. Future recruited identities and
downstream numbers can change on affected paths; old corrupted mappings are not
repaired and historical parity remains unproven. Remaining broad-audit failures
were not suppressed, and no full-package completion is claimed.


## Stability observation contract and GPU authorization

Stability now shares positive nonboolean integral round validation with
Brain.project_rounds, before entering read-only observation. It retains ordinary
projection calls so the second observation cannot lose its recurrent self-edge.
The cold-target expectation now matches the existing no-recruitment contract:
pool < k raises, pool == k measures but is untrustworthy. The exactly-k control
and trained/untrained contrast are retained in the CPU CI gate.

Focused stability suite: 25 passed. CPU workflow gate: 1026 passed, 1 skipped,
two expected sampled warnings in 85.70 seconds (`.cache/stability-gate.log`).
Changed Python files pass Ruff; diff whitespace check passes. No research result
or full-package completion follows from these software controls.

The user explicitly authorized GPU builds, execution, empirical validation and
performance work. This supersedes the earlier allocation of all GPU work to
Claude. Execution remains serial with an isolated extension cache; main/dev and
other sessions remain untouched. CUDA validation is pending at this checkpoint.


## CUDA compiler discovery repair

The first authorized GPU attempt used torch 2.12.1+cu130 on the RTX 3080,
CUDA toolkit 13.1, and automatically selected VS 2026/MSVC 14.50. CUDA rejected
that host compiler before tests ran (`.cache/gpu-gates-20260910.log`).

Compiler discovery now filters a configurable ASSEMBLIES_VS_VERSION range,
default [16.0,18.0), and the batch helper calls the Python discovery function.
The setup command then selected installed VS 2022/MSVC 14.44 successfully.
Three unit controls cover the default range, override, and no-match rejection.
The setup guide explicitly requires a successful fused load before pytest,
because the existing fixtures skip unavailable builds.

Reusing the first cache compiled CUDA successfully but failed linking main.o
built by VS 2026 (`.cache/gpu-gates-vs2022.log`): Ninja did not invalidate that
object after the compiler in PATH changed. A fresh worktree-local VS 2022 cache
built and loaded the fused extension. The guide now includes the MSVC version
in the cache path and requires fresh caches for other toolchain changes.
GPU parity results are recorded separately below; successful build is not parity.


## GPU gate: explicit storage and normalization arithmetic

The first executable gate returned 115 passed, 1 failed, 11 warnings in 45.30s
(`.cache/gpu-gates-vs2022-clean.log`), no skips. The failing case was named CSR
store parity, but its clipped, unscaled factory path actually selected
DenseOrganFiber. That kernel multiplies by a float32 reciprocal; the stored
reference divided by float64 degree. A diagnostic reran the same four brains:
both float64 and float32 division diverged, while float32 reciprocal multiplication
reproduced every final winner set (`.cache/normalized-parity-diagnostic.log`).
For brain2, episode1, division differed on 19 of 40 winners; later differences
were 25 and 18. This is a software counterexample, not a scientific estimate.

The gate now constructs CSR and dense-organ fibers explicitly, each against a
stored-weight reference with its actual arithmetic. The original division
mismatch remains a constructed negative in the normalized organ case. No kernel,
selector, learning rule, tolerance, or historical artifact was changed. The
factory documentation now disclaims identical trajectories from close drives;
VERIFICATION.md records the distinction and the missing margin-certificate gate.

Expanded gate: 122 passed, no skips, 11 warnings in 31.04 seconds
(`.cache/gpu-gates-arithmetic.log`). Files: test_fused_cuda,
test_hashed_substrate_parity, test_hashed_fsm_parity,
test_hashed_transducer_parity, test_hashed_aligner_parity, test_torch_parity,
and test_mixed_drive_indices. Warnings include the existing deep-count overflow
and sampled NumPy warnings; they were not suppressed. Tests include drive replay,
arithmetic-scoped trajectories and qualitative operations, not a uniform claim
of exact trajectories across all backends. Historical A1/capacity replay remains
open, as does the previously recorded broader package audit.

Final contract workflow rerun in the CUDA-enabled environment: 1029 passed,
1 skipped, two expected sampled warnings in 116.23 seconds
(`.cache/contracts-with-cuda.log`). Compiler-discovery tests are now in that
workflow. Changed Python files pass Ruff; git diff --check passes. No full-package
clean claim, scientific adoption, historical replay or performance speedup is made.


Before A1 migration replay, source inventory v3 adds Windows .cmd build scripts
(the CUDA setup entry point was previously omitted). The existing source-mutation
control now includes scripts/cuda-dev.cmd; runner and migration suites: 45 passed,
8.85s. Ruff passed. Version2 historical records retain their original meaning.


## A1 historical migration replay

At b350953, the shared runner completed the full registered A1 length (2000),
20 seeds 1..20 at p=.3 and .4. The immutable run/results and comparator receipt
are committed under research/results/runs/sequence.a1-horizon/
migration-a1-20260910-v3/. All 40 rows match
research/results/sequence/seq_a1_horizon_results_hashed_int8_timing.json,
including exact_fraction, accuracy, first_error and all prefixes. The comparator
records both artifact hashes. Source inventory3 includes the CUDA setup script.

All 40 brain/parameter cells were error-free to the finite 2000-step horizon.
The run's historical Gate3 remains FAIL: sampled NumPy seed1 at p=.3 first
errs at759 while the hashed interval is censored at2000. This result is preserved,
not replaced with migration success. The replay does not establish an unlimited
horizon, independent scientific adoption, or validity of sampled dynamics.

## Capacity replay inputs recovered before running

The figure control artifact is memory/capacity_scaling_results_figure_ctl.json,
introduced in fb934463e62e9fbf23338f4ee0b8170e177f9333. Its cell is B/4000,k60,
20 values per metric, M=8,16,32,64,128,192,256,384,512,768,1024. The registration's
Reproduce block gives T8,20 brains, masked readout, and dropping refraction for
the control83. The producer script at that commit defines brain identities
42+b in array order, measurement RNG1234 reset per cell, p.5,beta.1,clip20,
recall sample32,pair sample200,stim_sizeNone,no convergence gate. These sources
supply the reconstruction candidate; the historical file itself lacks a run
record, so original execution provenance is not authenticated by reconstruction.
The replay will compare all per-seed metrics and aggregate ceiling fields.


Capacity control replay at 5b26d64 completed under the shared runner. All 885
comparisons match the historical figure control: 11 checkpoints x 4 metrics x 20
seeds, plus 5 aggregate fields. The bracket remains [64,128), and the runner
explicitly refuses to treat interpolated 83.4 as resolved at that grid. The run,
results and hash-bearing comparison receipt are retained under
research/results/runs/memory.capacity-scaling/migration-capacity-20260910-v3/.
Both registrations now link to their replay evidence; the capacity Reproduce
command uses the maintained runner. A1 and this capacity cell have numerical
reproduction evidence, without authenticated historical execution provenance or
new scientific adoption. Further capacity variants and full-system gates remain.


## Winner margin: numerical agreement versus readout agreement

The capacity substrate gate formerly truncated to the shorter score vector and
reported only relative drive error. It now requires matching complete shapes and
records canonical winner agreement and margin-certified counts as JUnit properties.
The numerical tolerance remains separate; engine-selected trajectories are still
replayed rather than replaced with the diagnostic's canonical selection.

ir.selection.compare_winner_selection represents each finite binary-float or
integer input exactly with a common power-of-two scale, including mixed Python
lists without lossy NumPy coercion. Its sufficient condition is strict gap > 2E,
where E is the maximum observed absolute error. Empty/full selections are trivial
certificates, not assembly evidence. This is not a future-round error bound.

Nineteen controls cover allclose with changed winners, agreement without a margin,
equality at 2E, overflow/subnormals/uint64/mixed inputs, malformed shapes and counts,
and exhaustive small integer pairs. Together with specification links: 27 passed.
Lean build and leanchecker AssemblyIR.Selection passed: a pairwise separation
theorem and lifting to all selected/outsider pairs. Neither proves Python sorting,
input conversion, CUDA kernels or biological claims.

First GPU substrate audit: 20 passed in 8.64s. Across 12 observations per arm,
canonical agreements/certificates were NONE 11/2, B 12/9, C 12/4, G 12/9. Thus one
of 48 drive comparisons passed tolerance while canonical winners differed; only
24 had a separation certificate. These are diagnostic counts at a fixed test
fixture, not an ensemble result or a statistical confidence claim.

Final contract gate: 1049 passed, 1 skipped, two expected warnings in 104.29s.
GPU substrate rerun after exact mixed-value handling: 20 passed in 7.33s.
Log: .cache/margin-final-gates-native.log. Lean build and standalone leanchecker
passed; Ruff and diff whitespace checks passed. No projection kernel changed.


## Migration comparison identity hardening

Comparator version2 rejects duplicate historical A1 keys and duplicate JSON
members instead of silently replacing evidence. Integer identities, lengths,
error positions and capacity k compare exactly with integer type, while float
measurements retain their fixed tolerance. Nested booleans remain distinct from
integers; historical A1 integer fields and capacity reference seed keys reject
float/bool identities. Invalid reference input produces a failed CLI comparison.

Focused runner/comparator/specification checks:63 passed in11.68s. The final
capacity-coordinate check:12 comparator tests passed in0.36s. Ruff and whitespace
checks passed. Both committed replays were rechecked without running the GPU:
A1 remains40 matches, capacity885. New comparison-v2.json receipts beside each
run record version2 plus the comparator source SHA256; original receipts and
observations remain unchanged. This does not supply missing historical protocol
provenance or broader scientific validation.


## Capacity consumes its recorded execution settings

Capacity recorded arm_settings, device and distinctness bars but read module
ARMS/DEV/DISTINCT_GATE and literal .9 at execution. Those hidden lookups are
removed: cell execution requires arm settings and device, measurement allocates
on its stored tensors' device, and gating consumes both recorded thresholds.
Settings validate before neural construction. CLI options expose device and both
bars; changing bars still requires registration before scientific adoption.

Controls vary arm/device settings, demonstrate threshold-dependent acceptance,
and reject malformed settings before measurement. Focused capacity/runner/spec
checks: 69 passed in 27.14s. Ruff passed. Full historical control replay follows
before numerical equivalence is claimed for this change.


The full capacity control reran at 1b89ac1: all 885 comparisons match the historical
figure control, including every per-seed metric and aggregate fields. Its run,
results and version2 comparator receipt are retained under
research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/.
The registration links this replay. The [64,128) bracket remains unresolved;
no changed scientific bars or new adoption follows from making execution consume
its recorded inputs.

Final contract workflow: 1067 passed, 1 skipped, two expected sampled warnings
in 121.43s (.cache/capacity-consumed-contract-gate.log). Full-system completion
and further experiment migrations remain open.


## A1 consumes its recorded model and schedule

HorizonProtocol now freezes and validates the recorded sizes, k, probabilities,
learning/clip/refraction settings, presentations, length, checkpoints and device.
The FSM constructor, schedule and exact-state readout consume that object rather
than imported globals. It rejects implicit enlargement of the state arena by
requiring room for all assigned blocks. Requested probabilities absent from the
historical baseline reject before GPU execution instead of producing an empty
comparison that passes all(). Version2 records checkpoints and device explicitly;
version1 evidence remains readable but needs those fields for execution.

Focused protocol/runner/specification controls: 66 passed in 21.43s; Ruff passed.
Controls check copied configuration, malformed inputs, constructor arguments,
recorded grid selection and missing-baseline rejection. Full historical replay
follows before claiming numerical preservation.


A1 full replay at ae8c548 reproduces all 40 historical rows after protocol
consumption, with no smoke reduction: seeds 1..20, p=.3/.4, length 2000. Artifacts
and the version2 comparison receipt are retained under
research/results/runs/sequence.a1-horizon/horizon-record-consumed-20260910/.
The registration links them. The independent historical Gate3 remains FAIL;
this migration result does not establish an unlimited horizon or sampled validity.

Final contract gate: 1082 passed, 1 skipped, two expected warnings in158.12s
(.cache/horizon-consumed-contract-gate.log). No full-system completion or new
scientific adoption is claimed.


## Preregistered A1 combined-learning null

The new a1-learning-null adapter reuses HorizonProtocol/run_width, fresh brains,
the shared runner and existing paired Student-t statistics. Registration:
research/notes/sequence/PREREG_a1_learning_null.md. Null changes only beta and
strength to zero; the unchanged training/teacher-forcing schedule is retained.
20 paired seeds and both probabilities are required for study mode. The bars
and arm schedule are recorded; smoke results remain VOID. Initial software
controls caught missing serialized interval endpoints (2 failures), corrected
before any GPU run. Final focused controls: 65 passed in 10.29s. No null data has
yet been observed at this preregistration checkpoint.


The preregistered A1 null ran at 4e7e927 with 20 paired seeds at each p, fresh
arms, full 2000-step inputs. All four bars passed at both probabilities. Trained
accuracy was 1.0; null means were .27155 [.26380,.27930] at p=.3 and .25490
[.24680,.26300] at p=.4. Paired accuracy drops were .72845 [.72070,.73620]
and .74510 [.73700,.75320]. Paired exact-state drops were .97985
[.96646,.99324] and 1.0 [1,1]. These are nominal Student-t intervals over brains,
not independent timestep samples. Perfect label accuracy at p=.3 coexists with
imperfect exact-state recovery; the distinction is retained.

The run and raw rows are under research/results/runs/sequence.a1-learning-null/
a1-learning-null-20260910/. Artifact validation passed. This establishes this
instrument's sensitivity to the combined mechanism, not a new assembly theorem,
a component-wise causal effect, or an unlimited horizon. The active script guide
now includes this runner and removes obsolete --brains examples for migrated paths.

Final contract workflow: 1092 passed, 1 skipped, two expected warnings in101.51s
(.cache/null-final-contract-gate.log). The 40 freshly trained control rows also
match the historical A1 artifact. No projection kernel or prior evidence changed.
