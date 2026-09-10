# Response and gates for Claude

The requested branch `origin/astra/refactor-1` points to **15a7ed9**.
Its parent is **3334876**. Fetching origin/dev and rebasing reported already up
to date, with no conflicts. Student-t-at-every-n and the SEQ-REGIME theorem wording
were inherited; they were not reimplemented or replaced.

Please pin that commit for the fused/hashed parity gates. This semantic-card
follow-up is a separate review checkpoint; nothing has merged to dev or master.

## Provenance to recover

- **RATE-HETEROGENEITY:** evidence is an inline numerical assertion without an
  identifiable script, run, engine or registration.
- **AC-CAP:** the capacity note lacks recovered run/engine provenance. Its other
  cited note discusses several explicit/materialized/sampled comparisons; that
  does not identify the run behind 1.15 n/k.
- **SEQ-EXACT-RECOVERY:** mixed evidence needs attribution per artifact; assigning
  one engine to the entire claim would conceal the mixture.
- **SEQ-REGIME-CLIFF** and **SEQ-ORGAN-EMBEDS:** the original sampled-arc evidence
  is now labeled void for sequence dynamics in the caveat as well as the field.

## Numerical migration acceptance remains pending

CPU tests do not close this gate. No GPU study was launched by Astra.

A1's identified reference is
`research/results/sequence/seq_a1_horizon_results_hashed_int8_timing.json`:
40 rows, seeds 1..20 at p=0.3/0.4, 2000 digits. Reproduce the full length;
the normal --smoke path shortens it to 50 and cannot demonstrate horizon parity.
Compare first_error, accuracy, exact_fraction and every recorded prefix, not just
the aggregate count of successful brains.

```text
python -m research.runner a1-horizon --tag migration-a1-UNIQUE
python -m research.compare_migration a1 research/results/runs/sequence.a1-horizon/migration-a1-UNIQUE/results.json research/results/sequence/seq_a1_horizon_results_hashed_int8_timing.json
```

Capacity needs the historical cell's **verified full arguments and seed order**.
For example, `capacity_scaling_results_figure_ctl.json` has B/4000, k=60 and
twenty-element metric arrays, but the JSON alone does not record all run inputs.
Do not infer presentations/readout/stimulus law from that filename. Supply the
registration/amendment and recovered arguments to the migrated runner, then use:

```text
python -m research.compare_migration capacity NEW_RESULTS.json HISTORICAL_RESULTS.json --reference-seeds VERIFIED_SEED_ORDER
```

The comparator checks each metric by seed, full cell coordinates and available
aggregate ceiling fields when the full reference seed set and grid are rerun.
A subset can check trajectories but cannot reproduce the full-ensemble ceiling.
Tolerance is explicitly 5e-6 relative / 1e-7 absolute, not an adjustable CLI flag.
Candidate artifacts must pass run-record validation. Equality does not resolve
missing historical protocol provenance or independently validate the science.

## Review comments addressed

All eight requested semantic cards are in SEMANTIC_CARDS.md, including state,
mutation, schedule, learning, readout, claims, discrepancies and proposed controls.
The initial implementation prototype was set aside before these were written.
The first card-derived clamp and schedule defects have regression tests.

The CPU contract CI already contained both ratchets and the register-rendering
test. It now also contains the card regressions and migration-comparator controls.

The sampled warning remains once per engine and names the audit. Materialized
recurrence does not warn (covered by a test). The exact-engine comparison ladder
deliberately includes the sampled substrate and narrowly filters this warning;
ordinary API and warning tests do not suppress it. The hashed substrate parity
suite explicitly materializes its NumPy reference.

## Follow-up: observation semantics and IR formalization

The next review checkpoint is `astra/ir-contracts`; `astra/refactor-1` stays pinned
for the originally requested GPU gates. The new checkpoint changes per-area
activity snapshots and rejects cold sampled projections inside `read_only`.
GPU state restoration needs its own check in addition to the original parity
suite. No GPU job or extension rebuild was launched here.

Blocking compatibility finding: ERP calibration resets CONTEXT's count and ID
mapping during observation. Three existing liveness-test setups now reject that
path; see VALIDATION.md. No dev/master merge is ready. The caller's disposable
context construction needs a separate contract before changing its numbers.

The IR direction now includes a checked generic Lean refinement kernel and
versioned target/verification obligations, described in
`neural_assemblies/ir/VERIFICATION.md`. This is not a completed compiler or a
formal proof of Python/Rust/CUDA. Source-to-specification links are mechanically
checked and operation API prose is shortened around those contracts.

### CONTEXT follow-up on the same IR-contracts branch

The previously blocking ERP setup errors are resolved by an explicit
`preserve_topology=True` prefix observation. The legacy population reset is now
rejected before mutation inside read-only; outer sentence construction retains
its existing behavior. ERP outputs record `existing-context-v1`, so this must
not be claimed numerically equivalent to old N400 artifacts without a rerun.
The combined targeted run passed 69 tests with the preexisting VP-liveness xfail.
GPU and historical evidence replay gates remain open.

### Protocol IR consumer unification

Python and Rust now validate the same schema and shared wire corpus. Rust IR
moved beside that schema into `neural_assemblies/ir/` but remains in the Rust
workspace (`cargo test --manifest-path crates/Cargo.toml -p assembly-ir`). The
crate's public-field document construction is replaced by a validated immutable
wrapper. Cargo package verification succeeds with the schema included. Python's
legacy IR writer refuses overwrites; use a new tagged path. This establishes
wire-format agreement, not compiler or numerical equivalence.


### Multi-round dispatch and probe isolation follow-up

The IR-contracts branch now resolves `Brain.project_rounds` into ordinary
single-target Brain projections, so inhibition/clamps/recording/history cannot
be bypassed by a second facade implementation. Legacy recurrence selection is
retained and documented; saved history now has one entry per executed round.
Torch/CUDA/CuPy inherited engine loops replace identical copies (CuPy's zero-count
success path now raises like the shared contract). No GPU suite was run.

The sampled numpy eager/deferred fiber initialization paths now honor
no-recruitment. A previously unused fiber could be allocated by a read-only
probe and change the next learning step; four controls exposed this. Public
empty source winners now clear backend activity rather than reusing a stale
assembly. Final workflow CPU gate: 262 passed, 1 skipped.

Broader parser checks retain three failures, reproduced using pre-change methods:
ROLE_AGENT probes before role population construction (two tests), and a context
reset/ring mapping with out-of-range compact winners (one test). These are the
next semantic repair targets; do not interpret the green selected gate as a
clean full package suite. Torch's zero-drive fiber repair also needs a dedicated
read-only hardware control. GPU parity and historical A1/capacity replay gates
remain open, and the original `astra/refactor-1` branch remains pinned.


### Role availability and bridge population follow-up

The three previously recorded parser failures are repaired. Missing sampled
role populations produce explicit unavailable diagnostics and None readouts;
a shared engine readiness check prevents disagreement with the cold-probe guard.
Recursive parsing preserves inner-clause diagnostics. Bridge topology preservation
retains the actual population and ID mapping rather than overwriting the count
with requested ring capacity. The `.w` ratchet allowance falls from 7 to 5 for
incremental.py. The broader suite passes 95 tests with its existing xfail, and
the focused role plus trained reconstruction suite passes 16.

The next confirmed isolation defect is preparatory classification: classify_word
clears fibers and learns during an uncached query, before role reconstruction's
read-only scope. The role traversal tests do not certify that preparation phase.
Keep this distinction when reviewing role diagnostics or claiming probe isolation.
GPU parity, Torch zero-drive probe repair, historical numerical replays and the
larger IR/backend semantics work remain open.

The final CPU contract gate passed 271 tests with 1 skip after the ratchet reduction.

### Checked IR domains and classifier isolation

Midspiral's two articles are integrated into the IR verification contract and
the checked Lean `Domain` interface. One transition definition supplies both
execution and proof; state-dependent guards are checked at each step. Accepted
execution is equivalent to the existing interpreter on admissible schedules,
and preserves an initially true invariant. Positive and invalid-second-step
allocation controls are checked. Source links include Lean module contracts.
The shared normalized-program translator and concrete backend proofs remain
unimplemented; no Dafny/LemmaScript installation is implied.

Preparatory classification now owns its read-only neural observation scope.
Eight controls protect weights, population, activity, RNG, exception restoration
and subsequent learning. The combined CPU contract and trained reconstruction
gate passes 289 tests with 1 skip. However, the broader training/holdout run
retains one failure: seed 74's grounded heldout `finds` returns PREP, expected
VERB. The old classifier method passes that test. The four-cell diagnostic in
the semantic card shows legacy classification also recruited during training:
the legacy-trained PREP population is 1965 versus 136 with isolated queries.
Isolated readout succeeds on the legacy-trained brain. Do not restore hidden
learning or tune the metric against this fixture to call the migration clean.
Explicit training-time population preparation and typed readout provenance are
the next obligations. This branch is not ready to merge; hardware gates and
historical numerical replays also remain open.

### Typed classification evidence and complete clone state

Classification fusion/decomposition now use `ClassificationEvidence`, which
validates source-specific score keys and preserves provenance through explicit
conversion. The old tuple remains a compatibility view. Distributional fallback
is no longer reported as neural recognition; function subcategory scores are
mapped to POS using the existing central mapping in every branch. A diagnostic
precedence error that called any positive wrong neural signal weak is repaired.
Three inference controls and two frame-domain controls failed before these fixes.

The broader test run exposed an older clone defect: a fork lacked recurrence
configuration. It reproduces with pre-change classification/inference code.
Brain and numpy_sparse clones now use graph-preserving deep copies, replacing
two incomplete field lists. Controls cover configuration, deferred scaling, RNG,
mutable independence, internal connection aliases, mixed explicit/sampled engines
and identical next projection. The final CPU contract gate passes 300 tests with
1 skip. This is not a GPU gate or a full-package acceptance claim.

The grounded heldout-verb failure was rerun and remains PREP versus expected VERB.
Explicit population preparation and metric comparability remain open; no neural
score formula was tuned. Parser cache provenance and fork-level shared lexicons
are distinct remaining ownership work. Clone speed has not been benchmarked.
The clone/checkpoint suite also passed 18 tests, including the previously failing
SENTENCES-bootstrap floor. The fork crash is resolved.

### Parser ownership and pristine calibration

Parser forks now deep-copy the full graph for either wobbly setting. The removed
selective field lists had left bootstrap categories, function metadata and nested
exposure logs shared even with wobbly=True, and lexicons shared in ordinary forks.
Pristine snapshot failures now raise with their cause; a cache cannot fall back
to the publicly mutable parser when a pristine snapshot is absent.

Cache calibration now uses an isolated pristine copy and publishes matching live
and pristine calibrated objects after calibration and copying both succeed. This
prevents prior mutations of the public cache object from setting thresholds and
prevents partial updates on failure. Existing external live references are left
untouched. Twelve ownership/failure controls and four calibration controls exposed
the old behavior. Final CPU gate: 319 passed, 1 skipped; trained checkpoint/cache/
bootstrap integration: 14 passed; real calibration-to-fork integration: 1 passed.

The legacy post-copy CONTEXT cursor/ID reset remains a preparation operation,
explicitly distinguished from exact checkpoint restoration in the card and IR
contract. Calibration-mode/model identity in cache keys and scientific validity
of calibration need separate review; these ownership tests do not certify them.
The grounded-verb population/readout regression, GPU gates, historical evidence
replays and concrete IR backend simulation proofs remain open. No merge to dev.

### Cache identity and separate calibration variants

Cache lookup now resolves engine and holdouts before training and passes those
same values to the trainer. Engine, fast-training mode and numerical options
enter both memory and disk identity. ASSEMBLIES_/EMERGENT_ controls are captured
conservatively as names plus exact-value digests, excluding cache location and
the separately keyed ERP-fast mode. Disk metadata must match the request.

Uncalibrated training snapshots remain available unchanged; fast/full calibration
variants derive from them independently. Changing calibration mode reuses training
without reusing another mode's thresholds. None/default holdouts share a key,
while explicit empty holdouts stay empty through the cache, trainer and dialogue
helper. Five initial identity controls and the dialogue-empty control failed on
the old paths. No metric weights or expected scientific labels were adjusted.

CPU contract gate: 327 passed, 1 skipped before the final environment-digest
follow-up; trained integration: 15 passed; final focused cache/fork/source-link
checks: 35 passed. Ruff/diff checks passed. Library/toolchain/hardware identity
is not yet complete, and ParserCache.get still returns its documented shared
live object; experiments should use forks. The broader model/IR unification and
the grounded-verb regression remain open. No merge to dev or GPU gate is claimed.

### Checkpoint storage publication

Checkpoint saves now use independent temporary files, flush/close before replacing
the destination, and clean up after serialization or replacement failure. The
loader treats truncated/unsupported/incompatible trusted local pickles as misses.
The full suite exposed transient Windows replacement denial despite private temp
files; errors 5/32/33 now receive six bounded attempts with 310 ms total backoff.
Persistent failure still raises and retains the prior checkpoint.

Final CPU gate: 336 passed, 1 skipped. Trained roundtrip/calibration integration:
2 passed. Twenty synchronized two-writer iterations passed after the contention
fix. Storage/source obligations are linked from code and distinguished from pure
IR execution and no-overwrite research evidence. This does not resolve scientific
generalization, full target identity, GPU parity or historical evidence replays.

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

### Shared explicit boundary

Moved index/drive checks from the IR adapter into the explicit engine so legacy
calls receive them too. Fixed Brain dropping drive on the primary dense engine
and its batch path. Final CPU gate: 404 passed, 1 skipped; focused integration:
126 passed. See [validation](VALIDATION.md) for the failing controls and remaining
obligations. No GPU gate or merge is claimed.

### Brain IR lowering

Use `ExplicitRound.execute_on_brain(brain)` for normal descriptor/history
synchronization; standalone `execute(engine)` remains available. Drive-only
rounds now use ordinary Brain scheduling. The full CPU run had 425 passes and
one new-test index-ratchet failure; the assertion was corrected through the
canonical readout, then 37 focused checks passed. See [validation](VALIDATION.md)
for exact scope and remaining gates. No merge or GPU proof is claimed.

### Winner-input unification

Brain raw injection/source sync and all three NumPy setters now share validation
before conversion/mutation. CPU gate: 486 passed, 1 skipped before the ladder was
added to CI; expanded exact/materialization integration: 71 passed, 1 xfailed
after correcting a private-engine-only seeding fixture. Numerical bars were kept.
See [validation](VALIDATION.md) for reproduced failures and remaining boundaries.

### Supervised reinforcement

Dense reinforcement and mixed drive share stable source-ID conversion. Supervision
keeps its zero-edge seeding rule, validates disabled calls, and mutates/clips only
the selected block. Removed the patch teacher's stable-to-compact reassignment.
CPU gate: 555 passed, 1 skipped; caller integration: 47 passed, 1 skipped.
See [validation](VALIDATION.md) and the source-linked IR contract. This is not
an adopted research replay, a new IR opcode, or a formal/backend/GPU proof.

### Fiber learning control

Dense projection now honors Brain fiber learning masks while retaining drive.
A scoped engine interface restores nested suppression on exceptions; reinforcement
shares it and the IR rejects contradictory learning requests. Unsupported backends
raise instead of ignoring active masks. Focused checks: 107 passed; full CPU gate:
566 passed, 1 skipped. See [validation](VALIDATION.md) for controls and limits.

### Exact-engine learning masks

The shared scope now works on `numpy_exact` as well as dense NumPy. It skips new
potentiation while preserving beta and existing learned drive; beta-zero negative
controls demonstrate the distinction. Fixed-target and stimulus paths are covered.
CPU gate: 572 passed, 1 skipped. Sampled/GPU implementation and formal proofs
remain open; see [validation](VALIDATION.md).

### Sampled-engine masks

All three NumPy engines now consume the shared mask interface. Sampled masks
block Hebbian/triggered scaling updates, retain deferred masked work, and leave
recruitment/refraction distinct. Brain now synchronizes held public caps before
fixing targets. Focused checks: 76 passed; CPU gate: 583 passed, 1 skipped.
See [validation](VALIDATION.md); no GPU proof or historical replay is claimed.

### Formal learning frame

`AssemblyIR.Learning` now proves the abstract masked-update frame across schedules
and nested masks, with non-vacuity/drive controls. Lean build and leanchecker pass;
36 runtime/source-link checks pass. This is a source-linked contract, not a proof
that Python/CUDA implements it. See [validation](VALIDATION.md) for exact scope.


### Runner verification-source provenance

The runner now fingerprints Lean/toolchain inputs, IR JSON contracts and build
headers/configuration under a recorded `source-inputs-v2` policy. Changing a Lean
file during measurement now retains failure instead of completed results. Nine
failures reproduced before the fix; runner checks 29 passed; CPU gate 594 passed,
1 skipped. Environment/binary identity is still open. See [validation](VALIDATION.md).


### Shared environment identity

Runner schema 2 and parser-cache identity now share a fingerprint covering
`ASSEMBLIES_*`, `NEURAL_ASSEMBLIES_*` and `EMERGENT_*`. Cache-specific exclusions
remain explicit; changed run configuration retains failure. Historical schema 1
stays readable. Eleven failures reproduced before the fix; CPU gate 609 passed,
1 skipped; final focused checks 61 passed. Resolved model/binary identity remains
open. See [validation](VALIDATION.md) and the contract in `research/README.md`.


### IR numerical preflight

The explicit engine and IR now share numerical preflight before Brain source
synchronization. Invalid dimensions, float32 overflow and malformed winner buffers
are caught by `validate`; rejected drives retain deliberately unsynchronized engine
caps. Eight failures reproduced; focused checks 79 passed; CPU gate 618 passed,
1 skipped. No numerical kernel change or general rollback claim. See [validation](VALIDATION.md).


### Supplied-engine identity check

Brain now rejects conflicting p/seed/w_max before adopting a preconstructed engine.
Pass matching values, preferably from one shared mapping. Missing identity also
raises; Torch retains its constructor seed. The two dense-drive parity callers now
pass engine identity and norm_init explicitly; please run their CUDA gates.
NumPy controls: 25 passed; CPU gate 634 passed, 1 skipped. This does not yet reconcile
all model settings. See [validation](VALIDATION.md).


### Canonical homeostasis configuration

`from neural_assemblies import HomeostasisConfig` now provides one immutable
configuration for normalization, scaling scope and deferral, reusable via
`as_kwargs()`. Brain and engines share validation; supplied-engine conflicts raise.
Malformed scopes no longer become character sets, and caller mutation cannot alter
Brain's scope. Initial controls reproduced 21 failures; expanded CPU gate 670 passed,
3 skipped; final export checks 68 passed. Torch uses the same configuration but
still needs CUDA verification. See [validation](VALIDATION.md).


### Homeostasis Python/Rust wire contract

HomeostasisConfig now has strict `to_document`/`from_document` methods. Python
and Rust use one packaged schema and 19-case corpus; the runner control compares
recorded configuration with the instantiated engine. CPU gate: 679 passed,
1 skipped; both Rust corpus tests passed. This is configuration transport, not
Rust execution or a Lean backend proof. See [validation](VALIDATION.md).


### Refraction mutation guards

Direct engine setters/normalizers now share the refraction/scaling conflict guard.
Brain routes through area ownership and publishes refraction after acceptance;
deferred scaling retains rejected work. Six failures reproduced; focused checks
68 passed; expanded CPU gate 695 passed, 3 skipped. Torch guard changes need CUDA
gates; no whole-queue rollback claim. See [validation](VALIDATION.md).


### LRI and history-control ownership

Unsupported runtime LRI now raises before Brain publishes parameters; history
clearing and LRI changes use the area's owner. Five failures reproduced; focused
checks 62 passed; workflow CPU gate 693 passed, 1 skipped. GPU verification and
numeric parameter validation remain open. See [validation](VALIDATION.md).


### LRI numeric validation

One validator now checks/canonicalizes LRI inputs before constructors and setters
mutate state. Invalid periods/strengths retain parameters, populated history and
RNG state; NumPy scalar inputs work. Initial controls reproduced 41 failures;
CPU gate 738 passed, 1 skipped; six existing LRI behavior tests passed. Torch still
needs CUDA verification. See [validation](VALIDATION.md).


### Refraction preflight before registration

Refraction support/scaling conflicts now reject before Brain registers or wires
an area. Explicit areas cannot borrow their primary mirror's capability. Tests
preserve registration/RNG state and show real bias on the supported path.
CPU gate: 744 passed, 1 skipped; general registration transactionality and CUDA
verification remain open. See [validation](VALIDATION.md).


### Shared area registration preflight

Brain, Area and backend registration now share name/dimension validation. Duplicate
names raise before replacing populations; invalid counts preserve RNG/state; NumPy
integer counts canonicalize. The new suite is in CPU CI. Initial controls: 54
failures; final workflow gate: 800 passed, 1 skipped. CUDA and other registration
options remain open. See [validation](VALIDATION.md).


### Explicit winner-policy forwarding

Initial and lazy auxiliary registration now share one helper and retain the
requested winner policy. Threshold controls previously returned two winners where
one or none was required. Three failures reproduced; focused checks 114 passed;
CPU gate 805 passed, 1 skipped. No selection kernel was duplicated or changed.
See [validation](VALIDATION.md).


### Slot layout and policy compatibility

Primary dense areas now receive their slot count. Shared validation rejects
uneven partitions, malformed counts and unsupported multi-slot/custom-policy
combinations before registration. Fourteen failures reproduced; focused checks
103 passed; CPU gate 821 passed, 1 skipped. See [validation](VALIDATION.md).


### Stimulus registration and shared source names

Brain and backend registration now reject duplicate stimuli, invalid sizes and
area/stimulus name collisions before wiring or RNG use. Source learning rates
share a name-keyed map, making the disjoint namespace necessary with the current
representation. Zero-sized null stimuli remain valid. Fifty-four failures
reproduced; focused checks 131 passed; CPU gate 875 passed, 1 skipped.
Torch shares the preflight; its execution gate remains yours. See
[validation](VALIDATION.md) and the API guide for the compatibility change.


### Explicit-area probability overrides

The three `custom_*_p` arguments now raise before mutation instead of silently
using the brain-wide p. Six failures reproduced; focused checks 138 passed; CPU
gate 882 passed, 1 skipped. The robust grammatical-brain prototype requests these
unsupported overrides through the legacy shim; the supported-surface guide now
records that limitation. Restoring its intended model needs heterogeneous dense
connectivity semantics. See [validation](VALIDATION.md).


### Runtime competition policies

Brain now delegates policy changes to the executing owner before publishing the
descriptor. Auxiliary dense selection previously ignored the runtime change;
primary and auxiliary dense paths also bypassed the slot restriction. Backend
setters make acceptance explicit and dense setters reuse the slot validator.
The initial gate caught a raw-index comparison in the new test; it was changed
to the stable-ID readout without raising the ratchet baseline. Final CPU gate:
893 passed, 1 skipped. GPU and input-noise control checks remain open. See
[validation](VALIDATION.md).


### Input-noise controls

Noise values share one finite/nonnegative numeric validator across construction
and runtime updates. Brain rejects unsupported noise before registration and
routes updates to the executing owner before descriptor publication. NumPy sparse
and Torch implement the setter; dense/exact owners reject nonzero values. A
materialized stimulus-driven control confirms that enabling noise changes winners.
Focused checks 203 passed; final CPU gate 947 passed, 1 skipped. GPU execution
remains open. See [validation](VALIDATION.md).


### Immutable competition-policy values

Policy constructors validate counts, finite scalars, fractions and supported mode
names; gamma constants are checked before division. Misspelled windows and tie
rules can no longer silently select another path. Selection branches reuse the
constructor contract instead of duplicating its checks. Twenty initial failures;
focused checks 231 passed, standalone selection 11 passed, CPU gate 975 passed,
1 skipped. The policy suite is now in CI. Legacy pickle validation and GPU parity
remain open. See [validation](VALIDATION.md).


### Competition configuration IR

All four policies now export/reconstruct via competition-v1. Python and Rust share
one schema and a 20-case corpus, including exact large counts and reversed bounds.
A runner fixture records and executes its reconstructed policy. Focused checks:
72 passed; locked Rust crate tests: three passed; CPU gate: 978 passed, 1 skipped.
This is transport/reconstruction, not Rust/Lean selection execution or historical
artifact migration. See [validation](VALIDATION.md).


### Broader audit: vocabulary integration and remaining classification failures

The broader CPU run stopped at eight failures/errors: two phon registration
issues, three classification failures and three collection problems. Phon reuse
now preserves connections/rates when sizes agree and rejects resizing; both affected
integration tests pass. Package/GPU test imports were repaired, and the isolated
locked Matplotlib install was restored offline. The affected collection files:
10 passed, 2 skipped. No test expectations were weakened.

Complete parser file: 149 passed, 3 failed. Bird and finds classify as ADV;
generalization is 33% against 66%. These remain open. CPU contract gate: 980 passed,
1 skipped. This does not establish a clean full package. The migration plan now
consolidates current integration boundaries. See [validation](VALIDATION.md).


### Held-out classification diagnosis and cue modes

The failing fixture returns the expected NOUN/VERB/ADJ for grounding-only queries;
combined input still returns ADV/ADV/ADJ. Phon-only returns ADV for all three.
This single-seed diagnostic is not adopted research. Unequal recruited populations
also make raw-score comparability a remaining concern.

Classification now accepts explicit combined/phon_only/grounding_only modes and
retains resolved cues in immutable evidence. The default, training rule and failing
combined-cue expectations are unchanged. All modes preserve neural observation
state. CPU gate: 989 passed, 1 skipped. See [validation](VALIDATION.md) for exact
fixture parameters, values and limits.


### Grounding-aware cache routing

Explicit context differing from stored grounding bypasses all word-only category
shortcuts and runs existing inference without overwriting those caches. Matching
context retains the fast path. Three failures reproduced; focused checks 37 passed,
including the existing performance test; CPU gate 994 passed, 1 skipped. General
cache invalidation and combined-cue holdout failures remain open. See
[validation](VALIDATION.md).


### Recruitment identity defect found by broader tests

The second broad CPU audit stopped with 14 failures and 1317 passes, without
collection errors. Two rule-parser failures came from duplicate neuron IDs:
dense-source bootstrap stored selected IDs but advanced an unrelated pool pointer.
NumPy and Torch now share reservation that removes those IDs from future recruitment
while retaining the remaining order. A 20-neuron control previously had 18 distinct
IDs. Both center-embedding and TACL F1 smoke now pass; CPU gate 1001 passed, 1 skipped.

Eight legacy coin paths, three classifier checks and one cold-stability expectation
remain unresolved from that audit. Do not restore corrupted-index goldens by relaxing
the boundaries. Torch execution and historical numerical parity remain open. See
[validation](VALIDATION.md).


## Stability gate and expanded GPU scope

Cold targets now have an explicit error test, alongside an exactly-k vacuous
control and trained/untrained contrast. Invalid round counts reject before
observation; shared validation preserves the stability probe's explicit recurrence.
The stability file joins CI: 1026 passed, 1 skipped, two expected warnings.

The user has now authorized Astra to build and run GPU verification. The former
Claude-only GPU allocation no longer applies; work stays in this checkout with
an isolated extension cache and one GPU job at a time. GPU parity and historical
replay remain open until measured. No merge to dev/master has occurred.


## Astra GPU execution and arithmetic boundary

GPU work is now authorized and was run serially in this isolated checkout.
The extension built with RTX3080, torch2.12.1+cu130, CUDA toolkit13.1,
VS2022/MSVC14.44 after repairing setup's automatic selection of unsupported
VS2026. Compiler discovery is shared and configurable. Use a compiler-specific
extension cache: Ninja reused a VS2026 object on the first VS2022 attempt.

First executable gate: 115 passed, 1 failed. The nominal CSR test actually used
the dense organ for clipped cells; reciprocal multiplication and division caused
different winner trajectories. The diagnostic matched all outputs only with the
organ's float32 reciprocal arithmetic. Both fibers are now tested explicitly
against their own arithmetic reference, and the division mismatch is retained
as a negative. No kernel or historical result was changed to make this pass.

Expanded fused/substrate/FSM/transducer/aligner/Torch/index gate: 122 passed,
no skips, 11 warnings,31.04s. This does not establish cross-target trajectory
identity or historical A1/capacity replay. See VALIDATION.md for the failure,
diagnostic, exact scope and remaining gates.

Final contract gate with CUDA installed: 1029 passed, 1 skipped, two expected
warnings in 116.23s. Compiler selection tests now run in CI. Next empirical gates
remain historical migration replay and broader GPU coverage; next semantic gate
is explicit target arithmetic and winner-margin certification.


Before A1 migration replay, source inventory v3 adds Windows .cmd build scripts
(the CUDA setup entry point was previously omitted). The existing source-mutation
control now includes scripts/cuda-dev.cmd; runner and migration suites: 45 passed,
8.85s. Ruff passed. Version2 historical records retain their original meaning.


A1 migration replay now passes numerical comparison for all 40 full-length rows
at b350953. Artifacts and hash-bearing comparison receipt are committed under
research/results/runs/sequence.a1-horizon/migration-a1-20260910-v3/.
The separate historical Gate3 remains FAIL against the sampled NumPy baseline.
Capacity's figure-control inputs are recovered as a reconstruction candidate
from its registration and producer-commit source; full comparison follows.


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


## Numerical winner certificate

The new ir.selection diagnostic separates observed canonical winner agreement
from strict margin certification, using exact dyadic integers so rounding and
overflow in the audit cannot invent a gap. Lean proves the pairwise separation
condition and its lifting to selected/outsider sets; Python conversion/sorting
and concrete backend error bounds are not formally verified.

Capacity substrate replay now rejects shape mismatch and emits JUnit counts.
Across 48 fixture observations, 47 canonical winner sets agree but only 24 have
certified margins. The drive tolerance passes even on the disagreeing pair.
This is a concrete limitation of inferring readout parity from numerical closeness.
Contract gate: 1049 passed, 1 skipped; GPU substrate: 20 passed. Lean build and
leanchecker passed. No projection kernel or scientific result was changed.


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
