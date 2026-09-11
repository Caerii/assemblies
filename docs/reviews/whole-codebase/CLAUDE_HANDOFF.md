# Response and gates for Claude

The requested branch `origin/astra/refactor-1` points to **15a7ed9**.
Its parent is **3334876**. Fetching origin/dev and rebasing reported already up
to date, with no conflicts. Student-t-at-every-n and the SEQ-REGIME theorem wording
were inherited; they were not reimplemented or replaced.

Please pin that commit for the fused/hashed parity gates. This semantic-card
follow-up is a separate review checkpoint; nothing has merged to dev or master.

The current `astra/ir-contracts` continuation mechanically retracts the four
legacy `coin2024_*` goldens instead of rebaselining them. Verification stops before
their executors, the CLI names the retraction, and consumable metric keys are now
historical-only. Missing MNIST CSVs also skip before golden comparison rather than
accepting the synthetic perfect-score fallback. A new attractor coin study remains
open under a new protocol ID; the historical records are preserved as refusal
fixtures.

The three held-out parser failures are now split at their semantic boundary.
Grounding-only classification must recover NOUN/VERB/ADJ and passes; default
combined-cue classification remains a strict expected failure because the untrained
phon cue obscures bird/finds. This preserves the open integration problem while
stopping it from being misreported as absence of grounding representation.

## Provenance to recover

- **RATE-HETEROGENEITY:** resolved on the review branch by the preregistered
  `mechanism.per-fiber-plasticity` study. Its materialized NumPy artifact records
  source, producer, seeds and complete raw blocks; the unclipped copied-fiber ratio
  is 14.35x, with a label swap and equal-rate null. The old saturated inline number
  remains unreproduced and has been retired as evidence.
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


## Shared evidence JSON policy

Publication state checks already distinguished complete observations from
scientific adoption. Decoding and identity equality were weaker: duplicate JSON
fields could replace evidence, NaN/overflow could enter observations, and Python
equality could equate embedded boolean/float settings with reserved integers.
research.json_documents now owns strict decoding and deterministic encoding for
runner storage, artifact validation and migration comparison. It rejects duplicate
members, nonfinite values, overflow and nonzero underflow to zero; representable
subnormals and large integers round-trip. Record equality preserves JSON types.

Comparison version3 additionally records the shared source inventory fingerprint,
covering helper dependencies as well as the comparator file digest. Focused
runner/comparator/spec controls: 72 passed in 11.83s; Ruff passed. The full
contract workflow passed 1101 tests with one skip and two expected warnings in
100.02s (.cache/evidence-json-contract-gate.log). All five recorded run artifacts
pass the shared decoder and artifact validator. Version 3 comparison receipts,
produced at 4f09122, preserve the A1 40-cell match and capacity 885-value match
for the record-consumed replays. Earlier receipts and results remain intact.
This is document integrity, not authentication of historical execution or
scientific adoption. No new GPU run was needed for these decoding-only changes.


## Coin seed identity consolidation

The coin helper duplicated activation mapping, silently discarded unmapped IDs,
and dispatched through the primary engine. Attractor construction also carried
its own filtering map. Both now delegate to activate_assembly, preserving complete
membership and the area owner. The historical remap=False bypass explicitly
raises before mutation instead of installing stable IDs as compact positions.
The source links contract-coin-seed in IR/VERIFICATION.md.

Seven boundary controls cover nonidentity mapping, owner dispatch, returned-array
isolation, malformed and missing IDs, and rejection of the legacy bypass. Existing
attractor settling, membership, rounds-zero contrast, and beta-zero null tests
also pass (11 passed with three legacy/fairness selections excluded; all seven
new boundary controls separately passed). Full contract workflow: 1108 passed,
one skipped, two expected warnings in 119.53s, recorded locally in
.cache/coin-seed-contract-gate.log. Ruff and diff whitespace checks pass.

This does not repair legacy defaults or uniform compact seeding, recalibrate coin
bias, or validate old coin/PFA goldens. Those known failures remain open; their
numerical artifacts were not rewritten. Core owned-index migration is incomplete.


## Coin operation schedule and shared seed generation

The operation card now describes construction, owner-specific connection resets,
force-firing, both seed spaces, frozen settling and overlap readout. Mixed seeding
has one implementation; redundant oversize truncation was unreachable because the
two draws sum to at most k. Uniform seeding checks the owning engine's materialized
count before RNG consumption or activity writes. Training restores the previous
fixed flag even when projection raises. Invalid construction/control counts and
invalid flip modes, bias or round counts fail before their respective mutations.
Zero fires and zero settling rounds remain available as scientific controls.

Both legacy flip modes now reject before activity changes. The old default remains
inspectable but cannot emit new results; caller migration remains open in
SoftmaxContextCoin, PFANetwork, CoinFlipModel and NemoMarkovPFA. Historical coin
artifacts have not been overwritten or relabeled. Two obsolete legacy numerical
tests now assert explicit rejection; trained-attractor and beta-zero controls remain.

A local software replay compared pre-change e1c19e2 against this implementation:
180 exact matches of labels and complete winner-ID arrays (three brain seeds,
two modes, five biases, zero/ten rounds, three flip seeds). Local script and paired
outputs: .cache/coin-operation-replay.py and .cache/coin-operation-{before,after}.json.
This is refactor equivalence evidence, not a registered scientific measurement or
claim of calibrated fairness. No kernel or stored scientific evidence changed.

All 34 coin controls pass. The workflow now includes the whole coin construction
suite: 1135 passed, one skipped, two expected warnings in 102.15s
(.cache/coin-operation-contract-gate.log). Ruff and git diff --check pass.


## Explicit PFA seed-mixture configuration

Code inspection found four distinct caller problems: PFANetwork treats initial
seed fractions as probabilities; CoinFlipModel inherits that assumption; the
context coin learns during flip then overwrites context-driven activity; and
NemoMarkovPFA ignores probability weights and injects IDs across areas. The card
contract-pfa-choice records these differences. This increment migrates only PFA
and its CoinFlipModel wrapper, without certifying the other instruments.

SeedMixtureChoice is an immutable, JSON-serializable record for the independent
coin population, beta, training rounds, force-fires, settling rounds and seed mode.
PFA branching requires explicit configuration before constructing any populations.
The older flip_mode argument must agree with the configuration. Deterministic
PFA allocates no coin. The wrapper builds separate PFA and sampling coins from the
same configuration; its noise option still affects only the sampling coin.

API documentation includes a runnable example and explicitly disclaims calibrated
outcome probabilities and learned neural transition decoding. TransitionMap's
incorrect implication that normalized weights certify neural sampling was removed.
Old API tests now opt into the explicit construction. Four old direct-coin tests,
including unsupported fairness/seed-bias frequency assertions, were retired in
favor of the existing trained-attractor and true-null suite. Golden artifacts
were not changed or automatically reinterpreted. These are API/contract changes,
not reproduction or adoption of historical PFA probability measurements.

Focused coverage: 58 passed before the final serialization/cascade controls;
final PFA/configuration/transition subset: 26 passed in 3.94s. Full contract workflow:
1161 passed, one skipped, two expected warnings in 125.03s
(.cache/pfa-choice-contract-gate.log). Ruff and git diff --check pass. Full-package
legacy literature/computation callers still need explicit migration and scientific
reassessment. SoftmaxContextCoin and NemoMarkovPFA remain structurally unresolved.


## Complete non-slow package audit at 54164a8

The serial audit completed with the prepared CUDA environment and no early failure
limit: 2617 passed, 24 failed, 59 skipped, 143 deselected, 11 xfailed, 5 xpassed,
328 warnings and 10 passed subtests in 1649.37s (27m29s), exit code 1. This is the
non-slow package scope, not an all-tests or scientific-reproduction verdict.
package-audit-54164a8.json records the exact invocation, all failed nodes/messages,
active expected-failure markers corresponding to the five unexpected passes,
ten slowest cases, and SHA-256 identities for the local raw log and JUnit report.

Initial triage: 17 coin caller/instrument failures; two obsolete read-only
assumptions (cold NemoArcFSM and topology-resetting parse); three held-out
classifier failures; reciprocal restore overlap 0.8125 against the 1.0 golden;
and a stale group test table missing Z120. These categories guide investigation,
not test deletion or automatic rebaselining. The five unexpected passes cover
ERP calibration/range/aliveness and direct binding; their existing order-sensitive,
fragile or dead-probe caveats remain unresolved. A green observation alone does
not remove those caveats.

Read-only evidence audit at the same commit: 2602 tracked files, 2782 resolved
literal edges, 688 unresolved literals, 352 candidate orphan result files and
28 preregistrations without a resolved result link. These are syntax-based review
candidates: sibling run records and comparison receipts can be falsely flagged.
Typed graph relationships remain required. Source-byte checks recovered eight
of ten script/registration digests via CRLF conversion of Git blobs; the first A1
and capacity migration script digests remain unresolved under LF, CRLF and one
edit-based mixed-line-ending reconstruction. All three recorded input-artifact
hash checks matched current files. No inference of semantic source drift is made.
A future source snapshot must preserve recoverable bytes as well as identities.

No source was changed during the audit. No historical scientific artifacts,
thresholds or expected-failure markers were altered to improve the result.


## Reciprocal protocol duplication repaired

The full-audit reciprocal mismatch was reproduced: the cross-repository test
requested Brain(recurrent_projection=True) but invoked ops.project without its
explicit recurrent=True argument. The operation contract supplies no recurrence
by default. The canonical run_pnas_reciprocal already supplied the intended flag.
The test now delegates to that function; its golden and tolerance are unchanged.
Both PNAS program parameter records now retain recurrence and engine identity.
The reciprocal source-linked contract explicitly calls the score a training
round trip: both reciprocal legs retain plasticity, so this is not frozen recall.

Software diagnostics on sampled NumPy, seeds 42/43/44, n=5000, k=80, p=.05,
beta=.1, rounds=20: explicit source recurrence yielded .975/.9875/1.0;
without it .8125/.8625/.7625; beta-zero yielded 0/0/0. These are instrument
checks, not a registered study or fixed-connectome sequence evidence. New
mechanism-disabled controls and recurrence metadata checks enter the CPU workflow.
Targeted validation: 12 passed, 3 reference-dependent skips, 6 sampled-engine
warnings in 1.63s. Specification links, theory rendering/citations and both
ratchets: 27 passed in 40.57s. Ruff passed changed program/new tests. No engine,
golden, adoption bar or historical result was changed. The full package audit
was not repeated; only this identified failure is verified repaired here.


## Readable untrained arc control

The full-audit NemoArcFSM null attempted to observe an arc with fewer than k
materialized neurons. Read-only rejection was correct; that state is not a valid
learning-disabled observation. The test now explicitly materializes the arc before
its untrained read and checks population preservation. A separate cold-read case
asserts rejection without recruitment. The five arc-organ controls moved intact
(apart from this correction/new case) into test_nemo_arc_contract.py and now enter
the CPU contract workflow independently of unresolved Markov PFA tests.
NemoArcFSM.run links the initialization specification and documents the precondition.
No engine or run behavior changed.

Diagnostic, not a registered study: seeds 42/43/44, p=.05, n=2000, k=40,
beta=.1, full arc materialized before any transition teaching. For four a symbols
from q0, the untrained trajectories were q0/q1/q1/q0, q0/q0/q0/q0,
and q0/q0/q0/q1; ten trained presentations gave q1/q0/q1/q0,
q1/q1/q0/q0, and q1/q0/q1/q0. Seed 43 did not recover the target sequence;
this is retained rather than claiming reliability from the test fixture. Changing
the initialization changes the sampled construction and needs separate scientific
evidence before adoption.

Five focused controls passed in 2.19s; Ruff passed. Full contract workflow,
including the preceding reciprocal controls: 1170 passed, one skipped, six
sampled-engine warnings in 108.78s (.cache/nemo-null-contract-gate.log).
The full package audit was not repeated and broader failures remain open.


## Explicit ERP context reset and isolated observation

The old idempotence test assumed read_only could reset context identities and
recruit cold areas. Its first failure was correct rejection. ErpProtocol now names
context_reset: construction (unchanged default) or activity (preserve context IDs).
The runner consumes and records that immutable choice; descriptions name it at
either value. No automatic switch based on ambient scope was introduced.

Using activity reset under a single read_only scope got past the identity reset
but still differed: first-word P600 .9971 versus .9993. This diagnostic failure
is retained here. read_only permits activity changes; it is not a state-restoring
observation. Separate brain.probe scopes produced identical repeated fixture
results. The former test is renamed test_isolated_activity_reset_parse_is_idempotent,
initializes outside observation, and retains strict equality/population assertions.
The original non-isolated repeatability case remains xfailed, not hidden.

Focused ERP/reset/idempotence coverage: 29 passed, one expected failure in18.87s.
Full contract workflow:1177 passed, one skipped, six sampled-engine warnings in
113.70s (.cache/erp-reset-contract-gate.log). Ruff passed. Removed a duplicate
ERP protocol file argument from the workflow after that run; pytest already
collapsed that duplicate. No engine, scientific threshold, or historical result
changed. The result demonstrates fixture-level isolation, not ERP discrimination
or complete Python-side parser purity. Broader package failures remain open.


## Configurable cyclic benchmark and Z120 inventory

The group audit failure was a stale test table: GROUPS already contained Z120,
but its expected-order/solvability dictionaries did not. The test now covers that
control and checks complete registry-key agreement explicitly. The duplicated Z60
and Z120 constructors delegate to cyclic_group(order, generators), retaining both
named wrappers, repr-sorted elements, generator order, state labels and transitions.
Complete pre/post state-symbol-transition JSON tables for both wrappers matched.

The constructor rejects invalid integer controls and proper subgroups using the
gcd criterion, then independently verifies closure cardinality with a runtime
exception. Thus Python optimization cannot remove that final cyclic-group check.
All 650 generator pairs over orders 1 through 12 were compared with an independent
finite linear-combination oracle, including rejected subgroups. Additional controls
cover the trivial group, empty alphabets, normalized residues and malformed inputs.
These finite checks are not a Lean proof of the arbitrary-order implementation.

19 group tests passed in1.22s. Specification links, theory rendering/citations and
both ratchets:27 passed in41.53s. Ruff and git diff --check pass. Group controls
are now in the CPU workflow. No neural engine, study, golden artifact or scientific
threshold changed; no full-package rerun was needed for this symbolic-table-preserving
refactor. The other full-audit failures remain open.


## Recoverable runner source (2026-09-10)

Run schema 3 now reserves and writes a source ZIP before measurement. It retains
exact source-inventory bytes plus the entry script and registration, binds the ZIP
to run.json, and validates the archived inventory and individual digests before
publishing completion. Evidence validation checks the same contract without
extracting or importing code. Missing, corrupted, duplicate, unsafe and altered
archive contents are constructed negatives; schema 1/2 remain readable unchanged.
Specification: research/README.md#recoverable-source, linked from both implementations.

62 focused runner tests passed in 17.63s. Full workflow-selected contract gates:
1206 passed, 1 skipped, 6 warnings in 116.14s (.cache/source-archive-contract-gate.log).
Ruff on changed Python and git diff --check pass. All five committed runner
artifacts still validate. No historical records or scientific thresholds changed.

A real repository storage-only smoke (VOID) captured 1327 source files, 13786708
uncompressed bytes into 4389401 archive bytes in 6.754s, with successful validation.
Local artifact: .cache/source-capture/audit.source-capture/source-capture-20260910.
This single timing is diagnostic, not a benchmark or scientific result. Source
capture excludes datasets, ignored files, installed binaries and external dependencies;
it does not establish hermetic execution or repair the two historical unrecovered
script digests. The full-package audit's unresolved failures remain open.

Dedicated CUDA gates: 122 passed, 11 warnings in 36.05s, with the fused extension
loaded on the RTX 3080 (.cache/source-archive-gpu-gate.log). No kernel or
engine arithmetic changed. These gates do not replace the unresolved full-package audit.


## Shared transition domains and conditional schedules (2026-09-10)

FSMNetwork and PFANetwork now validate complete state/symbol declarations and all
transition endpoints before allocating brain infrastructure. Duplicate edges,
malformed labels, boolean/string/nonfinite/nonpositive weights and invalid mass
tolerances fail explicitly. Positive exact weights that underflow during binary64
conversion also fail. Partial tables remain supported; missing-edge steps now raise
before projecting or changing the symbolic state. Single-pass domain inputs are
snapshotted before validation so generators cannot vanish before construction.

TransitionMap owns the immutable ordered conditional branch schedule. PFANetwork
consumes it with one loop for binary and multiway choices, preserving their respective
seed streams. Direct remaining-mass sums replace subtraction from a rounded prefix.
The PFA module shrank by 36 net lines in this change, while the shared validation and
schedule layer grew. This is not a claim of net repository-wide line reduction.
Source-linked specs and code-derived cards are in IR/VERIFICATION.md under
contract-transition-domain and contract-branch-schedule. The misleading module-level
claim that SoftmaxContextCoin establishes smooth context-dependent probabilities was
removed; its implementation and NemoMarkovPFA remain unresolved, not certified here.

79 focused tests passed in 2.16s. The branch factorization was checked against exact
rational path masses for all 81 four-target integer-weight vectors over 1..3, alongside
a rounded-prefix cancellation control. These are finite mathematical checks, not a
Lean proof of arbitrary floating-point tables or a neural calibration claim.
Before/after replay matched all 36 labels and full winner arrays across three brain
seeds, binary/three-way branches, two selection modes and three flip seeds. Replay
inputs and outputs are local .cache/pfa-schedule-{replay.py,before.json,after.json};
these are sampled NumPy software diagnostics, not registered scientific evidence.

Full workflow-selected contract suite: 1261 passed, 1 skipped, 6 warnings in 121.11s
(.cache/transition-domain-contract-gate.log), including specification links, theory
rendering and both ratchets. Ruff on changed Python and git diff --check pass.
No historical numerical artifacts or adoption thresholds changed. The unresolved
full-package failures and neural Markov/context-coin redesign remain open.


## Arc Markov composition (2026-09-10)

The invalid NemoMarkovPFA/AlternatingMarkovNetwork bodies are retired with explicit
migration errors. ArcMarkovNetwork composes the shared selector with NemoArcFSM's
actual readout under an explicit decoded-state feedback protocol. MarkovChainModel
requires that protocol. The historical alternating architecture is not reproduced.
See [validation, including the perfect untrained two-state case and the subsequent
three-state controls](VALIDATION.md#arc-markov-composition-and-retired-invalid-instrument-2026-09-10).
SoftmaxContextCoin and probability calibration remain open. Historical goldens and
adopted result records are unchanged.

Validation: 1305 broad contract tests passed (one skip); 122 dedicated CUDA tests
passed; three subsequently added direct CUDA learning-null controls passed. No
production implementation changed after those broad gates. See the linked validation
section for per-seed counts, exact scope and logs. The full package is not declared clean.


## Context readout and noise contract (2026-09-10)

SoftmaxContextCoin is retired before brain mutation. Its explicit replacement,
ContextAttractorChoice, separates attractor construction, context teaching and
seeded read-only observation; overlap scores are not probabilities. Fixed teaching
and source-disabled controls expose real learning/context sensitivity. Both sparse
backends now consume positive noise at zero synaptic drive, and NumPy compiled
selection uses the shared noise/competition selector. Noise-only reads require a
materialized population. Native-noise capability is checked before construction.

See [the complete validation and limitations](VALIDATION.md#context-readout-seeded-observation-and-native-noise-2026-09-10).
Broad gate: 1346 passed, one skip, three test-helper failures (CUDA bfloat16 snapshot).
After correcting that helper, the final focused suite passed 71 tests, including
three-seed CPU/CUDA learning nulls. Dedicated fused/CUDA gate: 122 passed. All 36
saved PFA labels and full winner arrays replay unchanged. No production code changed
after the broad gate. No historical goldens or adopted numerical claims changed.
A useful noise-tolerance range is still unmeasured; the extreme-noise endpoints are
development diagnostics. Full-package and concrete proof obligations remain open.


## Noise study registration (2026-09-10)

A shared-runner context-noise experiment now records the complete grid and per-read
observations. Its primary std=1 hypothesis is fixed in
[the registration](../../../research/notes/memory/PREREG_context_noise.md); other
levels are descriptive, with no selected robustness-range claim. New brain seeds
101..120, independent backend reports, within-brain repeats and explicit nulls are
required. This registration checkpoint precedes measurement.


The registered study subsequently completed on both backends: all seven bars pass,
with 20 new brain seeds per backend. The 8800 raw observations, per-brain metrics,
ensembles, verdicts and source archives were checked. At std=1, both backends retain
about 92.6% target overlap and all observed labels; at std=3, labels remain perfect
while overlap falls to about 46.7% and joint recovery to zero.
See [the result, intervals and statistical limits](../../../research/notes/memory/PREREG_context_noise.md#results-2026-09-10)
and [curve](context-noise.svg). This is scoped noise-1 evidence, not a general
robustness range or population-perfect accuracy guarantee.


## Signed drive and distinct noise protocols (2026-09-10)

A balanced positive/negative drive is not silence. A constructed NumPy/CUDA control
failed on NumPy (old winners retained) and passed on CUDA (positive-drive winners).
NumPy now uses an all-zero predicate instead of testing the sum. The focused suite
passed 55 tests; the full workflow contract gate passed 1372, with one skip.
See [validation](VALIDATION.md#signed-zero-drive-semantics-2026-09-10).

The next semantic gap is the old cue-corruption suite, which is separate from the
new additive-noise study. The
[source-derived card](SEMANTIC_CARDS.md#contract-legacy-cue-corruption) records missing
recurrent training, mutable recovery, silently reduced corruption, process-dependent
seeds and a recovery bar weaker than the retained cue. Those tests have not yet
been rewritten or claimed as valid scientific evidence.

The final dedicated fused/CUDA gate also passed 122 tests in 38.57s. No production
source changed after either broad gate; full-package closure remains open.


## Cue recovery successor (2026-09-10)

The package's legacy cue-noise tests now exercise pure exact replacement and strict
read-only recovery, with reference-denominated overlap and an improvement score.
They require learned recovery to beat the delivered cue and fail zero-learning
and no-dynamics controls on CPU/CUDA. The sparse CUDA materialization count accessor
was missing; it now distinguishes cold and full populations.
See [validation and development-only scope](VALIDATION.md#explicit-cue-recovery-and-sparse-population-counts-2026-09-10)
and [the composable API example](../../api.md#explicit-cue-replacement-and-recovery).
Standalone historical noise experiments and scientific tolerance curves remain
separate work; old numerical claims are not relabelled as this protocol.

Validation: 1389 broad contract tests passed (one skip), 122 fused/CUDA parity
tests passed. The final wrapper guards against oversized cues/results then passed
19 focused tests. See the linked validation for exact ordering and scope. The new
API plus replacement tests use 21 fewer Python lines across changed files.


## Recovery result construction (2026-09-10)

RecoveryObservation now enforces membership invariants even when constructed
directly: nonempty unique reference, same-area unique cue/output, bounded sizes.
The observer shares that validator before activation. Valid failures remain
representable. All 28 focused tests pass; no backend code changed.
See [validation](VALIDATION.md#recovery-observation-construction-invariant-2026-09-10).
The standalone historical noise study now has a complete source-derived card;
it retains learning during recovery and drops raw per-seed observations. Its
migration remains distinct from the package's new frozen-observation controls.


## Historical noise trial consolidation (2026-09-10)

Consolidated the three standalone trial schedules while preserving their historical
learning behavior and pre-association references. All 27 pre-change replay cases
match complete schedules, winner trajectories, outputs, final weight digests and
engine ownership. The study now retains per-seed values for every one of its 43
cells, and names its primary/area engines and recovery/reference semantics.
See [validation and the remaining historical provenance limitation](VALIDATION.md#historical-noise-trial-consolidation-and-raw-retention-2026-09-10).
This is a numerically checked consolidation, not yet full shared-runner migration
or scientific validation of the older artifacts. The focused suite passed 28 tests.

Final checks: 39 replay/retention/ratchet tests passed, followed by 33 focused tests
including the new minimum-seed guard. Raw comparisons use canonical stable-ID
snapshots. Both obsolete ratchet allowances were removed; none were raised.
No backend code or historical scientific artifact changed.


## Shared result storage (2026-09-10)

Legacy ExperimentResult and the migrated runner now share strict exclusive JSON
storage; same-second saves refuse to overwrite, and unsupported/nonfinite data
cannot silently become strings or nonstandard JSON. Loads use the loss-aware
reader. Statistical flags remain booleans; the noise study represents explicitly
undefined tests as null plus their reason, with significant=False.
See [validation and compatibility limits](VALIDATION.md#shared-exclusive-result-storage-2026-09-10).
The focused storage/runner/historical replay suite passed 108 tests. Missing legacy
provenance and full experiment-runner migration remain separate obligations.

The full workflow-selected contract gate passed 1446 tests (one skip). Legacy
experiments with unresolved nonfinite statistics now stop at storage rather than
silently writing invalid evidence; the historical noise study has an explicit
undefined-statistic representation. Historical files remain unchanged.

The final fused/CUDA gate passed 122 tests in 35.25s. No implementation changed
after the gates. Full-package closure and legacy provenance migration remain open.


## Explicit execution status (2026-09-10)

Legacy result success is now a boolean at construction, assignment and storage.
Loading requires an explicit status, so neither "False" nor an absent flag can be
interpreted as successful execution. Failed outcomes remain representable; execution
success still does not certify scientific adoption. See
[validation](VALIDATION.md#explicit-legacy-execution-status-2026-09-10).
No backend or numerical experiment code changed.

The final combined suite passed 120 tests in 23.33s. The 27 specification/register/
ratchet checks also passed; exact ordering is retained in validation.


## Historical noise runner adapter (2026-09-10)

The old CLI now forwards to research.runner's historical-noise adapter and requires
--tag; --quick aliases VOID smoke. Seeds, grids and round counts are recorded and
consumed explicitly. Historical study verdicts are UNADOPTED, not scientific PASS.
The source-linked migration registration and validation precede the smoke run;
see VALIDATION.md#historical-noise-shared-runner-migration-2026-09-10.


The historical-noise migration now has archived VOID smoke evidence with exact
direct/adapter agreement, 1465 contract checks and 122 fused/CUDA parity checks
passing. Aggregate summaries no longer infer scientific PASS from execution or
missing metrics; unimplemented --full fails before work. Five summary controls
pass and are wired into CI. Aggregate call configurations remain unmigrated.
See VALIDATION.md#historical-noise-migration-evidence-and-aggregate-reporting.


## Legacy argument boundaries (2026-09-10)

The aggregate's six incompatible call configurations now fail together before any
experiment constructor. All eight producers reject unknown run arguments; seven
unused keyword captures were removed with module AST parity apart from those
signatures. The eight declared calls/order are preserved as one inventory. This
exposes previously ignored grids; it does not constitute their protocol migration.
The separate primitives/run_all.py has the same obsolete caller issue, now stopped
at the producer boundary. See VALIDATION.md#legacy-experiment-configuration-preflight-2026-09-10.


## Projection dead weight probe (2026-09-10)

H4 always returned 1 because area.connectomes does not exist. It now measures the
actual pre-evaluation recurrent matrix; the test against 1 was removed because
selection already biases beta-zero ratios. Fifteen source-177dbbc replay cases
preserve trajectories and weights. H3 is documented/tested as A-driven regeneration,
not cue recovery; all evaluation phases still learn. Full projection protocol
migration remains open. See VALIDATION.md#historical-projection-measurement-repair-2026-09-10.


## Projection runner preparation (2026-09-10)

The corrected projection protocol is now configurable and retains per-seed cells;
its old CLI forwards to a tagged shared-runner adapter. Source-linked version-2
migration registration precedes the smoke. Seed resolution and undefined-null
records are shared with historical noise. See VALIDATION.md#configurable-projection-migration-2026-09-10.


The projection migration smoke now has an archived source record and exact
direct/runner equivalence for six cells. Shared seed configuration is separated
from statistics in research/experiment_config.py. The full contract gate had 1516
passes plus one scanner-classification failure, resolved with an 80-test rerun;
no baseline relaxation or existing t-test implementation change. Scientific status
remains VOID and old provenance gaps remain open. Details in VALIDATION.md.

The dedicated fused/CUDA parity suite also passed 122 tests (84.41s).


## Projection stopping records (2026-09-10)

Version 3 distinguishes timeout from last-round convergence and keeps each seed's
status/time. Censored seeds block the ordinary scaling fit. Rule window/threshold
are recorded configuration; default trajectories and final weights are preserved.
The similar legacy scaling helper has a different initial activation schedule and
still requires its own migration. See VALIDATION.md#projection-convergence-stopping-2026-09-10.


The v3 smoke confirms all six H1 trials timed out at eight rounds, so the previous
scalar cannot be called a convergence time. The new recorded statuses block the
fit. Direct/v3 outputs match exactly; v2 elapsed work and all other raw measurements
are unchanged. Sixty-four focused/specification/ratchet checks pass. Artifact and
validation details are in the linked validation section.


## Shared convergence phase (2026-09-10)

Projection and scaling now share stopping and descriptive fitting while preserving
scaling's extra initial stimulus activation. Six new scaling and fifteen projection
replays preserve trajectories/weights; projection's complete v3 smoke also matches.
Scaling retains censored seed records and no longer infers complexity classes from
fit coefficients. The phase uses canonical snapshots and a constant-history streak.
Remaining scaling grid/runner migration is separate. See VALIDATION.md#shared-convergence-phase-and-scaling-correction-2026-09-10.


## Scaling runner preparation (2026-09-10)

The corrected scaling study now exposes its grid, seed identities, initialization,
evaluation and stopping schedule through a tagged adapter. Old CLI requires --tag;
--quick is VOID and full output UNADOPTED. Registration precedes smoke execution.
See VALIDATION.md#configurable-scaling-runner-migration-2026-09-10.


Scaling's recorded smoke now matches direct execution exactly, and its source
archive validates. The full configured contract workflow passed 1548 tests with
one skip after the accumulated convergence/runner changes. Both smoke cells retain
three timeouts each and remain VOID. Old aggregate-grid translation remains open.

Dedicated fused/CUDA parity also passed 122 tests in 58.38s.


## Convergence boundary checks (2026-09-10)

Fit inputs reject Boolean/fractional populations even when censored. Direct stopping
records reject ambiguous truthy statuses and invalid elapsed counts. Bounded
exhaustive checks cover 1024 history/window combinations; both archived projection
and scaling smoke outputs remain exact. Eighty-eight focused tests pass. See
VALIDATION.md#convergence-record-boundaries-and-bounded-equivalence-2026-09-10.


## Phase-grid reporting correction (2026-09-10)

The phase study no longer calls a mean-based crossing a phase boundary. It records
nominal interval status, all raw seed values, and explicit absence of a sampled
crossing. Six replay fixtures preserve dynamics; 26 focused/aggregate tests pass.
Evaluation still learns, and full runner/configuration migration remains open.
See VALIDATION.md#historical-phase-grid-interpretation-2026-09-10.


## Phase runner preparation (2026-09-10)

Phase grids, H3 k/beta, schedules, threshold and seed IDs are explicit and validated
before compute; duplicate realized assembly sizes fail. Its tagged shared-runner
adapter and migration registration are ready, with smoke still VOID and full
UNADOPTED. See VALIDATION.md#configurable-phase-grid-migration-2026-09-10.


The phase smoke now has a validated archive and exact direct/runner equivalence
for six cells, with both sampled crossings explicitly absent. Scientific status
remains VOID. The old aggregate grid is still not silently translated.


## Numeric configuration conversion (2026-09-10)

Fixed a nonzero rational silently resolving to zero in the new grid resolver.
Overflow and underflow now raise configuration errors; valid subnormals survive.
Phase configs and result parameters retain the actual normalized scalar values.
156 historical-study checks pass and the archived phase smoke remains exact.
See VALIDATION.md#numeric-grid-conversion-and-recorded-values-2026-09-10.


## Historical adapter composition (2026-09-10)

Four adapters now declare immutable HistoricalStudy objects and share execution/
CLI logic. Record identity/version/engine/mode are checked before construction.
All four archived smokes retain their exact observations, verdict and scope;
156 existing and seven new focused controls pass. Protocol parameter factories and
source/registration identities stay separate. See VALIDATION.md#historical-adapter-composition-2026-09-10.


## Recoverable declared run inputs (2026-09-10)

Before CLI parameter-file integration, fixed a runner provenance gap: input hashes
had no archived bytes. Schema 4 captures and validates the complete declared input
inventory; duplicate aliases fail before computation. Older archives remain readable
without retroactive recovery claims. Six existing study artifacts validate unchanged.
See VALIDATION.md#recoverable-declared-run-inputs-2026-09-10. CLI overrides remain open.

139 targeted checks pass, including both ratchets and register/specification gates; Ruff and diff checks pass. No new numerical or GPU claims.


## Historical CLI parameter files (2026-09-10)

All four historical adapters expose --parameters through their shared implementation.
Overrides retain exact file bytes and all resolved values; stale parse/capture hashes
fail before reservation. 93 adapter/runner checks pass. Domain checks remain in the
producers and customized runs remain VOID/UNADOPTED. Registered a phase smoke for
CLI/direct equivalence before running it. See VALIDATION.md#historical-cli-parameter-files-2026-09-10.


The configured phase smoke now passes direct/CLI equivalence and archive checks:
six cells, seeds1/2/3, only test_rounds overridden to 1. Exact configuration bytes
are retained; evidence source is a1cbecd and status VOID. The additional 200
historical/storage/specification/ratchet checks pass. Defaults are unchanged.


## Historical association trial semantics (2026-09-10)

Added a code-derived card and negative interpretation controls before consolidating
the duplicated setup. B corruption is unread by evaluation, which continues learning;
references precede association. Nine captured trials remain exact and the observed
engine is pinned. Removed two obsolete automatic-engine allowances. The file is
54 lines shorter. Outer experiment migration is next; see
VALIDATION.md#historical-association-trial-semantics-2026-09-10.

Final validation: 34 trial/specification/ratchet checks pass; Ruff and diff checks pass. No new GPU or scientific adoption claim.


## Configurable association harness (2026-09-10)

Association now uses the common tagged adapter and archived parameter files. Grids,
schedules and seed IDs are explicit; raw observations and paired differences persist.
Zero association rounds is an explicit control. Replaced misleading constant-difference
p=1 with undefined test fields and the actual paired-difference interval. Trial replay
fixtures stay exact. See VALIDATION.md#configurable-association-harness-2026-09-10.


Association smoke acceptance now passes: recorded source174f5f8, exact direct/runner
metrics/raw_data/parameters/success, valid archive and all seed vectors retained.
89 targeted checks pass. Scientific status remains VOID; old evidence is untouched.


## Shared paired reporting (2026-09-10)

Projection and association share the canonical keyed paired-difference report.
Projection becomes version 4 because its old constant-difference p=1 output changes;
association must remain exact. Other legacy paired_ttest callers remain open.
111 focused checks pass. See VALIDATION.md#shared-paired-reporting-2026-09-10.


Version 4 projection smoke passes the specified version 3 comparison and independent
paired-report recomputation. Association replay is exact. Archive source2fb207d;
48 final contract/register/ratchet gates pass. Other legacy paired callers remain open.


## Integrated migration checkpoint (2026-09-10)

Clean source9844e4b: all71 configured CPU contract modules passed locally:
1668 passed,1 skipped,6 warnings in188.77s. Skip is the unavailable optional CuPy
example. Dedicated fused/CUDA parity:122 passed,no skips,11 warnings in43.30s on
RTX3080 with the existing VS2022 build environment. No engine/dependency changes.
This does not supersede open full-package failures or prove general backend parity.
See VALIDATION.md#integrated-migration-checkpoint-2026-09-10.


## Historical merge trial controls (2026-09-10)

Six captured merge/recovery trials remain exact after shared setup extraction and
engine pinning. New controls show that C resets do not clear training or affect
positive-round drive, evaluation learns, and the maximum-overlap score can be perfect
while one parent is absent. Production file32 lines shorter; removed two unpinned
engine allowances. Outer harness migration remains open. See
VALIDATION.md#historical-merge-trial-controls-2026-09-10.

Final merge/aggregate/specification/ratchet validation:45 passed; Ruff and diff checks pass.


## Configurable merge harness (2026-09-10)

Merge now uses the shared tagged runner and archived configuration path. Reports
name mean/max parent overlap directly and retain every parent vector/seed; old
misleading report keys are absent. Six trial fixtures stay exact.29 focused tests
pass; smoke acceptance registered before execution. See
VALIDATION.md#configurable-merge-harness-2026-09-10.


Merge smoke acceptance passes: five cells, seeds1/2/3, sourceb8984cd, exact direct
metrics/raw_data/parameters/success and valid archive.84 targeted tests pass; status
VOID. This finishes the historical merge harness migration, not a scientific
validation of general composition or the broader library unification.


## Temporal mechanism position-pooling audit (2026-09-10)

Important correction: TM-9's supposed distractor overlap pools all noninitial
positions, including number-marked agreement words. A constructed control produces
contrast0.5 with zero distractor contrast. Arithmetic matches review commit3334876.
Suspended the0.11/0.22 mechanism interpretation in register/onboarding/notes, while
preserving separate prediction results. Old collection now fails before GPU work.
31 audit/register/specification/ratchet checks pass. Next is a position-specific
collector and registered rerun, not a readout study assuming the0.11 premise.
See VALIDATION.md#temporal-mechanism-position-pooling-audit-2026-09-10.


## Position-specific temporal observation boundary (2026-09-10)

The replacement CPU analyzer validates complete sentence/token/arc frames and
reports each position separately.45 checks pass, including the confounded negative,
positive signal and generated corpora at gaps1/2/3/6. It is not yet wired to GPU
collection; TM-9 remains suspended pending a registered rerun. See
VALIDATION.md#position-specific-temporal-observation-boundary-2026-09-10.

## Frozen temporal capture (2026-09-10)

capture_chain_arcs now records complete position-labelled frames, resets sentence
boundaries and advances frozen carry. Preflight rejects invalid corpus/seed inputs;
endpoint learned-tensor fingerprints reject plasticity or refraction mutation.
54 CPU checks and two real CUDA copy/induced schedule comparisons pass, no CUDA
skips. See VALIDATION.md#frozen-temporal-capture-2026-09-10 for scope and limitations.
This is instrument validation only. Next: integrate a versioned run artifact and
commit a replacement registration before collecting scientific g=0/g=1 data.
No suspended temporal mechanism claim is restored, and no dev/master merge occurs.

## Registered temporal-position runner and smoke (2026-09-10)

Preregistration commit2f9f584 precedes implementation45b44ab. The shared runner now
owns a fixed, fresh-seed position protocol with g0/g1/state-blind arms and raw frames.
42 focused CPU,19 ratchet/specification, and79 runner/source checks pass. A real CUDA
smoke on three seeds completed with108 frames per seed/arm; archive validation is
clean, duplicate tag reuse is refused, and the result is correctly VOID. The smoke
artifact is linked from PREREG_temporal_positions.md. Next run the fixed seeds82..101
study under the one-GPU rule, then adjudicate every bar before any register change.

## Position-specific temporal result (2026-09-10)

The fixed20-seed CUDA run completed from65cd4fd and its archive validates. All13,500
raw frames recompute exactly; paired corpora and g1/blind training fingerprints match.
TP-1 through TP-4 pass. D is.0258 g0,.1849 g1,.0022 blind; paired amplification is
.1591 [.1400,.1781]. g0 drops from.0479 at distractor1 to.0037 at distractor2; g1
retains.2032 then.1666. This answers the former readout-bottleneck premise: at gap2,
g0 has no representation at the second distractor immediately before agreement.
Register/onboarding/research notes now use the corrected mechanism while preserving
the historical pooled values as void. Next scientific task is registered gaps3..6
decay, then clip-window retention/new-learning and matched sequence baselines.

## Compressed raw-evidence contract (2026-09-10)

Schema5 at213a907 separates indexed results from deterministic gzip raw JSON. Both
compressed/decoded sizes and hashes are mandatory; exact file inventory and strict
decoded JSON are validated.111 focused checks pass. A real CUDA migration smoke is
exactly equal to the prior schema4 smoke after reconstruction and reduces observation
storage from452,602 bytes to22,752 results +22,371 attachment. It remains VOID.
Existing artifacts were not rewritten. Use `ExperimentOutput` for future large raw
studies and `load_json_attachment` for validated reads. The schema5 smoke artifact
is committed with this checkpoint; no dev/master merge occurs.

## Active evidence-graph gate (2026-09-10)

All19 tracked shared-runner results now validate and link from their recorded
registration; the schema5 smoke was the sole missing edge and is now linked. CPU CI
walks this active graph. A constructed valid-but-unlinked result fails until its exact
registration link is added.85 runner/graph checks pass. The legacy audit still has
367 candidate result orphans,28 preregs without resolved result links and741
unresolved textual references; these are inventory categories requiring disposition,
not a claim that every item is invalid. New runner evidence cannot add to that debt.

## Typed register evidence (2026-09-10)

All15 MEASURED entries now expose43 typed, resolving evidence files and explicit
provenance gaps. Roles distinguish artifact/registration/producer/analysis/log.
Thirteen entries retain a gap; seven have no artifact-role file. The renderer exposes
those facts and onboarding requires them. Missing/escaping paths and unacknowledged
absence fail tests;10 register tests pass. This makes the next migration queue
machine-readable without pretending legacy scripts or logs are immutable evidence.

## Single-pass ERP calibration and DIRECT retraction (2026-09-11)

The five ambiguous XPASS cases from the complete package audit are gone. ERP
calibration observes one sample set and relabels it after tuning; its former full
mode silently reparsed every frame and therefore compared a later mutable model
state with the fast mode. A unit gate fixes collection count at one, and the VP
liveness checks are ordinary passing contracts. Repeated parsing and raw-P600
saturation remain two distinct strict expected failures.

DIRECT's old overlap instrument is now retracted at every executable boundary:
the four public operations, recorder, golden and parity runner. Its readout was
insensitive to wiping the learned fiber. Historical values remain named as such;
a successor requires a new protocol ID, synaptic-asymmetry readout and wipe
negative control. The reconstructed failure-cluster run is 97 passed, 9 skipped,
2 xfailed and zero xpassed in 317.61 seconds. This closes the known audit inventory,
not the whole-library unification or a current complete-package run.

The cross-cutting methodology, specification, register, evidence-graph and parity
gate reports 73 passed and 11 skipped in 59.32 seconds.

The optional live-reference subprocess now seeds both the reference Brain and
NumPy's global RNG used by SciPy. Same-seed executions match exactly. The seeded
rounds=20 ensemble over seeds42..46 remains bimodal (0,.7625,.325,.8125,0;
0.3800 +/- 0.4909), so the near-chance claim is now a stable strict expected
failure. Verified against upstream/master 81e4297: one pass, one expected failure.

The non-strict compiled-fidelity test was an impossible zero-drive probe: both
policies exit before winner selection. It is removed in favor of the existing
active role-pathway sampling contract, which passes in1.88s and now documents
its live-drive precondition. Three newly exposed unused imports were removed.

The non-strict N/k "bridge strength" sweep was also measuring raw overlap between
consecutive assemblies, while sequence order lives in directed weights. That proxy
is removed; the module now names recall as its behavioral gate. Repetition and
three-item recall controls pass (2 tests,7.16s) with the expected sampled-engine
warning, so they remain API checks rather than scientific evidence.

All four remaining non-strict scientific xfails are now strict: dormant mutual
inhibition, the emergent-parser multi-mood path, and two shared-syntax word-order
cases. An AST methodology ratchet rejects any future non-strict pytest xfail and
has its own constructed negative. The combined gate is5 passed,4 strict xfailed
in36.05s; there are no non-strict expected failures left in package tests.

## Current-head package gate and expected-failure closure (2026-09-11)

The complete non-slow package gate at `c1f311b`, under the repository CUDA
developer-shell setup, is green: 3,244 passed, 65 skipped, 143 deselected,
7 expected failures, no failures/errors/XPASS, 318 warnings and 10 passing
subtests in 1,654.18 seconds. `package-audit-c1f311b.json` is the committed
receipt with exact expected-failure and duration inventory plus raw-artifact
hashes. This is a software baseline, not slow-suite coverage, backend simulation
proof, or a scientific adoption.

One of the seven expected failures used `unittest.expectedFailure`, outside the
pytest-only strictness ratchet. The mod-3 FSM now uses a strict pytest marker with
the A1 drift diagnosis. The AST gate also rejects both unittest decorator forms,
with a constructed negative. Focused verification is 6 passed and 1 strict
expected failure in 23.26 seconds; Ruff and diff checks pass. This corrects the
previous broader wording: there were no non-strict *pytest* xfails, but one
non-strict unittest expected failure remained until this checkpoint.

## Shared executable-round transport (2026-09-11)

`explicit-area-round-v1` now has its own strict packaged schema and 12-case
acceptance corpus. `ExplicitRound.from_document` uses that schema; Python and
Rust consume the same cases, and the public Python IR package exports the
instruction and validator. This does not reinterpret the permissive legacy
`projection.schema.json`. Backend/state checks remain a separate preflight.

The Python execution, Brain lowering, wire and specification gate is 123 passed.
The assembly-ir crate is 4 passed, with rustfmt and Clippy clean under denied
warnings. An isolated wheel contains the schema and corpus. Rust transport still
has no numerical executor, and no Lean-to-NumPy simulation proof is claimed.

## Pure formal explicit-round lowering (2026-09-11)

Added `formal/AssemblyIR/Projection.lean`: a scaled-integer semantic round, a
separate dense-kernel instruction, executable state admission, and a concrete
`Simulates` proof for their field lowering. The proof lifts to finite schedules
and winner observations. Frame laws cover frozen weights, non-target winners and
the selected target cap. Admission rejects registration, dimension, index,
duplicate-source, `k` and selector-output errors before returning state.

Positive controls compute drive `[4,0]` and weight 3 -> 4. A lowering that drops
learning remains at 3; three malformed cases reject. `lake build` with warnings
as errors and `leanchecker AssemblyIR.Projection` pass, with no `sorryAx`.
This is an internal pure lowering proof. The schema-to-Lean identity and concrete
NumPy/Rust/CUDA arithmetic simulations remain open.

## Package lint repair (2026-09-11)

The release/documented `ruff check neural_assemblies/` gate had 225 findings
(164 runtime, 61 tests). Reviewed cleanup makes it pass and removes 130 net lines
across 107 files. Calls with possible effects remain; dead pure expressions do
not. Two non-cosmetic fixes are the `typing.Sequence` versus assembly `Sequence`
name collision in the scaffold API and a noun-after-determiner test that formerly
discarded its own predicate. It now asserts the stated behavior.

Compilation, Ruff, diff checks and a focused cross-engine/sequence/prediction/IR
gate pass (40 passed, 1 skipped). A complete non-slow rerun is still required
before treating this broad cleanup as stable. Research remains a separate lint
debt of 715 findings; no claim of whole-tree lint cleanliness is made.

The complete rerun at `991642e` did its job: 3,256 passed, while one ten-test
cluster failed before computation. Ruff had removed the import from a handwritten
CuPy availability `try`, leaving its success flag unconditional in an environment
without importable CuPy. The kernel code itself was never reached.

Capability admission is now centralized as `core.backend.cupy_available()`.
It preserves the required torch-first Windows DLL order, imports CuPy, performs
a device allocation, and caches the result. Automatic backend selection and the
kernel tests share that decision. The focused gate is 24 passed, 10 correctly
skipped; Ruff, compilation and diff checks pass.

The complete rerun at `02cacc9` is green: 3,256 passed, 65 skipped,
143 deselected, 7 strict expected failures, zero failures/errors/XPASS,
318 warnings and 10 passing subtests in 2,166.44 seconds. The committed
`package-audit-02cacc9.json` receipt records the exact invocation, environment,
expected-failure reasons, slowest nodes and hashes of both raw artifacts. This
is the current non-slow package baseline; slow tests, unavailable CuPy execution,
scientific adoption and cross-backend semantic proof remain outside the claim.

## Engine admission errors are no longer erased (2026-09-11)

The lazy factory used to catch a mapped provider's `ImportError`, broadly reload
engines, and finally call the requested built-in name unknown. It now raises the
public `EngineUnavailableError` from the original cause. A provider that imports
without registering its mapped name has its own explicit admission message;
genuinely unmapped names retain the old unknown-name `ValueError`, and constructor
failures remain unchanged.

The implementation cites `VERIFICATION.md#contract-engine-admission`. Constructed
negative cases cover missing dependency, missing registration and invalid name.
The focused engine/spec/lazy-import gate is 68 passed; Ruff, compilation and diff
checks pass. A live `cuda_implicit` request on this machine now reports missing
`cupy` with `ModuleNotFoundError` preserved as the cause.

## Installed is distinct from usable (2026-09-11)

The root `GPU_AVAILABLE` flag only performed import-free CuPy discovery, so its
name overstated the evidence. `CUPY_INSTALLED` now names that predicate and the old
flag remains its exact compatibility alias. The lazy `cupy_available()` predicate
does the torch-first import and device allocation required for runtime admission.
Its code cites `VERIFICATION.md#contract-backend-capability`, which excludes parity,
memory sufficiency and extension compilation.

A fresh process confirms that root import loads no CuPy and that runtime availability
is false on this machine. The focused lazy-import/backend/admission/spec gate is
35 passed; Ruff, compilation and diff checks pass.

## Lean now consumes the shared round corpus (2026-09-11)

`AssemblyIR.Wire` independently decodes `explicit-area-round-v1`, retains exact
JSON decimals, and rejects lossy conversion to the caller's chosen integer scale.
`normalizeRound_identity` proves that successful conversion preserves target,
ordered sources and plasticity. Decoder cases are not copied into Lean: the
`check-wire-cases` executable consumes the same 12-case corpus as Python and Rust,
and the Lean CI job now executes it. A deliberately flipped verdict exits 1 and
names `area-source`.

Lean build, `leanchecker AssemblyIR.Wire`, and corpus execution pass. Python's
focused gate is 46 passed; Rust is 4 passed with fmt and Clippy clean. There is no
`sorryAx`. The remaining bridge is numerical: selecting a scientific scale and
relating integer semantics to float32 addition, multiplication, clipping, Rust
execution and CUDA arithmetic.

## Projection is now an inspectable operation value (2026-09-11)

`ProjectionPlan` freezes and validates every projection step before mutation, and
the public `project` function executes that exact sequence. Its attached
`OperationContract` names inputs, state reads, mutations, regime, observed outcome,
failure conditions, specification and constructed recurrent-learning control.
Incomplete contracts reject at construction; the registry is immutable and contains
only projection, so it does not imply the other operations are migrated.

The migration comparison reconstructs the former path on all three NumPy engines
and compares stable/compact winners, recruitment, owned RNG and a subsequent
read-only observation. Operation/spec tests are 127 passed, 4 skipped; public and
onboarding tests are 117 passed, 1 skipped. CI now includes the contract-object
suite. Reciprocal projection, association, merge and completion are next.

## Reciprocal projection is now an inspectable operation value (2026-09-11)

`ReciprocalProjectionPlan` freezes the one forward step and every following
forward, target-recurrent and return-edge step. Construction rejects invalid
names, self-projection, invalid rounds and implicit truthy clamp choices. Public
execution rejects absent areas and an empty source before borrowing a clamp or
calling the backend. The operation carries its exact contract and links the
disabled-learning round-trip control.

The migration gate reconstructs the former body on all three NumPy engines. It
compares both areas' stable and compact winners, recruitment, owned RNG, clamp
restoration and a subsequent read-only return observation. The contract-object
suite is 45 passed; related operation and round-trip coverage is 62 passed with 4
optional-reference skips; the CUDA-initialized Torch parity suite is 27 passed.
Contract construction now also refuses mutable plans and mutable, blank or
duplicate scientific surfaces. Association, merge and completion remain next;
this checkpoint is schedule equivalence and early rejection, not a new scientific
result or proof of autonomous bidirectional recall.

## Association is now an inspectable operation value (2026-09-11)

`AssociationPlan` freezes both sequential source phases and the joint phase. It
admits exactly the two coherent historical modes: two active fixed sources, or two
registered stimuli with evolving sources. Partial stimulus pairs, aliased areas,
bad phase counts, missing topology and empty fixed sources reject before clamp or
backend mutation. The removed `_associate_body` had about 75 lines of imperative
schedule code; the schedule now has one composable representation.

Both modes reproduce the removed helper on all three NumPy engines, including all
areas' stable/compact winners, recruitment, owned RNG, clamp restoration and the
next read-only observation. Contract objects are 74 passed; association-focused
conformance is 17 passed; the broader related operation selection is 62 passed
with 4 optional-reference skips; CUDA-initialized Torch parity is 27 passed.

The slow ventral consumer audit caught a separate methodological failure: an
empirical architecture-gap test used synthetic fallback for both sides, obtained
two perfect 1.0 scores and asserted a real-data gap. It and the full evidence
ladder now require the real MNIST CSVs before computation; absent data yields two
explicit skips in 0.52 seconds. The localization run reached 7 passes and that
failure in 494.48 seconds, then stopped. Merge and completion remain the operation
contract migrations; this change does not adopt a new association result.

## Merge partial-source semantics are now explicit (2026-09-11)

`MergePlan` freezes the first simultaneous parent step and all later recurrence
and return edges. It rejects aliased topology, bad rounds or switches, missing
areas/stimuli and inactive unstimulated parents before mutation. A one-stimulus
call must now declare its other parent `require-fixed`, `fix-current`, or
`evolving`, and the first and third modes verify current clamp state.

The caller audit found thirteen partial calls: twelve already lived inside
pinning scopes and now say `require-fixed`; `universality_composition` explicitly
says `evolving`. No unnamed static partial call remains, and an AST ratchet blocks
new ones. All touched scripts compile. The four source patterns reproduce the old
edge/state trajectory on all three NumPy engines. Contract tests are 111 passed;
the merge-focused package
selection is 17 passed. The expanded operation/public/specification/integration
gate is 345 passed with 4 optional-reference skips, and CUDA-initialized Torch
parity is 27 passed. The public workflow documents the compatibility change.

This preserves old arithmetic while making the protocol claim executable. It does
not rerun any registered study. Completion remains the last operation-card
migration before the shared plan layer covers the original five operations.

## Completion closes the original operation-plan set (2026-09-11)

`CompletionPlan` and `PreparedCompletion` now own the exact cue sampler, stable-ID
reference, compact-index cue and recurrent schedule. Seed and observation policy
are mandatory. `plastic` preserves the former path, `frozen` retains activity but
not weight changes, and `read-only` restores activity, recruitment, weights and
owned RNG. A prepared cue rejects cross-brain and stale-entry use before mutation.
The tracing API executes the same object instead of maintaining a second cue-size,
validation and sampling implementation.

All static callers name their policy. Research and compatibility paths use
`plastic` to preserve committed semantics; the onboarding investigation uses
`read-only`. The first broader boundary run caught that example because the
initial AST audit covered only package and research roots. The ratchet now includes
`examples/`, demonstrating that the new contract fails early on the old ambiguous
path.

The plastic migration is state-equivalent on all three NumPy engines. Focused
operation/trace/public coverage is 271 passed; the broader affected CPU selection
is 258 passed, 9 skipped and one expected failure. Package Ruff, compilation and
diff checks pass. The Visual Studio/CUDA Torch parity suite is 27 passed. No study
was rerun and no historical score was reinterpreted. Completion's linked negative
is the fixed-connectome teaching test that drops below .3 at beta zero and exceeds
.8 with recurrent learning; it remains instructional rather than adopted science.

The five original calculus operations now have discoverable immutable plans and
source-linked contracts. The next unification work is broader than these schedules:
organ contracts, full model semantics, numerical Lean/backend refinement, the
remaining experiment runner/evidence graph migration, index ownership and the
cpp/legacy disposition still remain open.

The full non-slow package audit at `29ab1f8` is also green: 3,405 passed, 65
skipped, 143 deselected, 7 unchanged strict expected failures, no failures/errors
or unexpected passes, 327 warnings and 10 passing subtests in 1,253.25 seconds.
`package-audit-29ab1f8.json` carries the exact command, JUnit accounting, xfail
reasons, slowest cases and hashes of both raw artifacts. This supersedes
`02cacc9` as the non-slow software baseline; slow tests and unavailable CuPy
execution remain outside the claim.

## Sampled recurrence can now fail before RNG consumption (2026-09-11)

The lazy NumPy engine now carries `SampledRecurrencePolicy`: `warn` preserves the
one-shot audit warning, `acknowledged` makes deliberate comparison explicit and
silent, and `forbid` rejects before the projection derives its child RNG or
mutates connectome state. The rejection test compares RNG, winners, recruitment
and recurrent weights. Materialized, fixed and read-only targets remain admitted
because they do not sample candidates.

The Brain and a supplied sampled engine must agree on the normalized enum.
Parity and operation migration fixtures say `acknowledged`; CUDA Torch parity is
27 passed with no sampled warning output. Focused policy/public/operation/specification
coverage is 293 passed, and constructor/clone/checkpoint/backend coverage is 88 passed with
3 optional skips. Default `warn` still names `PREREG_sampler_audit.md` and tells
the caller how to acknowledge a deliberate comparison. This addresses warning
noise without weakening the accidental-use guard.

## The first complete primary-path model profile is executable (2026-09-11)

`ModelSemantics` now names eight result-changing choices together: connectome
realization, candidate domain, stimulus drive, default tie rule, arithmetic,
normalization, clipped versus unbounded multiplicative plasticity, and the exact
weight ceiling. It is a frozen object with strict wire parsing and closed
categorical fields. Every primary CPU engine and
`torch_sparse` describes its actual default path; `Brain(model_semantics=...)`
rejects any requested/actual difference before topology registration.

This catches two concrete old confusions. The `ASSEMBLIES_STREAM_INIT=1`
compatibility switch changes graph identity and now fails a content-addressed
expectation. The default top-k tie behavior is distinct from the configurable
winner-policy `value_then_index` rule: exact/dense CPU use lowest neuron ID,
sampled NumPy exposes partition order, and Torch exposes backend top-k order.
Torch dense drive also has its own candidate-domain value because scoring all
neurons does not make their unmaterialized drives fixed.

The focused CPU gate is 171 passed and 3 optional skips; a broader
constructor/checkpoint/backend/operation gate is 234 passed and 3 optional skips;
CUDA Torch parity/profile is 29 passed; Ruff is clean. This resolves the requested object for connectome,
stimulus, tie and arithmetic semantics at the primary engine boundary. It does
not yet fold area-local mechanisms, operation schedules, readout definitions, or
the separate hashed-organ classes into one run protocol, and it does not prove
numeric refinement between engines.

## Model identity now survives into evidence (2026-09-11)

Run schema 6 requires the complete canonical `ModelSemantics` document whenever
`engine` names a Brain backend. The runner reconstructs that engine's default
profile using the declared normalization and weight ceiling, compares every
field, and rejects before tag reservation on omission or drift. The artifact
validator repeats strict parsing and canonical encoding checks. Its archive gate
now covers schema 6; the tests caught that the previous explicit `(4, 5)` check
would have stopped detecting a missing archived input after the version bump.

The real historical-association schema-6 smoke is linked from its registration
and validates. It reproduces the prior seeds1/2/3 artifact exactly for metrics,
raw data, producer parameters, success/error state, scope, and VOID verdict. Its
effective explicit-area profile records fixed dense content-addressed wiring,
all-neuron candidates, fixed Bernoulli afferent counts, lowest-ID ties, float32,
no normalization, multiplicative clipping, and ceiling 20.0. The producer's own
parameters continue to name the `numpy_sparse` Brain router and
`numpy_explicit` area owner.

The broader runner/adapter/experiment/research-contract/methodology/specification
gate is 158 passed and Ruff is clean. Hashed organs record `null` instead of borrowing a false Brain profile;
their organ-level semantic objects remain the next required boundary.

## Hashed-organ provenance checkpoint (2026-09-11)

Run schema 7 now records strict `execution_semantics` for both Brain and hashed
organ paths. `OrganSemantics` captures the reusable substrate plus state code,
training/inference schedule, exact tie jitter, effective refraction charges,
horizon, prediction gain and feature-register presence. Named profiles preserve
paired-arm differences. The memory, assigned FSM and transducer constructors
derive and compare the profile before allocating device state.

The migration exposed and fixed one semantic naming bug: temporal positions was
running `HashedTransducer` while recording `hashed_arc_fsm`; it now records
`hashed_transducer`. The four migrated hashed runners pass their recorded profile
back into the actual organ constructor. Schema 1-6 artifacts remain readable;
the committed schema-6 historical replay stays unchanged.

Evidence: 183 CPU contract/migration/evidence/specification tests pass; 33 CUDA
hashed substrate/FSM/transducer/semantic tests pass. The real fused-CUDA smoke at
`research/results/runs/sequence.a1-horizon/organ-schema7-smoke-20260911/`
validates cleanly, records the FSM's Binomial stimulus law and lowest-ID tie
rule, and is correctly `VOID`. Remaining work is the mixed-engine profile graph,
actual-record-driven construction throughout all migrated experiments, aligner
semantics, broader experiment migration and numerical refinement obligations.

Before the final two focused guard tests were added, the full CUDA-enabled
non-slow package suite also passed: 3,456 passed, 65 skipped, 143 deselected, 7
expected xfails and 10 subtests in 21m55s. Its 303
warnings expose a follow-up DX problem: many legacy tests deliberately use sampled
recurrence without setting the acknowledgement policy, so the intended safety
warning becomes noisy in the broad lane.
