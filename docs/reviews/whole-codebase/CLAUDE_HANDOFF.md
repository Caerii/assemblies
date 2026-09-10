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
