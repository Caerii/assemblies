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
