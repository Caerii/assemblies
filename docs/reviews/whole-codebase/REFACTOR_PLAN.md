# Assembly-calculus unification: implementation and evidence plan

Baseline: `3334876cf57991d27b1651c343f4640b4bcffaa6`, isolated branch
`audit/astra-codebase-20260909`. This is an active migration plan, not a claim
that the repository has been fully refactored or its science independently reproduced.

The purpose is to make the scientific meaning of a computation inspectable before
it runs and recoverable from its artifact afterwards. A newcomer taking an invalid
old path must receive an actionable error or a clearly unjudged result.

## Coverage and limits

`coverage.tsv` inventories all 2,501 baseline tracked files with content hashes.
All 1,183 Python files parse; 23 notebooks contain 133 code cells. The source
inventory is about 309,000 lines, including historical implementations. Structural
coverage is not semantic certification. The manifest marks selected source
inspection conservatively; those marks do not imply every line was reviewed.
No GPU study, dataset download, or historical result regeneration was performed.

The earlier report at `../2026-09-09-codebase-and-research-organization.md` is a
historical review. Claude's corrections at this baseline supersede its interval,
README wording, SEQ-REGIME wording, and ratchet-test findings. The theory and two
ratchet test groups now pass: 18 tests. The claim that no historical adopted
verdict changes with corrected intervals has not been independently recalculated.
The capacity protocol change was documented in an amendment; its missing runtime
identity is the issue, not an undocumented change.

## Implemented in the isolated migration

- Shared ensembles reject duplicate/nonfinite observations and retain seed keys;
  paired comparisons check key identity. Existing research interval helpers now
  delegate to that implementation. Unjudged studies cannot report PASS.
- The runner reserves an exclusive run directory before computation, records
  source contents, commit, registration, protocol, seeds, parameters and input
  artifacts, and preserves failures. A1 horizon and capacity scaling use it
  through both their new and original entry points.
- Capacity configuration is immutable; observations retain (arm,n,k,seed).
  Version 2 records ensemble intervals and no longer emits the unsound automatic
  slope verdict. Existing plotters need a versioned artifact reader.
- Artifact validation checks run identity and file edges; the historical
  literal-link audit emits candidate orphans and unresolved links for review.
  This is not yet a complete typed evidence graph.
- Every measured register entry names its engine or an explicit provenance gap.
  RATE-HETEROGENEITY remains UNRECORDED; parts of AC-CAP remain unresolved.
- Sampled recurrent projection warns once per engine and names the sampler audit.
  Compact-to-neuron conversion and public activation reject invalid indices.
  Public injected winners and supervision validate index spaces before mutation.
  Complete owned-index migration and full model-semantics configuration remain open.
- The CPU teaching example trains with recurrence on a fixed connectome, probes
  without learning, and includes a learning-disabled control across paired seeds.
- Real-data MNIST goldens refuse synthetic substitution. Missing expected golden
  metrics, absent criteria and nonfinite bound measurements cannot pass. Nested
  cross-language scaling observations now reach an actual comparator.
- Explicit engine requests outrank environment defaults. Training cache identity
  hashes source contents rather than size/mtime, invalidating old cache identities.
- Repository-owned CUDA discovery/check scripts and preparation instructions are
  present. A CPU research-contract pull-request workflow is present; its hosted
  execution has not been observed.

See [the executable research workflow](../../research_workflow.md) for current
commands and compatibility changes. These are instrument repairs, not new
scientific results. The full migration below remains open: most experiments still
have legacy entry points, operations lack complete constructed-control contracts,
parser state ownership is still fragmented, and cpp/legacy disposition is not
finished. No files were deleted merely because static imports did not reach them.

## Current integration boundaries (2026-09-10)

This table consolidates the later implementation increments. Detailed controls and
limitations live in [VALIDATION.md](VALIDATION.md), while source-linked contracts
live in [IR verification](../../../neural_assemblies/ir/VERIFICATION.md). A passing
contract suite establishes only the behavior those controls exercise.

| Boundary | Implemented and checked | Remaining acceptance work |
|---|---|---|
| Runtime configuration | Immutable homeostasis and competition policies; explicit area/stimulus identity; shared numeric and slot validation; executing-owner LRI, refraction, policy and noise controls | Complete model definition including graph/stimulus/tie/arithmetic semantics; remaining topology/plasticity controls; general registration transactionality |
| Execution IR | Sequential `ExplicitRound` validates and executes a restricted dense CPU profile; public Brain execution follows the same owner | Broader operation schedules and organs; backend adapters with explicit state relations; complete cross-compiler integration |
| Configuration transport | Protocol, homeostasis and competition schemas share Python/Rust acceptance corpora; runner fixtures reconstruct recorded configuration | Production experiment adoption; full model document; toolchain/executable identities and legacy document migration |
| Proofs | Lean composition/domain/frame rules and masked-update invariants; source links name implemented contracts | Concrete backend step simulation and readout compatibility; no abstract theorem certifies a numerical backend by itself |
| Observation and persistence | Isolated cloning/readouts; shared environment/source fingerprints; atomic checkpoint publication | The held-out classification regression and readout comparability; remaining parser state composition and old checkpoint compatibility |
| Research workflow | Two experiment adapters, immutable tagged runs, source/environment identity and explicit evidence status | Remaining experiment migrations; full historical replay of the two adapters; complete typed evidence graph and mechanism-specific negative controls |

The primary next gates are the broader CPU package audit, semantic repair of its
failures, and GPU/historical replay on the review branches. More passing contract
tests cannot substitute for those gates. No change has been merged to dev/master
as part of this isolated migration.

## What is actually in this repository

| Current code | Scientific role | Direction |
|---|---|---|
| `core/brain.py`, `area.py`, `engine.py`, engine subpackages | Runtime facade, topology, state, numerical execution | Retain public facade; one authoritative owner per mutable field; explicit model capabilities |
| `core/numpy_engine/_exact.py` | Fixed hash graph, graded stimulus counts, CPU exact drive | Reference execution path for small reproducible investigations; preserve independent conformance checks |
| `core/numpy_engine/_sparse.py` | Sampled/recruited or materialized execution, depending on state | Identify semantics per area; guard sampled recurrence; never label it only "exact" |
| `core/torch_engine/_memory.py`, hashed FSM/transducer implementations | Specialized batched GPU models | Keep specialization; adapt to shared protocol and evidence contracts, not a universal CPU-array engine interface |
| `assembly_calculus/ops.py` | Projection schedules | Make schedule, mutation, regime, probe and observed outcome explicit operation contracts |
| `assembly_calculus/emergent/` | Curriculum, parsing, prediction, evaluation and execution | Split state ownership and explicit collaborators gradually behind the existing facade |
| `assembly_calculus/emergent/training/` | Corpus compilation and training schedules | Reuse; freeze resolved schedules and exposure counts into protocol records |
| `assembly_calculus/emergent/evaluation/erp/protocol.py` | Existing immutable ERP protocol | Extend this successful pattern; do not invent a competing ERP configuration |
| `diagnostics.py`, `research/harness.py`, `research/experiments/base.py`, `_substrate.py` | Overlapping statistics, study orchestration, probe helpers | One statistics owner; one experiment entry point; one run artifact writer |
| `theory.py`, `ir/`, `parity/`, research registrations and results | Claims and several partial evidence registries | Typed edges and validation; keep literature reproduction distinct from newly adopted research |
| `programs/`, `language/`, `lexicon/`, `nemo/`, `reference/` | Several neural and hybrid language/computation systems | Label mechanism and execution route; retain useful independent reference implementations |
| `assembly_calculus/batched_trainer.py` | Additive bridge learner / sparse bounded context learner | Explicit alternative models or baselines; not an equivalent backend for fixed-graph multiplicative plasticity |
| `compute/hyperdimensional.py`, spiking simulators | Different computational models | Separate experimental surfaces; repair demonstrated failures only if the model remains supported |
| `cpp/`, `legacy/`, `scripts/`, notebooks | Accelerator implementations, hardware probes, history, demonstrations | Per-file disposition with importer/build/evidence reasons before any move or deletion |

Paths in this table are relative to `neural_assemblies/` unless otherwise stated.

## Confirmed hazards beyond the earlier review

1. `research.harness.study` accepted duplicate seeds, empty reported metrics, and
   missing registered metric names. Unjudged metrics rendered as PASS. The CPU
   probe returned `True` for an empty study with an accuracy bar.
2. `diagnostics.Ensemble` discarded keys; paired comparisons checked only length.
   A permutation of seeds could therefore become a different experiment.
3. `research/experiments/base.summarize` still used `1.96` after diagnostics was
   corrected. For values 0..19 its half-width was 2.592836 vs 2.768811. Other
   local statistical implementations, including `_substrate.py`, remain to audit.
4. `training/perf.resolve_engine` applied the environment before an explicit
   requested engine. The probe requested `numpy_exact` and received `numpy_sparse`.
5. Harness provenance recorded a file count as its source fingerprint. The cache
   fingerprint used size and integer timestamp; equal-size edits could evade it.
6. Dense batched training uses `W += beta * SRC.T @ TGT`. At p=0, a tiny CPU
   example changes from zero edges to 16 edges. Sparse batched training starts
   with an empty graph and creates bridges. These are model choices, not merely
   storage optimizations. Online/batched agreement within them proves a narrower claim.
7. The MNIST Brain wrapper chooses the reported implementation after observing
   its score relative to the reference. A stubbed native score .1 and reference
   score .9 returns .9 from `protocol_fallback`. The label is present in the
   wrapper, but the parity executor drops that backend label. This cannot prove
   native-backend conformance. Golden reproduction also needs actual dataset identity.
8. `compute/hyperdimensional.py` normalizes away count magnitude and then rounds
   normalized strengths back to counts. Encoding/decoding `[1,2,3,4]` returned
   `[]`. Sequence encoding also stores the sequence in metadata. Treat these as
   explicit HDC/storage operations, not neural sequence learning evidence.
9. `nemo/core/area.py` copies a short cue into a fixed-k buffer without clearing
   the tail, then supplies k to the kernel. This can include previous input in
   the next projection. The file advertises Hebbian learning without implementing
   a learned-weight update. Source finding; no CUDA reproduction attempted.
10. `programs/planning.py` uses symbolic BFS; `assembly_calculus/fsm.py` selects
    transitions from a dictionary; the Blocks parser can return a symbolic parse
    before neural parsing. Preserve these useful routes, but record which route
    answered an evaluation. Correct answers alone do not establish acquisition.
11. `cpp/python_implementations/billion_scale/billion_scale_cuda_brain.py` selects
    exponential random candidates independently of its updated weights. Its
    timing measures that workload. `cpp/cuda_kernels/simple_cuda_brain.cu` is a
    GPU arithmetic smoke test returning empty activated-neuron lists. Neither
    should be presented as scale evidence for the maintained neural dynamics.
12. `core/index_spaces.py` uses runtime-erased NewTypes and drops invalid indices.
    `_snap` independently passes invalid indices through. Public runtime boundaries
    still permit exactly the mistake the static types describe.

`probes.py` captures the original CPU observations; after repairs it should be
updated to report refusals rather than requiring the old defects to remain.

## One conceptual pipeline

`question -> operation contract -> preregistered protocol -> resolved run ->
per-seed observations -> judged result -> adopted claim`

Each arrow must have a resolvable identity. A claim is not a run, a run is not a
protocol, a backend name is not a model definition, and a metric value is not a
verdict. Unification means shared contracts for those boundaries.

The model definition must name graph semantics, stimulus law, tie rule,
plasticity, normalization/refraction and update order. Backend capabilities must
either implement that definition or reject it before allocating/training.
Selection fidelity is a separate field. No automatic fallback between models.

The run record must include schema version, script/function, commit and source
digest, model/backend identity, protocol version and content digest, registration,
seeds, tag, resolved parameters, input artifact/dataset identities, smoke/study
status, and completion/failure state. Keep observations separate from verdicts.
An incomplete run remains identifiable and cannot become adopted evidence.

## Migration sequence and acceptance gates

1. **Shared measurement repairs, in progress here.** Reject duplicate/mismatched
   seed keys, preserve pairing identities, refuse nonfinite readings, distinguish
   UNJUDGED, reject absent bars/metrics, consolidate legacy intervals, honor explicit
   engine arguments, hash source contents. Construct each misleading input in tests.
2. **Runner and immutable artifacts.** Extend the existing harness with one CLI
   and callable entry point. Reserve a unique tagged output before compute; never
   overwrite an existing run, including failure/interruption. Require >=3 unique
   seeds, >=20 for adopted hashed studies, and explicit smoke status for reduced
   grids. Migrate A1 horizon and capacity first. Use full cell coordinates including
   `(arm,n,k,seed)`; never key capacity by n alone. CLI must reject the old unsafe
   invocation before importing a GPU extension.
3. **Evidence graph.** Validate typed file/run/protocol/registration/claim edges.
   Inventory every historical orphan and provenance gap. Do not infer a missing
   engine or seed list from a filename. New artifacts fail on dangling edges;
   historic unresolved evidence is visibly unverified, not silently exempted.
   Populate the register engine field from registration and executable evidence.
4. **Model configuration and sampled-engine guard.** Begin with supported CPU
   configurations and one hashed adapter. Reject unsupported combinations. Warn
   once with the sampler-audit path at the actual sampled recurrent operation;
   registered sequence studies reject it. Do not silently materialize a huge area
   or change scientific defaults across old registrations.
5. **Operation contracts and first investigation.** Reuse `ops`, `read_only`, and
   the existing conformance suite. First demonstrate a control that breaks each
   claimed operation, then positive behavior in a stated regime. README/onboarding
   runs one fixed-connectome CPU training/partial-cue investigation with an explicit
   negative control and labels it an instructional demonstration.
6. **Runtime index boundaries.** Introduce owned area identity, typed immutable
   neuron IDs and typed compact indices at public input/readout boundaries. Separate
   active/materialized/ever-fired counts. Migrate callers under compatibility
   warnings; fail actual mixed-space/out-of-range inputs. Preserve backend-native
   buffers internally so safety does not imply full CPU copying on every GPU step.
7. **Research null checks.** Each adopted mechanism claim names its own lesion
   and affected observable. Beta=0 is not a universal null: fixed-graph similarity
   can exist without learning. Do not assert every number must move or reject every
   exact null as defective. Require a control that would expose the specific
   measurement failure; preserve failed bars and calibrate sensitivity across seeds.
8. **Dataset and toolchain paths.** Golden protocols refuse unavailable real data;
   pytest reports an explicit skip. Synthetic demonstrations remain separately
   named. Script Windows toolchain discovery/checks for vcvars64, ninja, CUDA_HOME,
   compiler and device; build only after active GPU users finish.
9. **State and module composition.** Extract parser runtime state, training state,
   and immutable observation/readout values before moving mixins. Reuse CorpusIndex,
   TrainingSchedule and ErpProtocol. Distinguish declarative phase selections from
   an executable ordered schedule; current executor uses a fixed sequence of `if`
   blocks. Turn `fast_training` changes into named protocol variants. Remove exact
   duplicates only after checking that historical evidence remains reproducible.
10. **Research progression.** Once the instrument passes these gates: separate
    temporal representation from readout at g=0/g=1, then test clip-window retention
    plus new learning through 200 presentations, then matched sequence baselines.
    A new weight rule is a preregistered scientific change, not a behavior-preserving
    refactor. No measured result is promised in advance.

## Developer-facing organization

The maintained entry should teach: choose model; build a brain; train an operation;
probe without mutation; compare against a constructed control; run seeds; inspect
the artifact; follow its evidence edges. Advanced registrations follow that first
investigation. Generate reference tables from configuration/contract objects where
possible so onboarding cannot contradict executable defaults.

Keep examples small and executable; distinguish API smoke checks from scientific
acceptance tests. Run fast CPU contracts on pull requests. Run the fixed-graph
conformance ladder separately, then hardware parity, then registered research.
The baseline workflow was release/manual publishing. The added research-contract
workflow provides a narrow CPU pull-request gate; it does not run all package or
research tests and does not validate GPU scientific results.
Keep historical tests and accelerator checks explicitly discoverable outside
the default `neural_assemblies/tests` collection.

The remaining work is substantial. This plan does not authorize indiscriminate
deletion or promise zero unknown defects. Its completion criterion is that the
supported research path enforces the distinctions above and every retained
alternative has a named role, evidence boundary and maintenance owner.

## Maintenance and proof direction added 2026-09-10

Keep model and protocol choices in immutable, validated configuration, with a
single environment/CLI adapter and resolved values in the run record. Avoid
creating switches for broken invariants. Keep compilation target descriptions
separate from scientific model choices, even where both affect performance.

The existing Assembly IR is the integration point for execution and verification
bridges. See `neural_assemblies/ir/VERIFICATION.md` for the executable-vs-evidence
boundary, generic Lean proof rules, and target schema requirements. Each bridge
must supply its state relation, step simulation and readout compatibility; a
shared file format alone does not establish semantic equivalence.

Prioritize actual duplicate behavior after its semantic card is written:
parser context resets, repeated training schedules, state synchronization and
parallel configuration adapters. Track maintained production lines separately
from tests, proofs and historical evidence. Retire old routes only after their
consumers migrate; moving code to another directory is not a reduction.
