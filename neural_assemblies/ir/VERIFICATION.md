# Assembly IR verification boundary

<a id="contract-ir-verification"></a>

## Status and ownership

The current v1 JSON schemas describe legacy brain/projection payloads, parity
reports and the restricted executable round. Python and Rust validate protocol
and explicit-round documents against the same packaged schemas and case corpora.
The legacy brain and projection schemas do not have equivalent executable
consumers. The separate `explicit-area-round-v1` profile below now lowers one
restricted instruction into the dense CPU engine; it is not a complete cross compiler.
Historical metric documents remain evidence artifacts, not executable programs.

The checked Lean kernel in [Refinement.lean](../../formal/AssemblyIR/Refinement.lean)
proves conditional rules for sequential schedules:

| Definition or theorem | Obligation it captures |
| --- | --- |
| `run` | Ordered interpretation of a list of operations |
| `lower` / `lower_compose` | Expand instructions; composing bridges preserves their order |
| `run_preserves` | A domain's invariant-preserving steps preserve it over a whole schedule |
| `Simulates` / `lower_preserves` | A local source/target simulation lifts to a complete schedule |
| `observations_agree` | Related states give equal measurements only with a readout compatibility proof |
| `visible_count_is_insufficient` | Equal visible counts do not establish equal hidden states |

These are generic proof rules, not proofs of the Python, Rust, Julia or CUDA
implementations. The pure explicit-round model below discharges `Simulates` for
its internal dense-kernel instruction; no implementation backend does. Neither
the sampling distribution nor floating-point arithmetic has been formalized here.

## One semantic definition, two consumers

The intended pipeline is a validated assembly program consumed by execution
lowerings and verification lowerings. A target description declares capabilities
and supported semantic profiles; it does not redefine an operation. Unsupported
instructions or profiles must be rejected before execution. The verification
side must come from the same normalized program, rather than a second handwritten
schedule that can drift. The restricted dense CPU execution lowering below is
a first step; a shared execution/verification consumer is still to be implemented.

Separate the following versioned inputs:

- **Model:** connectome identity/construction, stimulus law, winner/tie rule,
  arithmetic, learning, normalization, refraction, and index ownership.
- **Program:** ordered operations with state preconditions and mutation effects.
- **Observation:** reference assembly, cue/readout, candidate population and
  mechanism-disabling control. An observation does not inherit a learning claim.
- **Target:** backend/toolchain, supported operations and lowering revision.
- **Verification evidence:** exact program, specification and implementation
  digests; theorem and toolchain; hypotheses; relation proved; unresolved bridges.

Configuration can select supported values. It cannot opt out of seed identity,
no-overwrite, index validity, missing-evidence rejection or proof obligations.
Changing a scientific rule requires a named model/protocol revision in the run
record, even if the source code change is only one parameter.

<a id="contract-operational-benchmark"></a>

## Operational benchmark contract

The maintained throughput benchmark is an engineering diagnostic. It must
record the resolved engine, model semantics, and code revision, dimensions, round count,
independent seed identities, storage mode, per-seed timings, and summary
quantiles. It rejects fewer than three unique seeds and never replaces an
existing output. A benchmark result does not certify scientific validity,
backend refinement, or cross-machine comparability; those require their own
protocol and evidence.

<a id="contract-checked-domain"></a>

## Checked domain contract

[Domain.lean](../../formal/AssemblyIR/Domain.lean) implements the reusable
kernel/domain boundary described by the Dafny article. A `Domain` packages
`step`, `invariant`, state-dependent `valid`, an executable decision procedure
for `valid`, and a proof that allowed transitions preserve the invariant.
The interpreter and proofs use the same definitions and operation list.

`Domain.execute` checks each precondition against the intermediate state and
returns `none` on rejection. `execute_iff` proves both directions: execution
returns a result exactly when every step is admissible and the result agrees
with the existing `run` interpreter. `execute_preserves` proves that successful
execution preserves the invariant **provided it holds initially**. Construction
of a `Domain` requires its local preservation proof; it cannot be replaced by
a Boolean assertion that the domain is verified.

`Domain.checkedExecute` is the external entry boundary when the invariant
has a decision procedure. It checks the initial state before delegating to
`execute`. `checkedExecute_iff` proves acceptance exactly when the initial
invariant and all intermediate preconditions hold, with the same final state
as `run`. Thus an implementation that rejects every program would not satisfy
the contract. `checkedExecute_preserves` needs no initial-state premise from
the caller. The lower-level `execute` remains available for already-established
initial invariants and for domains without a decidable invariant.

A constructed boundary control rejects the empty program starting at allocation
four under capacity three; the lower-level interpreter accepts that state.
It also accepts the empty program starting at capacity three. These controls
expose the difference between checking transitions and checking the entire
entry contract. They do not constitute a backend refinement proof.

The constructed allocation controls admit `[1, 2]` under capacity three and
reject `[2, 2]` at its second instruction. The unchecked interpreter reaches
four and violates that invariant. This distinguishes a meaningful guard from
one that always accepts. This small domain is a kernel control, not a proof of
neural population allocation.

The interpreter is pure. Returning `none` does not promise rollback of external
effects. Mutable backend implementations must establish their own state and
failure relation, including RNG, allocation, clamps and learning. These proofs
do not certify JSON decoding, a Python-to-Lean translation, or backend execution.

The LemmaScript-inspired next bridge must consume the very same normalized
assembly program as execution. It must preserve instruction order, intermediate
preconditions and observations, and retain source locations for failures.
Until that bridge exists, the Lean domain is an executable specification kernel;
the Python/Rust wire validators are separate checked boundaries. No Dafny or
LemmaScript translator is installed or implied by this integration.

Acceptance of a future bridge requires all of the following:

- Source, normalized program, model profile, specification, target and toolchain
  identities accompany the proof artifact. A changed identity makes an old proof
  inapplicable; a passing parity report cannot replace it.
- Every supported instruction has an explicit state relation and local
  preservation obligation; unsupported semantics fail before execution.
- A shared corpus covers accepted and rejected schedules, with an intentionally
  broken translation or mechanism shown to fail. Differential tests remain
  evidence about the bridge, not a universal translation proof.
- Changed preconditions, invariants or observations appear in the review diff.
  Weakening the specification to make a proof pass changes the contract and
  cannot be represented as an implementation-only repair.

## What preservation means

A fixed-connectome integer lowering can seek bit equality. Floating arithmetic
needs a stated error relation and conditions under which winner selection is
unchanged. Sampling needs a probabilistic semantics; equal seeds across unrelated
RNG algorithms do not provide it. Quantization is not exact merely because two
quantized backends agree. The current Lean simulation rule handles a declared
state relation; probabilistic and approximate claims need additional mathematics.

For multi-target projection, specify whether all reads use the state at the
start of the round. Sequentially fusing targets is invalid if later targets see
earlier mutations where the source semantics used a shared snapshot. The same
issue governs batching, recurrent schedules and learning/readout reordering.

Lean certification belongs to an identified theorem and its checked hypotheses.
Passing a parity corpus belongs to a test report. Neither alone adopts a
scientific result, proves a capacity law, or justifies a faster implementation.
Benchmark time and memory separately after preservation gates pass.

<a id="contract-organ-semantics"></a>

## Hashed organ semantic identity

`OrganSemantics` composes one `ModelSemantics` substrate with the state code,
training schedule and inference schedule of an assembly memory, assigned-state
FSM or sequence transducer. It also records exact tie jitter, effective ARC and
STATE refraction charges, convergence gating, successor horizon and gain,
prediction gain and feature-register presence. These choices alter the transition relation and therefore cannot be
inferred from a hardware label such as `hashed_arc_fsm`.

The hashed substrate profile distinguishes fixed Bernoulli afferent counts from
zero-or-size drive, lowest-neuron-ID ties from deterministic hash jitter, and
inverse-indegree initialization from subsequent column scaling. The public pure
`describe_*` functions derive profiles without allocating a GPU. Each organ
constructor derives the same profile before device allocation and, when given an
expected `organ_semantics`, rejects every mismatch before constructing state.
Thus a run record can be passed back into the implementation as an executable
admission requirement.

`ExecutionSemantics` is the strict discriminated envelope used by evidence. A
Brain run has exactly one `default` model profile. An organ run has one or more
named organ profiles, allowing a paired null study or the capacity study's B/G
arms to state their different transition relations without pretending they are
one backend default. Unknown fields, incomplete nested objects, inconsistent
tie rules and invalid organ/state combinations reject.

An alignment run has exactly one `AlignerSemantics` profile. It separates the
nonlearning, gain-controlled stimulus anchors from the plastic LEX-to-FEAT cross
fiber and binds their normalization, clipping, storage, tie and schedule choices.
Schema 8 adds this third discriminator while retaining schema-7 Brain and organ
records. The Python constructor boundary is executable; a corresponding Lean and
Rust wire/refinement definition remains an open proof obligation.

Constructed controls pair each organ with a valid profile that differs in one
mechanism. Supplying that profile to the other configuration raises before any
device allocation. These tests establish that the boundary detects mismatch;
they do not prove the CUDA implementation refines the profile. CUDA parity,
mechanism-disabling nulls and observation contracts remain separate obligations.

## Source links and proof workflow

Definitions use `Specification: repository/path.md#stable-anchor` in docstrings.
`python -m research.evidence specifications` validates those links without
importing hardware backends, including Rust documentation and the Lean kernel's
module documentation. The links identify obligations; they are not badges
claiming the implementation is formally verified. Existing operation cards retain
unresolved discrepancies explicitly.

Run `lake build` in [formal](../../formal) to check the reusable kernel. The project
pins Lean 4.31.0, the newest installed toolchain found on this machine (4.30.0 is
also installed). Its core library is sufficient for these proofs; no mathlib
checkout is copied or upgraded. The local mathlib checkout found at
`F:/src/mathlib4` is pinned to 4.30.0, so it cannot be silently mixed into this
4.31.0 package. Add a pinned compatible dependency when a proof requires it.

## Design sources

[Midspiral's Dafny architecture](https://midspiral.com/blog/from-intent-to-proof-dafny-verification-for-web-apps/)
informs the reusable kernel/domain obligation boundary. Its integration and
specification limitations also apply here: a valid proof can target the wrong
requirement or be wired to the wrong executable.

[LemmaScript](https://midspiral.com/blog/lemmascript-a-verification-toolchain-for-typescript/)
informs the complementary verification pipeline derived from executable code,
and the need for differential checks at translation boundaries. It is a
TypeScript toolchain, not an existing translator for our Python or CUDA code.
No LemmaScript dependency or translator has been installed here.

[Ember Arlynx's target-specification work](https://ember.software/resume/) and
[Rust's custom-target interface](https://doc.rust-lang.org/rustc/targets/custom.html)
inform versioned target descriptions. Rust's target schema is compiler-version
specific; assembly targets likewise need a pinned semantic/schema identity.

<a id="contract-protocol-wire"></a>

## Protocol wire boundary

The authoritative schema is `v1/protocol.schema.json`. Both Python and Rust use
Draft 2020-12 validators compiled once from that file. Protocol IDs are nonempty
strings; required metrics cannot be silently supplied. Optional fields must have
their declared types. The schema's `format` remains an annotation in v1, so a
`recorded` string is not certified to be a valid date. Extension fields remain
allowed, and this permissiveness must not be interpreted as scientific validation.

Accepted documents round-trip as decoded JSON values without dropping metadata
or adding null properties. Rust uses arbitrary-precision JSON numbers to avoid
rounding large integer identities. Decimal/exponent numbers must fit finite
binary64 at both boundaries; exact decimal arithmetic is not promised. Python rejects nonfinite numbers and values
that cannot round-trip as JSON (including non-string object keys and tuples).
This is a decoded-value contract, not preservation of whitespace or original
numeric spelling; JSON duplicate-key detection is not implemented by this change.

The Rust `ProtocolDocument` is an immutable validated wrapper. Access the complete
payload with `as_value()` and its ID with `protocol()`. Construction and Serde
loading both validate; direct public-field construction from the former 0.1 API
is retired. A valid wire object still needs separate run provenance, metric
meaning and scientific acceptance checks.

`write_protocol_document` uses exclusive creation. Existing files raise
`FileExistsError`; choose a new tagged path for new evidence. It does not synthesize
a study run record. Studies should use the shared research runner.

Python and Rust tests consume `v1/protocol.cases.json`, including malformed
fields, missing requirements, metadata retention and a large integer. The Rust
crate now lives beside the schemas (`Cargo.toml`, `rust/lib.rs`) and remains a
member of the workspace at `crates/`. Cargo and wheel packaging can therefore
include the same canonical schema without generating or maintaining a second copy.


## Executable schedule obligation: projection recurrence

The source-linked projection card's P3 regression is a concrete instance of the
intent-to-proof workflow: the operation's recurrence argument determines the
source-edge schedule before execution. Disabling recurrence must remove every
self-edge, including when a backend helper has legacy recurrence enabled. The
regression checks dispatched edges and learned recurrent state on `numpy_exact`,
with the recurrent configuration as a positive control. No recovery claim is
inferred from this test.

An eventual projection lowering must preserve this schedule, including its
stimulus-only first round. It must separately discharge the neural transition
simulation obligation in `Simulates`; the schedule regression does not supply
that proof. Other direct users of `Brain.project_rounds` still have legacy
filtering and cannot be declared equivalent to the explicit operation contract.


<a id="contract-operation-objects"></a>

## First-class operation contracts and schedules

`assembly_calculus.contracts.OperationContract` is the package-level contract
shape for a named calculus operation. Every instance must name its inputs, state
reads, immutable plan type, mutations, regime requirements, observed outcome,
failure conditions, specification anchor and at least one constructed control.
Empty surfaces reject during contract construction. The public implementation
carries the exact object as `operation_contract`, and the read-only
`OPERATION_CONTRACTS` registry provides discovery without importing research prose.

Projection, reciprocal projection, association, merge and pattern completion are
migrated operations.
`ProjectionPlan` validates nonempty
names, a positive integral round count and an explicit Boolean recurrence choice,
then freezes the full ordered `ProjectionStep` sequence. Execution validates the
stimulus and target before the first call and sends exactly those inspected steps
to `Brain.project`. Round one is stimulus-only; each later step has the stimulus
and includes target recurrence exactly when requested. No global Brain recurrence
switch participates in this plan.

`ReciprocalProjectionPlan` separately requires two distinct registered areas, a
positive integral round count, an explicit source-clamp choice and an active
source assembly. It freezes the initial forward step and every subsequent
forward, target-recurrent and return-edge step. The public operation preflights
the entire plan before borrowing the source clamp, so bad topology or source
state cannot alter a clamp or call a backend. Clamp restoration remains scoped
around valid execution and preserves both facade and engine state.

The migration comparisons reconstruct the former direct-first-round plus
`Brain.project_rounds` tail on all three NumPy engines. It compares stable-ID and
compact winners, recruitment, owned RNG state, and a subsequent read-only
observation. The reciprocal comparison reconstructs the former forward/return
schedule and checks both areas by the same criteria. The recurrent-learning and
learning-disabled round-trip controls remain the behavioral negatives named by
the first two contracts.

`AssociationPlan` admits exactly two source protocols: two active assemblies
borrowed as fixed sources, or two distinct named stimuli that keep both sources
evolving. A partial stimulus pair rejects rather than silently changing both
source dynamics. Its three area names must be distinct. Pathway rounds are
positive; joint coactivation rounds are independently nonnegative, preserving
zero as the constructed no-coactivation control. The plan freezes both sequential
pathway phases and the joint phase, including exactly where target and source
recurrence begins.

Association migration comparisons cover both source protocols on all three NumPy
engines. They reconstruct the removed imperative helper and compare all three
areas' stable and compact winners, recruitment, owned RNG, clamp restoration and
the next read-only observation.

`MergePlan` freezes simultaneous parent drive and the later target schedule. The
parent-self, target-self and back-projection switches require explicit booleans.
With no stimuli it borrows both current parent assemblies as clamps; with two
stimuli both parents evolve. Exactly one stimulus requires the caller to declare
the other parent's state as `require-fixed`, `fix-current`, or `evolving`.
Preflight verifies that declaration against current state. This prevents a missing
stimulus from silently selecting a different transition system.

All thirteen statically visible partial-stimulus research calls now name their
mode. Twelve already execute inside a pinning scope and declare `require-fixed`;
the live composed parent in `universality_composition` declares `evolving`.
Migration comparisons cover all four historical source patterns on all three
NumPy engines.

`CompletionPlan` additionally owns the cue sampler and observation effect. It
requires a seed and one of three policies: live plastic execution, frozen-weight
execution, or a read-only Brain transaction. `PreparedCompletion` stores the
stable-ID reference separately from the exact compact-index cue, preventing those
two index spaces from becoming an implicit array convention. It rejects a
different brain or changed entry winners before injection. Traced and untraced
completion execute the same prepared cue and immutable recurrent steps. Static
callers must name both seed and policy. Legacy research callers explicitly select
`plastic` so the migration itself does not alter recorded protocols; the teaching
investigation names `read-only` because it is a measurement.

Migration comparisons reproduce the former plastic path on all three NumPy
engines. Policy controls distinguish weight mutation, retained activity and full
read-only restoration, including exception cleanup. Completion still reports a
min-normalized overlap against its entry snapshot; plan validity and schedule
identity are not evidence that recurrence reconstructed information.


## Repetition shares the ordinary transition boundary

`Brain.project_rounds` now executes the selected named-target schedule through
`Brain.project`, preserving inhibition, clamp synchronization, recording and
history handling at that boundary. Its source-linked contract records the
legacy recurrence selector separately from execution. This is still a Python
compatibility API, not a projection-JSON interpreter or a formally certified
backend lowering.

A future fused lowering must preserve the entire observed transition, including
per-round history when enabled and observation isolation. The new controls
exposed a sampled fiber being allocated during a probe even though its final
winners were restored. Thus neither final-winner equality nor the absence of
Hebbian updates suffices to discharge the state relation in `Simulates`.


## Population readiness is separate from measurement validity

The source-linked role and bridge-reset contracts distinguish requested capacity
from an allocated population. A target configuration cannot establish that
neurons and their ID mapping exist by assigning its requested size to a runtime
count. A preserving reset retains the actual population and identity mapping.

`ComputeEngine.probe_target_ready` checks whether a target can be projected into
without recruitment. It does not certify trained fibers, nonzero drive, or an
informative readout. Unavailable role observations retain explicit diagnostics,
including for inner clauses. Future IR observation consumers must preserve that
unavailability instead of interpreting it as a measured zero or successful label.
Preparatory neural classification now has its own read-only scope. Parser
category caches and distributional subcategory metadata may still be populated;
readiness alone does not certify a whole pipeline's mutation or provenance rules.


## Classification observations and score provenance

The source-linked classification contract uses existing neural populations and
an explicit stimulus-only schedule. It has constructed inhibition, exception and
subsequent-learning controls. No backend preservation theorem is inferred from
those tests. `ClassificationEvidence` now identifies source and validates its
score-key domain before fusion or decomposition. Conversion from core-area keys
to category keys is explicit; a distributional fallback cannot become neural
evidence merely by passing through the classifier. Source-specific strengths
remain distinct metrics. The legacy tuple and parser caches still lose source
information and require further migration. A future IR observation consumer must
retain the typed source distinction together with its model and protocol identity;
state-preserving execution does not by itself make metrics interchangeable.

## Snapshot ownership and prepared forks

Brain/engine clones retain their state graph and internal aliases. Parser forks
now copy the entire parser graph too, so lexicons and category metadata cannot
silently remain shared between cells. Pristine cache snapshot failure is an
error, and calibration publishes matching live/pristine state only after both
calibration and copying succeed.

The parser's legacy post-copy preparation still resets CONTEXT construction
counts/IDs and discards selected caches while retaining fibers. An IR bridge
must therefore represent copying and this preparation as separate effects;
it cannot claim the prepared parser is an exact state-preserving clone. These
Python controls establish tested ownership behavior, not a Lean simulation
proof or a certification of the calibration measurement. See the source-linked
parser-fork card for exact mutation and failure obligations.

## Training identity precedes reuse

Parser cache requests now resolve the engine and holdouts before lookup and pass
the same resolved values to training. Engine, fast-training mode and numerical
parameters enter memory/disk identity; disk metadata must also match before reuse.
Calibration variants have their own mode key and derive from an unchanged
uncalibrated training snapshot. Explicit empty holdouts remain empty through
the trainer and dialogue helper.

This is a concrete reuse boundary for the eventual model/program/observation
identities. It is not yet the complete target identity: external dependency,
hardware and arithmetic semantics still need the declared ModelSemantics/target
profile. A matching cache request cannot substitute for a scientific run record
or a preservation proof.

## Cache publication is a separate effect

Checkpoint persistence now writes and flushes a private temporary file before
replacement, with bounded handling of Windows replacement contention and cleanup
on failure. This is an external I/O effect, not the pure failure behavior of
`Domain.execute`. Any verification bridge claiming persistence guarantees must
model publication and failures separately. Replaceable trusted Python caches
remain distinct from exclusive scientific evidence artifacts and IR wire data.


<a id="contract-explicit-round"></a>

## First executable projection profile

`projection.ExplicitRound` owns the normalized `explicit-area-round-v1`
instruction. `explicit-round.schema.json` is its authoritative wire shape. The
strict decoder requires every field, rejects extra fields and other profiles,
and copies source and drive sequences into immutable tuples. Python and Rust run
the same positive and negative case corpus. This is separate from the permissive
historical v1 projection payload; that payload cannot silently become an
executable program.

| Component | Meaning in this profile |
| --- | --- |
| Target | One registered area in a standalone NumPy explicit CPU engine |
| Sources | Ordered distinct area names; self-source explicitly requests recurrence |
| Index space | Stable neuron IDs in `[0,n)`, with no duplicate winners |
| Connectome | Existing finite nonnegative dense float32 source-by-target matrices |
| Drive | Sum active source rows in source order, then optional float32 additive vector |
| Stimulus law | No stimulus inputs supported; external drive is supplied, not sampled |
| Competition | Dense k-WTA, descending drive and increasing neuron ID for ties; when k=n the backend returns neuron order |
| Learning | Explicit instruction Boolean, per-fiber beta, active pre/post pairs multiplied by `1+beta`, optional engine clip |
| Mutation | Selected fibers when learning; target winners, ever-fired flags and counts |
| Schedule | One target per instruction; a later invocation reads the preceding result |
| Unsupported | Stimuli, sampled/GPU engines, slots, custom policies, clamps, normalization and supervised slots |

The adapter validates inputs and relevant state before calling the existing
`NumpyExplicitEngine.project_into`. It contains no second implementation of
summation, winner selection or Hebbian learning. Direct engine calls now share index/source/drive validation. The stricter
profile restrictions (such as nonnegative weights and no clamps) stay in the IR. This engine-level entry must not be
used to mutate a Brain's private engine behind its facade. Use `execute_on_brain`
for the coherent Brain lowering described below.

Construction and preflight rejection do not mutate engine state. This does not
promise rollback after an unexpected backend failure or across a multi-round
program. Finite inputs do not prove absence of intermediate floating overflow.
The profile does not define connectome initialization or replace run records:
reproduction still needs initial state, beta/clip settings, code and NumPy identity.

Acceptance tests check hand-computed caps and full weight matrices, an explicit
learning-disabled null, dead-fiber versus intact observations, two recurrent
rounds, tie ordering and clipping, malformed wire/state rejection, and direct
backend agreement. Agreement alone is insufficient: the null and hand-computed
cases establish that the observation reacts to the tested mechanism.

This is the first execution lowering, not a formal translation certificate.
`Domain.checkedExecute` remains the reusable Lean obligation. To connect it,
formalize the numerical profile and its state/readout relation, then prove the
local `Simulates` obligation and identity of the normalized input consumed by
the proof. No generated Lean schedule or proof of Python/NumPy arithmetic is
claimed here. Rust validates the explicit-round transport but does not execute it.

<a id="contract-formal-explicit-round"></a>

## Pure formal explicit-round lowering

[Projection.lean](../../formal/AssemblyIR/Projection.lean) models the normalized
round fields and a separately named `DenseKernelRound`. Its state contains area
registration, population and cap sizes, winner neuron IDs and integer weights.
Drive is the declared external integer bias followed by source-row sums in source
and winner order. Winner selection and the per-fiber learning transform are
arguments, so a target must supply them rather than inherit an unnamed tie or
plasticity rule.

`Valid` is an executable state-dependent admission decision. It rejects missing
or duplicate sources, no input, a supplied drive with the wrong population
length, unregistered areas, malformed source winners, `k > n`, and selected caps
with the wrong size, duplicates or out-of-range IDs. `checkedRound_iff` proves
that checked execution returns a state exactly when this predicate holds and
that the result is the ordinary semantic step. This is a one-round pure failure;
it does not claim rollback for an effectful implementation.

`lowerRound_simulates` discharges `Refinement.Simulates` for the field mapping
into one dense-kernel instruction. `lowerRound_program` lifts it to every finite
schedule and `lowerRound_winners` proves equal winner observations. Separate
theorems prove that frozen rounds preserve all weights, non-target areas preserve
their winners, and target winners are exactly the selector result. Constructed
examples check a hand-computed drive and learned weight. A broken lowering that
clears the learning bit retains the old weight, so the proof suite distinguishes
the omitted mechanism. Short drive, duplicate-source and invalid-selector cases
all reject.

The formal drive uses arbitrary-precision integers. A translation must still
prove how NumPy float32 addition, multiplication and clipping refine them under
stated error bounds. Fiber existence, Python exceptions and Rust/CUDA execution
remain open.
`lake build` checks warnings as errors; `lake env leanchecker
AssemblyIR.Projection` independently checks the compiled declarations.


<a id="contract-explicit-program"></a>

## Explicit program composition

`ExplicitProgram` is the Python IR value for a finite ordered composition of
`ExplicitRound` instructions. Its `then` operation is associative because it
concatenates immutable round tuples; the empty program is the identity. The
class validates every round at construction and exposes canonical documents
only through fresh values. These are construction guarantees. Execution and
backend refinement remain obligations of the selected lowering.


<a id="contract-formal-round-wire"></a>

## Formal explicit-round wire admission

`AssemblyIR.Wire.decodeRound` is Lean's decoder for the complete
`explicit-area-round-v1` object. It checks the exact field set and profile,
nonempty names, ordered unique sources, explicit Boolean plasticity, numeric drive
items, and the requirement for at least one input. It retains drive values as exact
JSON decimals. `scaleNumber` converts each decimal to integer units at an explicit
base-10 scale and rejects any value that would require rounding.

`normalizeRound_identity` proves that successful numerical normalization preserves
target, source order and plasticity. The `check-wire-cases` executable reads the
same packaged 12-case corpus used by Python and Rust; changing one expected verdict
makes it exit unsuccessfully and name the mismatched case. Lean therefore shares
the wire acceptance examples rather than duplicating them as theorem literals.

This closes Lean wire-field identity for one instruction. The schema remains the
transport authority in Python and Rust; Lean's independent decoder is a conformance
consumer. This does not prove that arbitrary JSON-schema behavior is reproduced,
choose a scientific fixed-point scale, or relate scaled integers to NumPy float32,
clipping, Rust execution or CUDA arithmetic.


<a id="contract-explicit-inputs"></a>

## Shared explicit-engine input contract

`NumpyExplicitEngine.validate_projection_inputs` is the shared nonmutating
preflight for `project_into` and `ExplicitRound.validate`. It checks distinct registered sources,
valid target/source winner IDs, and the external drive before the clamped-target
return or numerical work. `set_winners` uses the same winner validator and checks
before replacing winners or counts. IDs are one-dimensional integral, unique and
within the area's population; partial and empty caps remain allowed. Projection
rechecks mutable winner buffers so in-place edits cannot bypass validation.

External drive must have shape `(target.n,)`, a real numeric dtype and finite
float32 representation. A missing drive is `None`; a supplied empty vector for a
nonempty area is an error. Signed additive drive remains supported. Clamping is
not an exemption from argument validity. Selected winner IDs are checked before
plasticity; validated source and selected IDs replace silent filtering in updates.

Both legacy calls and `ExplicitRound` use these checks; the adapter no longer
maintains its own copy of winner/drive checks. Feature eligibility (no clamps,
slots or custom policies) belongs only to the restricted IR profile. This does
not assert complete engine-state validation or rollback for arbitrary backend
errors. Valid inputs can still overflow during arithmetic; no performance or GPU
claim follows from these CPU tests.

`Brain.project` forwards explicit drive to either the primary dense engine or the
auxiliary dense engine. Targets with supplied drive bypass the batch API, whose
configuration cannot carry drive. Requests for unknown/inhibited targets or unsupported engines raise rather
than silently dropping the supplied input. Named external-drive targets now
participate in scheduling even without source edges, and drive-only projections
save winner history. This does not change the existing mixed sparse-source
versus supplied-drive rule. Brain's broader descriptor synchronization and
multi-target failure atomicity remain separate obligations.


<a id="contract-brain-round"></a>

## Brain lowering for the explicit round

`ExplicitRound.execute_on_brain` reuses profile and numerical input validation, then
lowers the same instruction to `Brain.project`. Named areas must share one dense
engine, whether primary or auxiliary. Their descriptor dimensions must match
engine dimensions, winner IDs must be valid, and fibers must have shared object
ownership. Clamps, custom policies, slots, active inhibition, mutual inhibition
and fiber plasticity overrides are rejected for this profile. A requested
learning step cannot silently override a disabled Brain or engine learning flag.

The instruction temporarily sets Brain's learning flag, restoring its prior value
in `finally`. Ordinary Brain projection synchronizes source winners, applies the
backend result to descriptors, saves histories and records activation. Returned
winner arrays are detached from live state. Drive-only instructions now use
ordinary Brain scheduling and history recording as well. They do not invent a
source edge or recurrent input to make the target execute.

Controls compare ordinary projection with IR execution on primary dense and
auxiliary dense engines, including full weight matrices, winner history, counts,
activation and returned-array independence. Further cases cover recurrence,
drive-only scheduling, inhibited-target rejection, malformed public winner
buffers, unsupported learning masks and restoring the learning flag on failure.
Malformed-length and float32-overflow drives must reject before synchronizing
source caps: controls deliberately give the engine a different cap from its Brain
descriptor and require the original engine cap to survive rejection. Standalone
`validate` rejects the same drive errors and malformed mutable engine winner
buffers without requiring an execution attempt. This covers input preflight,
not rollback after numerical work begins.

This is a tested lowering, not a Lean simulation proof. Profile checks do not
prove entire Brain state consistency, failure atomicity of a multi-target program,
or thread safety of temporarily scoped flags. Mixed-engine IR rounds remain
unsupported; ordinary mixed projection retains its existing behavior.


<a id="contract-winner-inputs"></a>

## Shared winner input boundary

`validated_indices(..., unique=True)` is the common numerical boundary for
assembly winner buffers: one-dimensional integers, distinct, nonnegative, below
the declared population and representable in uint32. Empty and partial caps are
valid. Generic index conversion leaves uniqueness optional because not every
index sequence represents an assembly.

Area assignment and the sampled, exact and explicit NumPy engine setters use
this boundary before mutation. Exact-engine ever-fired bookkeeping occurs only
after valid assignment. The explicit engine also rechecks buffers at projection.
Brain validates raw direct injections before dtype conversion, checks routing
names, and validates the whole injection map before applying any entry. Ordinary
projection similarly validates every source buffer before synchronizing any
source, so a mutated public buffer cannot be silently truncated by synchronization.

These guards establish numerical index validity, not ownership of an index space.
A compact position below n can still be beyond the sampled area's materialization
watermark; stable-neuron versus compact ownership remains a distinct obligation.
The sampled setter must also serve explicit-source mirrors, so n cannot simply
be replaced by its own compact-map length without resolving that ownership.
GPU-native setters are not certified by these CPU controls.

Failure-before-mutation applies to malformed winner values and unknown injection
routing names. This is not rollback for a later backend failure, invalid external
drive, inhibition change or multi-target execution. Existing partial-cue injection
and explicit clearing remain supported.


<a id="contract-supervised-reinforcement"></a>

## Supervised reinforcement is a distinct operation

`Brain.reinforce_connectome` reads the active source assembly and stable target
neuron IDs. It supports writable dense NumPy fibers whose axes span both full
populations. A sampled source's compact winners are mapped to stable IDs before
indexing dense rows. `_source_neuron_ids` owns this conversion and is shared with
mixed sparse-to-explicit drive. Empty mapping means no materialized neurons for
a sampled source; `None` means identity for a fixed-connectome engine.

The operation validates names, indices, beta and storage before its disabled or
empty-input no-op. Beta must be finite and nonnegative. A clip is finite and
positive or absent. Sparse/virtual/GPU storage is explicitly unsupported until
its writable coordinate semantics are specified; shape alone is insufficient.

At positive beta with learning enabled, selected zero synapses are seeded to one,
then selected weights are multiplied by `1+beta` and clipped. This is supervised
edge creation, not ordinary multiplicative Hebbian projection. Beta zero, the
global learning disable, or a fiber mask leaves zero edges zero. Only the selected
block changes; the old implementation clipped unrelated entries across the whole
matrix. Unclipped overflow is rejected before committing the updated block.
Malformed indices still raise when learning is disabled.

The patch teacher calls this operation using existing source winners. It no longer
writes `_snap` stable neuron IDs back into the compact winner field. Its current
all-explicit configurations had identity coordinates, but the helper was unsafe
to compose with sampled sources.

Controls cover full expected matrices, an unrelated over-clip sentinel, all three
learning nulls, malformed posts even when frozen, invalid beta, sampled source
mapping, unsupported compact storage, unbounded overflow and teacher source-state
preservation. No historical study is re-adopted from these software checks; this
operation is not yet an executable IR instruction or a Lean-proved lowering.


<a id="contract-fiber-learning"></a>

## Scoped fiber learning control

The [formal frame contract](#contract-learning-frame) specifies protected learned
values and permitted updates; concrete backend refinement remains unproved.

<a id="contract-plasticity-rate"></a>

### Plasticity-rate update boundary

The public per-fiber rate route accepts an existing area source, an existing area
target and a finite nonnegative real beta. It canonicalizes beta to a Python float
and validates the complete request before changing either the public area record or
the executing engine's store. Boolean, negative, NaN and infinite rates fail, as do
unknown endpoints. Bulk area/stimulus updates preflight every entry before applying
the first one, so a bad late entry cannot leave a partial schedule.

This contract establishes configuration routing and failure atomicity. It does not
establish that a configured rate changes a particular observable, that backends use
identical arithmetic, or that clipping leaves a rate comparison interpretable. The
registered per-fiber plasticity experiment tests the first question on a copied,
materialized fiber with an equal-rate null and an unclipped bar.

A Brain fiber mask disables learning on one directed area or stimulus fiber;
it does not remove that fiber's drive. Dense projection now implements this
control at the learning loops through `fiber_learning_allowed`. Brain scopes
active masked fibers onto their owning engines using `suppress_fiber_learning`
before dispatch. Both primary and auxiliary dense engines are covered. Nested
scopes accumulate suppression and restore the exact prior scope on exit,
including exceptions. Persistent Brain configuration is not modified by a scope.

`ComputeEngine.supports_fiber_learning_masks` defaults false. All three NumPy engines opt in: dense, fixed-connectome `numpy_exact`, and
sampled `numpy_sparse`. A learning-enabled projection requesting an
active mask on another backend raises before projection, rather than silently
ignoring the mask. Disabled global Brain learning needs no per-fiber suppression.
Masks on inactive routes are not forwarded. GPU-native mask support is not implemented.

Supervised reinforcement checks the same scoped engine predicate as well as
Brain's global/fiber gates and the target engine's global learning flag. An IR
instruction explicitly requesting learning rejects a contradictory engine mask;
it cannot claim to have executed its requested learning rule after suppressing it.

Controls preserve the blocked fiber's full weight matrix while showing its drive
changes the selected cap, show another fiber learns, re-enable learning, cover
stimulus fibers, nested scopes, exception cleanup, supervised writes, and IR
contradiction rejection. These are learning masks, not recruitment guards,
thread-safe transactions, or normalization/scaling guarantees on other engines.


The `numpy_exact` implementation applies the shared predicate in its single
`_apply_plasticity` function, used by both ordinary and fixed-target projection.
It suppresses only new stimulus exponents and area outer-product updates. It does
not change beta: this engine interprets stored potentiation through beta at read
time, so setting beta to zero would erase the contribution of existing learning.
Normalization remains the existing read-time scale; the mask does not alter it.

Controls pretrain fibers, then compare masked execution with a learning-disabled
reference at both normalization settings. Masked effective weights remain exactly
unchanged while an unmasked fiber increases, and re-enabling restores learning.
Stimulus exponents receive the same checks. A deliberately broken beta-zero
implementation loses previously learned activation and fails the observation
contract. Fixed-target cases preserve this engine's existing fixed-target learning
policy; this does not assert that all backends have the same clamp semantics.


The sampled engine filters permitted sources only at its learning boundary,
including the compiled and fixed-target callers, and separately guards the
explicit-source bootstrap update. Its global engine learning flag is checked at
that shared boundary too; the fixed-target caller previously bypassed it.
Triggered normalization receives only permitted area fibers. Deferred flushes
retain suppressed work in the queue, process permitted work, and apply retained
work after the scope ends instead of silently discarding it.

Recruitment, newly initialized connectivity, RNG consumption and area-level
refraction are not disabled by a fiber mask. Tests deliberately show recruitment
with a masked stimulus and use materialized populations when asserting whole-fiber
byte equality. Do not interpret masking as a read-only probe or reuse a learning
claim for the structural effects of recruitment.

Brain synchronizes public fixed-target caps alongside source winners before
fixing an engine target. This prevents a requested held cap from being replaced
by an empty/stale backend cap. Ordinary, fixed, compiled and explicit-bootstrap
paths, scaling on/off, deferred retention, recruitment and stale-cap controls are
covered by CPU tests. Native GPU backends and formal preservation proofs remain
open; no scientific sequence result from the sampler is adopted by these checks.


<a id="contract-learning-frame"></a>

## Formal learning-mask frame

[Learning.lean](../../formal/AssemblyIR/Learning.lean) states the common mask
contract independently of numerical representation. `LearningState` separates
per-fiber learned values from activity. `maskedStep` computes a proposed transition
on the original state, retains blocked weights, applies unblocked writes exactly,
and preserves the proposed activity result. It reuses `Refinement.run` for schedules.

| Theorem | Exact guarantee |
| --- | --- |
| `maskedStep_blocked` | A blocked learned value is unchanged by one step |
| `maskedStep_allowed` | An unblocked write equals the proposed write |
| `maskedStep_activity` | Activity uses the proposed transition on the original state |
| `masked_run_frame` | Blocked values are unchanged across any finite schedule under a fixed mask |
| `masked_scope_extension` | Adding an inner mask cannot release an outer protected fiber |

Together these properties prevent conflating masking with removal of input,
freezing all state, or replacing the proposed update with a no-op. Concrete
controls retain weight ten on the blocked fiber, increase an unblocked fiber to
eleven/twelve, and advance activity using the retained weight to ten/twenty.
Bypassing the mask changes the protected value; zeroing it before the proposal
loses its contribution to activity.

This is a pure, fixed-fiber frame contract. It does not prove exception unwinding,
thread safety, floating-point arithmetic, or any Python-to-Lean lowering. The
runtime scope's restoration is covered by Python controls, not this theorem.
A concrete backend must identify its learned-value representation, separate base
connectivity/recruitment, and discharge the simulation and observation obligations
in `Refinement.lean`. In particular, a sampled fiber's full physical matrix can
grow under a learning mask; treating that matrix as an invariant learned value
would be an invalid instantiation. No backend simulation is claimed here.

Run `lake build` and `lake env leanchecker AssemblyIR.Learning` from `formal/`.
The printed dependencies contain only `propext` (and none for the activity
projection theorem); there are no admitted proof holes in these theorems.


<a id="contract-engine-identity"></a>

## Supplied-engine construction identity

Before adopting a `ComputeEngine` instance, `Brain` calls the engine's nonmutating
`validate_brain_identity(p, seed, w_max)` check. Each parameter must match the
engine's value, including when the caller omitted it and Brain used a default.
Conflicts raise before projection fidelity is changed or any area/fiber is added.
This prevents a primary engine using one identity while Brain creates auxiliary
engines or mixed fibers using another. `None` remains a valid matching weight clip.

The default backend implementation reads `p`, `w_max`, and `seed` (or the legacy
`_seed` storage). Missing identity is rejected. A backend with different storage
must override the check, not guess a seed. Torch now retains its constructor seed
for this purpose. Engine-name construction still forwards the requested parameters
through the factory and is unaffected.

For example, a caller supplying an engine created with `p=.1, seed=41, w_max=8`
must supply the same three parameters to `Brain`. Keep them in one mapping to
avoid maintaining two symbolic copies:

```python
identity = dict(p=.1, seed=41, w_max=8)
engine = create_engine("numpy_sparse", **identity)
brain = Brain(engine=engine, norm_init=False, **identity)
```

Backend-only switches stay on the supplied engine. This is deliberately a construction-identity check, not a
complete model configuration: determinism, stimulus laws,
tie rules, mutable post-construction settings and prepopulated-engine adoption
remain separate obligations. GPU parity must still be verified on the CUDA setup.

Controls reject each mismatch on every NumPy engine before a fidelity setter can
run, reject unavailable identity, and check matching identity in newly created
auxiliary dense engines with both finite and absent clips.


<a id="contract-engine-admission"></a>

## Named-engine admission and failure identity

`engine_type(name)` resolves the registered class without constructing model
state, so `Brain` can validate declared capabilities before forwarding options.
Homeostasis has three independent, false-unless-opted-in capabilities:
`supports_norm_init`, `supports_synaptic_scaling`, and
`supports_synaptic_scaling_deferred`. An omitted `Brain.norm_init` resolves to
true on an opted-in named engine, which preserves the established sparse and exact
defaults, and false on the dense explicit engine. Any enabled unsupported option
raises before its constructor runs. Registry controls require every declared
capability to have an explicit parameter or deliberate option receiver, so a new
backend cannot silently accept or lose one.

Brain and `create_engine` call the same `validate_homeostasis_capabilities`
boundary. Direct factory use therefore cannot bypass semantic admission. The
legacy CUDA-implicit adapter also rejects every residual constructor option rather
than accepting a `**kwargs` value it will discard.

The CUDA-implicit and deprecated CuPy adapters override all three inherited
capabilities to false because their constructors do not forward the sparse
parent's homeostasis options. This prevents Brain from reporting mechanisms that
those adapters silently omit. Torch opts into normalization and scaling but
leaves deferred scaling false; its existing constructor rejection remains a
backend defense behind Brain's admission check.

Engine names in the built-in module map are known even when their optional
implementation cannot load. `ensure_engine` attempts only the module named by that
map and records its `ImportError`; it returns false for both unavailable and unknown
names because it is a membership probe. `create_engine` preserves the distinction:
a mapped implementation failure raises `EngineUnavailableError`, chained from the
original import error, while an unmapped and unregistered name raises the existing
unknown-name `ValueError`. Constructor errors after successful registration remain
constructor errors and are not relabeled. A provider module that imports but fails
to register its mapped name is also rejected here with that condition stated.

This boundary runs before engine construction or model-state mutation. It does not
claim that an importable backend has a usable device or satisfies parity; each
backend's runtime admission and conformance gates own those stronger claims.

<a id="contract-feedforward-inhibition"></a>

## Feedforward-inhibition configuration and admission

`FeedforwardInhibitionConfig(probability, weight)` is the immutable pair defining
whether a present area-to-area synapse is inhibitory and what signed weight it
receives. Stimulus afferents are outside this mechanism. Probability is finite
and in `[0,1]`; weight is finite and strictly negative. A nondefault weight at
probability zero rejects because it is inert configuration that cannot affect a
run. Values canonicalize to floats.

Engines opt into `supports_feedforward_inhibition`. Only `numpy_exact` currently
implements the complete law; sampled sparse, dense explicit, Torch,
CUDA-implicit and CuPy do not. Brain validates the pair and capability before
construction, stores the canonical configuration, and forwards both values
together only when enabled.
`create_engine` enforces the same rules, while supporting engine constructors
validate the pair again for direct callers. Torch removes and rejects these named
options before its general `**kwargs` processing, preventing the former silent
no-op path.

The sampled engine contains signed materialized-weight kernels, but its
unmaterialized candidate ranking still samples a positive binomial and its
recruitment split reconstructs positive edge counts. At probability one and
weight -0.75, the exact engine reports negative area-to-area drive while the old
sampled public path reported positive drive. Correct support requires a signed
candidate distribution and a reconstruction law that produces the same fixed
fiber; changing only the sampled score would create a second mismatch. Admission
therefore rejects the mechanism until both obligations have parity evidence.

Controls reject booleans, nonfinite values, out-of-range probabilities,
nonnegative weights and the dormant custom-weight case. Constructed spies prove
unsupported Brain and factory requests never invoke an engine constructor. The
exact engine exposes the canonical values it executes; the sampled engine has a
specific rejection control.

<a id="contract-projection-fidelity"></a>

## Projection-fidelity admission

`ProjectionFidelity.EXACT` and `ProjectionFidelity.COMPILED` select winner
candidate topology; they do not identify the connectome model or certify
scientific fidelity. Aliases normalize to one enum before engine construction.
Exact is the universal default. Compiled selection is admitted only when an
engine declares `supports_compiled_projection`; currently that is the sampled
NumPy engine, whose compiled mode restricts top-k to existing columns under its
documented freeze and materialization preconditions.

Brain construction, Brain's runtime setter and `create_engine` all validate the
same capability. The base engine setter rejects compiled selection, so a direct
call on an incapable backend cannot become a no-op. When Brain adopts an existing
engine, its normalized request must match the engine's current selection mode;
adoption never silently rewrites this model choice.

Controls require dense and exact Brains, the lower-level factory and a direct
exact-engine setter to reject compiled selection. The capable path normalizes
`fuzzy` to `compiled`, and an unknown runtime value leaves the prior mode intact.

<a id="contract-option-remainder"></a>

## Constructor-option remainder

A public engine constructor that accepts `**kwargs` must partition them into
named, validated mechanisms and an empty remainder before it creates model or
device state. Every unconsumed key raises `TypeError` with its name. A recognized
unsupported mechanism may accept only its canonical disabled value; any enabled
value raises. This permits a shared facade to send explicit defaults without
turning misspellings or newly introduced options into silent no-ops.

The exact engine applies the same rule at construction and area registration.
The Torch engine consumes projection topology, feedforward inhibition,
homeostasis, dense-drive and read-only configuration, then rejects all remaining
keys before CUDA device setup. Controls misspell normalization, slot count and
dense drive and require rejection before engine or area state exists. A positive
Torch control supplies every recognized option and reaches device setup, proving
that the remainder check has not forbidden the supported configuration surface.

<a id="contract-deterministic-allocation"></a>

## Deterministic-allocation admission

`deterministic` is a strict boolean execution policy. On capable engines, true
selects exact-fit buffer growth and the engine's deterministic sampling branch;
false permits amortized growth and faster sampling. It does not promise
bit-identical results across engine implementations, dependency versions or code
revisions. Scientific protocols must still record the engine and commit.

Engines opt into `supports_deterministic_allocation`. Sampled NumPy, Torch and
their derived GPU adapters implement the branch. Dense explicit has no lazy
allocation branch, while exact drive has no sampling stream; both reject true
rather than accept an inert option. Brain and `create_engine` validate before
construction, and direct engine constructors apply the same boundary. When Brain
adopts a capable engine, the requested policy must match the engine's stored
executed policy.

Controls reject integer, string and null substitutes for booleans, prove
incapable constructors are not entered, distinguish the universal false default
from an enabled request, and reject mismatched supplied-engine adoption.

<a id="contract-torch-execution-options"></a>

## Torch execution-option admission

`gpu_sampling` and `dense_drive` are Torch-specific execution choices. The former
selects the device used for truncated-normal candidate sampling; the latter
changes the candidate domain from sampled order statistics to all-neuron drive.
Because the latter changes `ModelSemantics.candidate_domain`, it is a model choice
and must be present in Brain construction when requested. An omitted Brain value
uses the selected backend's default; an explicit value on an incapable backend,
including `False`, is rejected as inapplicable rather than recorded as a no-op.

Both options are strict booleans. Brain, `create_engine` and direct Torch
construction validate them before device setup. Brain forwards them through the
same constructor path and records the effective values after construction;
deterministic Torch execution may make effective GPU sampling false even when the
requested default was true. Supplied Torch engines are admitted only when their
effective options match Brain's request. `readonly` is a separate inference
state, not one of these model-construction options, and remains governed by the
`read_only` context contract.

Controls cover invalid truthy substitutes, explicit inapplicable options,
requested dense-drive semantics and effective-policy recording. A positive Torch
admission case reaches device setup with all recognized options, so the boundary
does not merely reject everything.


<a id="contract-backend-capability"></a>

## Optional backend installation and runtime capability

`CUPY_INSTALLED` is the import-free package-discovery predicate used during root
package import. The historical `GPU_AVAILABLE` name remains an exact compatibility
alias for that weak predicate; it does not certify an import, device, allocation or
engine. `cupy_available()` is the cached runtime predicate: it loads torch first on
Windows, imports CuPy, and allocates a device array. Runtime test and engine-selection
gates use this predicate. Importing `neural_assemblies` alone must load neither
CuPy nor package submodules.

Runtime capability still does not prove numerical parity, memory sufficiency for a
study, or successful extension compilation. Those stronger claims require their
own backend and protocol gates.


<a id="contract-sampled-recurrence"></a>

## Sampled-recurrence admission

`SampledRecurrencePolicy` separates three intentions before recurrence would
sample candidates for an incompletely materialized `numpy_sparse` area: `warn`
emits one audit-linked warning, `acknowledged` admits a deliberate sampled-engine
comparison without noise, and `forbid` rejects the projection. Unknown values,
booleans and `None` reject during Brain or engine construction. Brain stores the
normalized enum and requires a preconstructed sampled engine to carry the same
value before adoption. Brain and engine expose read-only properties; their
internal synchronization path rejects a change after area registration.

The engine performs admission after validating the target but before deriving a
child RNG or changing projection state. The `forbid` control compares generator
state, winners, recruitment and recurrent weights before and after rejection.
Materializing the target to all `n` neurons removes the sampled condition and is
admitted under every policy. A fixed target and a `read_only` projection also
offer no sampled candidates and are admitted. Existing `warn` behavior remains
the default; parity and migration tests that intentionally exercise the sampled
backend explicitly select `acknowledged`.

This policy identifies the known lazy NumPy sampler hazard. It does not turn an
acknowledged result into valid sequence evidence, certify materialization parity,
or describe the separate Torch and hashed substrates. Engine identity, stimulus
law, tie handling and arithmetic remain distinct model-semantics obligations.


<a id="contract-model-semantics"></a>

## Complete primary-path model semantics

`ModelSemantics` is the immutable identity of eight choices that can change a
result while leaving an engine name unchanged: connectome realization, candidate
domain, stimulus-drive law, default tie break, arithmetic precision,
normalization, plasticity rule, and its numeric weight ceiling. Categorical
fields are closed enums. Construction
from a mapping requires the complete field set and rejects unknown fields; wire
serialization emits only canonical enum values.

Every `ComputeEngine` implements `describe_model_semantics()`. Brain records that
value after engine construction and before area or stimulus registration. If
`Brain(model_semantics=...)` supplies an expected object or wire mapping, all
fields must equal the engine description or construction raises with every
mismatched field. This makes a saved semantic profile an executable admission
gate. In particular, the legacy `ASSEMBLIES_STREAM_INIT=1` switch changes
`numpy_sparse` from content-addressed to stream-addressed and therefore fails a
content-addressed expectation instead of silently changing graph identity.

The three CPU profiles distinguish lazy order-statistic candidates from
all-neuron selection, dense storage from hash regeneration, conditioned lazy
stimulus drive from a fixed Bernoulli afferent count, partition-dependent ties
from lowest-neuron-ID ties, float precision, and inverse-indegree normalization.
Torch additionally names its backend top-k tie order, and its dense-drive option
is a distinct all-neuron domain whose unmaterialized drives remain sampled.
`numpy_exact` rejects arithmetic types outside float32 and float64 rather than
misreporting their precision.

This object describes the primary engine's default k-WTA path. Area-local winner
policies, operation schedules, observation mode, inhibition/refraction settings,
and result definitions remain explicit protocol or operation contracts. A model
profile does not establish numerical refinement between backends; that requires
the drive, winner, update, and error-bound gates named elsewhere in this file.

Research run schema 6 carries the same canonical document. For a registered
Brain engine, `run_experiment` reconstructs the engine's ordinary no-topology
path using the requested normalization and weight ceiling, compares all fields,
and rejects omission or disagreement before reserving a tag. A bespoke hashed
organ records `null` here and may not claim a Brain profile; its missing organ
semantics remain visible until an organ contract supplies them. Historical run
schemas remain readable without retroactive semantics.


<a id="contract-homeostasis-config"></a>

## Shared homeostasis configuration

`core._homeostasis.HomeostasisConfig` is the immutable construction contract for
normalization, column-scaling scope and deferred scaling. It is consumed by Brain,
NumPy sampled/exact constructors as applicable, and Torch's supported homeostasis
settings. It describes configuration, not equivalence of backend arithmetic.

`norm_init` and `synaptic_scaling_deferred` require booleans. Scaling is a boolean
or a list/tuple/set/frozenset of nonempty area names. Named scopes are copied into
frozensets; duplicates/order have no meaning and an empty scope canonicalizes to
False. Strings, dictionaries and non-name entries raise. Deferred scaling with no
enabled scope raises instead of becoming an ignored request. Backend restrictions
still apply: this object does not make deferred scaling available on Torch or
column scaling available on the exact engine.

Brain compares the supplied engine's homeostasis with this normalized request,
before adoption. Missing mechanism attributes mean disabled in the default
backend check; custom storage requires overriding `validate_brain_identity`.
Import `HomeostasisConfig` from `neural_assemblies`; a single
`config.as_kwargs()` can be supplied to both constructors. Legacy fields
remain available, but mutable caller collections cannot change Brain's scope after
construction or disagree with the engine's snapshot.

This does not lock legacy fields against subsequent direct assignment, reconcile
all backend switches, or prove that normalization/scaling are appropriate for a
scientific protocol. The separate refraction incompatibility and learning-mask
contracts still apply. GPU verification remains a required external gate.


<a id="contract-homeostasis-wire"></a>

## Homeostasis wire contract

`HomeostasisConfig.to_document()` and `from_document()` connect the runtime object
to the packaged `v1/homeostasis.schema.json`. Rust's `HomeostasisDocument` uses the
same schema and the same `homeostasis.cases.json` acceptance corpus. Both reject
unknown/missing fields, wrong versions, nonboolean flags, malformed scopes and
inactive deferred scaling. The profile is `homeostasis-v1`; all settings are required.

On the wire, scaling is boolean or a nonempty array of unique nonempty names.
Unlike convenient Python constructor inputs, empty arrays and duplicates are
rejected rather than normalized. Valid unsorted scopes are accepted and serialized
in sorted order. Disabled scaling has one wire spelling, False. The Python object
retains a frozenset; serializers produce a detached JSON list.

A runner can store `config.to_document()` in its parameters and reconstruct it
inside `measure(record)` before constructing a Brain. The integration control
compares the resulting engine configuration with the reserved run parameters.
The runner does not automatically validate arbitrary parameter subdocuments;
protocol adapters must explicitly decode the configuration they consume.

Rust provides validated configuration transport, not model execution. There is
no Lean lowering, backend refinement proof or complete assembly-program schema
implied by this bridge. Historical protocol documents retain their existing schema.


<a id="contract-refraction-boundary"></a>

## Refraction and normalization mutation boundaries

`core._homeostasis.check_area_homeostasis` guards direct sampled-NumPy and Torch
refraction setters and normalization operations as well as Brain configuration.
A refracted target rejects explicit column normalization regardless of its configured
scaling scope. The sampled scaling primitive rechecks current target state, so
previously queued work cannot normalize a target made refracted after scheduling.
A rejected queued fiber remains pending; this does not promise whole-queue rollback.

`Brain.set_refracted` dispatches to the area's owning engine and updates public
state only after backend acceptance. Unsupported primary/auxiliary dense-engine
requests therefore leave the descriptor unchanged. `Brain.normalize_weights`
also routes to the owning engine. No new normalization capability is implied for
backends that do not implement it.

Controls exercise direct and facade rejection, unchanged weights and descriptors,
queued-work retention, and legal normalization after disabling refraction. GPU
execution remains unverified. These boundaries do not make arbitrary direct
mutation of legacy fields safe or provide rollback for unrelated backend failures.


<a id="contract-area-controls"></a>

## Runtime area controls and ownership

`Brain.set_lri`, `clear_refractory` and `clear_refracted_bias` dispatch to the
area's owning engine, including auxiliary dense areas. LRI descriptor fields are
published only after backend acceptance. The base backend rejects nondefault LRI
requests instead of silently ignoring them; disabling with `(0, 0)` remains a
no-op on backends without LRI. History clearing remains a no-op where that history
does not exist, but it must reach the correct owner.

Controls reject unsupported LRI on exact/dense and auxiliary dense areas, check
unchanged descriptors, and exercise actual sampled LRI/history reset. Ownership
spies ensure clearing cannot operate on the primary engine's auxiliary-area mirror.
This does not promise rollback of arbitrary backend errors. Numeric input
requirements are given by the shared LRI parameter contract below.


<a id="contract-lri-parameters"></a>

## LRI numeric input contract

`core._homeostasis.validate_lri_parameters` is shared by public Area construction,
Brain runtime updates, sampled/Torch area construction and setters, and the base
unsupported-backend boundary. The refractory period is a nonboolean integral value
in `[0, sys.maxsize]`, the representable deque-window range. Strength is a
nonboolean real value convertible to a finite nonnegative Python float. NumPy
integer/real scalars are accepted and canonicalized to Python int/float; strings,
fractional periods, negative values, nonfinite strengths and booleans raise.
Zero period remains a disabled window; it does not force stored strength to zero.

Validation precedes parameter publication, history replacement, area registration
and RNG-consuming population initialization. Controls preserve an existing populated
history object, both descriptor/backend parameters and RNG state on rejected
construction/update requests. An accepted NumPy-scalar control retains exactly
the last two entries in a two-step window. Backend capability rejection still
applies after numeric validation; valid input does not create an LRI implementation.

This is input rejection, not a guarantee of rollback after arbitrary allocation or
backend failure. Legacy fields can still be mutated directly. Torch uses the shared
validator, but its CUDA execution remains a separate unverified gate.


<a id="contract-refraction-registration"></a>

## Refraction preflight before area registration

`Brain.add_area` checks refraction support and scaling compatibility before
creating the descriptor, consuming RNG draws, registering a population or wiring
fibers. `ComputeEngine.supports_refraction` defaults to False; sampled NumPy and
Torch declare support (CUDA inherits the sampled declaration). Explicit Brain
areas are owned by NumpyExplicitEngine, whose unsupported declaration is checked
without constructing the auxiliary engine. A primary mirror's refraction support
cannot make an auxiliary area implement the mechanism.

Controls reject scaled sampled areas, exact/dense areas and auxiliary dense areas,
then compare registrations, connectivity maps and all relevant NumPy RNG states.
The same name can be used in a supported retry. A positive control projects input
into a refracted sampled area and observes nonzero accumulated bias.

This closes refraction support/compatibility rejection, not all possible failures
of `add_area`. Other option preflight, duplicate names, allocation failures and
misdeclared custom-backend capabilities remain separate obligations. GPU support
is declared from the implementation but still requires the external CUDA gates.


<a id="contract-area-registration"></a>

## Shared area registration identity and dimensions

`core.registration.validate_area_registration` is the nonmutating preflight used
by Brain, standalone Area construction and NumPy/Torch area-registration methods.
Names must be nonempty strings and absent from both node registries.
Population size and cap size must be nonboolean integers with
`0 < k <= n <= 2**32`; the upper population limit keeps every neuron ID representable
at the shared uint32 boundary. Accepted NumPy integer scalars become Python ints.

Validation occurs before population replacement, wiring or random initialization.
Duplicate registration is an error, not a way to reset or resize a learned area.
Controls preserve the existing population/descriptor identities and RNG streams
across Brain and direct calls on all three NumPy engines. Standalone construction
and the logical ID-width limit are checked without allocating a huge population.
The requested size may still exceed available memory or a backend's tighter limit.

The check does not validate every area option, make registration generally
transactional, or establish GPU conformance. Stimulus registration is specified
below; dynamic resizing remains a separate contract. Existing failures unrelated to identity/dimensions
may still require broader option preflight or rollback.


<a id="contract-explicit-registration"></a>

## Explicit area option forwarding

Brain's initial and lazy auxiliary registration paths share
`_register_explicit_area`. It forwards the descriptor's dimensions, beta, LRI
settings, slots, winner policy and input noise setting to the owning dense engine.
Backend restrictions remain authoritative; a primary mirror accepting an option
does not establish support on the owner.

Previously both paths omitted the winner policy, so a recorded threshold policy
executed as ordinary top-k. Controls use a drive of `[9, 4, 3, 1]` with `k=2`:
threshold five requires one winner and threshold twelve requires none. These
outcomes distinguish policy execution from the old two-winner fallback. Primary
and auxiliary dense paths agree; the lazy-registration branch is checked too.

This preserves option meaning at registration. It does not establish complete
failure transactionality, checkpoint restoration, or support for every combination
of options. Numerical selection remains implemented by the existing dense engine.


<a id="contract-slot-options"></a>

## Slot layout and winner-policy compatibility

`core.registration.validate_slot_configuration` is shared by Area construction,
dense engine registration and the standalone slot selector. Counts are nonboolean
integers from zero (disabled) to the population size. Multiple slots must divide
the population evenly; a remainder cannot silently become unreachable neurons.
Multiple slots combined with a custom winner policy are rejected because the
current selection kernel implements slot top-k, not a composition of the two rules.
Zero/one slots retain ordinary policy selection in the dense engine.

Brain forwards slot counts to primary dense owners as well as auxiliary dense
owners. Other primary backends declare no slot support and reject the request
before registration. Controls distinguish best-slot selection from global top-k
on `[10, 0, 6, 5]`, cover invalid layouts/combinations on all dense entry paths,
and reject a trailing-neuron layout even through the standalone selector.

This changes previously ignored or ambiguous requests into errors; it does not
add a new selection algorithm or make direct mutation of registered legacy fields
safe. General option preflight and GPU conformance remain open.


<a id="contract-stimulus-registration"></a>

## Stimulus registration and shared source names

`core.registration.validate_stimulus_registration` is the nonmutating preflight
used by Brain, standalone Stimulus construction and NumPy/Torch registration.
Stimulus sizes are nonboolean integers in `[0, 2**32]`, canonicalized to Python
ints. Zero is a valid null stimulus; the upper limit matches the uint32 ID space
and does not promise that an allocation of that size will fit in memory.

Area and stimulus names must be nonempty strings and disjoint. Backend learning
rates and connection-probability overrides are keyed by source name, so allowing
the same name for both node kinds cannot represent independent source settings.
Duplicate registration raises rather than replacing connectivity or resetting a
learned source's rate. Validation precedes descriptor publication, wiring and RNG
consumption. Controls cover Brain and direct calls on all three NumPy engines,
including preservation of a previously customized source rate and RNG streams.

This intentionally rejects formerly accepted ambiguous names and replacement
calls. It does not make allocation failures transactional or validate mutations
made directly to legacy dictionaries. Torch uses the same preflight, but GPU
execution conformance remains a separate gate.


<a id="contract-explicit-probability"></a>

## Explicit-area connection probability

`Brain.add_explicit_area` constructs an auxiliary dense area using the brain-wide
connection probability. Its legacy `custom_inner_p`, `custom_out_p` and
`custom_in_p` arguments have never reached connectivity construction in this API.
Any non-None override now raises `NotImplementedError` before allocation, registry
mutation or RNG consumption. Zero is a requested override, not a default sentinel.
The error names every requested option. Callers must not omit these options unless
the brain-wide probability is the intended model.

Controls reject both zero and nonzero values for each override, preserve state
and RNG, and show the supported default path builds all-one connectivity at p=1.
This does not implement heterogeneous dense connectivity: precedence between one
area's outgoing default and another's incoming default, future fibers, and primary
mirror/owner consistency need a common connection-policy specification first.
The checkout-oriented `text_generation/robust_grammatical_brain.py` prototype
requests these overrides for its CORE areas and cannot use the maintained Brain
with that configuration. Its old silent fallback is not evidence for that model.


<a id="contract-runtime-policy"></a>

## Runtime competition-policy ownership

`Brain.set_competition_policy` resolves the area's executing owner and calls its
`ComputeEngine.set_competition_policy` method before publishing the accepted
policy on the Area descriptor. The default engine method rejects the operation;
NumPy sparse, fixed-connectome and dense engines, and Torch, explicitly implement
it. Storage remains backend-owned. An auxiliary area's primary mirror supplies
source activity; it does not select that area's winners and is not configured as
if it were the owner.

Dense runtime updates use the same slot/policy compatibility validator as dense
registration. Multiple slots with a custom policy must raise before either owner
or descriptor changes. Passing None restores the existing top-k selector on a
supported backend. Controls change a dense area's drive `[9, 4, 3, 1]` from two
winners to one under threshold five, then restore two winners by clearing the
policy. Both primary and auxiliary owners execute this behavior; direct NumPy
setters and rejected slot combinations are covered too.

This contract preserves routing and existing selection semantics. It does not
validate every parameter inside policy objects, certify GPU execution, or protect
direct assignment to legacy state fields. Input-noise controls follow the separate contract below.


<a id="contract-input-noise"></a>

## Input-noise value, capability and owner

`core.registration.validate_input_noise` defines a standard deviation as a
nonboolean real number convertible to a finite, nonnegative Python float. Zero
disables noise. Area construction, direct NumPy/Torch registration and runtime
setters use this validator. Invalid values raise before changing registered state
or consuming randomness; they cannot silently behave like disabled noise because
NaN or a negative value fails a later `std > 0` branch.

Brain checks the executing owner's capability before registering a noisy area.
NumPy sparse and Torch declare support. Dense and content-addressed NumPy engines
reject nonzero noise; the engine default accepts the disabled value only. Runtime
changes pass through the executing engine before publishing the descriptor, so an
auxiliary dense area cannot acquire a noise setting only on its primary mirror.
This does not introduce noise into an engine that previously lacked it.

Controls cover invalid numeric values on public/direct NumPy paths, state/RNG
preservation, unsupported registration and runtime updates, and enabling/disabling
supported noise. A seeded materialized all-connected area changes its selected
neurons under noise compared with an otherwise identical clone. This is a software
execution control, not statistical validation of a noise law. Finite configuration
does not guarantee finite arithmetic at every extreme scale, and GPU execution
and direct legacy-field mutation remain separate verification tasks.


<a id="contract-policy-values"></a>

## Competition-policy values

The frozen dataclasses in `compute.winner_policies` validate and canonicalize their
parameters at construction through `_validate_fields`. Counts are nonboolean,
nonnegative integers; scalar parameters are finite nonboolean reals. Accepted
NumPy scalar values become Python ints/floats. Fraction parameters lie in `[0, 1]`;
relative maximum counts cannot be below minimum counts. Thresholds may be signed,
and zero winner counts remain valid null policies.

The supported tie rule is `value_then_index`. An arbitrary string previously
selected a fallback sort without describing its tie semantics; it now raises.
E%-WTA windows are exactly `epsilon` and `sigma`, with nonnegative sigma_c. Gamma
construction requires finite `0 <= d_ms <= tau_m_ms` and `tau_m_ms > 0`, including
zero delay and the full-delay boundary. A window typo can no longer silently run
the epsilon mechanism.

Parameter validation is owned by policy construction rather than duplicated in
the selection branches. The policies do not promise k winners if fewer candidates
exist, and E%-WTA's existing silent-input and minimum/cap rules remain unchanged.
Controls reject malformed policies before selection and preserve null/signed
boundary behavior. This does not validate every feature vector, certify backend
tie parity or GPU execution, or validate arbitrary objects reconstructed by
bypassing constructors (including legacy pickle state).


<a id="contract-competition-wire"></a>

## Competition configuration transport and reconstruction

`ir.competition.policy_to_document` and `policy_from_document` implement the
`competition-v1` profile for all four supported policy classes. Its shared schema
is `ir/v1/competition.schema.json`; Rust embeds that exact file in its opaque,
validated `CompetitionDocument`. Every default is explicit. Unknown fields, modes,
profiles and omitted settings fail instead of acquiring reader-specific defaults.

Counts must use JSON integer values, with arbitrary precision; floating-point
encodings of counts are rejected even if mathematically integral. Continuous
parameters must fit finite binary64. Besides the schema, both readers enforce
`max_winners >= min_winners` and these numerical representation requirements.
Rust compares exact integer spellings for bounds, retaining adjacent counts beyond
u64/binary64 precision. Python reconstructs through the validated policy classes.
The shared corpus includes all four policies, large adjacent bounds, null maximum,
unknown/missing fields and invalid parameter values. Valid Rust documents preserve
the original JSON value; Python normalizes continuous scalars to floats while
preserving the represented values.

Selection controls compare each policy before and after Python reconstruction.
A runner integration records the document, reconstructs it inside measurement and
executes threshold selection for each recorded fixture seed. The expected one
winner distinguishes the policy from default top-k. These checks establish
transport/reconstruction and one CPU execution path, not Rust or Lean selection
refinement, GPU parity, or migration of existing experiment artifacts.


<a id="contract-phon-registration"></a>

## Phonological stimulus reuse

`CoreParserMixin.add_phon_stimulus` is the shared word-input registration path.
Multiple vocabulary sources may encounter the same word. If its named stimulus
already exists with the requested size, reuse it and bind the word mapping without
rewiring fibers, resetting source learning rates or consuming connectome RNG.
If the requested size differs, raise before replacing the stimulus. Changing
phon_weight after registration is not an implicit resize operation.

This parser-level ensure operation is distinct from Brain.add_stimulus, whose
strict duplicate rejection remains intact. Controls preserve original connections,
stimulus identity and a customized source beta on reuse. Corpus vocabulary and
scaled-vocabulary integration tests exercise the formerly failing call paths.
This does not make grounding-context replacement transactional or establish
classification/generalization quality.


<a id="contract-classification-cues"></a>

## Explicit classification cue selection

`classify_word` and `classify_word_evidence` accept keyword-only cue_mode with
values combined (the unchanged default), phon_only, and grounding_only. Selection
occurs before observing neural state. Phon cues must already be registered;
grounding cues come only from the explicitly supplied context and registered
feature names. Resolved names are deduplicated in schedule order. No mode adds
stimuli, consults category labels to choose input, or learns during observation.
Unknown modes raise before entering the neural observation context.

ClassificationEvidence retains cue_mode and an immutable tuple of resolved cues,
alongside its existing source and score domains. Legacy tuples intentionally lose
this metadata. Manually constructed older evidence may leave mode unspecified only
with no cue names. The word-only category cache continues to use the default query;
explicit cue variants are not cached under that ambiguous key. Existing neural
zero-evidence distributional fallback remains source-labeled in every mode.

Controls trace actual projection inputs, preserve weights/activity/RNG for all
three modes, and check immutable cue provenance. This makes alternative queries
composable and inspectable. It does not claim their scores are calibrated across
unequally recruited areas or silently replace the existing combined-cue protocol.


<a id="contract-classification-cache-context"></a>

## Classification cache context

`classify_word_cached` uses word-only category, bootstrap and distributional caches
only for the default context: no explicit grounding, or grounding equal to the
word's stored GroundingContext. An explicitly different context goes through the
existing classify_word_bootstrapped inference path before any word-only shortcut.
Its answer is not written into those word-only caches. Thus a one-off context
cannot inherit or replace a category cached for the default context.

Controls construct stale answers in all three shortcut maps, trace the supplied
context to inference, preserve the maps, and retain the matching-context fast path.
A trained-parser control compares against uncached inference and preserves neural
state. This is a cache routing repair, not a change to fusion or neural selection.
It does not provide general cache invalidation after training or in-place mutation
of stored grounding, and it does not introduce cue variants into the word-only key.

<a id="contract-state-path-active-sources"></a>
<a id="contract-active-source-routing"></a>

## Active-source routing

State-to-prediction bootstrap may project only areas with current winners. Source
eligibility therefore reads `Area.active_count`; cumulative recruitment or a
backend materialization extent does not establish an active drive. Empty SUBJ and
OBJ areas are seeded from the first active core before their fibers are initialized.
The source list is recomputed after seeding. An inactive but historically recruited
core must not displace an active core or become a prediction source.

The constructed control gives an inactive core positive ambiguous `w`, gives the
next active core zero `w`, and gives empty syntactic areas positive `w`. The active
core must seed both syntactic areas and the inactive core must never project. This
fails the former `.w > 0` implementation in both source-selection phases.

The same rule governs composition programs: a source participates in merge only
when it has current winners. `area_has_active_winners` is the shared predicate
used by the two-part and grid-patch MNIST merge paths. Its counterexample gives a
silent area positive historical population and an active area zero historical
population; only current activity may determine the branch. The `.w` source
predicates are absent under the repository ratchet.

<a id="contract-population-counts"></a>

## Area population counts

`Brain.population_counts(area)` is the public product of three distinct
quantities: current active winners, cumulative neurons that have ever fired, and
the optional lazily materialized extent. Active count comes from the public area
snapshot; cumulative and materialized counts come from the area's executing
engine. A dense engine returns `None` for materialized because every neuron exists
and no lazy extent applies. Unknown areas raise rather than returning zeros.

Callers must select a named field. Compiled ring capacity requires a non-`None`
materialized extent and rejects a dense owner; stimulus preallocation uses
cumulative recruitment. Clearing activity changes only `active`. Controls verify
all three after sparse training and inhibition, the dense `None` case, and an
unknown area. The `.w` alias is not part of this interface.

<a id="contract-pre-kwta-observation"></a>

## Pre-k-WTA observation

A global pre-k-WTA sum is inseparable from the number of candidates over which
it was accumulated. `Brain.pre_kwta_observation(area)` therefore returns one
immutable `PreKwtaObservation(total, candidate_count)` with `mean` derived from
that exact pair. It returns `None` only when neither component was recorded,
rejects a one-sided record, and rejects zero, negative or nonfinite components.
Unknown areas raise. Changing current activity, cumulative recruitment or lazy
materialization after recording cannot change the observation's mean.

Every maintained normalized-energy consumer reads this value. A backend that
does not produce a pre-k-WTA observation may use the explicitly documented
winner-energy fallback where one exists; it may not substitute area size,
winner count or one as a divisor. ERP adapters turn a missing or invalid
observation into `Measured.undefined`, preserving the reason. Controls construct
a valid observation, each malformed half-record and an area whose ambiguous
`.w` changes without changing the recorded candidate count.


<a id="contract-initial-recruitment"></a>

## Initial dense selection and recruitment identity

When dense explicit-source input initializes a sparse population, selected winners
are stable neuron IDs. NumPy and Torch use `index_spaces.reserve_initial_neuron_ids`
to reserve those IDs before later lazy recruitment. The resulting pool has the
selected IDs as its prefix, in selection order, followed by the original pool's
unselected IDs in their original order. The pointer then advances past that prefix.
No random draw is added. With no original pool, the remainder uses identity order.

The old implementation advanced a pointer into an unrelated random permutation.
It could recruit selected IDs again while permanently skipping other neurons,
so compact positions no longer mapped injectively to neuron identities. Controls
force initial selection from the end of the pool, check disjoint selected/pending
sets, and materialize the remaining population to verify all n IDs remain unique.
Invalid reservations reject before modifying input arrays. The helper is for
initial population construction, not resetting learned mappings.

This corrects future recruitment identities and can change downstream numerical
results from affected paths. It does not repair persisted corrupted mappings or
establish historical result parity. Source-based training-cache invalidation still
applies; GPU execution is a separate gate.


<a id="contract-observation-rounds"></a>
## Observation round counts and stability

`validate_round_count` accepts positive nonboolean integral counts and returns
an integer. Both `Brain.project_rounds` and `assembly_stability` reject invalid
counts before neural observation or mutation. Stability executes ordinary
projection calls: the second observation includes target recurrence regardless
of the compatibility schedule in `project_rounds`.

A read-only target with fewer than k materialized neurons is unmeasurable and
raises. Exactly k permits measurement but yields an untrustworthy stability
score: there are no alternative winners. A larger pool permits a meaningful
contrast, without guaranteeing an assembly. `test_parse_errors.py` checks cold
rejection, the exactly-k vacuous control, and trained versus untrained separation.
These are executable software contracts, not a formal proof or research adoption.


<a id="contract-hashed-normalization"></a>
## Hashed fiber normalization is part of the execution target

AreaFiber uses float32 division by in-degree. DenseOrganFiber precomputes a
float32 reciprocal and multiplies by it in organ_drive_kernel. These expressions
are equal over real numbers but need not round identically. Canonical tie-breaking
resolves equal computed drives; it cannot make different rounded drives equal.
The recurrent_fiber factory selects DenseOrganFiber for a finite clip without
column scaling, so storage selection can change winner trajectories.

The multi-episode fused test explicitly constructs both fiber classes and compares
them to stored-weight references with their declared normalization arithmetic.
It retains a constructed negative: four episodes, three rounds, beta .1, clip20,
n2048, k40, seeds 0x5EED1234+17*b, normalized, cue RNG5. For brain2 in episode1,
substituting division for reciprocal multiplication changes final winners.
The original division-reference failure is not historical parity acceptance.

For a general cross-target winner guarantee, drive-error bounds alone are
insufficient: a separating k/k+1 gap greater than twice a uniform error bound is
a sufficient mathematical condition for preserving the winner set. The diagnostic below checks this bound for observed score pairs; uniform
backend error bounds and concrete kernel proofs remain open. Exact CPU/GPU
trajectory equivalence does not follow.


<a id="contract-winner-margin"></a>
## Numerical winner separation

`ir.selection.compare_winner_selection` audits one pair of equal-length finite
score vectors under descending-score/ascending-index selection. Returned indices
address those vectors, not Assembly neuron IDs. It converts
binary floats and integers to exact integers with one common power-of-two
scale. The result records observed winner agreement separately from certification:
reference boundary gap > twice the maximum observed absolute error. Equality is
insufficient. Ties can agree without certification. Empty or all-selected sets
have no competing index and certify only that trivial selection.

`formal/AssemblyIR/Selection.lean` proves that a strict pairwise gap and error
bounds preserve separation, then lifts the statement to every selected/outsider
pair. This is not a proof of Python conversion/sorting or the CUDA kernels.
Executable tests cover an allclose-but-different counterexample, boundary equality,
subnormals, overflowing float differences, exact uint64 values, invalid inputs,
and exhaustive small integer vectors. The capacity substrate replay reports
agreement and certified counts in JUnit properties; its drive tolerance remains
a separate gate. It rejects mismatched vector sizes instead of truncating.

The diagnostic does not certify future steps, arbitrary backend tie policies,
biological meaning or scientific adoption. Exact integer conversion is an audit
cost outside the projection hot path. Lack of a certificate is inconclusive,
not proof of disagreement. No probabilistic confidence interval is implied.


<a id="contract-migration-identity"></a>
## Migration comparison identity

`research.compare_migration` compares integer-valued historical fields exactly,
with matching integer type: seed identities, lengths and error positions are not
continuous measurements. Float-valued observations retain the existing fixed
5e-6 relative / 1e-7 absolute tolerance. Booleans remain distinct, including in
nested arrays. Historical A1 rows must have unique (seed,p) keys. JSON input must
have unique object keys; neither duplicate rows nor duplicate object members may
silently replace conflicting evidence. Ambiguous inputs reject the comparison.

These checks establish numerical comparison scope, not validity of a historical
protocol or its provenance. Run records, protocol review and true negatives remain
independent obligations. Tests include duplicate conflicting references and integer
changes small enough that the former floating tolerance accepted them.

Comparison version 2 receipts also record the comparator source SHA-256, alongside
both artifact hashes. Earlier unversioned receipts retain their historical scope.

The version-4 `capacity-paired` comparator projects a version-3 run into its explicitly keyed
control and refracted cells, then compares each against its own historical artifact
using one independently supplied seed order. Treatment must cover exactly its
historical checkpoint grid. Control may contain later checkpoints, but every legacy
checkpoint must be present; this permits one paired run to use the treatment's
larger grid without discarding control observations. Both aggregate ceilings are
still compared when every reference seed is present. The receipt names and binds
the candidate and both references, the explicit seed order, and the comparator's
Git blob. The evidence graph recomputes every tracked receipt. A missing condition,
reference, cell key or per-seed value fails.


<a id="contract-capacity-execution"></a>
## Recorded capacity settings are execution inputs

The capacity experiment consumes arm_settings, device, distinct_gate and
distinct_low_bar from the run record. These values do not fall back to module
constants during execution. Arm settings must cover exactly the requested arms
and explicitly name boolean norm_init and synaptic_scaling. Distinctness bars
must be finite nonnegative numbers, with the fractional lower bar at most one.
Invalid settings fail before constructing neural state. Measurement allocations
follow the actual stored tensors' device. Constants remain CLI defaults only.

Tests verify altered arm/device settings reach the cell, distinctness thresholds
change acceptance, and invalid recorded settings never reach measurement. This
contract does not adopt a changed threshold scientifically: registration and
adoption remain separate obligations. Existing version2 records already contain
these fields; their historical values are preserved.


<a id="contract-capacity-comparison"></a>
## Paired refraction-capacity comparison

Protocol version 3 represents the Hebbian control and refracted treatment as two
complete named CapacityProtocol values in one immutable run. The conditions are
exactly `control` and `refracted`. Both use masked, ungated readout and must agree
on every field except activation and strength of refraction; the control strength
is zero and treatment strength is positive. Each condition receives its matching
OrganSemantics profile. Missing, additional or cross-condition drift rejects before
GPU construction.

Both conditions use the same ordered unique brain seeds. The independent pair-
sampling generator restarts from the same recorded measurement seed for every
condition and cell, so treatment/control measurements refer to the same sampled
item pairs. Results are keyed by condition and `(arm,n,k)` rather than position.
The version-2 single-condition document and output remain supported unchanged.

This contract establishes paired execution identity. It does not adopt a capacity
bar, infer that refraction is the only possible cause, or turn a post hoc contrast
into a preregistered scientific result. The registration owns the comparison bar;
the result register separately verifies retained sample-aligned movement.


<a id="contract-horizon-execution"></a>
## A1 horizon execution inputs

HorizonProtocol freezes and validates recorded model sizes, learning parameters,
probability grid, schedule, checkpoints and device before GPU construction.
run_width consumes this object, including k for the state-block readout. The
experiment cannot silently substitute global probabilities or learning defaults.
A requested probability absent from the historical comparison rejects before
execution, instead of allowing an empty all() comparison to claim PASS.

Protocol version2 adds explicit checkpoints and device. Version1 artifacts remain
readable evidence but require those fields to be supplied when reconstructing an
executable configuration. The digit generator and mod3 machine remain fixed
protocol semantics. Software controls do not adopt alternate scientific settings.


<a id="contract-evidence-json"></a>
## Evidence document decoding and record identity

research.json_documents supplies shared encoding and decoding for the runner,
artifact validator and migration comparator. Reading rejects duplicate object
members, nonfinite constants, float overflow and nonzero float underflow to zero.
Finite representable subnormals and arbitrary JSON integers remain supported.
Encoding is deterministic and refuses nonfinite floating values.

Embedded and reserved run records compare their deterministic encodings, so
boolean/integer/float substitutions do not pass Python's coercive equality.
This is document integrity checking, not a signature or scientific validity proof.
Historical source/registration hashes are not authenticated by this JSON check.
Comparison version3 also records the shared source-inventory fingerprint, so
comparison helper dependencies are covered alongside the comparator file digest.


<a id="contract-coin-seed"></a>
### Coin seed identity

`assembly_calculus.pfa._seed_winners` reads stable neuron IDs and the target
area mapping. It delegates to `activate_assembly`, mutating only activity in the
area and its owning engine. It does not recruit, learn, or discard unmapped IDs.
Malformed or missing IDs fail before activation. `remap=False` is rejected:
legacy numerical artifacts do not authorize treating identities as positions.
Attractor training and mixed-seed flips use this same boundary. This contract
neither validates legacy uniform seeding nor proves coin fairness or settling.
Controls: `tests/test_coin_seed_contract.py`, including a missing-ID seed and a
primary engine that raises if called instead of the actual owner.


<a id="contract-coin-operation"></a>
### Random-choice operation schedule

Code-derived card: construction forms two stimulus-cued snapshots with a recurrent
connection reset after each; attractor construction then materializes the population
and force-fires each snapshot equally. Resets and activation address the actual
area owner. A force-fire temporarily fixes winners, restoring the prior fixed flag
on success or failure. Construction mode and training/firing counts are checked
before population registration. Zero force-fires is a valid untrained control.

A flip reads the snapshots, bias, mode, seed, area noise configuration and owning
engine population extent. `k_split` seeds sampled stable IDs. `compete` at neutral
bias without input noise seeds a uniform compact k-subset of the complete population;
otherwise it uses the same mixed-ID seed, falling back to uniform only if empty.
Uniform seeding requires a full population and never interprets unrecruited positions
as neurons. Mode, finite bias in [0,1], and nonnegative integer rounds are checked
before mutation. Zero rounds is a seed-only control. Legacy construction flips are
rejected, since the historical path lacks a valid recurrent attractor instrument.

Settling disables plasticity, performs the requested recurrent projections, then
compares stable-ID overlap against both stored snapshots; ties select label zero.
Activity changes are intentional. This is not a calibrated Bernoulli probability,
a claim of fairness, or a guarantee that arbitrary training settings form attractors.
Controls live in test_coin_seed_contract.py and test_coin_construction.py.


<a id="contract-pfa-choice"></a>
### PFA branch experiment

Code-derived card: PFANetwork delegates single-target transitions to FSMNetwork's
symbolic table and neural state encoding. For branching transitions it chooses a
binary coin label, or cascades binary labels, and assigns a symbolic successor.
It does not decode that successor from a learned transition network. Transition
weights currently parameterize initial seed mixtures, not calibrated outcome laws.
Normalizing those weights alone cannot make the neural choice a valid PFA sample.

SeedMixtureChoice makes this experimental interpretation explicit and binds coin
population, plasticity, training/firing schedule, settling rounds and seed mode.
PFA branching requires this immutable configuration before constructing any areas;
FSM n/k/beta/rounds do not silently configure the independent coin. Deterministic
PFA construction allocates no coin and requires no choice configuration. Legacy
flip_mode arguments may only agree with the explicit choice; conflicting schedules
are rejected. Existing probability fields are retained as target weights for
compatibility, with no claim that measured transition frequencies equal them.

The context coin additionally learns during its read and overwrites context-driven
activity. The historical NemoMarkovPFA copied coin IDs into the arc and ignored
transition weights; it is now retired. The distinct ArcMarkovNetwork composition
has its own contract below. The context replacement has its own contract below.


<a id="contract-pnas-roundtrip"></a>
### PNAS reciprocal round-trip protocol

`programs.pnas_extended.run_pnas_reciprocal` owns the parity protocol: source
training supplies recurrence explicitly, followed by two plasticity-enabled
reciprocal projections and a stable-ID overlap against the original source.
The score is measured after bidirectional training, not a frozen recall. Its
parameter record names the recurrence option and sampled NumPy engine. The
cross-repository test delegates to this function instead of rebuilding a schedule
under a Brain-level default. Golden values and tolerances remain unchanged.
A beta-zero control must move the score; it does not certify theorem equivalence
or fixed-connectome sequence behavior.


<a id="contract-nemo-arc-observation"></a>
### Refracted-arc observation initialization

NemoArcFSM.run reads inside brain.probe: learning, bias charging and recruitment
are prohibited. A cold arc with fewer than k materialized neurons cannot be read.
A mechanism-disabled control must instead initialize a population before observing
it, e.g. brain.materialize_area(fsm.arc_area) without transition teaching. This
initialization is not proof of learned transitions. In particular, materializing
before training changes the sampled construction and needs its own evidence.
The separate test_nemo_arc_contract.py covers cold rejection, initialized untrained
readout, trained behavior, unchanged readout bias, and accumulated training bias.
The distinct ArcMarkovNetwork controls are specified separately below; they do not
validate the retired historical Markov PFA.


<a id="contract-erp-context-reset"></a>
### ERP context reset and observation scope

ErpProtocol.context_reset selects construction (legacy default) or activity.
The runner executes that choice and includes it in erp_protocol; descriptions name
it even at the default. Activity reset preserves context IDs through the existing
_reset_context_winners(preserve_mapping=True) boundary. It neither disables other
recruitment nor restores other areas' activity. Read-only prohibits learning and
population changes but permits activity evolution. Separate brain.probe scopes
restore the initial brain state for repeated observations of an initialized parser.
No cold-read exemption or automatic initialization is introduced. The isolated
fixture's repeatability does not establish ERP discrimination, calibration, or
whole-parser purity outside the brain. Tests: test_erp_context_reset_contract.py
and test_parse_idempotence.py::test_isolated_activity_reset_parse_is_idempotent.


<a id="contract-cyclic-group"></a>
### Configurable cyclic benchmark

word_problems.cyclic_group builds the additive residues modulo a positive integer
order. Integer generators are normalized modulo that order without reordering the
alphabet. The gcd of the order and generators must be one; otherwise they generate
a proper subgroup and construction raises. A closure enumeration independently
checks the requested cardinality, with a runtime exception rather than an assert
that optimization can remove. Identity is zero, composition is modular addition,
and the group is abelian (hence solvable). Closure's existing repr-sorted element
order is preserved so Z60/Z120 state labels and transition tables remain unchanged.
The named wrappers remain compatible. Controls include a 60-state request with
only even generators and exhaustive pairs of residues for orders 1 through 12,
compared against all linear combinations. These finite tests do not constitute a
Lean proof of the general constructor or a neural word-problem result.


<a id="contract-transition-domain"></a>
### Transition domain before neural construction

Code-derived card: FSMNetwork stores a symbolic transition table and trains state
and symbol encodings; PFANetwork delegates only its deterministic edges to that
FSM and selects branching successors from coin labels. Previously neither checked
that all edge endpoints and symbols were declared; invalid branching targets could
escape the FSM's subset entirely. A missing deterministic edge could stimulate a
symbol before raising KeyError. These are bookkeeping failures, not failed neural
learning, and must be detected before changing brain state.

TransitionMap now requires nonempty string edge labels and finite positive real
weights at most one (booleans and numeric strings are not weights). Duplicate edges
are rejected rather than silently interpreted as distinct choices. validate_domain
checks unique named state/symbol declarations, initial-state membership and every
edge. Both constructors consume this validation before allocating any area or
stimulus. Empty alphabets and partial tables are allowed; states must include the
initial state. Step on an absent edge raises before neural activity changes.

<a id="contract-branch-schedule"></a>
### Conditional branch schedule

TransitionMap.branch_schedule preserves declared target order and returns immutable
(target, conditional weight) pairs. Each weight is its positive target mass divided
by math.fsum of remaining masses; the final target has weight one and serves as the
fallback. This factorization accepts positive relative masses; PFANetwork separately
requires each complete group to sum to one within its explicit tolerance. Summing
the tail directly avoids catastrophic cancellation in 1 minus a rounded prefix.
In exact arithmetic, calibrated conditional Bernoulli choices would reproduce the
normalized target masses. That conditional statement does not calibrate the neural
seed-mixture selector; its observed outcome law remains an empirical question.

PFANetwork consumes this schedule through one loop for binary and multiway branches.
Binary choice retains the caller seed; multiway choice retains one generated seed
per attempted decision. Deterministic transitions allocate no coin. Branch schedules
are symbolic data, not evidence of a learned neural transition circuit or softmax law.


<a id="contract-arc-markov"></a>
### Decoded-state arc Markov experiment

Historical card: NemoMarkovPFA built two state areas, a symbol area, a refracted
arc and a legacy coin. It copied the coin's stable IDs directly into the arc's
compact winners, reused one area's state IDs in another, ignored transition weights,
and returned a table-selected successor after discarding the neural output. Clearing
refraction repeatedly also erased the proposed accumulated mechanism. Those bodies
are retired. NemoMarkovPFA and AlternatingMarkovNetwork now raise before touching a
brain and name the replacement; Git history retains the invalid instrument.

ArcMarkovNetwork is a different explicit protocol, arc-symbol-feedback-v1. A validated
closed transition domain compiles each outgoing target's ordinal into a branch symbol
for NemoArcFSM. Each state has at least one declared outgoing group; no missing row
silently becomes a self-loop. Training constructs the assigned state code and fully
materializes the arc, then executes the recorded number of teacher-forced presentations.
The zero-presentation control has the same initialized population and no transition
teaching. Protocol geometry, local density, beta, refraction and presentations are
immutable, explicit and JSON-serializable. Branching independently requires
SeedMixtureChoice; deterministic graphs allocate no coin.

At sampling, the shared conditional selector supplies a branch index. That index
selects a stimulus, not a successor label. NemoArcFSM.run observes the learned
transition, returning its nearest-overlap state readout. The enclosing brain.probe
restores neural activity, recruitment, RNG and learning state. Only the decoded
label advances outside the probe and becomes the next step's canonical state cue.
Reset changes this feedback label only; it does not clear learned refraction.
No stable ID crosses an area boundary. MarkovChainModel compiles trace frequencies
through this same implementation and requires the explicit protocol.

This is decoded-state feedback with branch-symbol inputs. It is not the historical
alternating-area architecture, continuous neural state carry, a calibrated Markov
sampler, a proof of a softmax law, or evidence for arbitrary long sequences. Tests
must distinguish three boundaries: conditional-weight consumption, genuine arc
readout instead of target lookup, and sensitivity to disabled transition teaching.
A perfect small untrained fixture is possible and must be retained when observed.
The separate legacy SoftmaxContextCoin is retired; see contract-context-choice below.

<a id="contract-trace-counts"></a>
### Trace count identity

A `TraceStep` keeps three different counts. `num_winners` is the size of the
observed assembly now. `num_ever_fired` is the target area's cumulative recruited
population and must come from `Area.get_num_ever_fired()`, never the ambiguous
`Area.w` alias. `num_first_winners` is the engine projection's new-recruit count;
an observation-only cue injection records zero because no projection occurred.
Assigning a partial cue may change current winners but must not reduce cumulative
recruitment or reuse a stale new-winner count from training.

Control: `test_pattern_complete_trace_reports_recovery` trains a population larger
than the retained half cue and verifies that round zero preserves the former
cumulative count, reports the cue size separately, and reports zero new winners.

<a id="contract-trace-sweeps"></a>
### Trace sweep model identity

Every tracing sweep configuration carries both its engine and sampled-recurrence
policy into Brain construction and emits both fields with each observation row.
The sampled NumPy default remains `warn`; callers making a deliberate sampled
comparison must say `acknowledged`, while `forbid` stops recurrent work. Unknown
policy names fail during Brain construction before areas, stimuli or observations
exist. A row without the model choice that produced it is not a portable result.

Controls exercise an acknowledged teaching sweep without warning, retain its
engine and policy in output, and reject an unknown policy before projection.


<a id="contract-seeded-observation"></a>
### Seeded read-only observation

Brain.read_only(seed=...) preserves the existing no-learning/no-recruitment and
activity/RNG restoration scope. An optional nonnegative integer seed creates a
NumPy SeedSequence and one child stream for each distinct backend RNG, in owner
enumeration order. Each child's state is installed using that RNG's bit-generator
type. Original generator objects and states are restored even after exceptions;
nested scopes restore the enclosing stream. None preserves the original behavior.
The policy is read-only-seed-v1. It controls backend observation randomness, including
native input noise, not global process RNGs or a new connectome seed. Different
backends need not share identical trajectories or noise laws. This strict API does
not use the legacy probe-isolation environment escape hatch.

<a id="contract-noise-only-observation"></a>
### Materialized noise-only observation

The NumPy sparse and Torch sparse engines previously preserved the incumbent assembly
whenever accumulated synaptic drive was zero, even with positive configured input
noise. That discarded the independent noise contribution before winner selection.
Zero signal means every accumulated drive entry is zero, not that signed entries
cancel in their sum. A balanced positive/negative vector must still undergo
selection. The zero-signal shortcut now applies only at zero noise. Positive-noise zero-signal
selection requires a fully materialized population; a partial population raises
instead of pretending to sample the absent population. Existing no-input scheduling
semantics are unchanged: a zero-sized stimulus can explicitly schedule this read.
Compiled NumPy selection uses the same noise/competition selector as ordinary
selection, retaining its existing-column candidate population. Noiseless zero-drive preservation is retained. The fixed-target teaching path is
unchanged. These are backend selection semantics, not a claim that noise calibrates
an attractor's outcome frequencies.

<a id="contract-context-choice"></a>
### Context-conditioned attractor readout

Historical card: SoftmaxContextCoin coupled context to an outcome area without
clamping the intended target, learned during flip, then overwrote context-driven
activity with a random seed and ran outcome-only recurrence. It also used stable
IDs as compact positions. Those operations did not implement the advertised
context-dependent softmax law. The class now raises before brain mutation and names
the separate replacement; historical numerical artifacts remain unchanged.

AttractorConfig owns only two-attractor construction. SeedMixtureChoice extends it
with the existing mixture/read schedule, preserving that API and numerical replay.
ContextChoiceProtocol instead specifies assigned disjoint context codes, an integer
presentation count for each context/outcome pair, a separate coupling beta, read
rounds and native noise amplitude. It takes AttractorConfig, rejecting unused
seed-mixture settings. Teacher forcing clamps the actual context and target assembly
while potentiating context->outcome; no recurrence is included in these coupling
writes. Noise is enabled only after teaching. Zero counts and zero coupling beta
are valid controls. Context and outcome populations are fully materialized.

ContextAttractorChoice.observe clears activity inside seeded read_only, cues the
chosen context and holds it throughout the requested context-plus-recurrence rounds.
A zero-sized stimulus schedules outcome updates even in source-disabled controls.
The context is never overwritten by a new random seed, and reading does not teach.
Context and recurrence gates are explicit boolean controls. The observation reports
both stored-attractor overlaps, their absolute margin and a label; tied overlaps
produce None, including silence. These are similarity scores, not probabilities,
and arbitrary parameter choices need not produce useful attractors or readouts.
This is context-attractor-v1, not numerical reproduction of historical softmax goldens.


<a id="contract-cue-recovery"></a>
### Cue replacement and recurrent recovery

replace_neurons is pure: a nonempty unique stable-ID reference, explicit unique
population containing it, exact nonnegative integer replacement count and explicit
nonnegative integer seed produce an immutable cue in the same area. Replacements
are distinct and outside the reference. Impossible counts raise, never clamp.
Sorted IDs and one NumPy generator make input order irrelevant. Permutation prefixes
make increasing replacement counts nested at a fixed seed and population. This
changes active membership; it is not additive input noise or independent dropout.

observe_recovery accepts a separate reference and cue, validates their area/IDs,
requires a fully materialized population and performs explicit recurrent rounds
inside strict seeded read_only. It temporarily releases a fixed target, activates
the cue through the stable-to-compact boundary, and restores activity, clamps,
learning and RNG ownership on exit. Disabling recurrence returns the delivered cue
as a no-dynamics control. No stimulus, training or materialization is performed.
Existing configured native noise remains in force and must be named in experiments.

The observation retains reference, delivered cue and recovered snapshot. Both
scores divide intersection size by REFERENCE size, unlike min-size overlap, so an
uncorrupted subset does not score as full recovery. Improvement is final minus cue
score. Tests must show improvement and fail an initialized learning-disabled or
no-dynamics control; correct labels alone do not establish recovery. No general
corruption tolerance or biological basin claim follows from this API.

TorchSparseEngine.materialized_count now reports its compact population size,
including zero before growth and n after materialization. Its previous inherited
None incorrectly advertised the dense-engine convention. Dense engines retain
None (full population allocated by construction); an unknown sparse area also
returns None, so callers validate area identity first.

This observer rejects cues and recovered winner sets larger than the reference.
Reference-denominated overlap is coverage, not precision: an oversized set could
otherwise score 1 despite extra winners. Variable-cardinality readout requiring
precision/recall analysis must use a separate explicitly defined measurement.
An oversized result raises inside read_only so state is still restored.

RecoveryObservation validates these membership invariants on direct construction
as well as through observe_recovery: a nonempty unique reference, unique cue and
recovered IDs in the same area, and neither set larger than the reference. Invalid
observations cannot reach score computation. Empty cues/results remain valid failed
observations against a nonempty reference; negative improvement is retained. The
same validation runs before activating the cue and again on the returned snapshot.


### Shared legacy and runner document storage

The contract-evidence-json boundary also owns write_new_document. Encode and validate
before creating directories or opening the output, then create exclusively with
UTF-8 and flush. Existing files are never replaced. The migrated runner and legacy
ExperimentResult.save use the same function; legacy loads use load_document.
Unsupported objects/arrays and nonfinite numbers raise instead of default=str or
nonstandard JSON. Experiments must explicitly represent their arrays and undefined
statistics; historical files are not silently repaired. Same-second legacy filename
collisions therefore fail without modifying the first result. This storage change
does not add missing run provenance or certify scientific validity of legacy data.


Common legacy t-test significance flags are native booleans. For the historical
noise study, a test already classified as degenerate retains that classification
and false significance, and represents undefined t/p/d as JSON null. Its display
names the reason. The underlying statistical helpers retain their computational
NaN convention for existing analysis callers; other producers must explicitly
resolve that representation before strict storage. No undefined statistic is
converted into zero, a significant result, or a fabricated numeric value.


<a id="contract-legacy-execution-status"></a>
### Legacy execution status

ExperimentResult.success is a native boolean describing execution, never a
scientific verdict. Construction, loading and ordinary assignment reject strings, numbers, null
and NumPy boolean objects rather than interpreting truthiness. The same check runs
before serialization because legacy results remain mutable. A post-construction
invalid status cannot create an output directory or file. False execution outcomes
and their messages remain representable. This does not validate arbitrary nested
metric schemas or convert legacy completion into scientific adoption.

Persisted legacy result documents must explicitly contain success; loading a missing
status may not inherit the constructor default True. Non-object documents also
raise at this boundary. Historical missing statuses are not inferred or rewritten.


<a id="contract-result-sensitivity"></a>
### Registered result sensitivity

A MEASURED Result carries at least one retained sensitivity check or a specific
sensitivity gap. They may coexist when a composite claim has verified and uncovered
facets. Each check names a repository-relative JSON artifact already
present as a typed artifact evidence edge, an explicit unique sample-identity
vector, treatment and control scalar vectors, a directional relation and a finite
strictly positive minimum effect. All three vectors are nonempty and equal-length.
Values are paired only in the explicit sample order; duplicate identities,
container-valued observations and nonfinite numbers reject the register.
Paths use RFC 6901 escaping for object-key tokens and `*` only as an explicit
list expansion, allowing keyed result cells without confusing `/` in an identity
for a path separator.

For all-greater, every treatment-control difference clears the minimum. For
all-less, every control-treatment difference clears it. For all-different, every
absolute difference clears it. A single failing sample invalidates the check;
averaging cannot hide a dead seed. Unsafe, missing or malformed artifacts and JSON
paths also fail validation. A sensitivity gap keeps a legacy claim visible but does
not satisfy the evidence. A retained contrast establishes instrument movement under
the named control; scientific interpretation still depends on the registration,
protocol, regime and result caveats.


<a id="contract-weight-normalization"></a>
### Weight-normalization mutation

`normalize_weights(target, source)` is a live weight mutation, not an acknowledgement
that normalization would be desirable. An engine may expose it only when it owns
mutable connection storage and can normalize every selected source-to-target
connection. Engines with fixed or regenerated connectomes must reject the operation
with `NotImplementedError`; they must not return a successful no-op. Sparse NumPy
and Torch implementations normalize their stored weights and invalidate cached
drive state. The contract is exercised by negative exact/dense calls and a positive
sparse mutation test. This contract says nothing about whether a particular
normalization schedule is scientifically appropriate for a registered experiment.


<a id="contract-stim-preallocation"></a>
### Stimulus-vector preallocation

`preallocate_stim_targets(target, min_columns)` is an optional storage
preparation operation for lazy stimulus fibers. A capable engine must extend
the vectors while preserving existing values and semantics. Engines with dense
or non-mutable stimulus storage reject direct calls with `NotImplementedError`;
composition code consults the explicit `supports_stim_preallocation`
capability and omits this optimization when it is not applicable. The
operation changes capacity only; it does not train weights, change winners, or
establish a scientific result.


<a id="contract-overlap-space"></a>
### Overlap index-space boundary

`overlap` accepts two `Assembly` snapshots or two explicitly same-space winner
arrays. Mixing an `Assembly` with a raw array is rejected at runtime because
the raw array's compact-versus-neuron-ID space is otherwise unstated. Static `CompactIdx`/`NeuronIds` annotations provide the stronger check for two
raw arrays, and the branded ndarray values also retain their space at runtime,
so mixed branded pairs are rejected before scoring. Cross-area `Assembly` overlap remains allowed when
the caller intentionally compares stable neuron IDs.

Both `Assembly` snapshots and raw winner arrays must contain one-dimensional,
integer, unique indices. Duplicate entries would otherwise be collapsed by the
set-based metric and could turn malformed activity into a plausible overlap.


<a id="contract-readout-threshold"></a>
### Readout threshold

`fuzzy_readout` treats its threshold as a decoder criterion and requires a
finite probability in `[0, 1]`. Invalid thresholds reject before lexicon lookup,
including for an empty lexicon. `None` remains the valid outcome for an empty
lexicon or a best overlap below the criterion; it is a decoder uncertainty, not
a claim that the neural operation failed.

When multiple labels have exactly equal overlap, both `fuzzy_readout` and
`readout_all` break ties lexicographically by label. Decoder output therefore
does not depend on dictionary insertion order; tie policy is part of the
measurement semantics rather than an accidental container property.


<a id="contract-batched-next-token"></a>
### Batched next-token inference

`BatchedLM` is admitted only from an engine that explicitly advertises
`supports_batched_next_token`. Backend shape coincidence or private-field
presence is not a conformance test. Unsupported engines fail before importing
CUDA-specific implementation code; a capable engine must expose the frozen
connectome and vocabulary-drive state required by the batched predictor. This
contract governs execution availability and does not certify agreement with a
sequential readout; parity remains a separate measured gate.


<a id="contract-lexicon-build"></a>
### Lexicon construction

`build_lexicon` validates the target area, positive round count, unique word
labels, and an exact word-to-stimulus mapping before projecting the first word.
Unknown stimuli therefore cannot leave a partially trained brain. Recurrent
reset is dispatched through the target area's owning engine, which matters for
brains mixing sparse and explicit areas. Each projection uses recurrence and
the reset only separates successive lexicon entries; it does not establish a
readout accuracy claim.


<a id="contract-reset-area-connections"></a>
### Area-connection reset ownership

`Brain.reset_area_connections(area)` dispatches to the engine that owns the
named area. It forgets learned area-to-area weights while preserving
stimulus-to-area fibers according to that engine's reset contract. Code using a
mixed sparse/explicit brain must call this facade rather than reaching through
the primary engine; otherwise a successful call can mutate unrelated storage
or leave the intended owner unchanged.


<a id="contract-area-fix"></a>
### Area fixed-assembly ownership

`Brain.is_fixed`, `fix_assembly`, and `unfix_assembly` resolve the named area's
execution owner and keep the public `Area` descriptor synchronized with that
owner. Per-area code must use these façade methods rather than calling a
primary engine directly; this is required when one Brain mixes sparse and
explicit areas.


<a id="contract-assembly-attention"></a>
### Typed assembly attention (design target)

The package implements a pure snapshot `attend` readout; it does not mutate a
Brain or learn query-key fibers. Its future Brain-backed IR node must represent
query, key, value, target, compatibility projection, sparse selection,
optional recurrent refinement, and causality as distinct fields. Multihead
composition is an explicit product of independent heads followed by a merge.
The implementation gate requires no-compatibility, value-shuffle, and
future-token-leakage controls; backend agreement alone is not evidence of an
attention claim.
