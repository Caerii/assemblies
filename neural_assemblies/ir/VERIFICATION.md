# Assembly IR verification boundary

<a id="contract-ir-verification"></a>

## Status and ownership

The current v1 JSON schemas describe brain/projection payloads and parity
reports. Python and Rust now validate protocol documents against the same packaged schema.
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
implementations. No backend currently discharges `Simulates`. Neither the
sampling distribution nor floating-point arithmetic has been formalized here.

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
instruction. Its strict document decoder requires every field, rejects extra
fields and other profiles, and copies source and drive sequences into immutable
tuples. This is separate from the permissive historical v1 projection payload;
that payload cannot silently become an executable program.

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
claimed here. Rust currently consumes the protocol wire schema only.


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
has its own contract below. The context coin remains unresolved.


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
The separate legacy SoftmaxContextCoin remains unresolved.
