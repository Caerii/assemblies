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
Names must be nonempty strings and absent from the applicable area registry.
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
transactional, or establish GPU conformance. Stimulus namespaces and dynamic
resizing are separate contracts. Existing failures unrelated to identity/dimensions
may still require broader option preflight or rollback.
