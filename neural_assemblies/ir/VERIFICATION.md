# Assembly IR verification boundary

<a id="contract-ir-verification"></a>

## Status and ownership

The current v1 JSON schemas describe brain/projection payloads and parity
reports. Python and Rust now validate protocol documents against the same packaged schema.
Brain and projection schemas do not yet have equivalent executable consumers. They do
not yet define one executable projection semantics or a complete cross compiler.
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
schedule that can drift. This next integration is still to be implemented.

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
