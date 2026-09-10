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
importing hardware backends. The links identify obligations; they are not badges
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
Preparatory classification still has a separate, documented mutation defect;
these readiness checks do not certify that entire pipeline as observational.
