# Assembly IR verification boundary

<a id="contract-ir-verification"></a>

## Status and ownership

The current v1 JSON schemas describe brain/projection payloads and parity
reports. Python and Rust validate only parts of those documents today. They do
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
