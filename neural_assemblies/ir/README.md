# Assembly Calculus IR

One directory owns the language-neutral schemas and their Python/Rust protocol
consumers. Protocol documents carry parity evidence; they are not executable
assembly programs. A separate `ExplicitRound` now executes a restricted dense
CPU projection profile; the legacy brain/projection payloads still lack a unified
compiler. See [verification obligations](VERIFICATION.md).

```text
neural_assemblies/ir/
  competition.py           policy configuration export and reconstruction
  projection.py            restricted executable dense CPU round
  protocol.py              Python validation, loading and exclusive writing
  Cargo.toml               assembly-ir crate (member of crates/ workspace)
  rust/lib.rs              validated, lossless Rust protocol wrapper
  rust/homeostasis.rs      validated homeostasis configuration transport
  rust/competition.rs      validated competition policy transport
  rust/projection.rs       validated explicit-round transport
  v1/
    protocol.schema.json   canonical protocol contract for both languages
    protocol.cases.json    shared acceptance and round-trip corpus
    competition.schema.json strict competition configuration contract
    competition.cases.json shared Python/Rust acceptance corpus
    homeostasis.schema.json strict runtime configuration contract
    homeostasis.cases.json shared configuration acceptance corpus
    explicit-round.schema.json executable round wire contract
    explicit-round.cases.json shared Python/Rust acceptance corpus
    brain.schema.json      brain payload description
    projection.schema.json legacy projection payload description
```

Python validates the packaged schema with `jsonschema`. Rust embeds that same
file with `include_str!`; keeping the crate here also includes the schema in
Cargo packages. Both compile the validator once. Neither keeps a second list of
field rules in implementation code.

The Rust wrapper exposes `protocol()` and `as_value()` instead of writable public
fields. Decoding validates before construction and preserves extension metadata.
Python's `write_protocol_document` rejects existing paths. New study evidence
should use the research runner, which additionally writes provenance.

From the repository root:

```bash
uv run pytest neural_assemblies/tests/test_protocol_wire.py -q
cargo test --locked --manifest-path crates/Cargo.toml -p assembly-ir
python -m research.literature.cross_lang.runner --protocol cross_lang.pnas_scaling
python -m research.literature.cross_lang.runner --julia
```

The Julia command requires Julia and the reference environment. Rust currently
checks the wire contract; it does not execute the Julia/Python numerical protocol.
Lean's reusable lowering rules live in `formal/AssemblyIR/Refinement.lean` and
still require concrete backend simulation proofs.

`formal/AssemblyIR/Projection.lean` instantiates those rules for a pure,
scaled-integer explicit-round model and a dense-kernel instruction. It proves
the field lowering for one round and finite programs, winner observation
agreement, frozen-weight and other-area frame laws, and exact checked admission.
Its selector and learning transform remain explicit parameters. This is a proof
of the pure IR lowering, not of NumPy float32 or a Rust/CUDA executor.

`formal/AssemblyIR/Domain.lean` adds a checked domain interface: state-dependent
preconditions, one transition definition shared by execution and proof, and
invariant preservation for accepted schedules. Its `checkedExecute` entry also
validates the initial invariant and proves exact acceptance, including empty
schedules. See the
[checked domain contract](VERIFICATION.md#contract-checked-domain) for its
Midspiral/Dafny and LemmaScript integration, negative controls, and remaining
translation obligations. Check the kernel with `lake build` from `formal/`.


For the first executable profile, import `ExplicitRound` from
`neural_assemblies.ir`, then use a standalone `NumpyExplicitEngine` and
`ExplicitRound(target="T", from_areas=("S",), plasticity=False).execute(engine)`.
All named areas and their initial state must already exist. Decode persisted
instructions with `ExplicitRound.from_document`; Python and Rust validate the
same `explicit-round.schema.json` and case corpus before construction. Rust
transport validation is not a Rust execution backend. Unknown features are errors.
See the [profile contract](VERIFICATION.md#contract-explicit-round) for numerical
semantics, constructed controls, and the remaining Lean/backend bridge.

For a Brain, call `instruction.execute_on_brain(brain)` instead of passing its
private engine to `execute`. This uses normal projection synchronization and
history, supports primary or auxiliary dense areas, and returns a detached winner
array. See the [Brain lowering contract](VERIFICATION.md#contract-brain-round).


`formal/AssemblyIR/Learning.lean` gives the shared mask frame: protected learned
values stay unchanged across schedules, while allowed writes and activity keep
their specified meaning. Its [contract and instantiation limits](VERIFICATION.md#contract-learning-frame)
distinguish this checked model from a proof of a concrete numerical backend.


Homeostasis configuration now has a shared Python/Rust wire contract:
`HomeostasisConfig.to_document()` produces `homeostasis-v1`; `from_document()`
validates it and reconstructs the runtime configuration. Rust exposes
`assembly_ir::homeostasis::HomeostasisDocument`. Both consume the packaged schema
and acceptance corpus. See [the exact contract](VERIFICATION.md#contract-homeostasis-wire)
for canonical scope encoding and the limits of this configuration-only bridge.


Competition settings are exported with `competition.policy_to_document(policy)`
and reconstructed with `competition.policy_from_document(document)`. The
`competition-v1` profile includes every setting explicitly. Store that document
in a runner parameter and reconstruct the policy from the recorded parameter in
the measurement function. Rust's `CompetitionDocument` validates transport; it
does not execute selection. See the
[wire contract](VERIFICATION.md#contract-competition-wire).


For numerical readout audits, use
`selection.compare_winner_selection(reference_scores, candidate_scores, k)`.
`winners_agree` describes the observed canonical sets; `margin_certified` means
the reference boundary gap exceeds twice the observed error (or the selected
set is empty/full). A failed certificate is inconclusive. Exact integer units
preserve binary-float and integer inputs, including subnormals and mixed Python
values, without overflowing score subtraction. This is an offline diagnostic,
not a projection backend or an assurance about future rounds. Its
[source-linked contract](VERIFICATION.md#contract-winner-margin) explains the
Lean theorem and the unproved implementation bridge.
