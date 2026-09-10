# Assembly Calculus IR

One directory owns the language-neutral schemas and their Python/Rust protocol
consumers. Protocol documents carry parity evidence; they are not executable
assembly programs. Brain and projection schemas describe payloads but do not yet
have a unified compiler. See [verification obligations](VERIFICATION.md).

```text
neural_assemblies/ir/
  protocol.py              Python validation, loading and exclusive writing
  Cargo.toml               assembly-ir crate (member of crates/ workspace)
  rust/lib.rs              validated, lossless Rust protocol wrapper
  v1/
    protocol.schema.json   canonical protocol contract for both languages
    protocol.cases.json    shared acceptance and round-trip corpus
    brain.schema.json      brain payload description
    projection.schema.json projection payload description
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

`formal/AssemblyIR/Domain.lean` adds a checked domain interface: state-dependent
preconditions, one transition definition shared by execution and proof, and
invariant preservation for accepted schedules. Its `checkedExecute` entry also
validates the initial invariant and proves exact acceptance, including empty
schedules. See the
[checked domain contract](VERIFICATION.md#contract-checked-domain) for its
Midspiral/Dafny and LemmaScript integration, negative controls, and remaining
translation obligations. Check the kernel with `lake build` from `formal/`.
