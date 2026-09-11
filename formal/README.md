# Assembly IR proofs

This Lean package proves small semantic boundaries used by the executable
Assembly IR. The code that implements each boundary links to its section in
[`neural_assemblies/ir/VERIFICATION.md`](../neural_assemblies/ir/VERIFICATION.md).
The proofs do not certify a numerical backend unless a concrete refinement theorem
says so.

Run the library and independent declaration checker from this directory:

```bash
lake build
lake env leanchecker AssemblyIR.Wire
```

Run Lean against the same explicit-round acceptance corpus used by Python and Rust:

```bash
lake exe check-wire-cases ../neural_assemblies/ir/v1/explicit-round.cases.json
```

`AssemblyIR.Projection` defines pure scaled-integer round semantics and a field
lowering. `AssemblyIR.Wire` decodes the JSON instruction independently and rejects
decimal values that are not exact at the caller's chosen scale. Connecting that
integer model to NumPy float32, clipping, Rust execution, or CUDA requires a further
state relation and error-bound proof.
