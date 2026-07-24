# Cross-language Assembly Calculus parity

Compare metrics across Python, Julia ([AssemblyCalculus.jl](https://github.com/Caerii/AssemblyCalculus.jl)), and Rust backends using [assembly_ir/v1](../../assembly_ir/v1/) documents.

## Python baseline

```bash
python -m research.literature.cross_lang.runner --protocol cross_lang.pnas_scaling
```

## Export IR metrics

```bash
python -m research.literature.cross_lang.runner --export /tmp/pnas_scaling_python.json
```

## Julia (manual)

1. Vendor or clone [Caerii/AssemblyCalculus.jl](https://github.com/Caerii/AssemblyCalculus.jl).
2. Run capacity/scaling experiment with `n=5000, k=80, p=0.05, beta=0.1, seed=42`.
3. Emit JSON matching `assembly_ir/v1/protocol.schema.json` (see `julia/export_metrics.jl.example`).

## Rust

```bash
cd crates/assembly-ir
cargo test
```

Parse and validate protocol JSON; reference kernel TBD.
