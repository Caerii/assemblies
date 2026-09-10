# Assembly Calculus IR (v1)

Language-neutral interchange format for Assembly Calculus brains, projection
steps, and literature parity protocols. Enables cross-language golden tests
(Python, Julia, Rust).

## Layout

The schemas are package data, next to the Python that validates against
them (`neural_assemblies.ir.protocol.schema_path`). They lived at the
repository root as `assembly_ir/` until 2026-09-09.

```
neural_assemblies/ir/
  protocol.py            # validate + export protocol documents
  v1/
    protocol.schema.json   # parity golden + thresholds
    brain.schema.json      # areas, connectomes, stimuli
    projection.schema.json # fused projection step (sources, drive, renorm, slots)

.reference/
  AssemblyCalculus.jl/   # vendored Julia reference (git clone)
  AssemblyCalculus.jl.README

crates/assembly-ir/      # Rust JSON types + validation
```

## Consumers

| Backend | Path | Status |
|---------|------|--------|
| Python | `neural_assemblies/ir/` | validate + export |
| Julia | `research/literature/cross_lang/julia/pnas_scaling.jl` | reference script |
| Julia (full) | `.reference/AssemblyCalculus.jl` | vendored; wire metrics export |
| Rust | `crates/assembly-ir/` | schema types + JSON I/O |

## Engine features (v1)

- **Explicit winner IDs**: global neuron indices `0..n-1` (not sparse remap)
- **Slot-WTA**: `Brain.add_area(..., slot_count=N)` for CLASS digit slots
- **Supervised reinforcement**: `Brain.reinforce_connectome(src, dst, post_neurons)`

## Cross-language parity

```bash
python -m research.literature.cross_lang.runner --protocol cross_lang.pnas_scaling
python -m research.literature.cross_lang.runner --julia   # if julia on PATH
cd crates/assembly-ir && cargo test
```

Clone Julia package:

```bash
git clone https://github.com/Caerii/AssemblyCalculus.jl.git .reference/AssemblyCalculus.jl
```
