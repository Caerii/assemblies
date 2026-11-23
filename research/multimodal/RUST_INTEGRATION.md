# Integration into Caerii/assemblies

Verified repository structure (master, Nov 23 2026):
- `crates/Cargo.toml` workspace
- `crates/na-kernels/Cargo.toml` PyO3 0.22 + numpy 0.22 + rayon
- `crates/na-kernels/src/lib.rs` Python module `na_kernels`
- `neural_assemblies/tests/test_rust_kernels.py` byte-identical initialisation tests

The new code intentionally extends the **existing `na-kernels` extension** rather than creating another PyO3 crate. It does **not** replace `Brain.project` and is *not* equivalent to the library's multiplicative synaptic update rule. This is the exact additive baseline used by `optimized_backends.py`; proving value beyond episodic lookup and native Brain requires separate experiments.

## Patch to apply in a local checkout

Copy `rust_associative.rs` to `crates/na-kernels/src/associative.rs`.
In `crates/na-kernels/src/lib.rs`, add near the imports:

```rust
mod associative;
```

Inside `fn na_kernels(...)` immediately before `Ok(())`, add:

```rust
    associative::register(m)?;
```

No `Cargo.toml` changes required.

## Build + validation

From repository root:

```bash
# In the existing Python development environment with Rust and maturin installed:
python -m pip install maturin
maturin develop --release --manifest-path crates/na-kernels/Cargo.toml
# Copy rust_bridge.py, optimized_backends.py, associative_workbench.py,
# multimodal_online_merge.py and both new test/benchmark scripts into an
# isolated experiments directory, or add this directory to PYTHONPATH.
python -m unittest test_rust_bridge.py -v
python benchmark_rust_bridge.py --n 2048 --k 16 --items 256 --queries 512
```

Run `pytest neural_assemblies/tests/test_rust_kernels.py` in the repository too. Do not interpret skips as a passing native build. For performance comparisons, pin thread count, warm up both backends, and report the raw per-repeat measurements.

## Precise semantics

- Inputs: stable neuron IDs as `np.uint32[k]`; invalid/out-of-range/duplicate IDs rejected.
- Learning: additive `float32` Hebbian coactivation, `min(cap, w + lr)` per pair.
- Prediction: score by summing rows in neuron-ID order; select top-k, lowest-ID tie-break.
- Correction: **explicit verified=True only**, decrease obsolete coactive edges by lr then potentiate corrected edges.
- Storage: the existing Python sparse map and the Rust `BTreeMap` have different actual heap footprints; `bytes_used` reports only packed-CSR equivalent, not resident set size.
- Rust implementation uses ordered per-row `BTreeMap` to support continual writes; it is **not** yet a packed CSR read path. Benchmark and profile before claiming HBM/FPGA readiness.

## Follow-up engineering

1. Make a flattened CSR snapshot for frozen retrieval, leave mutable delta rows for live learning, and compact only at versioned checkpoints.
2. Add a reproducible randomized parity test exercising repeats, correction, tie cases, clipping, and source-order permutations.
3. Add a *separate* native Brain-compatible kernel that multiplies `(1+beta)` with matching float32 rounding, plus explicit connectome identity/observation semantics; do not silently substitute additive learning.
4. Run modality-aligned paired video/text embeddings and held-out compositional relations, followed by a mixed time-stratified continual stream with frozen readout.
5. Use release + PGO / architecture-specific build only for machine-local diagnostics; keep published wheels portable.

## v4 compiled CSR path

`RustAssociativeMap` additionally exposes `compile_csr()`, `csr_ready()`,
`drive_csr(source)`, and `predict_csr(source)`. Compilation constructs a sorted,
packed source-row index from the existing online `BTreeMap` state. Updates
immediately invalidate the snapshot; callers must explicitly compile again.
This is deliberately a **strict checkpoint** policy, not a silent stale-cache
fallback. If the workload has frequent writes, benchmark recompilation cost;
there is no claim that CSR is always faster for every query mix.

`csr_snapshot.py` provides the corresponding pure-NumPy reference and
`test_csr_snapshot.py` verifies exact drive/winner parity, source-code
permutation, correction, and stale-snapshot detection. Rust native execution
must still be checked with a built extension.

## v5: Hybrid CSR + copy-on-write delta (new)

Copy `rust_hybrid.rs` to `crates/na-kernels/src/hybrid.rs`. Add `mod hybrid;`
near `mod associative;` in `crates/na-kernels/src/lib.rs`, and call
`hybrid::register(m)?;` inside `fn na_kernels(...)`. Run:

```bash
maturin develop --release --manifest-path crates/na-kernels/Cargo.toml
python -m unittest test_hybrid_csr.py test_rust_hybrid.py -v
python benchmark_hybrid.py --n 2048 --k 16 --initial 128 --writes 64 --reads-per-write 8
```

Python reference: `hybrid_csr.HybridCSRMap`. Rust binding:
`na_kernels.RustHybridAssociativeMap`. The frozen CSR stores the last complete
checkpoint, and a dirty BTreeMap source-row REPLACES the corresponding frozen
row when reading. Corrections operate on these complete logical rows.
Compaction never mixes different versions of the same row, and each read
sees its latest logical state. A checkpoint is built before the active frozen
arrays are replaced. There is no background concurrency or transactional
crash recovery yet.

**Important scientific boundary:** This is an additive Hebbian associative
*control*, NOT the native Brain's multiplicative, recurrent, fixed-connectome
simulation, and it has not demonstrated one-shot multimodal generalization.
Native Rust compilation and execution are UNVERIFIED in this environment;
Python reference tests must not be mistaken for Rust tests. The Rust path's
`stats()` returns `(version, checkpoint_version, dirty_rows, compactions,
base_nonzero_edges)`. Its Python-equivalent memory estimates exclude heap
allocation overhead, which needs separate RSS profiling in native benchmarks.

The first CPU mixed-load sweep found the mutable Python baseline faster than
copy-on-write CSR (see benchmark_hybrid.json). The hybrid design is primarily
a storage/lifecycle architecture until actual Rust measurements demonstrate
a performance advantage. With the current threshold of 64 dirty rows, the
benchmark triggered 42 compactions over 192 total writes, which explains
much of its overhead. Do not use this workload to claim an acceleration.