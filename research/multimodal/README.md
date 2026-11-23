# Assembly-based multimodal online binding: experimental prototype

Target repository: [Caerii/assemblies](https://github.com/Caerii/assemblies) (`master`, inspected November 23, 2026).

## Scope

This is a small NumPy sensory-adapter prototype **built to call the existing library**, not a second assembly-calculus engine. `FlyHash` deterministically converts a dense vector into a `k`-sparse population code with a fixed random sign-sparse expansion and winner-take-all. `AssemblyCortex` injects these codes into `Brain(engine="numpy_exact")` and invokes the library's `merge()` primitive in three named hierarchy levels. `PlasticityGate` exposes a conservative policy decision and telemetry, but does **not** implement anti-Hebbian erasure or a novel learning mechanism; live native `merge` uses the library's own Hebbian update.

### Run

With a local checkout of the repository, copy `multimodal_online_merge.py` into `examples/04_multimodal_online_merge.py`, then:

```bash
uv sync --no-group gpu
uv run python examples/04_multimodal_online_merge.py
uv run python examples/04_multimodal_online_merge.py --benchmark
uv run python examples/04_multimodal_online_merge.py --native
python -m unittest discover -s . -p 'test_multimodal_online_merge.py' -v
```

`--native` requires `neural_assemblies` to be importable and exercises a real `Brain` / `merge` route, but **has not been executed here**, because the full repository package is not available in the local sandbox. Integration errors and physical behavior must be checked in the repo before treating this as merged work.

### Contracts and guardrails

- Fixed encoder projection, deterministic `top-k`, finite nonzero input checks.
- Explicit area registration: stable neuron IDs and no silent confusion with compact indices.
- Native mutating `merge()` with recurrence and backprojection switched off for a conservative first test. Actual biological claims require ablations on these flags.
- The `surprise` and `conflict` values are *heuristics* from sparse-overlap and are **not** recorded N400/P600 signals.
- `review` means *no automatic corrective training*. It does not alter any existing connectivity.
- The hierarchy prototype is a sequence of actual merges, **not** a demonstration of exact structural retrieval.

## Research gates before F2, foundation-model, or continual-learning claims

1. **Native integration:** run `--native` at small `n`, establish actual source-and-target readout, and compare learned vs `brain.frozen()` or `brain.read_only()` negative controls.
2. **Real modality encoders:** compare noisy real image embeddings and paraphrase embeddings against same-modality hard negatives; random projections alone do not establish semantic grounding.
3. **Crossmodal consistency:** learn matched object-word pairs, evaluate held-out combinations, shuffled-pair controls, swaps, and object permanence across frames.
4. **Depth:** train/read depth 1-6 trees and check exact reconstruction and collision with layer recycling and matched single-area baselines. Merely counting parent winners proves nothing.
5. **Continual learning:** stream 10k concepts, test time-stratified recall, adaptation cost, area saturation and ablate refraction; no unsupported 'zero forgetting' claims.
6. **Real physiological proxies:** use the repository's existing ERP adapters with independent validation; do not equate WTA settles directly with human EEG.
7. **F2 implementation:** independently measure bandwidth, accumulator hazards, WTA latency, resource use, frequency and total end-to-end projections, not theoretical peak estimates.

## Current evidence

The `--benchmark` route uses 20 independent random seeds and synthetic vector perturbations. This tests *sensor code robustness only*, not learning, cross-modal composition, neural assembly memory, hierarchy, or FPGA throughput. Keep the paired seed-level data; do not report one seed as a scientific result.

## v2: optimized exact-control backends (2026-11-23)

The `optimized_backends.py` module adds two independent, locally executable
reference backends, in addition to the `AssemblyCortex` adapter that invokes
`neural_assemblies.core.brain.Brain` and the native `merge` operation when the
actual package is installed:

- `IndexedHebbianMap`: a source-neuron-indexed sparse synaptic store. It is
  **algorithmically equivalent** to the additive, clipped dense control in
  `associative_workbench.py` (not to the native multiplicative-plasticity Brain
  implementation). Its scores and selected k winners were checked against the
  dense control over multiple streams and noisy cues.
- `InvertedEpisodicMemory`: posting lists indexed by source-neuron ID. Under
  constant-k source codes, maximizing intersection is equivalent to maximizing
  Jaccard, so it performs the **same nearest-memory lookup** as the original
  episodic baseline, but without scanning every stored item on every query.

### Reproduce

```bash
python -m unittest discover -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python benchmark_optimized.py
python associative_workbench.py --benchmark --seeds 8 --output experiments_v1.json
```

### Example local microbenchmark

The `benchmark_optimized.json` file records a single synthetic CPU run with
n=2048, k=16, 256 stored pairs and 128 query codes. The indexed synapse
implementation used ~0.54 MB packed-array-equivalent storage against 16.8 MB
for the dense float32 matrix, with identical prediction fingerprints; query
throughput was slower in Python due to dictionary traversals. The postings
baseline returned identical predictions to the naive episodic scan and was
far faster in this small measurement. These are **single-run diagnostics**, not
confidence intervals, not native engine parity, and not meaningful F2 speed
predictions. Packed-array sizes exclude Python object overhead; actual resident
memory for dictionary backends is higher.

### Important next engineering boundary

To benchmark the real library, install `neural-assemblies` in a normal
network-enabled development environment and execute
`python multimodal_online_merge.py --native`. Then add paired native
Brain/accelerated backend tests for: neuron-ID mapping, exact vs sampled
connectomes, fixed-target plasticity, refraction, and read-only observation.
Those tests must pass **before** claiming an accelerator replacement or
investing in HLS. No local full-repository clone or native package execution
was performed in the restricted execution container used to create v2.

## v3 Rust associative accelerator (2026-11-23)

This package now contains `rust_associative.rs` (PyO3 extension source for the
existing `crates/na-kernels` crate), `rust_bridge.py`,
`test_rust_bridge.py`, `benchmark_rust_bridge.py`, and
`RUST_INTEGRATION.md`. Read the integration document before copying files into
the main repository. **The Rust code was not compiled in this sandbox** (Rust
toolchain unavailable); all Python baseline tests pass and native Rust parity
checks are explicitly skipped when the extension cannot be imported. The source
is a candidate patch, not a measured Rust speedup. Its online BTreeMap is a
mutable sparse-synapse representation, not a compact CSR inference engine.

## v4: immutable sparse CSR checkpoints (2026-11-23)

Implemented `csr_snapshot.py`, `test_csr_snapshot.py`, and `benchmark_csr.py`.
The reference compiles mutable additive Hebbian rows into sorted CSR arrays,
records the source write version, and rejects stale snapshots after corrections
or training. `rust_associative.rs` now includes matching `compile_csr`,
`csr_ready`, `drive_csr`, and `predict_csr` methods for the existing PyO3
extension. **These Rust methods are not compiled in this environment.**

Local CPU diagnostic (`n=2048, k=16`, 256 associations, 256 queries, five
repeats, Python 3.13, `OPENBLAS_NUM_THREADS=1`): median checkpoint compilation
11.85 ms, mutable Python lookup 4,172 queries/s, Python CSR lookup 4,495
queries/s. Serialized CSR storage 536,616 bytes; Python object overhead
not counted. These are diagnostic measurements on an additive synthetic
baseline, **not** native Rust throughput and **not** a Brain-engine speedup.

Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 python benchmark_csr.py
python -m unittest discover -p 'test_*.py' -v
```

Current full local suite: 21 tests run, 20 passed, 1 skipped (native Rust
extension unavailable). See `RUST_INTEGRATION.md` for the compile/validation
gate and the required distinction from multiplicative plasticity.

## v5 — Hybrid CSR with online learning

`hybrid_csr.py` is an executable NumPy reference for an immutable CSR base
plus dirty, copy-on-write source rows. Newly learned edges and verified
corrections are visible immediately; reads always select either a dirty
row or the corresponding CSR row, never both. `checkpoint()` rebuilds and
atomically publishes a compact immutable view.

`rust_hybrid.rs` implements the corresponding PyO3 class for the existing
`na-kernels` crate. See `RUST_INTEGRATION.md` for installation steps.

Run CPU conformance + mixed-load benchmarking:

```bash
python -m unittest discover -p 'test_*.py' -v
python benchmark_hybrid.py
```

First CPU benchmark (n=2048, k=16, 128 initial items, 64 writes, 8 reads
per write, 5 sequential repeats) gave median **0.1075 seconds** for
`IndexedHebbianMap`, **0.1552 seconds** for hybrid checkpoint threshold 64,
and **0.1309 seconds** for threshold 256. All prediction signatures agreed
across methods. Thus the NumPy hybrid is **slower** in this mixed load;
its value is a durable representation boundary suitable for a native
read-optimized implementation, not measured acceleration yet.

The original assembly engine's multiplicative rule and inhibitory dynamics
are NOT simulated by these controls. Native Rust compilation and performance
remain to be verified after integration into a checkout of Caerii/assemblies.

## v6 — research governance and multiplicative learning contract (2026-11-23)

Start with [`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md), the dependency-ordered research roadmap, preregistration requirements, baseline suite, and explicit limits on conclusions. The machine-readable DAG is [`research_milestones.json`](research_milestones.json); validate it with `python check_milestones.py`.

[`multiplicative_reference.py`](multiplicative_reference.py) contains two independent algorithms for a controlled *existing-edge* multiplicative Hebbian fiber (`w <- min(w_max, float32(w*(1+beta)))`): explicit dense and sparse-row. This is **not** the full `Brain` engine. Its role is to establish a narrow algebraic parity checkpoint before testing genuine native `Brain` behavior. Run `python -m unittest -q test_multiplicative_reference` for randomized weights, drive, winner, clipping, input and read-only parity. The previous additive CSR prototypes remain isolated as a different learning rule.

A fully passing Python test suite does not establish a native Rust build or native `Brain` parity. Those are independently gated milestones P1 and P2.