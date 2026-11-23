# Multimodal assemblies — continuation point

**Canonical branch:** `research/multimodal-continual-assemblies` in `Caerii/assemblies`.

## What this branch contains today

- `RESEARCH_PROGRAM.md`: P0–P8 dependency graph, exit gates, baselines, falsifiable claims.
- `README.md`: historical v1–v6 prototype results and commands.
- `RUST_INTEGRATION.md`: proposed PyO3/Rayon kernel integration and build requirements.
- `multiplicative_reference.py`: independent explicit dense/sparse fiber implementations (not native Brain).
- `test_multiplicative_reference.py`: deterministic dense/sparse learning conformance.
- `native_contract_probe.py`: direct `Brain(engine="numpy_exact")` smoke run, pending execution inside full checkout.

## External prototype snapshot

The companion prototype created in ChatGPT during 2026-11-23 is named
`assemblies_multimodal_prototype.zip` (v6). Its files include:
`multimodal_online_merge.py`, `associative_workbench.py`,
`optimized_backends.py`, `csr_snapshot.py`, `hybrid_csr.py`,
`rust_associative.rs`, `rust_hybrid.rs`, `rust_bridge.py`,
`multiplicative_reference.py`, all corresponding test and benchmark scripts,
the research milestone validator, the experiment records, and the documentation.

**Archive committed:** The compressed v6 companion **source** snapshot is stored under
`archive/v6-companion-sources.tar.xz.base64`; recover it with instructions in
`archive/README.md` and the recorded SHA-256. Most companion modules have not
yet been integrated as first-class source files; their bytes are preserved in
Git and must be migrated in small verified commits. The source snapshot excludes
historical generated JSON diagnostics, retained in the conversation ZIP.

## Evidence status

- Local standalone v6 suite (separate companion files): 33 tests run,
  30 passed, 3 skipped (native extensions unavailable) on 2026-11-23.
- The new branch-local multiplicative reference and the native probe
  have been committed but not executed from a complete branch checkout.
- No compiled Rust / native Brain parity, CUDA parity, F2 measurements, or
  multimodal foundation-model benefit has been established.
- Previous performance numbers are diagnostics or assumptions, not benchmarks
  proving acceleration against the native `Brain` runtime.

## Next five concrete actions

1. Run `uv sync --no-group gpu` in the branch checkout.
2. Run `uv run python -m pytest research/multimodal/test_multiplicative_reference.py -q`
   and `uv run python research/multimodal/native_contract_probe.py`.
3. Port the remaining companion source and tests into `research/multimodal/`,
   preserving commit-by-commit provenance.
4. Add strict numeric and state parity gates against native Brain, including
   exact/sparse distinction, stable vs compact IDs, recurrence, read-only
   isolation, refraction, and per-fiber initialization.
5. Only then compile Rust modules and benchmark parity, RSS, p50/p95 latency,
   write/read mixes, and consolidation overhead.

## Working rule

Every future change goes on this same branch unless the user explicitly
chooses a new branch. Keep this file updated with the latest HEAD SHA,
last validated commands, failures, and next experiment. Never claim a test
passed unless executed in the referenced environment.

## Latest checkpoint — 2026-11-23 (R1a)

Committed on this same branch:
- `retention_stream.py`: deterministic multiplicative-fiber continual learning, exact/Jaccard recall, source-target coverage, early-item retention, stronger exact episodic control, stream digest.
- `test_retention_stream.py`: deterministic stream, schema/bounds, invalid parameters, dense-positive-control checks.
- `RETENTION_R1A.md`: negative and positive exploratory results, run commands, SHA-256 hashes of local output files and interpretation boundaries.

Latest checkpoint before this handoff update:
`0cd7bbba12f884840ed06a5725ba680478c24d86`.

**Validation caveat:** the local validation sandbox ran 11 new tests against
the v6 companion multiplicative implementation, not a full checkout of this
GitHub branch. The branch-local `multiplicative_reference.py` has a compatible
but independently authored API. Native `Brain` and Rust execution remain
unverified. Regenerate diagnostics on the branch before publication.

**Result:** with n=128, k=6, 3 training presentations and 3 seeds, exact
recall at 32 learned items was 0% at p=.1, 2.1% at p=.3, 12.5% at p=.6,
and 80.2% at p=1.0. These are exploratory toy-model measurements, not
native-runtime results. Consult `RETENTION_R1A.md`.

**Immediate next tasks:** run branch-local test and native probe on a full
checkout; increase seeds and add p/k/round sweeps with held-out conditions;
instrument weight saturation and target support; implement a controlled
native-Brain-versus-explicit-fiber comparison; compile existing Rust
PyO3 extensions before making performance claims.
