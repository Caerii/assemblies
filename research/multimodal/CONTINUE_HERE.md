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

## Additional local reproduction (same session)

A local mirror of the branch's standalone multiplicative fiber plus the R1a
diagnostic passed **11/11** tests using Python 3.13/NumPy 2.3.5/pytest 9.0.2.
The p=.6, n=128, k=6, beta=.5, three-presentation, three-seed reproduction
gave exact recall at loads 8/16/32 = 0.416667/0.229167/0.125000,
matching the recorded exploratory run. This was NOT a full Git checkout,
and does not establish the native Brain conformance gate.

## R1b checkpoint — native Brain identity-retention protocol (2026-11-23)

On the canonical branch:
- `native_retention.py` implements three native Brain arms: recurrent
  plasticity, feedforward plasticity, and no-plasticity recurrence.
- `test_native_retention.py` checks configuration, winner IDs, summary
  accounting, and mock observation scope usage.
- `NATIVE_R1B_PROTOCOL.md` defines hypotheses, confounds, commands,
  and reporting rules **before running** the native experiment.

**Important status:** These R1b scripts are committed but have not been
executed within a complete `Caerii/assemblies` checkout. This environment
has no mounted native package and cannot resolve the GitHub host via its
local shell. No native recovery results or speedups are asserted.

Primary next step: run
`uv run python -m pytest -q research/multimodal/test_native_retention.py`
and the R1b CLI command in `NATIVE_R1B_PROTOCOL.md`; audit failures and
push a subsequent commit to this branch. In particular validate that nested
`brain.read_only()` and `brain.frozen()` produce reproducible observations
without persistent connectome or recruitment mutation, and check whether
clearing activity in the observation scope truly isolates each cue.

After R1b: reproduce the native repo's MASKED refraction readout as R1c,
rather than evaluating refracted memory through an unmasked readout.
Then compare actual target binding and compositional held-out queries.

## R1c checkpoint — paired refraction readout (2026-11-23)

Added on this SAME branch:

- `refraction_paired.py`: a paired, within-trained-brain masked-vs-net
  half-cue recall protocol using the native `Brain.set_masked_readout`
  switch, graded stimulus components, materialized `numpy_sparse`
  recurrence, deterministic probe indices and `read_only`+`frozen`
  isolation.
- `test_refraction_paired.py`: mocked protocol tests for sample pairing,
  scope usage, invalid configurations and masked flag restoration on errors.
- `REFRACTION_R1C_PROTOCOL.md`: pre-execution study design, controls,
  comparator and interpretation limits.

A separate corrective commit (3e7df1e) preserved original native
winner order for half-cue selection. Do not revert to sorted cues,
which differs from the older `refraction_memory_numpy.py` protocol.

**Current evidence status:** R1c is source-committed only. Neither its
tests nor the native study have been run against a complete repository
checkout in this session. The earlier registered refraction study
`research/notes/memory/PREREG_refraction_memory.md` is prior
evidence—not a result newly established by R1c.

**Next executable gates:**

```sh
uv run python -m pytest -q research/multimodal/test_native_retention.py research/multimodal/test_refraction_paired.py
uv run python research/multimodal/refraction_paired.py --n 2000 --k 60 --loads 8 16 32 --seeds 42 43 44 --output /tmp/refraction-r1c.json
```

Compare against the established `research.runner refraction-memory-numpy`
study, review discrepancies in read-only state management, and add native
state-invariance tests before treating any R1c values as established.

## R1d checkpoint — native role/content recombination (2026-11-23)

New files on `research/multimodal-continual-assemblies`:
- `role_recombination.py`: balanced held-out role/content split; native
  `Brain` + `project` + `bind` + `read_binding` protocols; pre-exposure
  unbound control; one-shot bindings; old-binding retention and role-local
  ranked readout; native stable neuron IDs throughout.
- `test_role_recombination.py`: schedule and leakage guards, config checks,
  role-local ranking, mocked native API call ordering.
- `ROLE_R1D_PROTOCOL.md`: preregistration, limitations, execution commands,
  strong episodic baseline and separate zero-shot gate.

R1d has been **committed but not executed on the native Brain runtime** in this
environment; the full repo isn't mounted here and the shell cannot resolve
github.com. Mock tests have not been run from a checkout either. Do not
claim a measured R1d accuracy, model speedup, or compositional benefit.

Immediate next gate:

```sh
uv sync --no-group gpu
uv run python -m pytest -q research/multimodal/test_role_recombination.py
uv run python -m research.multimodal.role_recombination \
  --seeds 11 17 23 --contents 8 --output /tmp/role-r1d.json
```

Before extending to full compositionality, separately validate R1b/R1c,
audit the native read-only scopes, run a same-memory-budget episodic
control, then hold out complete sentences without writing test pairs.
The old ZIP remains archival; Git branch is the continuation authority.
