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

## R1d follow-up — collapse diagnostics

R1d now records the fraction of **distinct** lexical assemblies and role
bindings (before and after held-out one-shot writes). This is necessary
because uniform role-area attractors could otherwise masquerade as a
successful top-1 retrieval, especially with deterministic tie-breaking.
The diagnostic and its mock regression test were committed separately.

**Latest experiment is still NOT run on native Brain.** Do not infer any
accuracy from these commits. The next investigator should execute
`test_role_recombination.py` and the R1d CLI on a complete checkout,
then inspect both exact retrieval and assembly distinctness before writing
a scientific claim. Run R1b and R1c native gates as well.

The same branch remains canonical; no new research branch was created.

## R1d audit + R1e structural calibration (2026-11-23)

R1d has been hardened against two evaluation artifacts:
1. Familiar retention is now scored against a **fixed original candidate
   dictionary** both before and after held-out learning. The expanded
   dictionary score is reported separately, with an explicit
   `candidate_expansion_penalty`.
2. Top-1 is now strict: **ties count as failure**, even if the
   deterministic smallest-ID tie-break chooses the target. The old
   tie-broken diagnostic remains exposed under `tie_broken_top1`.

Added `zero_shot_baselines.py` and `test_zero_shot_baselines.py`.
The fixed invertible role-permutation oracle achieves 100% on held-out
pairs by construction, whereas a pair-only exact-key memorizer has 0%
held-out coverage. This is a structural control, NOT evidence of native
neural zero-shot transfer or a learned mechanism. See
`R1D_EVALUATION_AUDIT.md` for interpretation and commands.

Local standalone structural-calibration + decoder-artifact tests: 9 passed.
These were not executed from a full GitHub checkout and native R1b/R1c/R1d
runs remain outstanding. The underlying Rust extension is not compiled.

**Immediate reproducible gate:** run the branch-local pytest suites,
then run `python -m research.multimodal.role_recombination` on a full
checkout and inspect candidate-expansion penalty, strict winner margin,
role distinctness, and any failures. Do not claim one-shot/generalization
performance from mocked protocol tests.

## R1f checkpoint — leakage-resistant zero-shot target benchmark (2026-11-23)

Committed to THIS branch:
- `zero_shot_gate.py`: train-only adapter protocol, independent role
  transforms and held-out target codes; exact/overlap/strict-rank1 scoring;
  strong privileged oracle, exact pair memory, input-copy, learned sparse
  Hebbian and train-only shared cyclic-shift baselines.
- `test_zero_shot_gate.py`: leakage, mutation protection, code-shape,
  tie policy, deterministic splits, learned transform and negative controls.
- `ZERO_SHOT_R1F_PROTOCOL.md`: scientific gates, measured synthetic
  control outcomes, and explicit identifiability limitations.

A local mirror ran **23 tests passing** on Python/NumPy; this is
NOT a full native repository test run. For the cyclic-shift family,
the hypothesis-constrained `ShiftRoleLearner` learned the shared role
operator from six train pairs per role and achieved 100% exact
held-out target recovery over seeds 0,1,17,42; the sparse coactivation
pair model got 0% exact held-out recovery. This result is deliberately
synthetic, and does not demonstrate neural composition.

For random-permutation targets, exact held-out recovery was 0%
for unprivileged sparse Hebbian, while the fully privileged
KnownPermutationOracle succeeded by construction. Random mapping
is not identifiable from a few samples: DO NOT treat its failure as
a meaningful criticism of neural assembly calculus.

**Next execution gate:** run the committed `test_zero_shot_gate.py`
and R1b–R1d native tests in a full branch checkout. Then implement
a native `Brain` adapter for R1f that receives only train examples and
no hidden role map. Compete against the train-only ShiftRoleLearner,
not an artificially weak memory baseline; report costs and failed seeds.

No new branch created; keep committing to
`research/multimodal-continual-assemblies`.

## R1g checkpoint — native projection lower bound

Committed `native_zero_shot.py`, `test_native_zero_shot.py`, and
`NATIVE_R1G_PROTOCOL.md` to the SAME research branch. The adapter
uses `Brain`, materializes all relevant areas, injects stable lexical
neuron IDs, and measures held-out role-conditioned outputs using the
existing R1f independent target evaluator.

**No native experiment was run here.** This adapter deliberately ignores
target supervision; it is an unsupervised native projection LOWER BOUND,
not a learned shared role operator. Mock scheduling tests are committed,
but not a substitute for native runtime execution.

Next: execute the branch-local tests and adapter in a full checkout,
resolve any native API/shape errors, then implement teacher-guided native
training using only the observed pairs. Compare to `ShiftRoleLearner`
and report failure if native assembly calculus does not learn the
held-out cyclic role transform.

## R1h checkpoint — supervised native role projection (2026-11-23)

Committed `native_teacher_drive.py`, `test_native_teacher_drive.py`,
and `NATIVE_R1H_PROTOCOL.md` on this same branch. Unlike the R1g
unsupervised lower bound, R1h uses **train-only teacher external drive**
on a dense explicit native Brain role area; it never supplies hidden
role permutations or held-out target codes to inference.

**Unverified:** The native implementation and mock tests have not run
from a complete checkout in this session. This is an experimental
training schedule, not a measured compositional result. It may merely
clamp winners without learning a reusable transform. Read the protocol
and run its commands before scientific interpretation.

Next: execute R1b–R1h tests and numerical runs on the full repo;
inspect source->target weight changes and read-only invariance, compare
against the cyclic-shift learner and Hebbian controls, and preserve
failures on this branch. Do not create a new branch.

## R1i checkpoint — paired teacher controls and fail-closed local validation (2026-11-23)

Committed on this SAME branch:
- `native_teacher_drive.py`: beta parameter permits a no-plasticity arm at beta=0.
- `teacher_control_matrix.py` and `test_teacher_control_matrix.py`: matched correct-teacher, within-role shuffled-teacher, no-plasticity conditions; identical test targets and seeds; strict rank-1; duplicated-read consistency.
- `native_teacher_controls.py`: native CLI connecting R1h to the paired control matrix.
- `run_local_gates.py` and `test_run_local_gates.py`: local, Actions-free, fail-closed runner for R1b–R1i; exit statuses, logs, elapsed time, SHA-256 hashes, and Git HEAD preserved.
- `R1I_NATIVE_CONTROLS.md`: complete experimental protocol and interpretation caveats.

LOCAL evidence: 12 tests passed (7 control-matrix contracts and 5 runner contracts) in the standalone Python/NumPy sandbox. A separate dry run correctly marked two selected gates not-executed and all_passed=false. **Neither these tests nor the dry-run validate the native Brain engine.** This environment has no installed neural_assemblies or Rust toolchain and cannot clone via its shell.

Next execution on a full checkout of this branch:

```sh
python -m research.multimodal.run_local_gates --output /tmp/assemblies-native-gates-20261123
```

Preserve run.json, per-gate logs and Git HEAD in the next commit, including failures. Follow with R1i paired seed/gain/presentation sweeps, target-clamp fidelity, before/after connectome fingerprints, and same-budget strong learned-rule baselines. NO native accuracy or compiled Rust speedup is currently claimed from R1i.

## R1j checkpoint — complete explicit synaptic audit (2026-11-23)

On this same canonical research branch we added:

- `learning_audit.py` and `test_learning_audit.py`;
- active source→target Hebbian edge deltas and teacher-clamp selection
  diagnostics inside `native_teacher_drive.py`;
- `full_explicit_fingerprint` of **all** directed area-to-area matrices
  (including reverse and recurrent edges), checked around every query;
- fail-closed native R1i evidence requirements and
  `test_native_teacher_controls.py`;
- repaired fake Brain ownership to match the shared native explicit engine,
  with both beta-zero and beta-positive controls;
- automatic full-subtree discovery in `run_local_gates.py`;
- the protocol/evidence record `R1J_NATIVE_AUDIT.md`.

Local pure-Python mirror: **24 tests passed**, including mutated-read,
missing-evidence, beta-zero and wrong-connection tests. This was NOT
a full GitHub checkout and does NOT constitute native Brain test results.

Next exact command on a complete checkout:

```sh
python -m research.multimodal.run_local_gates --output /tmp/assemblies-native-gates-r1j
```

Follow with the fuller R1i native run in `R1J_NATIVE_AUDIT.md`.
Check whether teacher forced targets actually potentiate eligible
`LEX -> role` edges, whether read-only preserves the whole explicit
graph, and whether any held-out generalization survives strong matched
baselines. Do not claim new speed/capacity/zero-shot native measurements
until that experiment completes. Keep all subsequent commits on
`research/multimodal-continual-assemblies`.

## R1k checkpoint — hidden-label seed leak fixed (2026-11-23)

**CRITICAL scientific integrity correction:** Previously `evaluate()`
and `run_conditions()` passed `Config(seed=dataset_seed)` to
unprivileged models, allowing trivial reconstruction of every
supposedly hidden target by instantiating `Dataset(cfg)`. The exploit
was demonstrated on the OLD local interface: 100% held-out exact
for both random permutation and cyclic-shift families, with ZERO
training.

**FIXED on this SAME research branch:** distinct `Config.seed`
(private dataset generation) and `Config.model_seed` (public model
initialization), immutable `ModelConfig`/ `learner_view`, a shared
geometry validator, rejection of equal seeds, and oracle-only
privileged Dataset access. Both single-model and three-condition
factory calls now receive the restricted public config. Python
reflection/brute-force attacks remain out of scope; this is
API-level leakage prevention, not a secure evaluation sandbox.

New code: `baseline_sweep.py` and `test_baseline_sweep.py`.
Across twelve dataset seeds and four n/k settings, the synthetic
shared-shift-rule learner achieved 100% exact held-out recovery;
plain Hebbian pair mapping, exact-key pair memory, and identity
copy each achieved 0%. See `R1K_SEED_ISOLATION.md` for complete
metrics, hashes, scientific limitations and reproducibility commands.
An **independent local mirror passed 48 tests** across the
seed-isolation and sweep suites, not the full Git checkout.

**Still blocked:** native Brain tests/full checkout, Rust compilation,
and any honest native zero-shot/composition performance claim.

NEXT: rerun `python -m pytest -q research/multimodal` on a native
checkout, regenerate R1k's raw JSON and compare the canonical rows hash,
then run the R1i teacher-control matrix with teacher weight audits.
Keep committing further work to this branch, update this handoff and
preserve failed seeds.


## R1L checkpoint — twenty updates × twenty controlled stages (2026-11-23)

**Same branch, no new branches.** Latest checkpoint before this handoff:
\`bc65c7d27432e1cf5f28f40bcfd4d265b38c125f\`.

Completed and committed:

- \`continual_twenty.py\` — 20 sequential writes with a role-rule shift after
  round 10, disjoint held-out probe codes, corrupted examples at rounds 7
  and 16, prequential before/after metrics and read-state SHA checks.
- \`analyze_twenty.py\` — seed-level paired bootstrap, explicitly rejecting
  per-round pseudoreplication.
- \`campaign_twenty.py\` — exactly 20 controlled ablation stages over
  memory-window size, wrong labels, shift timing and n/k geometry.
- \`verify_twenty_evidence.py\` — decodes the Git-archived compressed 20-stage
  manifest, verifies SHA-256, and optionally recomputes all 8,000 learning
  updates. The first replay check exposed JSON tuple/list representation
  differences; canonical serialized comparison fixed the false mismatch.
- Four corresponding test modules plus the evidence archive at
  \`evidence/r1l-20stage-manifest.json.xz.base64\`.
- \`R1L_TWENTY_ROUNDS.md\` — detailed protocol, 20-stage table, statistics,
  validity caveats, raw JSON hashes and exact replay commands.

**Measured synthetic results:** Default n=128, k=8, 20 seeds, 20 sequential
updates per seed: mean exact held-out recall during rounds 11–20 was
79.5% for last-three rule evidence, 85.0% for last-one, 15.5% for
all-history and 0% for plain pairwise Hebbian / copy. Last-three minus
all-history paired seed mean +0.64, percentile 95% bootstrap interval
[+0.615,+0.665]. At round 20, all-history retained 50% old-rule accuracy
while last-three retained 0%; this is an engineered stability–plasticity
tradeoff, not native assembly behavior.

**20-stage campaign:** 20 stages × 20 seeds × 20 updates = **8,000**
learning updates, completed. The Git-tracked compressed campaign manifest
decodes to JSON SHA-256
\`35b42065dfff39a16133ad6857302c6bc9e616ea2ab54e5e24d5c694a44980d1\`.
The Git blob SHA-1 for the base64 archive exactly matches local source
bytes: \`6ecccb5fc4d324da332b909b5595de3eb6a602e2\`.

**Validation:** The eight newly committed Python source/test files were
compared to the exact locally tested Git blob hashes. A combined local
research-only mirror (prior R1k plus R1L) passed **82/82** tests, including
a full regeneration of the 8,000-update campaign. An older locally copied
\`teacher_control_matrix.py\` caused eight temporary import failures:
the **GitHub branch version was already correct**. The mirror was reconciled
and rerun successfully; do not attribute these failures to Git.

**Critical boundary:** This evidence is SYNTHETIC and hypothesis-aware.
It establishes neither native \`Brain\` compositional transfer nor
Rust/CUDA/FPGA acceleration. The complete native repository and toolchain
were unavailable in this execution environment. The seed space is small and
models share the evaluator process; publication-grade blinding remains open.

Next executable work:
1. On a complete checkout of THIS branch:
   \`python -m pytest -q research/multimodal\`
2. Verify evidence:
   \`python -m research.multimodal.verify_twenty_evidence --recompute\`
3. Run native checks through:
   \`python -m research.multimodal.run_local_gates --output /tmp/r1l-native-gates\`
4. Port the twenty-round prequential protocol to native teacher-driven
   \`Brain\` with matched baselines, independently logged weight changes,
   read-only matrix fingerprints, correct/no-plasticity/shuffled controls.
5. Only after native accuracy and semantics pass, compile and benchmark the
   existing Rust kernel candidates. No GitHub Actions dependency.

## R1m checkpoint — drift resilience under identical 20-round streams (2026-11-23)

Committed under `research/multimodal/`:
- `robust_shift.py` — train-only, two-confirmation surprise quarantine and decayed evidence learners (gamma=0.5 and gamma=0.8).
- `test_robust_shift.py` — stream-fingerprint parity against `continual_twenty.py`, read-only state checks, poison/switch behavior and deterministic replay.
- `R1M_ROBUST_ADAPTATION.md` — comparative measured results, caveats, replication commands.
- `evidence/r1m-20rounds.json.xz.base64` — entire compressed evidence artifact, reversible to JSON with recorded SHA-256.

**Source parity:** All 20 private training, withheld-probe and hidden-rule digests
matched the original R1l campaign exactly. The combined R1l/R1m local
test suites passed **32/32**. The four committed R1m files were
independently retrieved from GitHub and each exactly matched the tested
local payload.

**Measured synthetic post-change exact held-out accuracy, 20 seeds,
rounds 11–20:** quarantine2 0.750, exponential decay 0.5 = 0.810,
decay 0.8 = 0.545. Existing R1l last-example learner = 0.850,
recent-three = 0.795, global-history = 0.155, Hebbian pairs = 0.0.
The quarantine model recovers to 1.0 by round 14 and resists the
isolated wrong example at round 16; it is slower to adapt overall.
None of the new policies beats the last-example learner on the
chosen aggregate metric. Do NOT imply an OOM intelligence gain.

**Evidence:** original raw R1m result JSON SHA-256
`8018bf2c4bab9671fd05ecb337b52f677009e32fd49facf1acf85f542406ef6a`.
This result is SYNTHETIC and assumes the cyclic-shift hypothesis family.
Native `Brain` validation and Rust acceleration are not established.

**Next scientific gates:**
1. Reproduce R1m's SHA and 32 tests in a full checkout.
2. Predeclare a multi-regime family (unknown drift time, changing roles,
   multiple corruption rates, gradual drift), using fresh private seeds.
3. Make train-only change detection compare fairly to decay/window
   baselines with equalized state size and compute.
4. Pass R1b–R1j native Brain correctness/isolation experiments before
   transferring this prequential campaign to actual assembly circuits.

All future changes remain on `research/multimodal-continual-assemblies`.

## R1n checkpoint — unknown drift, multi-change, paired corruption (2026-11-23)

Canonical implementation now committed to THIS SAME branch:

- `unknown_drift.py`: deterministic 48-update streaming task with
  3 hidden rule-change points (13,27,39), two role areas, isolated or
  consecutive same-role corrupted training examples, eight fixed
  train-only baseline policies, disjoint probe and training cues.
- `test_unknown_drift.py`: geometry guards, private-stream pairing,
  fixed corruption timing, read-only prediction, adversarial mutation tests.
- `analyze_unknown_drift.py` and its tests: paired seed-level corruption
  effects and descriptive bootstrap ranges; reject mismatched streams.
- `verify_unknown_drift.py` and its tests: fail-closed compressed-evidence
  validation, SHA-256 verification and exact full campaign replay.
- `R1N_UNKNOWN_DRIFT.md`: complete exploratory results and limitations.
- `evidence/r1n-20seed.json.xz.base64`: complete 80-trial result, 3,840
  sequential environment updates (30,720 model-training calls).

**Provenance:** committed all eight experiment artifacts atomically and
verified each Git blob SHA-1 matched its independently tested local source.
Raw JSON SHA-256:
`ebfc57c6c6c0580d3933707b6445110ebbeacf86c1cc1b364491e685eeafbd06`.
Full evidence replay passed locally. **66 tests passed** across
R1l/R1m and new R1n experiment/analysis/evidence modules. This was a local
research checkout mirror, NOT the full native Brain repository checkout.

**Result:** clean three-change post-change accuracy for last-example
learner 95.83%, decayed-evidence γ=.5 89.44%, two-confirmation
quarantine 79.17%, all-history 8.89%; two-confirmation avoids the
isolated corruption penalty but drops 9.72 percentage points under
consecutive same-role corruptions. All learners *assume cyclic-shift*
structure. The exploratory dual-timescale policy did not beat existing
baselines. Do not claim native composition or acceleration.

**New automated gate:** `r1n-evidence` has been added to
`run_local_gates.py` for replay verification without GitHub Actions.

Reproduce:
```sh
python -m pytest -q research/multimodal
python -m research.multimodal.verify_unknown_drift --replay
python -m research.multimodal.run_local_gates \
  --gates protocol-tests r1n-evidence --output /tmp/assemblies-r1n
```

Next: run the full Git checkout native R1b–R1j tests; add a genuinely
held-out shift/change/corruption generator and process-isolated models
before a publication-level leakage claim; compare memory and inference
costs; only after native correctness adapt the same streaming regime to
real synaptic/refraction assembly circuits.
