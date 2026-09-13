# Engineering and science backlog

This is the deferred work list for the Assembly Calculus repository. Items are
ordered by semantic leverage. A task is complete only when its implementation,
source-linked contract, negative control, and relevant gate are all present.

## First: remove methodological ambiguity

- [ ] **Unify experiment entry points.** Migrate the remaining research scripts
  to `research.runner.run_experiment` and one CLI adapter. The adapter must own
  `--seeds`, `--tag`, `--engine`, smoke/study mode, immutable output paths,
  protocol version, resolved parameters, and run provenance. Refuse overwrite,
  fewer than three study seeds, duplicate/nonregistered seed identities, and
  ambiguous engine names. Preserve committed evidence with a tagged,
  numerical migration comparison.
- [ ] **Complete the evidence graph.** Require every results artifact to carry
  script, commit/source digest, engine, protocol version, seeds, tag, parameters,
  and semantic profile. Require every register evidence edge and every
  `PREREG_*.md` result link to resolve. Add orphan detection and make the graph
  gate fail on dangling or unreferenced maintained results.
- [ ] **Close all register sensitivity/provenance gaps.** Keep the engine field
  accurate for every measured entry, and migrate each entry from prose-only
  caveats to retained treatment/control vectors. Every adopted result needs a
  beta-zero or mechanism-disabled null whose measured number moves.
- [ ] **Finish operation semantic cards.** For projection, association,
  reciprocal projection, merge, completion, attention, binding, memory, FSM,
  transducer, and parser operations, diff executable state reads/mutations,
  schedule, learning rule, readout, and failure conditions against the
  docstring and register. Turn each discrepancy into a regression test or
  corrected claim. Keep a constructed true negative for every contract.
- [ ] **Make configuration the single source of truth.** Expand the validated
  immutable semantics envelope so connectome mode, candidate domain, stimulus
  law, tie rule, arithmetic, normalization, plasticity, schedule, and
  observation policy are never inferred from scattered flags. Serialize the
  resolved configuration in every run and reject partial profiles.

## Index spaces and runtime composition

- [ ] **Make compact and stable indices unspellable everywhere.** Replace
  remaining public `.w` uses with `active_count`, `recruited_count`, or an
  explicitly named materialized extent. Add typed wrappers at every public
  boundary and reject raw mixed-space arrays in runtime paths. Remove or isolate
  legacy consumers after migration, with a written disposition for each.
- [ ] **Unify backend conformance semantics.** Define one executable conformance
  matrix for projection, association, reciprocal projection, merge, completion,
  and attention across explicit NumPy, exact NumPy, sparse NumPy, Torch, CUDA,
  and hashed substrates. Record stimulus distribution, candidate domain,
  normalization, arithmetic, and tie behavior. Each case must include a broken
  configuration that it demonstrably rejects or fails.
- [ ] **Finish the learned assembly-attention operator.** Keep the pure snapshot
  readout separate from the Brain-backed operator. Specify typed query/key/value
  supports, multihead composition, causality, refinement, and separate
  compatibility/value readouts. Add no-compatibility, value-shuffle, and future
  token leakage controls before making sequence claims.
- [ ] **Consolidate duplicate state and parser paths.** After cards exist,
  merge repeated context resets, training schedules, synchronization logic, and
  parallel configuration adapters. Delete only after all consumers and evidence
  migrate; moving a duplicate does not count as reduction.

## Proof and cross-compilation

- [ ] **Connect Assembly IR to concrete execution.** For each Python, Rust,
  Torch/CUDA, and future Rust cross-compiler target, provide a state relation,
  step simulation theorem, readout compatibility theorem, and differential test.
  The current Lean proofs establish pure checked lowering only; they do not
  prove NumPy float32, clipping, Rust execution, or CUDA equivalence.
- [ ] **Keep schemas and proof sources linked.** Every new IR opcode must have a
  canonical schema, shared acceptance corpus, Python validator, Rust transport,
  Lean model/theorem, source-linked verification anchor, and a malformed true
  negative. Add source/proof identity checks so a proof cannot certify a stale
  schema.
- [ ] **Integrate cross-compilation target descriptions.** Define a language-
  neutral target/profile layer inspired by Rust target specifications. Keep
  compilation capabilities separate from scientific model semantics, and test
  round trips across Python, Rust, Lean, and generated target descriptions.
- [ ] **Evaluate Dafny/LemmaScript-style refinement workflow.** Document how
  intent, executable contracts, lemmas, and generated/runtime checks map onto
  Assembly IR. Prototype one operation end to end before expanding the toolchain.

## Scientific program

- [ ] **Temporal carry decay law.** Extend the registered position-specific
  chain-corpus instrument through gaps 3--6 with at least twenty hashed seeds;
  fit and report a preregistered decay model. Keep the corrected position
  instrument distinct from the void pooled 0.11/0.22 numbers.
- [ ] **Clip-window retention.** Test weight rules that preserve exact
  transition-organ behavior from 20 through 200 presentations, with both
  retention and new-learning bars. Preserve failed homeostasis attempts and
  identify the mechanism that relocates the arc.
- [ ] **Derive or qualify the square capacity law.** Relate the measured
  approximately `0.40 (n/k)^2` ceiling and drifting exponents to Willshaw and
  sparse-Hopfield assumptions; state which assumptions hold and which constant
  is empirical.
- [ ] **Add sequence baselines.** Run trigram and a small recurrent network at
  matched step counts on the chain corpus, paired with bigram and oracle, and
  report gaps 4--6 under the same instrument and seeds.
- [ ] **Re-run noise robustness as registered science.** Separate stimulus,
  readout, and recurrent noise; define noise distributions and nulls before
  running. Report per-seed intervals, engine, materialization, and failure mode.
- [ ] **Audit perfect scores.** Sweep every adopted metric enough to prove it can
  move. Treat sealed areas, dead fibers, frozen winners, synthetic fallbacks,
  and beta-zero invariance as fake-perfect failures.

## Performance and GPU operations

- [ ] **Run the fused CUDA and parity suites in the documented developer shell.**
  Add an environment diagnostic for `vcvars64`, `ninja`, `CUDA_HOME`, compiler,
  and device capability; serialize one GPU job at a time. Do not call a CPU
  smoke run a GPU science result.
- [ ] **Measure end-to-end throughput at scale.** Extend the README performance
  table with reproducible sizes, seeds, backend, materialization semantics,
  wall time, memory, and confidence intervals. Include plots/artifacts rather
  than isolated peak timings.
- [ ] **Reduce gate latency without weakening coverage.** Profile phase-level
  wall time, cache immutable construction/proof results where valid, replace
  pure-function composition tests with invariant/construction checks, and
  retain empirical tests for backend/stateful behavior. Re-measure matched green
  runs before changing worker defaults.
- [ ] **Make optional accelerators composable.** Route all dynamic Torch/CUDA/
  CuPy access through typed lazy boundaries, keep CPU imports lightweight, and
  add parity tests for every newly shared operator.

## Research tree and legacy disposition

Current inventory baseline (2026-09-12): `research.evidence audit` sees 2,821
tracked files, 3,306 resolved literal edges, 1,023 unresolved/ambiguous
references, 368 candidate orphan result files, and 19 preregistrations that are
explicitly pending results. These are inventory signals, not automatic deletion
decisions; triage them by role and provenance before tightening the gate.

- [ ] **Repair or explicitly retire the remaining static-debt clusters.** The
  latest inventory is 451 experiment files, 1,165 Pyright errors. Start with
  `gpu_writeback_gemm_prototype.py` (40), `primitives/run_all.py` (30),
  `primitives/test_unified_phenomena.py` (30), the two hashed parity scripts
  (26 each), `run_all_experiments.py` (24), and `surprise_gain_recall.py` (24).
  For each file write: retained role, consumer, contract/evidence link, owner,
  and migration or archival decision. Do not add broad directories to the
  maintained gate without cleaning their dependencies.
- [ ] **Review every tracked `cpp/` and legacy-tree file.** For each item,
  record stay/move/archive/delete, current consumers, build path, specification,
  and replacement. Remove only with a passing replacement and evidence audit.
- [ ] **Inventory orphaned results and registrations.** Reconcile the roughly
  1,500-file research tree against register entries, preregistrations, scripts,
  and run records. Mark historical/void artifacts explicitly; never silently
  delete failed bars or cite-only source material.
- [ ] **Keep CHILDES cite-only and protect working-tree process rules.** Preserve
  one worktree per session, explicit-path commits, tagged runs, no concurrent
  GPU jobs, and no engine edits while imported by a running process.

## Acceptance checklist for every future item

- [ ] A newcomer can discover the contract from the code definition.
- [ ] Invalid or semantically misleading use fails loudly at the nearest boundary.
- [ ] A true negative demonstrates that the test would fail on a broken setup.
- [ ] Static checks, focused tests, and the appropriate full gate pass.
- [ ] Scientific changes have a preregistration/amendment, per-seed values,
  interval, engine/protocol provenance, and retained failed bars.
- [ ] The change is committed by explicit path and pushed to `dev`.
