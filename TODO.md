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
  - Progress 2026-09-12: the adapter refuses all five (argparse `choices`
    for engines, `auto` rejected, `validate_registered_seeds`, `mkdir` as
    reservation). Two execution kinds added so the last unspellable
    scripts could migrate: `computed_baseline` (`BaselineSemantics`) and
    `reference_nemo_numpy` (declared `ModelSemantics`), run schema 9.
    Migrated: `seq_a3_oracle_ceiling.py` (`a3-oracle-ceiling`; per-seed
    bigram and oracle vectors equal the committed gap-2 file, receipt
    `research/results/comparisons/oracle-chain-gap2-20260912.json`),
    `seq_arc_refraction_reference.py` (`arc-refraction-reference`,
    twenty seeds registered in `PREREG_arc_refraction_reference.md`);
    `word-capacity` and `word-capacity-ladder` registered. Remaining:
    `seq_a3_transducer.py` (its five arms still parse `sys.argv`; the
    migration comparison needs a ~45 min GPU replay of seeds 62..81 at
    gap 2), and a written disposition for `seq_s5_arc_drift.py` and
    `seq_s5_arc_clip.py` (post hoc diagnostics).
  - Later the same day: `seq_a3_transducer.py` is a runner module
    (`a3-transducer --arm {induced,strength,successor,temporal,register}`,
    one frozen organ profile per configuration, the strength arm's
    reference cell a declared input artifact; the numpy study is kept as
    `legacy_numpy_study`, void under the sampler audit). The two arc
    diagnostics have their disposition in the experiments README. The
    temporal-arm replay on seeds 62..81 reproduces the committed file's
    160 per-seed values (receipt
    `research/results/comparisons/a3-temporal-replay-20260912.json`). What
    remains under this item is the migration boundary already written
    (two S5 forensic scripts, the wander diagnostic) and the numpy A3 study,
    void under the sampler audit.
- [ ] **Complete the evidence graph.** Require every results artifact to carry
  script, commit/source digest, engine, protocol version, seeds, tag, parameters,
  and semantic profile. Require every register evidence edge and every
  `PREREG_*.md` result link to resolve. Add orphan detection and make the graph
  gate fail on dangling or unreferenced maintained results.
  - Progress 2026-09-12: the gate already fails on a tracked runner result
    its registration does not link, on a dangling register evidence edge,
    and on an invalid comparison receipt; receipts of kind `baseline` now
    validate. Of the 19 "pending" registrations the audit lists, 15 carry a
    Result section with outcomes inline and no artifact link: the phrase
    heuristic mislabels closed notes. Disposition lines are being added to
    each note; the classifier should then report "inline results, no
    retained artifact" as its own category.
- [ ] **Close all register sensitivity/provenance gaps.** Keep the engine field
  accurate for every measured entry, and migrate each entry from prose-only
  caveats to retained treatment/control vectors. Every adopted result needs a
  beta-zero or mechanism-disabled null whose measured number moves.
  - Progress 2026-09-12: retained checks now on 11 of 15 MEASURED entries
    (added SEQ-EXACT-RECOVERY, AC-CAP, SEQ-ORGAN-EMBEDS,
    REFRACTION-NEEDS-LOAD, SEQ-REGIME-CLIFF, and, from the twenty-seed
    arc-reference study with all four registered bars passing,
    ARC-CONJUNCT-EXPOSURE and REFRACTION-PROPORTIONAL, and, from the
    registered tie census, KWTA-TIE-FRAGILE, and from the paired-anchor
    replay, CAP-ANCHOR-RATIO: 13 of 15; and on 2026-09-13 from the nk
    sensitivity replay, CAP-RATIO, whose three registered bars pass with the
    (8000, 120) ceiling noted as a cliff interpolation: 14 of 15; and
    REFRACTION-CANCELS-CONVERGENCE from the convergence study, 15 of 15).
    **Every MEASURED entry now holds a retained treatment/control check.**
    The convergence study took three rounds to settle: the twenty-brain run
    failed "converges and holds" (RC-2), "feedforward holds" (RC-4) and the
    transition bracket (RC-5); the full-curve instrument (Amendment 1) then
    failed three more bars on mis-set thresholds while showing the real
    structure; and Amendment 2 stated that structure as bars and confirmed
    all ten on a FRESH seed block. The result: a refracted recurrent area
    relocates on a fixed period equal to the clip arithmetic
    ln(w_max)/ln(1+beta) + (1-1/w_max)/beta = 40.93 rounds, measured 41.60
    and 41.52, first relocation on round 42 on all forty brains. Six failed
    bars are retained and still printed on every run. What remains under this
    item is the bias-masked capacity numbers of
    `PREREG_refraction_capacity.md` (logs, no artifact).
- [ ] **Finish operation semantic cards.** For projection, association,
  reciprocal projection, merge, completion, attention, binding, memory, FSM,
  transducer, and parser operations, diff executable state reads/mutations,
  schedule, learning rule, readout, and failure conditions against the
  docstring and register. Turn each discrepancy into a regression test or
  corrected claim. Keep a constructed true negative for every contract.
  - Progress 2026-09-12: R2, C3, F1, F2, T1/T2 have regression tests; D1
    is a claim-scope note (SEMANTIC_CARDS.md, "Card resolutions"). The
    parser card (E1..E11, training and parse entry points) is written from
    the executable bodies; its eleven discrepancies are resolved by
    `tests/test_parser_card_regressions.py` and corrected docstrings (see
    SEMANTIC_CARDS.md, "Parser card resolutions"). Finding: the beta-zero
    core->role control does not move the role readout's gap (0.978 against
    0.983), so on the test parser `parse` measures fixed image separation,
    not a learned binding; the language line's role-binding claims must
    answer this before any adoption. Cards that still lack a
    discrepancy-by-discrepancy resolution: attention (design target only),
    binding (cards exist at the plan level).
- [ ] **Make configuration the single source of truth.** Expand the validated
  immutable semantics envelope so connectome mode, candidate domain, stimulus
  law, tie rule, arithmetic, normalization, plasticity, schedule, and
  observation policy are never inferred from scattered flags. Serialize the
  resolved configuration in every run and reject partial profiles.
  - Progress 2026-09-12: partial profiles are rejected at construction
    (missing or unknown fields) for every record type. Observation policy
    is now a closed enum (`ObservationPolicy`: plastic, frozen, probe,
    read-only, none), required in the record of every run that reads a
    substrate (Brain engines, the vendored reference) and refused for
    organ, aligner and baseline runs; run schema 10; the completion plan
    spells its modes with the same enum; every migrated Brain-engine entry
    point declares its policy beside the context manager it uses.
    Remaining: an explicit schedule document for Brain-engine runs (today
    the schedule is the immutable operation plans plus `presentations` and
    `rounds` in the parameters, not a validated field), and a check that
    the declared policy matches the code path (today a per-experiment
    reading obligation).

## Index spaces and runtime composition

- [ ] **Make compact and stable indices unspellable everywhere.** Replace
  remaining public `.w` uses with `active_count`, `recruited_count`, or an
  explicitly named materialized extent. Add typed wrappers at every public
  boundary and reject raw mixed-space arrays in runtime paths. Remove or isolate
  legacy consumers after migration, with a written disposition for each.
  - Progress 2026-09-12: raw-w reads that meant recruitment are migrated in
    the simulation modules and eight experiment files; every kept site has
    a disposition in the ratchet baseline (writes mirroring the engine sync,
    a snapshot restore, engine-internal materialized counts, two deliberate
    demonstrations). Remaining: the field itself (`Area.w`, the engines'
    state objects, `Brain`'s sync sites), typed wrappers at the public
    boundaries, and rejection of raw mixed-space arrays in runtime paths.
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
  - DONE 2026-09-13: registered as Amendment 1 of
    `PREREG_temporal_positions.md` (geometric model on the per-offset carry,
    five bars per gap, the gap-2 replay as the instrument comparison); the
    runner takes `--gap 2..6`; all five twenty-seed runs are recorded
    (`temporal-positions-gap{2..6}-2026091{2,3}`). Result: the carry decays
    geometrically at a POOLED r = 0.709 [0.689, 0.729] whose four per-gap
    fits all contain it, while the amplitude falls with the chain's length
    (A 0.207 to 0.124) -- one shape, not one amplitude, and that is the
    whole pooled residual. The measured horizon is FIVE distractors: DL-3
    clears the 0.02 bar through gap 5 and fails at gap 6, where the carry is
    still positive on 20 of 20 brains. TP-3 fails from gap 3 on because its
    estimand is a mean over all positions; the position-specific DL-5 passes
    at every gap. Both failures are recorded, neither amended. The pooled
    numbers of the old harness stay void. Register: SEQ-TEMPORAL-CARRY
    carries the law, the horizon and a second retained check (the decay is
    per-brain at gap 6; the tail against blind is not).
- [ ] **Clip-window retention.** Test weight rules that preserve exact
  transition-organ behavior from 20 through 200 presentations, with both
  retention and new-learning bars. Preserve failed homeostasis attempts and
  identify the mechanism that relocates the arc.
  - Progress 2026-09-13: the mechanism that relocates is identified, confirmed
    and now measured as a LAW. A refracted recurrent area relocates on a fixed
    period equal to the clip arithmetic
    `ln(w_max)/ln(1+beta) + (1-1/w_max)/beta`, which holds across five cells
    (beta 0.05 to 0.20, w_max 5 to 100, twenty brains each) to 4.1% in the
    worst cell and 1.6% in four of five, with the measured w_max ratio 2.354
    against a predicted 2.34 (`PREREG_refraction_period_law.md`, PL-1 to PL-5
    all pass). The unrefracted control relocates ZERO times in every cell
    though its weights clip on the same schedule, so the clip alone does not
    move an assembly. **This is the deadline any retention rule has to move,
    and it says how:** for small beta the period is
    `(ln w_max + 1)/beta`, so tenure is bought with beta and not with the
    weight ceiling (a fourfold beta change moved it 3.6x, a twentyfold w_max
    change only 2.4x). Still untested: whether the period depends on the
    refraction strength, and any weight rule that preserves behaviour past the
    deadline.
- [ ] **Presentation schedule as a control variable.** At a FIXED round budget
  per item, the arrangement of those rounds moves assembly-memory recall as
  much as the plasticity rule does, and the two rules want opposite
  arrangements. Registered and measured in
  `research/notes/memory/PREREG_presentation_schedule.md`: an unrefracted
  memory needs ORDER (interleaving holds the hub statistic at chance and takes
  usable load from below 8 items to between 64 and 128; splitting alone does
  nothing), and a refracted memory needs SHORT EPISODES (one 16-round episode
  recalls 0.198 at M = 256 against 0.925 for four 4-round episodes; order adds
  0.075). Idle spacing is a no-op by construction because the substrate has no
  decay term, and the registration asserts that bit for bit rather than
  measuring it. Open: PS-7 failed, so the instrument does not reproduce the
  capacity line's published cell and every number is internal to its six arms;
  Amendments 1 and 2 closed the instrument question: a parity test shows the
  single-episode write is BIT-IDENTICAL to `AssemblyMemory.store`, and PS-7 and
  PS-9 failed on a misread premise (a published `M*` equal to the grid's first
  point is a sentinel for "already collapsed there", not a recall near the
  bar), so the scoping caveat is lifted and these numbers are comparable to the
  capacity line. What is NOT established, with two failed bars behind the
  withdrawal, is WHICH items a massed schedule loses: the write-order sign
  reverses between M = 8 and M = 32, so only the hub statistic carries the
  mechanism. Adding a decay constant would make genuine time-mediated spacing
  possible and would turn the refraction bias into a relative refractory
  period; that is a substrate change and needs its own registration.
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
  - Progress 2026-09-12: the toolchain diagnostic exists
    (`scripts/check_cuda_toolchain.py`, `scripts/cuda-dev.cmd`; device
    capability is not probed). The runner now takes a machine-wide exclusive
    lock for every device engine before reserving a tag and refuses a
    second device job outright (research/README.md#one-device-job; test
    `test_second_device_job_is_refused_before_reservation`). Studies run
    from a worktree pinned at a commit
    (`%TEMP%/assemblies-runs-20260912`) so edits here cannot void them.
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
  - Progress 2026-09-12: dispositions for the eight named files are in
    `docs/reviews/whole-codebase/STATIC_DEBT_DISPOSITIONS.md` (MIGRATE 1:
    `surprise_gain_recall.py`, whose registration lives in its docstring;
    RETAIN-AS-DIAGNOSTIC 4; ARCHIVE 3). The three ARCHIVE files are moved
    under `legacy/experiments/archived_2026_09/` with history: the two
    first-generation hashed parity scripts (ImportError since 637593ee,
    replaced by `test_hashed_substrate_parity.py`) and
    `primitives/run_all.py` (its registry entry point removed).
- [ ] **Review every tracked `cpp/` and legacy-tree file.** For each item,
  record stay/move/archive/delete, current consumers, build path, specification,
  and replacement. Remove only with a passing replacement and evidence audit.
  - Progress 2026-09-12: the inventory exists
    (`docs/reviews/whole-codebase/CPP_LEGACY_INVENTORY.md`: 293 files in 25
    clusters; STAY 38 files, ARCHIVE 115, DELETE-CANDIDATE 140, every
    consumer cited). Decisions pending: the universal-brain-simulator stack
    under cpp/ (~140 files, no consumers, broken includes, superseded by the
    torch engine), `legacy/artifacts/` (114 MB of GIFs, no consumers), and
    `cpp/core_cpp/`, whose maintained wrappers import a module no tracked
    build produces.
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
