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
  - Progress 2026-09-13: the REFERENCE construction is now ported and proven
    against the clone itself. `assembly_calculus/attention_area.py` is
    mdabagia/nemo's `AttentionArea`, and
    `test_attention_area_parity.py` loads `.reference/mdabagia-nemo/brain.py`
    and checks the port bit for bit: the set rule, the undo, and the recurrent
    read all match exactly on a shared connectome.
    It carries two properties nothing else here has. Potentiation ASSIGNS
    `1 + plasticity` to positive weights rather than scaling them, so it
    SATURATES and needs no clip -- weights are binary. And `decay_weights`
    REVERSES the last update exactly, so a link can be bound, read and
    released, which is what attention needs and what permanent compounding
    plasticity cannot express.
    ONE DELIBERATE DIVERGENCE, recorded rather than hidden: `change` is
    assigned and not accumulated, so a second bind without a release makes the
    undo over-subtract and leave `1 - plasticity**2` where `1` was. The
    reference permits that; this port REFUSES it, and a test demonstrates the
    reference really does corrupt its baseline there, so the guard is justified
    rather than arbitrary.
    The pure snapshot readout in `attention.py` is untouched and stays what its
    docstring says it is. What remains under this item is the measurement --
    the port is infrastructure and has no registration and no register entry,
    so nothing here claims attention DOES anything yet. The scientific question
    it opens is what TRANSIENT binding buys that permanent binding does not.
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
    all pass). Amendment 1 then DISCRETISED it, `ceil(ln(w_max)/ln(1+beta)) +
    (1-1/w_max)/beta`, since a weight needing 31.43 rounds of growth clips on
    round 32; on three further cells chosen before the run and sharing no
    coordinate with the first five that cuts the mean error from 4.05% to
    0.78% and is closer in 3 of 3 (PD-1, PD-2 pass). Neither form lands inside
    a 95% interval: at 0.02 to 0.12 rounds wide the measurement is sharper
    than either approximation, which is recorded rather than smoothed over.
    Amendment 2 then found the law's SCOPE: sweeping the refraction strength
    at one operating point gives 59.49, 47.15, 41.60 and 40.78 rounds at
    s/beta 0.25, 0.375, 0.5 and 0.625, a 25.9% spread, so the period depends
    strongly on strength and the formula's lack of a strength term is a
    condition rather than a generality. Both candidate closed forms are
    refuted (ST-1, ST-2, ST-3 fail) and the dependence stands as four measured
    points with none adopted. The law is exact where the repository works,
    s = 0.5 beta, which is the adopted operating strength. A dissociation
    worth chasing: strength is a capacity SWITCH flat over 0.3-0.6 beta, yet
    tenure moves 13% between 0.375 and 0.5 beta inside that plateau, so a
    strength chosen for capacity is not thereby chosen for tenure.
    Amendment 3 then used strength as the tenure knob to test whether the
    schedule study's episode penalty IS the tenure, with two strengths inside
    the capacity plateau so the contrast was controlled (PL-9 passes, the
    short arms differ by 0.0078). It is NOT: the penalty grows as the episode
    becomes a SMALLER share of a tenure (+0.8016, +0.8328, +0.8641 at 38.5%,
    33.9%, 26.9%), the reverse of the prediction, with non-overlapping
    intervals at the extremes. The tenure reading is withdrawn from
    `PREREG_presentation_schedule.md`. The effect is a THRESHOLD between 8 and
    4 rounds per episode (0.185, 0.180, 0.856, 0.987 across 1x16, 2x8, 4x4,
    8x2), not a gradient. Best remaining candidate, post hoc: each episode's
    first round is driven by the stimulus alone, so the episode COUNT sets the
    anchor-to-recurrence ratio that [[CAP-ANCHOR-RATIO]] already identifies as
    setting capacity at formation. Testing it needs episodes held fixed while
    total rounds vary. The unrefracted control relocates ZERO times in every cell
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
- [ ] **Reproduce the sequences paper's ordered recall, which currently does
  NOT work here.** Registered and pinned 2026-09-13 in
  `research/notes/sequence/PREREG_ordered_recall_reproduction.md`. Measured
  ZERO steps after the cue across 64 knob combinations at length 8, and across
  lengths 3/8/16 x repetitions 3/10 x sampled/materialized. The maintained
  parity test appeared to show it working because `ordered_recall` snapshots
  after firing the cue, so its two assertions both describe the CUE and hold
  with zero advancement; and it runs on the sampled connectome, where cue
  retrieval reads 0.92 against 0.30 materialized. Six bars registered (OR-1 to
  OR-6) including the mechanism-disabled null the claim demands. Order of
  work: compare against the reference implementation first to separate a bad
  port from a different regime, then fix, then the sequence-length-limit
  study. That study is the prize: the paper reports a 20-40 assembly limit
  that "varies with the parameters", and `c* = ln(w_max max(1,kp)/base) /
  ln(1+beta)` gives 26.5 / 33.8 / 41.0 at w_max 5 / 10 / 20, so the limit may
  be the weight clip we already have in closed form. It needs recall that
  advances, so it waits.
- [x] **Autonomous chain recall, built on the mechanism that works.** DONE
  2026-09-13, `research/notes/sequence/PREREG_autonomous_chain.md`, all five
  bars pass on twenty fresh brains. Driving the refracted arc with a SINGLE
  CONSTANT symbol makes the chain autonomous: the tick carries nothing, so
  every advance comes from the state through the arc. Recall is exact at
  length 32 and at length 128 on 20 of 20 brains, four times the top of the
  20-to-40 band the papers report. Refraction is the entire mechanism and the
  null is total: with strength 0 not one brain takes a single correct step.
  The reason is measured rather than inferred -- the arc assemblies of distinct
  states are IDENTICAL without refraction (overlap 1.0000) and disjoint with it
  (0.0000, chance 0.0100), because the tick appears in all 32 transitions and
  each state in one, which is [[ARC-CONJUNCT-EXPOSURE]] at its extreme. That
  contrast is now the entry's second retained check. The perfect score moves
  four ways (undertraining at 5 and 10 presentations, arc density 0.05 and
  0.02) and the failure modes are distinguishable: refraction off collapses the
  ARCS, undertraining and sparsity leave them disjoint.
- [ ] **Where the sequence-length limit actually is: state FORMATION.** The
  study above localizes it away from the arc and the sequence mechanism. What
  remains is the one difference from the papers' setting: our states are
  teacher-forced onto disjoint blocks, theirs are formed by projection where
  they can overlap and interfere. Replacing the teacher-forced blocks with
  projection-formed states and finding where the chain breaks is the next
  study, and it is the one that would say whether the reported 20-to-40 limit
  is state collision. Until it runs, nothing here refutes or explains that
  limit and no such claim is made.
  - Progress 2026-09-13: the substrate can now express it. `HashedArcFSM`
    takes an explicit `state_code` and decodes by MEMBERSHIP rather than
    integer division, so a code whose states share neurons is representable at
    all; a supplied code no longer widens the area underneath itself.
    `test_hashed_fsm_state_code.py` pins on the device that the two readouts
    are the same function on a disjoint code -- identical cued winners,
    identical decoded states, identical run through a trained chain -- which is
    what lets the collidable arm and the disjoint arm be compared on one
    instrument. Both arms pass an explicit code so both go through membership.
    `--states` runs the six cells; SC-1..SC-6 evaluate automatically. Two
    pre-data corrections recorded in the registration: the crowding split was
    off by one ((L+1)k = 16100, so THREE of five arms force overlap, not two),
    and SC-5 inherits AL-1's position-locked defect so SC-6 asks the same
    question with the wrap/stall/scatter classifier. Not yet run.
  - RESULT 2026-09-13 (`chain-states-20260913`): **the chain tolerates state
    collision.** All six arms 20/20 exact at L = 160, n_arc = 3000, across a
    sixteenfold load range (0.25 to 4.03 states per neuron; at the tightest
    area a disjoint code is impossible). SC-3 FAILED, which the registration
    named in advance as the stronger outcome: the disjoint teacher-forced code
    is NOT what carries this repository's sequence results. Adopted as
    `SEQ-STATE-COLLISION-TOLERATED`, whose sensitivity check proves the
    treatment reached the organ (arc overlap moves on every brain) while the
    outcome did not move at all. THREE things this does not settle, all
    recorded in the registration: collision at a MARGINAL cell (this ran where
    there is margin; n3000-L384 at 5/20 is the test that matters),
    CORRELATED collision (random k-subsets are uncorrelated; projection forms
    assemblies whose overlap tracks input similarity, and the papers' states
    are projection-formed), and higher load than 4.03. The item stays open on
    those three.
  - REVISED 2026-09-13 (`chain-margin-20260913`, Amendment 4): the null above
    was a CEILING EFFECT. At the marginal cell (n_arc = 2000, L = 256, 14/20
    exact) the same sweep is devastating -- mean consecutive-correct falls
    251.8, 220.8, 180.8, 142.0, 110.0 as load runs 0.40 to 6.42, losing 142 of
    256 steps. So the chain tolerates state collision only where it has margin.
    `SEQ-STATE-COLLISION-TOLERATED` is rescoped accordingly; the id reads too
    strongly on its own and says so in its caveat.
    Three of the five MC bars cannot carry that conclusion and this is recorded
    rather than glossed: all five read `exact/20`, every random arm is 0/20
    roomy and crowded alike, so MC-3 passed VACUOUSLY (0 <= 0) and MC-2 passed
    by comparing the tightest random arm against BLOCKS, mixing crowding with
    the separate cost of randomness (3.7 steps of 256, which exact/20 magnifies
    into 14 -> 0). The dose-response lives in `mean correct`, which no bar
    reads. Third time `exact@L` tie-fragility has bitten this registration.
    MC-4 FAILED: arc overlap moves 0.0868 across the random arms against a 0.05
    bar, so the cost is NOT attributed to state collision specifically.
    STILL CONFOUNDED: Amendments 2 and 4 differ in margin AND in load (4.03 vs
    6.42). The cheap run that separates them -- sweep the roomy cell to
    n_state 2500 and 2000, load 6.44 and 8.05 -- is not yet run.
  - RESOLVED 2026-09-13 (`chain-load-20260913`, Amendment 5): **it is the
    MARGIN, not the load.** At matched load 6.4 the roomy cell sits at 1.000 of
    L and the marginal cell at 0.430; the roomy cell then holds at 1.000 all
    the way to load 13.42 -- more than twice the killing load, 161 states in
    1200 neurons at pairwise code overlap 0.083. State collision costs nothing
    until the chain is already marginal and then costs enormously: an AMPLIFIER
    of an existing limit, not a limit of its own.
    LM-4 FAILED (the roomy cell has no breaking point in the sweep) and the
    registration's pre-declared reading of LM-2-pass-with-LM-4-fail
    ("uninformative") is WITHDRAWN with the reasoning written out: LM-4 guarded
    against a treatment that never reaches the organ, and this treatment
    demonstrably did (state overlap tripled, arc overlap moved monotonically,
    one brain broke). The decisive evidence is the load-matched comparison,
    which does not rest on LM-4.
    OPEN: what "margin" IS. The two cells differ in chain length and arc size
    jointly and nothing separates them. Named hypothesis, not run: the ARC is
    the bottleneck, so state crowding is costly exactly when the arc has no
    room -- consistent with MC-4 failing and LM-5 confirming that crowding the
    STATES moves the ARC in every run.
  - ANSWERED 2026-09-13 (`chain-arcb-20260913`, Amendment 6): **the ARC is the
    bottleneck.** Holding the chain and the state crowding fixed at the
    combination that collapsed to 0.430 of L and giving the arc 1.5x the
    neurons restores 0.997; 2x restores every brain; the disjoint control is at
    0.998-1.000 at both arc sizes, so the rescue is specific to the crowded
    code. State collision is a way of SPENDING ARC CAPACITY, free until the arc
    has none left -- which also accounts for the effect flagged unexplained in
    three runs, that crowding the STATES moves the ARC overlap.
  - MECHANISM 2026-09-13 (`chain-order-20260913`, Amendment 7): the
    end-of-chain break follows the TRAINING position, not the chain position.
    Chain order breaks at 0.996 of L; REVERSING the sweep breaks at 0.010 of L.
    Arc overlap is 0.1295/0.1315/0.1297 across arms, so no arm is more crowded.
    TO-3 FAILED in the OPPOSITE direction to its prediction: shuffling is far
    worse (0.403 of L against 0.995), and both fixed orders are TIGHT while
    shuffled spreads 15 to 512. Hypothesis, not yet measured: a fixed order
    trains each transition against the same accumulated bias every sweep so its
    assembly consolidates, while shuffling gives it a different landscape each
    time and it never settles -- CONSISTENCY of the training context mattering
    more than its average quality. The direct test is arc-winner overlap
    between successive presentations of one transition, per order. Not run.
  - Amendment 8 registered and wired: the arc-capacity curve, 4 arc sizes x 8
    lengths, graded and interpolated (L* = where mean correct crosses 0.95 of
    L), n_state pinned at 51300, CHAIN order set from Amendment 7's result
    rather than the plan's assumption of shuffled. Not yet run.
  - RESULT 2026-09-13 (`chain-capacity-20260913`, Amendment 8): **all five AC
    bars FAIL, and there is not yet a capacity law to cite.** AC-5 fails as
    anticipated -- at n_arc 3000 and 4000 the curve never crosses 0.95 inside
    the grid, so L* is CENSORED and AC-2/AC-3/AC-4 read an undefined quantity.
    Without AC-5 they would have compared grid EDGES and reported a scaling
    law, which is exactly the trap `exact_length` fell into in Amendment 1.
    The law cannot be fitted above n/k = 20 here: topk_select packs a 16-bit
    index, so n_state <= 65536 bounds the chain at 639 at k = 100, and reaching
    the crossing for larger arcs needs a smaller k -- a different study.
    AC-1's failure matters more: **the capacity surface is NOT smooth**, and
    not only in exactness but in the GRADED measure. arc1000 runs 0.995 /
    0.677 / 1.000 at L = 192/256/320; arc4000-L64 is 0.711 with EVERY brain
    failing tightly at 0.59-0.80 of L (19/20 wrap) while arc3000-L64 is exact
    on all twenty. That break sits mid-chain, so it is a different mode from
    Amendment 7's end-of-chain artifact.
    An underload reading (refraction needs load) was proposed and WITHDRAWN:
    failures do not track relative arc overlap (2.24x works, 2.48x fails, 2.54x
    works, 2.79x fails).
    NEXT, and it comes before fitting anything: re-run the hole cells on the
    fresh block 82..101. If arc4000-L64 and arc1000-L256 reproduce as the n3000
    hole did across two blocks, the surface genuinely has holes; if they move,
    they are draws. Fitting L*(n_arc) through a surface with holes would be
    fitting a curve through something that is not a function.
  - ANSWERED and ADOPTED 2026-09-13 as `SEQ-INTEGER-ARC-LOAD`
    (`chain-iload-20260913`): **recall peaks when the arc load DIVIDES
    EVENLY.** Arc stability is a function of the FRACTIONAL PART of L*k/n_arc
    alone -- a V minimised at the half (0.748 at frac 0; 0.420, 0.303, 0.245,
    0.216, 0.262, 0.352, 0.584) -- and at integer load every brain recalls
    every step while frac 0.375 and 0.625 fall to 0.66 and 0.53. Confirmed out
    of sample at a THIRD arc size on seeds the diagnostics never used, with the
    load axis sampled inside Nyquist: two complete periods repeat cell for
    cell. IL-1/2/4/5 pass; IL-3 FAILS and is kept, its premise (that the half
    is the worst case) being false -- the quarter-ish fractions are.
    BOTH earlier "periods" (40 in L at n_arc=1000, ~16 at 1500) were ALIASES of
    one true period of 1.0 in LOAD, measured and reported before I noticed.
    SAMPLE THE LOAD AXIS, NOT L.
  - SETTLED 2026-09-13 (`chain-bphase-20260913`, Amendment 10): **PHASE LOCKING
    dominates; the mechanism is an INTERACTION.** chain+integer 1.000,
    chain+half 0.991, shuffled+integer 0.030, shuffled+half 0.033 -- an integer
    load buys NOTHING under a shuffled schedule. BP-2 fails narrowly on the
    statistic it read and not to zero: the stability gap survives shuffling at
    0.078 against the 0.10 bar, against 0.517 under chain order. About 85
    percent schedule, 15 percent load. `SEQ-INTEGER-ARC-LOAD` was CORRECTED,
    its claim having stated a balance mechanism this refutes for the outcome,
    and the broader reading -- that integer load is a property of the PROBLEM,
    predicting the same arithmetic in the memory line -- is WITHDRAWN. Because
    it needs a rigid repeated order it is substantially a statement about how
    these experiments PRESENT data, so the biological reading is weak.
  - EXPLAINED 2026-09-13 by the lag diagnostic: **the arc ROTATES** by about
    frac(L*k/n_arc) per presentation, and overlap at lag l peaks when l*frac is
    near a whole turn (r = -0.816 over 35 points). At frac 0.5 that is
    ALTERNATION -- odd lags 0.216-0.252, even lags 0.525-0.694 -- so the cell
    with the LOWEST consecutive stability holds TWO well-formed assemblies and
    recalls 0.991. Consecutive-overlap "stability" was measuring alternation
    and calling it instability. Two ways to succeed: a slow rotation whose
    neighbours overlap (frac 0.125), or a fast one that CLOSES at a short lag
    (frac 0.5). Even at integer load there is no fixed point -- it drifts to a
    plateau near 0.58 against chance 0.0625.
    A threshold defect came first and is recorded: the preceding probe's
    same-assembly cutoff of 0.8 sat ABOVE the best cell's consecutive overlap
    of 0.748, so its distinct-assembly count saturated at 19.8 and nearly
    produced a false refutation. Fourth instance in this registration of a
    statistic set by my own choice rather than by the system, after
    `exact_length`, the tautological AL-4, and the two-point median.
    NOT RUN, and the sharpest prediction the rotation makes: more presentations
    should cure fractional load in proportion to the ORBIT PERIOD -- twice the
    rounds at frac 0.5, about eight times at frac 0.375 -- while integer load
    is unaffected. That differential is hard to obtain by accident.
  - Amendment 3 ran on the fresh block 82..101 (`chain-limit-am3-20260913`):
    AL-6 CONFIRMS (median first error at 0.99-1.00 of L in every edge cell),
    AL-8 and AL-9 pass on the corrected instrument, AL-7 FAILED on a two-point
    median in one cell (its "every landing earlier than expected" clause held
    everywhere; no forward jump exists in either block). The limit replicates
    cell for cell across independent seed blocks, INCLUDING the n3000
    non-monotonicity (5/20 at L = 384, 18/20 at L = 512, both blocks), so that
    hole is structural rather than seed noise and now needs an explanation.
  - Amendment 1 ran (`chain-limit-v3-20260913`) and localizes the limit, but
    its instrument was wrong in five ways, all recorded in the registration:
    the chance denominator read a module constant (reported 20.9x-8.2x chance
    when the truth is 2.1x-3.3x, reversing the trend), AL-4 was a tautology
    whose control was in fact violated (the state area was L_max k, one block
    short, so the L = 512 column ran wider than every other), and
    `total_correct` is position-locked so AL-1 could not tell a wrap from a
    death and fired on three coincidences. AL-3 FAILED informatively: a
    fourfold arc ratio buys 2.4x the exact length, sublinear rather than the
    registered 4-16 band. The retained traces show the chain runs to within a
    handful of steps of the END and then jumps BACKWARD to an early state,
    often continuing correctly from there; no mechanism is claimed for that.
    Amendment 3 states it as bars AL-6..AL-9, to be confirmed on the fresh
    seed block 82..101.
- [ ] **Wake the inhibition primitives the papers depend on.** All three roles
  the two papers use are implemented here and unexercised: inter-area mutual
  inhibition is provably dormant (a strict xfail records 1361 training
  projections and 12 parse projections reaching it with never two members
  active, role exclusivity being carried by a Python set instead); transient
  inhibit-and-release, which is `mitropolsky2025simulated`'s generation
  trigger, exists as area gating with unit tests but no experiment, no
  registration and no register entry; and long range inhibition exists only on
  the Brain path, not on the hashed substrate that carries the fast sequence
  work. No preregistration in this repository has inhibition as its subject.
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
