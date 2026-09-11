# Semantic cards: operations and organs

Code baseline: 15a7ed9, parent 3334876. These cards were derived from executable
bodies and their called helpers, then compared with prose and the register.
The original cards describe baseline behavior, not desired behavior or independently
reproduced science. Resolution sections below record subsequent changes. Source
docstrings link stable contract IDs; `python -m research.evidence specifications`
checks their destinations without importing CUDA modules. CUDA arithmetic is followed through the Python dispatch; kernel-level
equivalence still requires Claude's hardware gates.

Each discrepancy has an ID. A future contract must resolve the discrepancy with
a regression test or corrected claim; renaming the function is not resolution.
The uncommitted protocol-class prototype was set aside until these cards existed.

The attention card below separates the implemented pure snapshot readout from
the still-unimplemented learned Brain-backed operation. It exists to keep the
transformer analogue distinct from the current projection and sequence APIs.

Register crosswalk: the five legacy operation functions do not each have an
adopted Result entry certifying their whole postcondition. Their mathematical
prose and parity goldens therefore cannot substitute for such a contract.
The memory card relates to REFRACTION-ANTI-MERGING and
REFRACTION-CANCELS-CONVERGENCE; the FSM card to SEQ-EXACT-RECOVERY and
SEQ-REGIME-CLIFF; the transducer card to SEQ-TEMPORAL-CARRY. Those entries'
preconditions and provenance caveats remain part of any claim made from a run.

<a id="contract-assembly-attention"></a>

## D: typed assembly attention (snapshot readout implemented; learned path is a design target)

`neural_assemblies.assembly_calculus.attention.attend` is a pure readout over
immutable snapshots. It computes overlap compatibility, stable softmax
weights, deterministic top-k key selection, and a bounded weighted value
assembly. It does not mutate a `Brain` or claim to learn query-key fibers.

The learned Brain-backed operator remains a design target. It must make the
following state and choices explicit before code is accepted:

- **Inputs:** a query assembly, a key population or key assemblies, a value
  population or value assemblies, a target area, head count, rounds, and a
  causality policy.
- **Compatibility:** a named query-to-key projection or learned fiber. Its
  drive, normalization, arithmetic, and tie rule are model semantics, not
  performance flags.
- **Selection:** sparse k-WTA over the compatible candidates. The selected
  key/value support and its index space must be observable; compact indices
  cannot be passed as neuron IDs.
- **Value transfer:** a separate selected-key-to-value projection followed by
  a target merge. Query-key compatibility and value readout are separate
  measurements and require separate null controls.
- **Refinement:** optional recurrent target rounds. A refinement round must
  not silently change the query or key set unless the schedule says so.
- **Causality:** a causal decoder may read encoder state and prior decoder
  state only. Bidirectional encoder state is allowed during encoding; future
  target tokens are forbidden during next-token inference.
- **Composition:** multihead attention is a product of independent typed
  heads followed by an explicit merge. Heads may not share mutable state by
  accident.

The first implementation contract should include a matched no-compatibility
control, a value-shuffle control, and a future-token leakage control. Agreement
between backends alone is insufficient: each control must make the claimed
readout move or fail loudly. The operator should reuse `ProjectionStep`,
`MergePlan`, `Measured`, and the execution-semantics envelope rather than
introducing a parallel protocol vocabulary.

<a id="contract-projection"></a>

## P: projection

Code: `assembly_calculus/ops.py:project`, `core/brain.py:project_rounds`.

- **Reads:** named stimulus, current target winners, afferent weights, backend,
  brain normalization/fast-path recurrence flags and plasticity state.
- **Schedule:** always one stimulus-only projection, even when rounds <= 0.
  With recurrent=True, the remaining rounds explicitly include target recurrence.
  Otherwise the tail delegates to Brain.project_rounds, whose recurrence is
  configuration-dependent. A protocol argument and a runtime optimization flag
  both participate in choosing the experiment.
- **Mutates:** target winners; enabled stimulus and participating recurrent
  weights; backend recruitment/history and homeostatic state.
- **Learning:** backend-defined Hebbian update, clipping/normalization/refraction.
  The operation does not establish equivalence of those rules across engines.
- **Readout:** final neuron-ID snapshot, with no stability or recovery test.
- **Claim/diff P1:** the opening docstring promises a stable assembly; no
  stability criterion is evaluated. The default can train no recurrent synapses.
  Fix the claim now; explicit training/activation contracts are the migration.
- **P2:** zero rounds still executes a step. Acceptance: invalid schedules fail
  before any state mutation. Keep a regression exposing this until corrected.
- **Control:** same seeded model and cue protocol with learning disabled; additionally
  compare stimulus-only and recurrent training. Matching final winners alone is
  insufficient to demonstrate learned recurrence.

<a id="contract-reciprocal-projection"></a>

## R: reciprocal projection

Code: `ops.py:reciprocal_project`; fixed-target paths in numpy engines.

- **Reads:** live source winners and source/target afferents.
- **Schedule:** optionally fix source; source -> target once; then source -> target,
  target -> target and target -> source for rounds-1.
- **Mutates:** target state and enabled fibers. Return unfixes source in finally.
  It does not restore the source's preexisting fixed flag.
- **Learning:** learning into the fixed source depends on the fixed-target
  plasticity policy. A scheduled back edge does not establish that it learned.
- **Readout:** target snapshot only; source restoration is not measured here.
- **R1:** pre-fixed sources become unfixed. Acceptance: preserve incoming clamp
  state on success and exception. Test this explicitly.
- **R2:** operation name cannot stand in for a bidirectional learning claim.
  The caller must record fixed-source learning policy and probe both directions
  after removing their direct cues. The PNAS reciprocal golden is a separate
  experiment and must not be treated as this function's postcondition.
- **Control:** disable the reverse learning fiber while leaving forward training
  and forward recall live; reverse recovery must distinguish that intervention.

<a id="contract-association"></a>

## A: association

Code: `ops.py:associate` and `_associate_body`.

- **Reads:** two parents, shared target, optional parent stimuli and live weights.
- **Schedule:** A alone for max(1,rounds) steps, B alone for the same count,
  then co-firing for rounds or explicit cofire_rounds. Each individual phase
  introduces target recurrence after its first step. Co-firing includes it
  immediately. With no stimuli both parents are fixed; supplying either stimulus
  switches both parents to the evolving-source branch.
- **Mutates:** target and, in evolving-source mode, parent winners/weights;
  enabled participating fibers; clamp flags are unconditionally cleared on exit
  in fixed-source mode.
- **Readout:** only the final co-fired target snapshot. Neither separately evoked
  representation nor an association delta is returned or measured.
- **A1:** “association” is a scheduled intervention, not an observed increase in
  similarity. The experiment must separately evoke A and B before/after and
  specify whether target state is reset and whether probing learns.
- **A2:** one missing parent stimulus changes both parents' dynamics. Acceptance:
  a contract must reject partial specification or name this mixed mode.
- **Control:** cofire_rounds=0 retains single-parent training while removing the
  association phase. Confirm the readout distinguishes it before claiming success.

<a id="contract-merge"></a>

## M: merge

Code: `ops.py:merge`.

- **Reads:** current parent winners, optional stimuli, target and fiber states.
- **Schedule:** both parents drive target; parent_self controls each parent's
  recurrence. From step two, target_self and back_project independently add
  target recurrence and reverse edges. No-stimulus mode fixes both parents.
- **Mutates:** target, potentially parents, participating weights and clamp flags.
  With fixed parents, reverse learning remains policy-dependent as in R.
- **Readout:** final target snapshot; no test of conjunction specificity, parent
  recovery or retrieval against competitors.
- **M1:** simultaneous projection, learned conjunction and bidirectional merge
  are distinct claims even though one function name covers their schedules.
  Record all three edge switches plus source fixation.
- **M2:** shares the preexisting-clamp restoration defect with R/A.
- **Control:** disable back_project for parent-recovery claims; remove one parent
  for conjunction claims. A single beta-zero null does not answer both.

<a id="contract-separation"></a>

## D: separation measurement

Code: `ops.py:separate`, `contracts.py:SeparationPlan`.

- **Reads:** two registered stimulus fibers and one target area with its current
  recurrent state.
- **Schedule:** project stimulus A to the target, reset only the target's
  recurrent connections, then project stimulus B with the same round budget.
- **Mutates:** target winners and recurrent weights; the reset is deliberately
  destructive and must not be interpreted as a biological learning rule.
- **Readout:** immutable neuron-ID snapshots for A and B plus normalized overlap.
- **D1:** this isolates order-dependent attractor carryover; it does not measure
  capacity, long-term retention, or robustness to noise.
- **Control:** identical stimuli, unknown topology, and invalid rounds fail
  before the first projection; a matched beta-zero arm is required for any
  learning claim.

<a id="contract-completion"></a>

## C: completion

Code: `ops.py:pattern_complete`, `Brain.read_only`, `Assembly.overlap`.

- **Reads:** the current winners as the reference, not a separately supplied
  stored representation. Subsamples compact indices with random.Random(seed).
- **Schedule:** inject floor(k*fraction) winners once, then recurrent projection.
  Winners are free to change; the cue is not clamped on subsequent steps.
- **Mutates:** live winners and, unless the caller disables it, synaptic and
  homeostatic state. read_only additionally restores winners and suppresses
  recruitment on supported sparse engines; that changes the candidate population.
- **Readout:** overlap against the snapshot taken at entry.
- **C1:** an attractor/recovery claim is not guaranteed by this schedule. The
  docstring's generic 0.8 expectation has no regime or success check attached.
- **C2:** repeated probes can train the object being measured; changing current
  winners changes the reference. Contract needs explicit reference, cue policy,
  measurement learning policy and supported substrate.
- **C3:** the manually clamped cue in existing conformance tests is a different
  protocol. Its retained-cue overlap floor does not apply here.
- **Control:** same reference/cue construction on a matched learning-disabled
  brain; test that a frozen target cannot masquerade as successful free recall.

<a id="contract-memory"></a>

## H: refracted associative memory

Code: `core/torch_engine/_memory.py:AssemblyMemory`, `recurrent_fiber`;
`_hashed.py:HashedArea.project`, DenseOrganFiber/AreaFiber/StimulusFiber.

- **State:** winners, ever-fired mask, accumulated refraction bias, recurrent
  potentiation counts/store, stimulus potentiation during each write, item count.
- **Write:** create a stimulus fiber, inhibit current winners (not bias/weights),
  run rounds of stimulus plus recurrence. Optional convergence gate stops each
  brain at its first repeated winner set.
- **Learning:** counts price a clipped multiplicative update on present edges.
  DenseOrganFiber is chosen with a clip and no column scaling; the alternative
  AreaFiber stores episodes and may rescale columns. These are distinct numerical
  implementations whose equivalence is gated, not assumed.
- **Read:** replace winners with a supplied cue; freeze learning and bias charging
  for self.rounds recurrent steps. Default masked readout removes bias for
  refracted memory. Winners remain advanced after recall; this is not a fully
  transactional host read.
- **Claim/diff H1:** REFRACTION-ANTI-MERGING describes a specific masked half-cue
  capacity protocol, not every constructor configuration or net readout.
  in_regime reports one inequality; it does not validate all adopted conditions.
- **H2:** capacity means a thresholded, gated curve over stored items, not a
  property returned by store(). Grid censoring and fill censoring differ; version
  2 now records separate flags. A 64-bit distinctness hash can collide.
- **Control:** strength=0 with matched cue and readout, plus masked/net readout
  contrast. Preserve the distinction between representation and accessibility.

<a id="contract-transition-machine"></a>

## F: assigned-state transition machine

Code: `_hashed_fsm.py:HashedArcFSM`, `_arc_core.py:HashedArcCore`,
`_hashed_transducer.py:StackedStimuli`.

- **State:** assigned disjoint state blocks; current STATE and ARC winners;
  symbol drive; state->arc, arc->state and symbol potentiation; arc bias.
- **Write:** inhibit arc; cue the from-state block; symbol plus state drive one
  plastic arc step; directly cue the destination block and observe arc->state
  coactivity. The destination is teacher-forced, not inferred during training.
  train() traverses the transition dictionary in insertion order per presentation.
- **Run:** set symbol, conjoin, advance state; freeze=True stops learning and bias
  charging but state/arc winners still advance. The transition table is not read
  to choose the emitted state at step time.
- **Readout:** largest overlap with assigned state blocks. Ties choose the first
  label; a label can be returned with no valid state-block evidence.
- **Claim/diff F1:** correct argmax label and all-k exact state recovery differ.
  A1 records both; the register's exactness claims must use the latter.
- **F2:** run() masks negative-symbol outputs after calling step(); padding must
  not be assumed to skip state transitions without tracing StackedStimuli.
- **Control:** disable arc->state learning while preserving the symbol/state
  conjunction; compare label accuracy and exact_fraction against trained controls.

<a id="contract-sequence-memory"></a>

## S: ordered sequence memory

Code: `assembly_calculus/ops.py:sequence_memorize` and
`assembly_calculus/ops.py:ordered_recall`; immutable schedules live in
`assembly_calculus/contracts.py:SequenceMemorizePlan` and
`OrderedRecallPlan`.

- **State:** an ordered tuple of stimulus names, one target area, recurrent and
  transition fibers, and (for recall) refractory history.
- **Write:** each stimulus is projected for the declared rounds and repetitions;
  the optional Phase-B ratio and temporary beta boost are part of the schedule.
  The plan validates every input and topology before the first projection.
- **Read:** recall clears refractory state, activates the cue, then self-projects
  for the declared step budget. It stops on a cycle, a known-assembly novelty
  failure, or the budget; it does not consult the training transition table.
- **Outcome:** an ordered `Sequence` of immutable neuron-ID snapshots. A full
  sequence is not implied by completion of the budget: recovery and retention
  remain measured outcomes.
- **Controls:** scalar/empty input, unknown topology, invalid phase schedules,
  and recall with refractory period zero must fail before mutation. A beta-zero
  or disabled-recurrent arm is required before adopting a learning claim.

<a id="contract-transducer"></a>

## T: transducer

Code: `_hashed_transducer.py:HashedTransducer`, `_arc_core.py`.

- **State:** LEX/ARC/STATE/OUT winners and fibers; word stimulus banks; grounded
  output signatures; optional feature register and successor stimuli; arc bias.
- **Grounding:** each word separately grounds LEX and OUT; records OUT signatures.
- **Tick:** word -> LEX for rounds; one ARC step from LEX and prior STATE,
  optionally the feature register and a prediction-gain bonus.
- **Write:** target stimulus and ARC jointly train OUT. Induced-state mode advances
  STATE from ARC (optionally future-word stimuli); copy mode instead copies the
  current ARC winners into STATE after writing.
- **Emit/readout:** frozen output projection; also advances or copies STATE.
  overlaps() compares to grounded signatures; rank() randomly breaks lexical ties
  using the caller's RNG. emit() is a state transition, not a passive query.
- **Learning:** hashed count-priced fibers; different default stimulus law and
  tie jitter from HashedArcFSM. Shared ArcCore does not establish full model parity.
- **Claim/diff T1:** SEQ-TEMPORAL-CARRY concerns registered copy/prediction/readout
  configurations. It is not a theorem about the default induced-state constructor.
- **T2:** state_mode, horizon, prediction gain, register visibility and rank RNG
  are protocol/model inputs, not performance flags.
- **Control:** g=0 vs g=1 with representation and readout measured separately;
  a second readout on identical stored arcs separates missing information from
  extraction failure. No new result is adopted by this card.

<a id="contract-hashed-aligner"></a>

## A: hashed cross-situational aligner

Code: `_hashed_aligner.py:HashedAligner`,
`_scheduled_aligner.py:ScheduledAligner`, `_hashed.py:StimulusFiber`,
`PresentFiber`, `AreaFiber`.

- **State:** fixed hashed LEX and FEAT areas; one phon anchor per word, one
  perceptual anchor per feature, one plastic LEX-to-FEAT fiber, cached anchor
  winners, cross-fiber potentiation counts and optional pinned-winner
  observations. Seeds and symbolic names jointly identify every graph.
- **Two fiber families:** anchors use Binomial afferent counts, optional initial
  inverse-indegree normalization, an explicit drive gain and normally no
  plasticity. The cross fiber uses multiplicative Hebbian counts, optional
  initial normalization and column scaling, with either a present-only store or
  dense counts. These cannot truthfully be represented by one substrate profile.
- **Write:** for every sentence in shuffled order, train every word against every
  scene bundle. LEX is reset to its cached phon-anchor assembly. FEAT starts at
  its cached bundle-anchor assembly, then performs `rounds_word` simultaneous
  stimulus-plus-LEX rounds. Each round reads the previous LEX winners, selects
  FEAT under deterministic hash jitter, and writes previous-LEX by new-FEAT
  coactivity. The destination is perceptually anchored but not teacher-forced.
- **Readout:** bundle references are frozen stimulus-only FEAT assemblies.
  Word reconstruction freezes learning, anchors LEX from the word, then projects
  LEX through the cross fiber alone. Capacity uses the argmax overlap with the
  entire bundle inventory after filtering words below the exposure minimum.
- **Numerical domain:** default anchors have gain `1/p`, zero stimulus beta and
  deterministic tie jitter. The present-only store is valid only without a
  finite clip. Column scaling with a finite clip is refused because those
  operations do not commute. The pricing table must cover potentiation count,
  not episode count.
- **Claim/diff A1:** drive parity with sampled NumPy does not imply winner parity
  because their stimulus laws and tie rules differ. Hashed and scheduled paths
  should agree in the same arithmetic and schedule; capacity evidence still
  depends on the registered corpus, exposure filter, inventory readout and
  thresholded curve.
- **Control:** change one of anchor gain, rounds per pair, stimulus plasticity,
  cross-store representation, normalization, column scaling or tie jitter and
  require constructor rejection against the recorded alignment profile before
  CUDA loading. A mechanism null must also move the alignment statistic; backend
  agreement alone is insufficient.

<a id="contract-word-capacity-protocol"></a>

## A2: word-capacity protocol value

Code: `word_capacity_protocol.py:WordCapacityProtocol`,
`word_capacity_run.py:measure`, `word_capacity.py:corpus`, `run_cell` and
`capacity_report`. Registration: `PREREG_word_capacity.md`.

- **Inputs:** selected lexical cells, their `(n, k, stimulus_size)` definitions,
  vocabulary grid, FEAT geometry, connection probability, plasticity, rounds,
  corpus composition, exposure filter, readout threshold, seed transforms,
  corpus seed scope, early-stop margin, interpolation band, W2/W3 thresholds,
  launch budget and the registered FEAT ladder. The value is frozen and
  round-trips to the complete JSON parameter record; unknown or missing fields
  are invalid.
- **Schedule:** `corpus_seed_offset` selects each synthetic experience and
  `training_seed_offset` selects its presentation order. Every sentence trains
  each word against each perceived bundle for `rounds_per_pair`. Scheduled runs
  may batch brains up to `launch_budget_bytes`; batching does not alter their
  logical schedules. Protocol 3.3 is per-brain and therefore admits the
  scheduled backend only; the lower hashed batch path requires a distinct
  `shared-batch` protocol.
- **Mutation:** corpus construction is pure. Training mutates only the aligner's
  cross-fiber counts under the separately recorded `AlignerSemantics`; reporting
  reads frozen reconstructions and does not train.
- **Readout:** words below `minimum_exposures` are excluded. Every remaining word
  chooses the bundle with maximum FEAT overlap. A per-seed interpolated crossing
  of `threshold` gives V*, and Student-t ensembles plus censoring produce W1-W3.
- **Admission:** protocol 3.3 permits cell, vocabulary-grid and FEAT selections
  from the registered design. Changing its other fields requires a new protocol
  version. Invalid dimensions, duplicate cells, unordered grids, impossible
  exposure rules, nonfinite values, incomplete records, malformed curves,
  unselected cells and unknown engine names fail before CUDA.
- **Control:** a changed category count must change the generated category
  inventory; changed `p` and `beta` must reach the actual NumPy aligner; a full
  scheduled-CUDA cell-A replay must remain exactly equal across all 140 committed
  observations. These controls distinguish a live parameter from recorded
  decoration.

<a id="contract-word-capacity-ladder"></a>

## A3: word-capacity FEAT ladder

Code: `word_capacity_ladder_run.py`; shared execution and measurement are the A2
protocol and A aligner above. Registration:
`PREREG_word_capacity.md#part-1----the-feat-ladder`.

- **Domain:** only registered lexical cells A and C, an ordered subset of the six
  registered FEAT rungs, and a protocol-declared vocabulary grid. The first rung
  and `feature_area` must agree. Protocol 3.3 fixes every other A2 field.
- **Execution:** each `(cell, FEAT rung)` becomes its own immutable A2 selection;
  the scheduled backend runs one independently generated corpus per seed. New
  studies require twenty seeds. Historical ten-seed replays must be smoke/VOID.
- **Observation:** output contains every per-seed curve and the corresponding
  interpolated V* ensemble and censor count. It does not silently convert the
  historical F1/F2 interpretation into a newly adopted verdict.
- **Controls:** cell B, unknown/reordered/duplicate rungs, a mismatched first
  rung, fixed-constant drift and the old direct `ladder()` entry all fail before
  CUDA. The A/1000x50 migration cell must reproduce its 70 committed values.

## Consequences for the architecture

The shared units are immutable model semantics, executable schedules, explicit
state ownership, observation protocols and claim/evidence links. “One backend
interface” cannot collapse fixed graph vs sampled recruitment, assigned vs induced
state, masked vs net readout, or teacher forcing vs autonomous transitions.

First migrate one complete card through construction, execution, observation,
recording and a demonstrated negative control. Then extend it to the next card.
A contract may return a candidate snapshot; it may report scientific PASS only
when that card's measured acceptance conditions have actually been evaluated.

## First resolutions after writing the cards

P1 and C1: corrected the unconditional formation/recovery wording. P2: project
rejects nonpositive/noninteger rounds before projection. R1/M2: source clamps are
scoped and both facade and engine flags are restored on success and exception.
The regression file is `test_operation_semantic_cards.py`: seven failures and
six passes on the baseline after preparing valid source assemblies, then all
thirteen passed after repair. Other discrepancies above remain open.

No new operation-protocol classes were introduced by this follow-up. The earlier
uncommitted prototype remains outside the package in the worktree's ignored cache.

## Reciprocal plan resolution (2026-09-11)

The later `ReciprocalProjectionPlan` makes card R's schedule an immutable value.
It rejects blank or identical area names, nonpositive or nonintegral rounds,
implicit truthy clamp values, missing areas and an empty source before the first
backend call. The public function performs this preflight before borrowing a
clamp. Its valid schedule remains one forward step followed by the declared
forward, target-recurrent and return edges, with the caller's facade and engine
clamp state restored on success or exception.

The migration test reconstructs the former body on sampled, fixed-connectome and
explicit NumPy engines and compares both areas' stable and compact winners,
recruitment, owned RNG state and the next read-only return observation. The
contract links the existing disabled-learning round-trip control. This establishes
schedule identity and mechanism sensitivity for that control; it does not turn a
target snapshot into evidence of autonomous bidirectional recall or settle the
fixed-target plasticity policy across backends.

## Association plan resolution (2026-09-11)

`AssociationPlan` resolves A2 by admitting either two active fixed sources or two
distinct named source stimuli. A one-sided or aliased stimulus pair now fails during construction
instead of silently leaving both sources evolving. Three distinct area names,
positive pathway rounds and nonnegative joint rounds are validated before
topology; absent areas, absent stimuli and empty fixed sources then reject before
the first clamp or backend mutation. Zero joint rounds remains the explicit
no-coactivation control.

The immutable steps retain the original working schedule: each source trains its
pathway sequentially, target recurrence begins on the second step of each pathway,
and the joint phase includes target recurrence from its first step. Stimulus-driven
sources retain their self-edges; fixed sources do not need them. The obsolete
imperative `_associate_body` is removed.

Migration tests reconstruct that helper for both source protocols on all three
NumPy engines and compare all areas' stable and compact winners, recruitment,
owned RNG, clamp restoration and the next read-only observation. The linked
coactivation sweep distinguishes zero, registered and excessive joint training.
This preserves A1: the operation returns a candidate target snapshot; an
association claim still requires the separately cued before/after measurement.

## Merge plan resolution (2026-09-11)

`MergePlan` makes the three edge switches and every round inspectable. It rejects
aliased areas or stimuli, invalid rounds, truthy nonbooleans, missing topology and
empty unstimulated parents before mutation. Both-unstimulated calls retain the
legacy scoped clamp; both-stimulus calls retain the evolving-parent schedule.

M1's partial-stimulus ambiguity is now explicit. Exactly one stimulus requires
`unstimulated_source_mode`: `require-fixed` verifies a clamp owned by the caller,
`fix-current` borrows a clamp around the current assembly, and `evolving` verifies
that the current parent is free. An omitted or contradicted mode rejects before
the backend. The thirteen static research callers were audited: twelve existing
pinning scopes now declare `require-fixed`, while `universality_composition` names
its deliberately live composed parent `evolving`. Their edge schedules are
unchanged.

Migration tests reproduce both-fixed, both-driven, partial-fixed and
partial-evolving schedules on all three NumPy engines, comparing all area states,
recruitment and RNG. A constructed schedule control removes return edges when
`back_project=False`; the existing weight-level control remains responsible for
the scientific two-way-connectivity claim.

## Completion plan resolution (2026-09-11)

`CompletionPlan` resolves C2 by making both stochastic cue identity and observation
mutation explicit. A caller must supply an integer seed and choose `plastic`,
`frozen`, or `read-only`. The first preserves historical learning, the second
suppresses weight changes while retaining activity and recruitment, and the third
restores activity, recruitment, weights and owned RNG state through the Brain
transaction. Invalid fractions, nonpositive rounds, missing policy, unknown or
empty areas and fractions that retain zero neurons all reject before cue injection.

`PreparedCompletion` holds two deliberately different spaces: the immutable
reference contains stable neuron IDs while `compact_cue` contains engine indices.
It is bound to the exact brain and entry winner state, so cross-brain or stale
injection rejects before mutation.
The sampled cue and recurrent `ProjectionStep` sequence are shared by traced and
untraced execution. An AST ratchet requires every statically visible caller to
name the seed and policy. Existing compatibility and research callers say `plastic`
to preserve their prior arithmetic, while the teaching investigation says
`read-only`; changing a registered protocol to a measurement mode requires an
amendment and rerun.

The migration test reconstructs the removed implementation on all three NumPy
engines and compares recovered stable IDs, live compact winners, recruitment,
owned RNG and the next read-only observation. Separate controls exercise all three
mutation policies and exception restoration. These tests establish admission and
schedule identity. They do not establish attractor recovery: the score remains
min-normalized and the free cue is not clamped. The linked fixed-connectome
teaching control proves the score falls when recurrent learning is disabled; it
is an instructional mechanism check rather than a preregistered result.

<a id="contract-read-only"></a>

## Read-only observation: executable obligations

Implementation: `Brain.read_only`, `ComputeEngine.validate_probe_target`, and
owner-declared `ActivityState` fields. This contract concerns projection inside
an observation scope; arbitrary user mutation, adding areas, changing policies,
or directly editing fibers inside the scope is not a supported transaction.

- **Requires:** each sampled target already has at least k materialized neurons.
  Reject a cold target before projection, including in a multi-target step.
- **During:** plasticity and recruitment are disabled; winners may respond.
  A sampled backend selects from its recruited population. This is a different
  candidate set from a full fixed connectome, and must be named in the protocol.
- **Restores on normal exit and exception:** facade and backend winners, clamp
  flags, population/firing counts, ever-fired masks, refractory/refraction
  activity, saved activity histories, local generator state and nesting flags.
  Restore existing mutable buffers so references held by callers remain valid.
- **Returns observational outputs:** latest drive scores and their population
  denominator remain available; they are measurements, not retained learning.
- **Control:** `frozen()` may deliberately initialize a sampled population.
  A read-only probe must still change winners inside its scope when the drive
  warrants it; retaining every winner would be a dead measurement.
- **Evidence:** `test_probe_state_contract.py` exposed five failures before this
  repair; `test_read_only_probes.py` checks nesting, exceptions, weights and
  probe order. CPU checks cover sparse and fixed NumPy paths; GPU state support
  still needs hardware gates. These are software obligations, not a scientific
  result or proof that every caller uses an informative readout.

C4 resolution: the previous cold-area exemption restored only visible counts,
leaving a materialized backend behind. Cold read-only projection now raises.
Initialize explicitly before probing; do not catch the error and call training
inside the observation scope. Exact pre-kWTA measurements now report the number
of candidates over which the summed drive was measured.

<a id="contract-context-observation"></a>

## CONTEXT: construction versus prefix observation

Code: `IncrementalMixin.build_context_incremental`, `_reset_context_state`,
`erp.adapters._settle_context_into_prediction`, `ErpProtocol`.

- **Construction:** the outer incremental parser resets the recruitment cursor
  and ID mapping, retains learned fibers, then accumulates each word. This is
  legacy disposable-context construction, not a read of a fixed population.
- **Nested observation:** prefix surprise is measured after the outer parser has
  accumulated context. The prefix loop clears activity and reuses the existing
  population/IDs (`preserve_topology=True`), inside the caller's read-only scope.
- **Requires:** initialized sampled target populations, including PREDICTION;
  the observation may not bootstrap an absent population as a side effect.
- **Mutation:** winners move during the prefix computation. Population count,
  mapping and allocation cursor remain unchanged. Read-only restores activity.
- **Rejected misuse:** a population/ID reset inside read-only raises before any
  clearing, including when the caller catches the exception inside the scope.
- **Protocol revision:** ERP outputs carry `ErpProtocol.observation_version =
  existing-context-v1` plus the resolved configuration. This change is not a
  claim of numerical equivalence with historical N400 artifacts.
- **Evidence:** `test_context_observation.py` exercises the real prefix loop and
  projection engine with initialized fibers, checks identities during/after the
  probe, and keeps an explicit construction-reset control. The ERP subset's
  three setup errors disappear; its preexisting VP-liveness xfail remains.

The generic `_reset_area_activity` helper had only one caller and duplicated the
CONTEXT winner/ID reset. It is removed; sentence construction now composes that
shared reset with a count reset. Disposal of learned context fibers is not part
of either operation.


## P3 resolution: operation recurrence is explicit

`ops.project(recurrent=False)` now passes no self-edge to the multi-round
helper. `True` still supplies self-edges on rounds 2 through T, via ordinary
projection. Global `Brain.recurrent_projection`, `norm_init`, and legacy scaling
gates cannot introduce a self-edge into the False operation. The helper's
historical filtering remains for its other callers; this is not a migration of
lexicon training or a claim that its fast path preserves every Brain side effect.

The regression observes actual `numpy_exact` backend source edges and the
recurrent potentiation store, for both operation values crossed with both global
recurrence and normalization flags. The old path fails when False encounters
both global flags enabled. Comparing final winners alone would miss this error.
This intentionally changes that previously ambiguous operation configuration;
results produced with it need a new protocol revision and a rerun.


<a id="contract-projection-rounds"></a>

## Multi-round projection: dispatch card and obligations

Source baseline: `9cf2c3b`, `Brain.project_rounds`, `Brain._project_impl`,
`ComputeEngine.project_rounds`, and the torch/CUDA/CuPy overrides.

- **Reads:** target, incoming stimulus/area maps, live source/target winners,
  clamp flags, plasticity, recurrence/normalization settings and backend state.
- **Selects:** the sparse path keeps only edges to the named target and removes
  its self-edge unless global recurrence and (norm_init or full scaling) allow
  it. The explicit path previously executed the entire supplied map, without
  the self-edge filter. Lexicon callers intentionally rely on sparse filtering.
- **Mutates:** the backend runs several projections; the sparse facade receives
  only the final result. It appends one history entry, using compact indices
  directly, instead of the ordinary path's per-round neuron-ID history.
- **Bypasses:** sparse rounds skip Brain inhibition, facade-to-engine clamp sync,
  activation recording and the normal result application helper. The docstring's
  claim to fall back to repeated `self.project` calls is false for sparse areas.
- **Other invalid inputs:** zero rounds returns None from most engines and fails
  after dispatch; some backends instead accept it. Extraneous destinations are
  ignored for sparse targets and executed for explicit targets. Empty resolved
  inputs can produce backend-dependent zero-drive behavior.

Acceptance: select one target schedule, validate it before mutation, then use
ordinary projection semantics for each round. Preserve the existing self-edge
selection as an explicitly documented legacy policy while callers migrate.
Known destinations outside the named target are excluded on every backend;
unknown names and empty resolved schedules are rejected. Positive integer rounds
are required. Inhibition, clamps, read-only preflight, final activation summaries,
per-round histories and learned state must match executing the resolved schedule
through ordinary projection. Construct controls with a closed target/fiber and
with an unsynchronized target clamp; final-winner agreement alone is insufficient.

The previous embedded historical capacity narrative is available in git and in
`research/notes/memory/recurrence_ceiling_on_exact_drive.md` and
`research/notes/memory/ceiling_n_scaling_on_exact_drive.md`. It does not prove that
normalization licenses recurrence or that any backend loop is a valid lowering.


### Multi-round resolution and additional observation defect

The named-target helper now resolves and validates its inputs without mutation,
then executes each round through `Brain.project`. Its legacy self-edge policy is
unchanged, including explicit targets retaining supplied self-edges. Each result
uses the shared application path; history therefore contains one entry per
executed round in the same index space as ordinary projection. Inhibited rounds
execute no target transition. The helper no longer claims to be a fused kernel.

Testing a later learning step after a probe exposed two further problems:

- Empty facade source winners were not synchronized, so the old backend assembly
  could still deliver drive and learn. Empty activity now propagates through the
  shared source synchronization path; a duplicated source-flag assignment was
  removed.
- The sampled numpy engine's eager and deferred initialization could construct
  an unused fiber under `read_only`. Both now honor `_no_recruitment`. A probe
  reads the currently materialized fiber, including a missing fiber's zero
  contribution; it does not fill in missing structure. The frozen-plasticity
  control still constructs the fiber, and a subsequent learning step after the
  read-only probe is compared with an unobserved brain.

This does not certify all GPU read-only paths. Inspection found a separate
zero-drive fiber-repair branch in `TorchSparseEngine.project_into` that needs
its own hardware regression; it has not been exercised or changed here.
Existing measured results are not re-adopted by these software regressions.


<a id="contract-engine-rounds"></a>

### Engine-level repetition

`ComputeEngine.project_rounds` owns the sequential backend loop. It requires a
positive integer count before calling `project_into`. Every step receives the
same target, source lists, plasticity flag and activation-recording flag; the
last result is returned. An exception stops the loop and does not roll back
previous steps. Brain-level controls are not part of this low-level API.

Torch and CUDA overrides repeated this exact loop. CuPy did too for positive
counts, with an additional zero-round success path. Those copies are removed;
all inherit the base method. Invalid counts now raise consistently. The old
comments claiming only the final result was copied to CPU did not describe the
loop: each iteration already called the ordinary `project_into` method.
CPU checks cover repeated-step equivalence and invalid-count rejection; hardware
parity suites remain required for the GPU implementations.


<a id="contract-role-reconstruction"></a>

## Role reconstruction: availability before observation

Source baseline `9d58bdd`: `RoleBindingMixin.parse_roles_by_reconstruction`.
Classification and learned voice gating select role slots. Inside read-only,
core assemblies are projected into those slots to record temporary winners;
replaying noun candidates is scored against those winners. Only a strict
occupant-versus-runner margin assigns a noun role. The ACTION assignment follows
successful verb traversal, not a noun-style reconstruction margin. Caller-supplied
filler roles are external inputs, not newly recovered evidence.

The method assumed role populations already existed. Lexicon-only parsers
therefore raised the cold-probe guard, while older behavior silently recruited
during readout. Acceptance: unavailable populations produce None for attempted
readouts and explicit per-area diagnostics without construction. One engine
readiness predicate must also drive the low-level guard, so the two cannot drift.
A populated role control must still yield readout evidence; suppressing every
traversal is not a fix. Unknown input failures must not be swallowed as generic
unavailability. The traversal/recall observation scope restores dynamical state.


<a id="contract-context-bridge-reset"></a>

### Bridge reset: requested capacity is not allocated population

At `9d58bdd`, `_reset_context_for_bridge(preserve_topology=True)` assigned the
requested ring capacity to both facade and backend `w` while retaining the old
ID mapping. Pre-growth can allocate fewer neurons than requested; disabling the
ring optimization exposed winners outside a 176-entry map. Requests smaller
than the current population could also shrink its count without changing IDs.

Acceptance: a topology-preserving reset clears activity and retains the actual
backend population and mapping, independently of requested capacity. It cannot
allocate or discard neurons by assigning a configuration value to a count.
The explicitly destructive reset remains a separate path, forbidden inside
read-only. A subsequent projection and neuron-ID snapshot must remain valid.


Resolution: role availability is checked through `ComputeEngine.probe_target_ready`,
also used by the low-level guard. Diagnostics include `unavailable_areas`; recursive
parsing retains `inner_role_diagnostics` rather than discarding them. Bridge reset
preserves actual population and IDs, with requested capacities below/above the
population covered by regression tests.

Remaining preparation defect: role classification occurs before the read-only
scope. `CategoryClassificationMixin.classify_word` resets core fibers and projects with
plasticity enabled unless its caller has disabled it. A diagnostic at n=1000,
k=20, seed=13, three rounds increased the queried phon->NOUN_CORE weight sum
from 2660.16748046875 to 2994.654296875. This is an instrument counterexample,
not a multi-seed scientific result. The populated role harness uses fixed
category inputs and therefore does not certify classifier isolation. The next
repair must separate classification observation from learning and construction.


<a id="contract-word-classification"></a>

## Word classification: neural query without training

Source baseline `c578f20`: `CategoryClassificationMixin.classify_word`,
`classify_word_cached`, and `classify_distributional`.

The direct neural classifier resets each nonempty core lexicon's area fibers,
then projects phon/grounding cues with the Brain's current plasticity setting.
Its tail uses legacy recurrence selection. Maximum overlap with any stored word
in that core is the score; this is not a mean or a probability. The first maximum
in CORE_AREAS order wins ties. No positive score triggers distributional fallback
when corpus statistics exist, otherwise UNKNOWN. The fallback returns category
scores, whereas neural scores are keyed by core area; this legacy distinction
must not be mistaken for one calibrated metric.

Acceptance: neural classification must preserve fibers, neuron identities,
activity, clamp flags, RNG and subsequent training behavior. It uses an explicit
stimulus-only schedule over existing populations, clearing temporary activity
and releasing target clamps inside read-only. It cannot initialize a cold
population with a nonempty lexicon. Grounding names are resolved without adding
stimuli. Constructed inhibited-area and zero-evidence controls must return no
neural recognition; familiar words and grounded holdouts remain positive checks.

Cache writes and distributional subcategory metadata are parser bookkeeping,
not neural learning. Their provenance/invalidation remains a separate contract.
This repair does not certify category accuracy or statistical equivalence of
engines, and historical numbers using the mutating classifier require a rerun.


### Classification observation resolution

The neural branch now resolves registered cues once, clears temporary target
activity and clamps, and repeats stimulus-only projection inside one read-only
scope. It does not reset fibers, add stimuli, recruit or potentiate. The initial
five controls failed on `c578f20` and pass after repair, including inhibited
populations that previously returned stale recognition and an exception after
projection. Category/subcategory cache writes remain distinct from neural state.
The preparatory neural mutation defect recorded in the preceding role card is
resolved by this shared classifier path; broader parser cache purity is not claimed.

The legacy tuple API changes score-key space on distributional fallback. The
typed evidence contract below now resolves this for fusion and decomposition;
other tuple consumers and parser caches still require an explicit provenance
migration. Neural state isolation alone did not solve that distinction.


### Heldout regression retained: query-driven population growth was hidden training

An exploratory software diagnostic crossed legacy/isolated classification during
training with legacy/isolated observation, on copies of each trained brain
(n=3000, k=30, seed=74, compiled dialogue fixture). The held-out verb was `finds`.
These are single-fixture debugging results, not adopted scientific evidence.

| Training classifier | Observation | Label | VERB overlap | PREP overlap | PREP population after query |
| --- | --- | --- | --- | --- | --- |
| isolated | isolated | PREP | 0.166667 | 0.333333 | 136 |
| isolated | legacy | VERB | 0.166667 | 0.033333 | 206 |
| legacy | isolated | VERB | 0.166667 | 0.033333 | 1965 |
| legacy | legacy | VERB | 0.166667 | 0.033333 | 1965 |

On the isolated-trained brain, the legacy observation also increased the queried
VERB stimulus weight sum from 25860 to 25971.201171875; isolated observation left
it at 25860. The legacy-trained brain was already altered by classification calls
during its training pipeline. Thus the old successful check depended on a
construction history that its explicit training schedule did not state.

`test_holdout_verb_classifies_via_grounding` remains failing with the isolated
classifier, and its expected VERB label has not been weakened. The noun holdout
passes. Explicit population preparation and comparability of overlaps across
unequal candidate populations are the next scientific/protocol obligations.
Changing a score formula to fit this one observed fixture would not discharge them.

<a id="contract-classification-evidence"></a>

## Classification evidence and conversion

`core.classification.ClassificationEvidence` carries category, source, raw
scores, cue mode and resolved stimulus names. `classify_word_evidence` owns query dispatch; `classify_word` is its legacy
tuple view. Neural results retain core-area keys, distributional results retain
POS keys, and absent evidence supplies UNKNOWN with no scores. Construction
rejects unknown domains, mismatched keys and nonfinite/negative scores. Scores
are copied and read-only; conversion to POS keys returns a fresh dictionary.
These strengths are not calibrated probabilities or comparable by construction.

Fusion reads the source before converting. A distributional fallback cannot
be renamed `lexicon_readout` or `phon`, counted as an independent neural signal,
or credited as correct neural readout by the decomposition report. The legacy
tuple still loses source and cue information and must not be used for new evidence
fusion. Three constructed controls failed using the pre-change inference module:
fallback source/score loss, fallback credited to neural readout, and a strong
wrong neural answer classified as weak because of conditional-expression precedence.

Inspection found another score-domain discrepancy: function subcategories were
converted to POS only in the high-confidence ungrounded distributional branch.
Other branches could return AUX/COMP/MARKER beside ordinary POS scores. All
branches now use the existing `FUNC_SUBCAT_TO_CORE` and `CORE_TO_CATEGORY` maps
before accumulating distributional scores. Frame fusion uses those same maps.
Two controls (grounded AUX and low-confidence ungrounded AUX) failed before the
repair; the existing high-confidence ungrounded case is a positive control.
Original function-subcategory metadata remains separate for gating.

This changes mislabeled reports and mixed-domain scores. It does not establish
neural generalization, tune fusion weights, or close the population-preparation
regression above. Cache results and whole-pipeline mutation remain separate work.

<a id="contract-brain-clone"></a>

## Brain and sparse-engine cloning

Forking reads the complete instrument state: configuration, primary/secondary
engine, populations, connectomes, RNGs, activity, inhibition, refraction, scaling
and diagnostic state. It must not project, reconstruct from constructor defaults,
or consume random draws. The returned object retains internal ownership aliases
(facade connectomes refer to the fork's engine connectomes) while mutable state
is independent of the parent. Identical next inputs must produce identical next
transitions before either copy is independently changed.

The old Brain clone enumerated attributes and omitted recurrence/normalization,
the mixed-connectome RNG, and diagnostic counts; it also discarded the secondary
engine. Sparse-engine clone reconstructed through its constructor and copied a
second hand-maintained field list, omitting deferred scaling and other runtime
state. Thus identical winners at fork time did not imply identical instruments.

The clone contract uses graph-preserving deep copy for the current Python state
objects. Any future specialized copy must prove these ownership and next-step
properties before replacing it. No speedup or GPU clone conformance is implied.
Parser ownership and its post-copy preparation are specified separately below.

<a id="contract-parser-fork"></a>

## Parser forks and pristine cache ownership

`fork_parser_instance` reads the complete parser graph and creates independent
mutable state, retaining aliases within the copy (including objects that refer
to the copied brain). This includes lexicons, bootstrap/distributional categories,
function metadata and nested exposure logs. The `wobbly` compatibility argument
selects no weaker ownership policy: whether a later protocol replays episodes
does not determine which state it is allowed to contaminate in another cell.

After copying, the existing preparation step clears the fork's incremental
circuit and wobbly memory, and resets its CONTEXT construction cursor/IDs while
retaining fibers. This is a prepared fork, not an exact all-state checkpoint
restore; the reset occurs only on the copy. Separating this legacy preparation
schedule from fork construction remains part of the protocol/IR migration.

`ParserCache.fork` must copy the pristine snapshot, never the publicly returned
mutable parser. Failure to create a snapshot, or a missing snapshot at fork time,
raises an error. No silent fallback to the live parser is allowed. Snapshot
failure preserves its original exception as a cause for diagnosis.

The old selective-copy implementation let mutations reach the source and sibling
forks. Constructed controls cover both wobbly settings, six mutable state groups,
internal aliases, an uncopyable snapshot, and an absent pristine cache entry.
Cache calibration operates on a private copy of the pristine training snapshot.
Only after calibration and a fresh snapshot both succeed does it publish matching
live/pristine calibrated objects. Previously it calibrated the publicly mutable
object and left the pristine snapshot uncalibrated; an earlier caller's mutations
could therefore determine the thresholds, while forks received different state.
Failures must leave both original cache objects and calibration status intact.
Existing external references to the old live parser are not modified by this
replacement. Tests cover contaminated live state, calibration/snapshot failures,
and the first calibrated `get` returning the new published object.

The semantics of post-copy preparation remain a separate protocol obligation;
successful copying or calibration does not prove the ERP measurement informative.

<a id="contract-parser-cache-identity"></a>

## Resolved parser cache requests

Cache lookup and training must consume the same resolved engine, holdout set,
fast-training choice and numerical parameters. None selects default holdouts;
an empty set means no holdouts. Engine resolution occurs before lookup and the
resolved name is passed to the trainer. The memory key and disk path include
these choices and a named environment signature, with disk checkpoint metadata
checked against the requested identity before reuse.

Repository-controlled ASSEMBLIES_/EMERGENT_ environment values are hashed exactly,
retaining case distinctions without persisting raw values. Cache location and
ERP-fast calibration mode are excluded from training identity; calibration mode
has its own key. Extra training cache misses are preferable to merging distinct
instruments. This does not yet capture arbitrary external library/toolchain or
hardware changes; a complete model/target identity remains an IR obligation.

Uncalibrated training entries remain unchanged. Fast/full calibration variants
are derived independently from their pristine snapshot and cached separately.
Requesting an uncalibrated parser must not return a previously calibrated variant.
Calibration publication preserves the parser-fork failure-isolation contract.

<a id="contract-checkpoint-storage"></a>

## Checkpoint storage publication

These files are replaceable, trusted local Python caches, not adopted result
artifacts or language-neutral IR programs. Each save owns a unique temporary
file in the destination directory, serializes and flushes the complete checkpoint,
closes the file, then replaces the destination. Serialization or replacement
failure preserves the previous destination and cleans up that save's temporary
file. Concurrent successful writers may publish either complete checkpoint;
neither may write into the other's temporary file.
Windows replacement errors 5/32/33 receive at most six attempts and 310 ms of
total backoff. Other errors fail immediately. Persistent denial propagates after
cleanup; this policy does not promise publication when a destination stays locked.

The loader treats missing, truncated, unsupported or structurally incompatible
pickle caches as misses. A decoded object must be a ParserCheckpoint, and the
cache caller separately checks its request metadata. This is not validation of
untrusted pickle input, numerical evidence, or scientific claims. File flushing
and replacement do not establish directory durability across arbitrary power
loss. Research results retain their separate exclusive/no-overwrite policy.

Constructed controls cover incomplete/unsupported pickle streams, partial
serialization, replacement failure, and synchronized concurrent writers. The
previous implementation used one fixed `.pkl.tmp` name, retained failed temp
files, and did not handle EOF or unsupported-protocol exceptions as misses.

<a id="contract-mixed-drive-indices"></a>

## Sparse-source drive into an explicit target

The source winners are compact indices; rows of the dense mixed connectome are
stable source neuron IDs. `_sparse_sources_drive_to_explicit` reads winners,
the engine's mapping and dense weights, and returns the sum of the corresponding
rows without changing neural state. The shared index validators and conversion
define the boundary. A backend mapping of None declares identity indexing; an
empty sampled mapping is not permission to reinterpret nonempty winners as IDs.

Negative, fractional, multidimensional, oversized or unmapped compact winners
raise before conversion. Mapped IDs must fit the source population and selected
matrix rows. No index is passed through as a fallback or silently dropped.
Empty sources contribute zero. Missing/non-dense fibers retain the legacy zero
contribution behavior; explicit capability/topology validation remains separate.

Eight invalid-input controls failed before repair. A non-identity mapping with
deliberately different compact-row weights is the positive control, and an empty
source is the zero control. Supervised `reinforce_connectome` now shares source
coordinate conversion, with its distinct zero-seeding and selected-block mutation
specified in the [reinforcement contract](../../../neural_assemblies/ir/VERIFICATION.md#contract-supervised-reinforcement).
That write path has separate controls; neither path is certified on other backends.

The same investigation exposed an earlier coercion in `Area.winners`: fractions
and large integers were already truncated/wrapped before the drive function saw
them. The setter now validates one-dimensional integer positions in [0, n) using
the shared validator on the area's array backend, before changing winners or
counts. It does not assert that a position has been materialized; the consuming
engine/mapping establishes that stronger bound. Existing mutable winner buffers
and the legacy meaning change of `.w` remain separate ownership/count work.
Five setter controls verify rejection leaves the previous activity unchanged.
No GPU validation/performance claim follows from the CPU checks.


### Classification cue protocol resolution

The default combined-cue behavior remains unchanged. Explicit phon_only and
grounding_only modes now resolve inputs before read-only observation and retain
the mode/names in evidence. A word having a registered phonological stimulus does
not establish that those inputs were learned. The held-out fixture diagnosed in
VALIDATION.md distinguishes grounding-only success from combined-cue failures;
this is not an automatic reason to change the default or adopt new accuracy claims.
See [the cue contract](../../../neural_assemblies/ir/VERIFICATION.md#contract-classification-cues).


<a id="contract-legacy-cue-corruption"></a>
### Legacy cue-corruption tests: observed code, not an adopted robustness claim

Source: `neural_assemblies/tests/test_noise_robustness.py` (reviewed at ea871be).
This file is not the additive-drive experiment in PREREG_context_noise.md.

- State read: one sampled NumPy brain at seed 42; a stimulus-trained assembly;
  the current compact winners and the recruited population count.
- Training mutation: `project(..., rounds=10)` omits `recurrent=True`, despite the
  module's stated recurrent-training protocol. The helper's default does not train
  recurrent structure. Lexicon cases also re-project before each corruption.
- Perturbation: replace floor(k*fraction) current positions with other recruited
  compact positions, using Python random.Random. If fewer alternatives exist,
  silently reduce the replacement count. The requested fraction is not guaranteed
  to be the delivered fraction. This is neither full-population corruption nor
  independent Gaussian input noise.
- Recovery mutation: eight ordinary A->A projections, with learning/recruitment
  available. These are not frozen measurements of the trained attractor.
- Readout: overlap with the pre-corruption snapshot, or a fuzzy lexicon label.
  The mild-corruption case only checks final overlap >.6 after retaining roughly
  .8 of the cue; that bar does not require improvement over the corrupted input.
  The chance comparisons use k/n despite replacement from the recruited population.
- Randomness: the lexicon case uses hash(word), so the same explicit seed need not
  reproduce its perturbation across processes. Different corruption fractions use
  different seed streams rather than nested corruptions of one cue.
- Claim discrepancy: monotonicity of these single-seed readouts, even if a test
  passes, cannot establish a basin of attraction, frozen recovery, a noise law or
  the robustness of the library. No mechanism-disabled recovery control is present.

Required successor contract: record the perturbation space and actual replaced
count; reject impossible requested corruption; use stable explicit seed identities;
train recurrence explicitly on a materialized population; separate learning from
read-only recovery; measure improvement relative to the delivered cue and construct
an initialized no-recurrence/no-learning control before selecting acceptance bars.
Do not relabel old numerical outcomes as this successor. This card records pending
work; the legacy tests have not been rewritten or empirically revalidated here.


### Cue-corruption successor (2026-09-10)

The historical card above remains the record of the pre-refactor tests. The package
test file now uses pure replace_neurons and strict observe_recovery, specified in
[the new contract](../../../neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery).
Exact membership replacement, explicit recurrent attractor construction, frozen
observation, reference-denominated scores and initialized learning/no-dynamics
controls replace its former implicit protocol. Increasing corruption uses nested
permutation prefixes at a fixed seed; input ordering does not alter the draw.
The standalone historical research experiments have not been migrated by this change.


<a id="contract-historical-noise-study"></a>
### Standalone historical noise study

Source reviewed: `research/experiments/stability/test_noise_robustness.py` at a3d2b4e.
It is distinct from the package tests replaced above. Its three trial functions
request explicit areas and therefore must not be described simply as lazy-area
experiments from the Brain constructor's default engine.

- Training is one stimulus-only round plus establish_rounds stimulus/self rounds.
  H1 then corrupts the current winners and reapplies stimulus/self; H2 uses self
  only. Both recovery loops retain ordinary plasticity. They measure a combination
  of restoration and further learning, not strict observation of an attractor.
- H3 trains A and B separately, stores their references, then co-stimulates A/B
  with A->B for establish_rounds. The stored B reference predates association;
  corruption is generated from that reference, not the post-association assembly.
  Recovery re-stimulates A and projects A->B, with learning available. Its scores
  therefore conflate association drift, driven recovery and ongoing learning.
- Perturbation samples distinct non-winners from range(n), replacing floor(k*f).
  It uses an explicit per-trial generator, unlike the old package hash(word) case.
- H4 reuses H2 with k=floor(sqrt(n)). The outer runner sweeps noise and sizes,
  reports summaries, one-sample chance tests and effect sizes. It returns
  raw_data={}, losing the per-seed observations needed to audit those summaries.
- CLI exposes only --quick; seed count changes to five without the shared runner's
  VOID status. Results do not use the new immutable run/source/protocol envelope.

Migration must preserve this historical protocol for numerical reproduction while
introducing a separately registered frozen-observation protocol. It must retain
per-seed and delivered-cue values, name actual backend owners, separate pre/post-
association references, and supply initialized learning/no-dynamics controls. A
simple replacement of its recovery loop would change what the study measures.
No standalone-study migration or new scientific adoption occurred in this audit.


### Historical noise trial consolidation (2026-09-10)

The three historical trial functions now share brain construction, recurrent
establishment and corruption helpers; H1/H2 differ by the recovery stimulus
schedule. H3 retains its original ordering and pre-association references.
The primary engine is pinned to numpy_sparse; actual explicit area ownership is
numpy_explicit, verified by replay. Recovery still learns. Results now retain the
ordered seed identities and values for every cell, with owner/learning/reference
metadata. The old card's raw_data={} describes the pre-change implementation.

The mutation protocol has not been replaced by observe_recovery. Full shared-runner
migration, no-overwrite storage and frozen-protocol registration remain open. The
current implementation's replay fixture does not reconstruct missing historical
seed/source provenance or certify the old scientific artifacts.

The outer historical study now refuses fewer than three seeds before computation.
This does not replace the pending shared-runner and scientific-protocol migration.


<a id="legacy-aggregate-summary-2026-09-10"></a>
### Legacy aggregate summary (2026-09-10)

Source: research/experiments/run_all_experiments.py, generate_summary and
print_summary. Reads ExperimentResult metrics; does not inspect registration,
seeds, protocol or engine. Writes an in-memory summary and prints it. Before this
change, missing metrics became zero, noise/scaling/capacity passed unconditionally,
phase_diagram disappeared, empty inputs passed, and execution failure was ignored.
These are quick runs, hence scientific status must be VOID regardless of metrics.

Contract: preserve every supplied experiment and its actual metrics/parameters,
keep explicit execution success separate from scientific status, and reject an
empty summary. Do not manufacture thresholds or missing measurements. A failed
execution must remain visible. Summary storage uses the exclusive strict JSON
writer. Tests construct empty, failed and perfect-looking results first.

This does not validate the aggregate execution schedule: its obsolete noise
arguments now raise instead of silently selecting defaults, and other producer
configurations still need migration. A request for unimplemented --full now fails
before computation instead of silently substituting a quick suite.


<a id="legacy-experiment-configuration"></a>
### Legacy experiment configuration (2026-09-10)

The aggregate quick launcher names eight producers. Its projection, association,
merge, phase-diagram and scaling calls supply grids/trial counts absent from their
run signatures. These five producers never read **kwargs and therefore executed
default internal grids instead. Noise has the same caller mismatch but now raises.
Coding-capacity and biological calls match their current explicit parameters;
their unused **kwargs still accepted typos. Signature compatibility alone does not
validate the scientific meaning or adequacy of those two protocols.

Contract: unknown arguments fail at the producer boundary, before its body runs.
The aggregate validates all configurations before constructing any experiment,
running trials or saving output. Collect mismatches with experiment names and
unsupported parameter names. Preserve the original declared calls as data; do not
translate grid names into scalar parameters or silently discard them. An invalid
suite is a configuration error, never a partially completed scientific result.

Implementation: run_all_experiments.py QUICK_EXPERIMENTS and validate_suite;
explicit run signatures in all eight listed producers. This removes only unused
keyword capture, not trial algorithms, defaults, random draws or schedules. The
inventory remains intentionally inadmissible until the six calls are migrated
under source-linked protocol records and replay checks. Successful preflight
alone grants no scientific PASS, provenance or seed-count guarantee.


<a id="historical-projection-measurement"></a>
### Historical projection measurement (2026-09-10)

Source: research/experiments/primitives/test_projection.py, all four trial helpers
and ProjectionExperiment.run. Default primary backend resolves numpy_sparse;
explicit=True areas are owned by numpy_explicit. Every evaluation phase continues
Hebbian learning. None is a frozen readout or proof of a fixed-point attractor.

H1 trains stimulus+self until three successive overlaps exceed .98, or 100 rounds.
Its convergence_time conflates timeout with convergence on the last round. It then
runs 20 self-only rounds with learning and measures overlap with the trained set.
H2 compares 30 stimulus+self rounds against 30 stimulus-only rounds, then uses the
same learning-on autonomous persistence schedule. Historically every unknown mode
string silently selected stimulus-only. H3 establishes A, then repeatedly updates A
from stimulus and B from A. Randomizing B before repeating this schedule does not
supply a partial cue: no B-source fiber is projected. Its result is A-driven B
regeneration while learning, not B autonomous completion or representational fidelity.

H4 trains stimulus+self for T rounds and intends to inspect pre-evaluation recurrent
weights, then measures learning-on persistence. The old implementation checks
area.connectomes, which does not exist, and always returns weight_ratio=1.0. Three
source-177dbbc probes reproduced that dead branch. Read the actual shared dense
brain.connectomes[A][A].weights instead. The ratio includes zeros in both means:
mean W over selected source/target pairs divided by mean W over the entire matrix.
Use the pre-evaluation winners and reject missing/invalid/zero-mean measurement
inputs, never substitute 1. This ratio is selection-biased: beta=0 does NOT imply
an expected ratio of 1. The previous test against 1 is not a justified learning null.

The outer study hardcodes H1/H3 size grids, H4 round grid and default schedules,
uses base_seed+offset, summarizes away raw observations, and lacks run provenance.
These must be migrated separately; the obsolete aggregate grid is not its protocol.
This correction changes H4's dead observable but preserves all trial projections,
winner trajectories and final weights. Fifteen pre-change development fixtures
cover three seeds and five schedules; old ratio=1 values are retained as evidence
of the defect, not accepted as the corrected measurement's expected answer.


### Projection configuration and evidence migration (2026-09-10)

The historical projection outer study now accepts H1/H3 size grids, H4 training
counts, train/test/max rounds and explicit ordered seed identities. Defaults retain
the corrected version-2 protocol. It retains every per-seed cell and records engine
owners, learning-on evaluation and the weight-ratio definition. Both historical
noise and projection use one seed resolver and one undefined-null record adapter.
A constant convergence-time response yields explicit undefined fit statistics.

The CLI now forwards to the tagged shared runner. The source-linked registration
research/notes/memory/PREREG_historical_projection_migration.md was committed before
the smoke. Its six cells match the direct path exactly; smoke is VOID, full outputs
UNADOPTED. This closes the outer configuration/raw-retention/provenance migration
listed above for new runs, not the timeout ambiguity, frozen-readout distinction,
old paired-test interpretation, or obsolete aggregate grid translation.


<a id="projection-convergence-stopping"></a>
### Projection convergence stopping (2026-09-10)

Projection H1 and legacy scaling both test three consecutive strict overlaps >.98,
but scaling has an extra initial stimulus-only activation. Neither protocol may be
replaced with the other. This change covers projection only and leaves every
projection call and evaluation phase intact under the default rule.

Version-3 contract: training_rounds is elapsed training work, converged records
whether the stopping rule was met, and convergence_time is null on timeout.
Convergence on the final allowed round is true; a timeout at the same round is
false. The window counts consecutive comparisons, requiring window+1 snapshots.
The threshold comparison remains strict; threshold/window are explicit recorded
configuration. Reject invalid rule values before constructing a brain.

Retain each seed's stopping status and nullable convergence time. Report capped
training rounds and the observed convergence indicator separately. Do not fit a
convergence-time scaling regression when any seed is censored; do not drop failed
seeds or substitute the time limit as an observed convergence event. Constant
uncensored responses retain the prior explicit undefined-fit representation.
This is a stopping observation during learning, not proof of asymptotic stability.


<a id="shared-convergence-phase"></a>
### Shared learning-on convergence phase (2026-09-10)

Scaling first projects stimulus-only once; projection does not. Both then repeatedly
project stimulus plus the area's self-fiber, testing the most recent consecutive
strict-overlap comparisons. Evaluations remain learning-on self-only projections.
Share only this identical phase and its stopping record; callers own initialization
and evaluation. Use stable-ID Assembly snapshots, retaining only the previous one
and a comparison streak. A failed comparison resets the streak. Window W requires
W+1 snapshots; the initial activation outside the phase is not a comparison sample.

The phase returns the last trained snapshot and explicit elapsed/converged/nullable
event-time fields. It mutates brain activity and weights by the supplied projection
schedule; it is not read-only. It retains no complete winner history. Default
schedules/weights must match projection's 15 and scaling's six pre-change fixtures.

The old scaling fitter classifies asymptotic complexity from the coefficient of a
linear log10(n) fit: slope<.5 became O(1), slope<2 O(log n), slope<5 O(log^2 n),
otherwise polynomial. These are not valid inferences: multiplying log(n) by any
positive constant changes that coefficient without changing its asymptotic class.
Remove these labels. A shared fit is explicitly descriptive linear-in-log10(n),
with no fit for censored observations and undefined inferential statistics for a
constant response. Do not silently drop seeds. Legacy scaling's outer grid/raw-data/
CLI migration is separate from this phase correction.


### Scaling outer protocol migration (2026-09-10)

The corrected scaling study now consumes explicit ordered seeds, population grid,
initial stimulus count, training limit, evaluation count and stopping rule. It
constructs and validates configs before computation, retains raw cells, records
k=floor(sqrt(n)), and forwards the old CLI to a tagged shared-runner adapter.
Default initialization remains one separate stimulus-only round. This closes the
new-run grid/configuration/provenance migration noted above; it does not repair
old artifacts or the obsolete aggregate grid. Full outputs remain UNADOPTED.


Shared convergence boundary checks: population sizes and observed event times are
positive integer counts (booleans are invalid). Validate every cell before deciding
that censoring makes a fit unavailable. Direct ConvergenceObservation construction
requires an Assembly, positive elapsed count and an explicit native boolean; a
string "False" may never become a convergence event by truthiness. Normalize valid
integer counts for serialization. Exhaustive tests compare the streak implementation
to the window definition for every eight-comparison Boolean history and windows1..4;
this is bounded equivalence evidence, not an unbounded formal proof.


<a id="historical-phase-measurement"></a>
### Historical phase-grid measurement (2026-09-10)

The phase trial applies one stimulus-only initialization, 30 stimulus+self rounds,
then 20 self-only evaluation rounds. All phases learn. It compares the trained
snapshot to final activity; it does not prove a frozen fixed point or an asymptotic
phase transition. The actual explicit area owner is numpy_explicit.

The old grid printed stable when mean persistence>=.95 and selected the first such
beta per sparsity as a phase_boundary, omitting sparsities without a crossing.
A sample mean above .95 can have an interval spanning .95. Replace that label with
above_threshold when the nominal interval's lower bound>=threshold, below_threshold
when its upper bound<threshold, otherwise unresolved. Preserve all raw seeds and
all grid rows. Report only the lowest sampled beta with above_threshold status;
retain null when none qualifies. This descriptive selection is not a simultaneous
confidence statement, monotonicity proof or identified physical phase boundary.

Capture the threshold, learning status and engines. Pin the primary engine while
using canonical Assembly snapshots. Replay must preserve the six source-f0a0de8
trial schedules, winners, weight hashes and persistence values. H3 currently fixes
k=100; validate n>=100 before any grid work instead of failing midway through it.
Full grid/schedule/runner migration remains a separate next step.


### Phase configuration resolution (2026-09-10)

The phase study now exposes ordered sparsities, betas, connection probabilities,
H3 k/beta, initial stimulus rounds, training/evaluation rounds and threshold.
Resolve all trial configs before timing or computation. Fractional sparsities map
to floor(sparsity*n); reject zero/invalid k and duplicate resolved k values so distinct
labels cannot disguise duplicate model cells. Record requested and actual sparsity,
resolved assembly sizes, full schedules and explicit seed identities. Defaults
preserve the corrected historical grid; n<100 is valid when H3 k and all grid sizes
are explicitly made compatible. Raw observations remain learning-on persistence.


<a id="experiment-numeric-resolution"></a>
### Experiment numeric resolution (2026-09-10)

Real-valued experiment grids resolve to binary64 before trials. Ordinary rounding
is part of that numeric model; overflow, nonfinite values and nonzero-to-zero
underflow must raise configuration errors. Preserve representable subnormals and
explicit zero. Check uniqueness after conversion, so collapsed grid values cannot
masquerade as separate conditions. A resolved config and its recorded parameters
must contain the same normalized scalar values; validation that discards the
converted value is insufficient. This concerns input representation, not numerical
error guarantees for the backend's subsequent arithmetic.


<a id="historical-study-adapter"></a>
### Historical study adapter (2026-09-10)

Four adapters duplicate the same parser, required tag, smoke alias, default seed
identities, owner-engine check, producer construction and result-status wrapper.
Only protocol identity/version, registration, source script, parameter factory,
producer and interpretation differ. Represent those differences as one immutable
HistoricalStudy specification; keep numerical protocol factories separate.

The common adapter validates protocol, version, engine and mode before constructing
a producer. A record for one study must not label another study's output, and an
unknown mode must not silently become a study. The shared runner still owns tag,
seed, exclusive-storage and source-archive validation. Preserve script provenance,
all resolved parameters and seeds, VOID/UNADOPTED output and scope text exactly.
CLI tests retain an injectable runner boundary; registrations remain study-specific.
This is execution composition, not a merger of scientific protocols or backend
semantics. Replaying all four archived smokes must preserve their observations.


Historical adapters accept `--parameters` with a repository-relative UTF-8 JSON
object. Override only keys in the selected smoke/full parameter factory; replace
whole values (including grids), never mutate the defaults. Execution identities
(seed IDs, tag, engine, mode) remain CLI controls and cannot be overridden by this
file. The canonical JSON reader rejects duplicate keys and lossy nonfinite,
overflowing or underflowing numbers. Domain validation remains in each producer
before trial computation; parsing does not certify a scientific configuration.

Bind the exact bytes parsed to the runner's captured input inventory using
`expected_input_digests`. A mismatch fails before reservation or producer execution.
Record every resolved parameter and archive the original file. This detects a
changed file between CLI resolution and capture; it does not promise detection of
all concurrent mutate-and-restore races. Historical outputs remain VOID in smoke
and UNADOPTED in study mode. An override does not inherit scientific registration
of a different grid, nor silently replace the migration's registered default cell.


<a id="historical-association-trials"></a>
### Historical association trials (2026-09-10)

Code source before refactor: research/experiments/primitives/test_association.py
at 5672fda. Both trial functions create explicit A/B, add sa/sb, establish A then
B separately using stimulus+self for establish_rounds, and snapshot both before
association. Co-stimulation projects sa->A and sb->B with A->B, plus B->A only in
the bidirectional condition. No recurrent fibers participate during association.
Every project call retains the backend's normal Hebbian learning and clipping.

The association readout replaces B winners, then repeats sa->A and A->B for
test_rounds, comparing the final B to the pre-association B snapshot. No B fiber
is active in evaluation, so after the first projection the result is independent
of the replacement B cue. This measures driven regeneration while learning,
including any association drift, not autonomous partial-cue completion or frozen
recall. The supplied corruption RNG is consumed but cannot influence this readout
when test_rounds is positive. Preserve that API behavior in this refactor.

The identity trial always associates bidirectionally, then evaluates A using
sa->A plus A->A, followed by B using sb->B plus B->B. These phases also learn;
original references remain the pre-association snapshots. A nonsignificant paired
difference cannot establish bidirectional/unidirectional equivalence. The outer
historical harness retains unresolved seed, raw-data and statistical-reporting
migration work; this card does not adopt its hypotheses or old result files.

Consolidate identical establishment/association scheduling in one helper and pin
the observed primary engine numpy_sparse (explicit areas owned by numpy_explicit).
Before changing it, capture nine source traces: three brain seeds for bidirectional,
unidirectional and identity trials at n60,k6,p.2,beta.1,w_max20 and schedules3/3/3.
Require exact projection calls, winner sequences, final weight hashes and returned
values after refactoring. Construct disjoint B replacements and verify equal
post-projection states; inspect evaluation weight updates to disprove frozen-readout
interpretation. These are software semantic controls, not scientific noise evidence.


<a id="historical-association-harness"></a>
### Historical association harness (2026-09-10)

Resolve ordered unique brain seeds and all grids before timing or trial computation.
Explicit seed identities are not offset; implicit IDs retain base_seed+i. Expose
establishment, association and evaluation rounds. Require positive establishment
and evaluation counts; allow zero association rounds as a disabled-association
control. Preserve the original default grids and trial invocation order, including
the corruption RNG stream (even though positive-round regeneration ignores B).
Size cells use k=floor(sqrt(n)); every derived configuration is validated first.

Retain every per-seed scalar and paired bidirectional-minus-unidirectional difference
in ordered raw_data with seed IDs. Summarize through the shared Student-t ensemble.
A paired test is a one-sample test of these differences against zero. Constant
differences yield undefined t/p/d and an explicit degeneracy reason, never a fake
p=1 or significance claim. Preserve marginal summaries; no multiplicity correction,
equivalence claim or scientific adoption is implied. Record statistical version,
all resolved grids/schedules and actual engine ownership. The shared adapter owns
exclusive tagged execution, archived inputs and VOID/UNADOPTED status.


<a id="paired-study-reporting"></a>
### Paired study reporting (2026-09-10)

For migrated historical protocols, construct both ensembles with the same explicit
ordered seed IDs and use the canonical paired_delta operation. Refuse mismatched
lengths, duplicate keys, fewer than three seeds and nonfinite observations. The
caller must supply values in that shared seed order; this function does not infer
alignment from unlabelled values. Return the differences, their Student-t summary,
and the one-sample test against zero through reported_null_test. Constant differences
retain their descriptive interval but have undefined t/p/d, an explicit reason and
significant=false. No equivalence or population certainty follows from these values.

Association reporting already has this meaning and must remain exact after adopting
the helper. Projection's old paired_ttest fallback reports p=1 for all constant
differences. Version 4 corrects that reporting and adds the paired-difference summary;
all numerical trials and unrelated summaries remain unchanged. The legacy generic
paired_ttest still has other callers; do not silently claim those were migrated.


<a id="historical-merge-trials"></a>
### Historical merge trials (code inspection, 2026-09-10)

Source9844e4b: research/experiments/primitives/test_merge.py. Both trials create
explicit A/B/C and establish A then B using separate stimulus+self schedules.
The composition trial trains A->C alone, then B->C alone, then both into C, on
one brain with continuing learning. C winner replacement between phases neither
clears the learned fibers nor enters subsequent drive (C has no outgoing fiber).
The description that resetting C prevents carryover is therefore false: learned
A->C weights persist into the joint phase, as do B->C weights and source learning.

Readouts compare C_AB against earlier C_A/C_B snapshots. merge_quality is their
average; composition_score is their maximum. With disjoint parents, a candidate
identical to one parent scores1 on the maximum and.5 on the average while retaining
none of the other parent. Neither scalar certifies representation of both parents.
The separate parent overlaps are already returned and must remain visible.

The recovery trial has a different training history: it skips the separate A-only
and B-only C training and goes straight to joint training. It then performs20
A-only evaluation rounds followed by20 B-only rounds, still learning and replacing
C winners before each. These are sequential driven readouts, not isolated frozen
partial-cue completion. Do not merge these schedules under a helper that changes
initial conditions or claims the two trial functions test the same prepared state.

Before refactoring: capture both trial histories and final weights across three
seeds; construct disjoint-parent and replacement-invariance controls. Preserve
legacy outputs as historical observables, correct their interpretation, and specify
any replacement metric/version before measurement. Outer grids, seeds, exclusive
storage and raw data remain unmigrated. This card is code analysis, not adoption
of old merge hypotheses or an empirical experiment result.


<a id="historical-merge-harness"></a>
### Historical merge harness (2026-09-10)

Expose establishment, merge and readout schedules, ordered grids and explicit seed
IDs. Resolve and validate every configuration before timing or trials; explicit seeds
are not offset. Default duration/size grids and trial invocation order stay unchanged.
merge_rounds governs each isolated-parent C training phase and the joint phase in
the composition trial. Thus its duration sweep does not isolate joint training time.
Positive merge rounds ensure trained reference snapshots exist; beta=0 is available
as a learning-disabled control. Recovery readout rounds are separately configurable.

Version parent-overlaps-v1 reports the old arithmetic as mean_parent_overlap and
max_parent_overlap, preserving individual overlaps for every cell and every seed.
Do not retain misleading composition_score/merge_quality aliases in the new report.
The trial helper's legacy keys remain for captured historical replay. Neither renamed
scalar becomes a scientific success condition. Chance tests remain limited to the
old mean-overlap and recovery observables; do not add implicit hypotheses for every
reported column. Undefined tests use the shared explicit null reporter.

Store ordered raw vectors, seed IDs, full schedules/grids, engine ownership and the
size rule k=floor(sqrt(n)). The common adapter owns tag, source/input archive and
VOID/UNADOPTED interpretation. Descriptive historical summaries do not adopt a
merge theorem, erase sequential training history or imply a frozen readout.

<a id="contract-binding"></a>

## D: role binding (shared target schedule)

`ops.bind` stores a source assembly in a shared target area. Its resolved
schedule is one feed-forward source-to-target projection followed by an optional
short target-recurrent tail; a source snapshot is replayed when current, with a
stimulus or already-live source as explicit fallbacks. The source may be clamped
for the schedule and its prior clamp state is restored. The target snapshot is a
neuron-ID observation; the operation mutates target winners, participating
weights, and engine history.

The operation rejects unknown topology, invalid round counts or clamp flags, and
missing source activity before projection. `read_binding` reuses the same
schedule inside `brain.read_only()` and therefore cannot create the binding it
measures. Binding strength and pathway drive are separate readouts and must not
be conflated.

<a id="contract-convergence"></a>

## Convergence learning schedule

`ConvergencePlan` is the shared immutable schedule for the two maintained
learning helpers. It owns the epoch budget, per-epoch projection rounds, the
minimum consecutive-history window, the overlap threshold, and whether the
pattern-driven variant includes recurrence. Both helpers return the final
snapshot, epochs used, and the last observed persistence; exhausting the budget
is not reported as convergence. The plan is validated before the first epoch,
so malformed schedules cannot partially mutate a brain.

<a id="contract-consolidation"></a>

## Pair consolidation (sleep replay)

`consolidate_pair` replays two current assembly snapshots across their
bidirectional fibers. The immutable plan names the areas, verifies that each
snapshot belongs to its area, chooses one or both directions, and fixes the
round budget before mutation. Each enabled direction activates its stored
source and invokes the reciprocal projection schedule; outputs are post-replay
snapshots. Unknown areas, stale area/snapshot pairing, invalid flags, and an
empty direction schedule fail before replay.

<a id="contract-source-binding"></a>

## Multi-source teacher binding

`assembly_calculus.binding.bind` is a distinct parser-facing operation from the package-level `ops.bind`. It co-fires one or more active source areas and optional teacher areas into a free target for a validated positive round schedule. Sources are pinned during the pairing; the target remains free so the teacher can determine its winners. It returns whether an active pairing was applied, rather than an Assembly snapshot. Unknown or duplicate topology, invalid rounds, and absent source activity are rejected before projection.

<a id="contract-binding-recall"></a>

## Multi-source binding recall

`assembly_calculus.binding.recall` is the readout paired with the multi-source teacher binding operation. It activates the named source snapshots, optionally clears the target, and re-drives the target inside `Brain.read_only()` with plasticity and recruitment disabled. It returns a target snapshot when a source is active and `None` for an inactive source. Unknown topology and malformed source lists fail before entering the readout scope.

<a id="contract-consolidation-protocol"></a>

## Generic consolidation protocol

`consolidate` executes a nonempty ordered tuple of `PathwayReplay`, `MergeReplay`, or `MultiProjectReplay` steps for a positive number of passes without resetting area connections. It optionally clears activity and optionally performs the destructive area/index preparation used only for episodic-reset protocols. The plan returns strengthened pathway edges; malformed or empty schedules are rejected before replay.
