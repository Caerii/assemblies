# Semantic cards: operations and organs

Code baseline: 15a7ed9, parent 3334876. These cards were derived from executable
bodies and their called helpers, then compared with prose and the register.
They describe current behavior, not desired behavior or independently reproduced
science. CUDA arithmetic is followed through the Python dispatch; kernel-level
equivalence still requires Claude's hardware gates.

Each discrepancy has an ID. A future contract must resolve the discrepancy with
a regression test or corrected claim; renaming the function is not resolution.
The uncommitted protocol-class prototype was set aside until these cards existed.

Register crosswalk: the five legacy operation functions do not each have an
adopted Result entry certifying their whole postcondition. Their mathematical
prose and parity goldens therefore cannot substitute for such a contract.
The memory card relates to REFRACTION-ANTI-MERGING and
REFRACTION-CANCELS-CONVERGENCE; the FSM card to SEQ-EXACT-RECOVERY and
SEQ-REGIME-CLIFF; the transducer card to SEQ-TEMPORAL-CARRY. Those entries'
preconditions and provenance caveats remain part of any claim made from a run.

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
