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

Register crosswalk: the five legacy operation functions do not each have an
adopted Result entry certifying their whole postcondition. Their mathematical
prose and parity goldens therefore cannot substitute for such a contract.
The memory card relates to REFRACTION-ANTI-MERGING and
REFRACTION-CANCELS-CONVERGENCE; the FSM card to SEQ-EXACT-RECOVERY and
SEQ-REGIME-CLIFF; the transducer card to SEQ-TEMPORAL-CARRY. Those entries'
preconditions and provenance caveats remain part of any claim made from a run.

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
