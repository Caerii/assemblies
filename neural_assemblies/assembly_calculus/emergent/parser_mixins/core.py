"""
CoreParserMixin -- core setup, training, classification, and batch parsing.

Extracted from the 48-area emergent NEMO parser (Mitropolsky & Papadimitriou
2025) on numpy_sparse.  Categories emerge from grounding patterns: a word
presented with VISUAL grounding forms its assembly in NOUN_CORE, a word with
MOTOR grounding forms in VERB_CORE, etc.

All operations compose existing assembly_calculus primitives:
    project, reciprocal_project, merge, sequence_memorize,
    readout_all, _snap

Architecture (counts are authoritative in ``..core.areas``, which asserts the
total at import; keep this summary in step with it):
    Input:   8 modality areas, driven by PHON + grounding stimuli
    Layer 1: 8 CORE areas + 2 LEX areas
    Layer 2: 7 THEMATIC role areas + ROLE_SCENE + 4 SYNTACTIC areas
    Layer 3: 5 PHRASE areas + 3 VP_COMPONENT areas
    Control: SEQ, MOOD, TENSE, POLARITY, NUMBER, ERROR
    Advanced: CONTEXT, PRODUCTION, PREDICTION, DEP_CLAUSE

WHAT "EMERGENT" MEANS HERE, precisely.  Nothing in training tells the model
that "dog" is a noun.  Training presents a word form together with whichever
sensory modality accompanied it, and the routing table
``areas.GROUNDING_TO_CORE`` sends each modality to a different core area.  A
word's part of speech is then just: which core area holds a stable assembly
for it.  What is BUILT IN is the routing table and the area inventory; what is
LEARNED is the assignment of words to categories and the bindings between
them.  A reader evaluating the claim should hold that line firmly -- the
architecture is given, the lexicon and its structure are not.

References:
    Mitropolsky, D. & Papadimitriou, C. H. (2025).
    "Simulated Language Acquisition with Neural Assemblies."

COMPOSITION
-----------
The training stages this class used to inline now live next to it, one module
per stage: lexicon, roles, gating, phrases, classify. ``CoreParserMixin`` keeps
construction, area setup, the ``train`` ORCHESTRATION, and ``parse``.

The split is not cosmetic. Every silent failure found in this file sat at a
stage boundary -- a connectome reset at the end of ``train_phrases``, a
vocabulary registered only in ``__init__`` and never reconciled with the
corpus, words skipped by the parse loop with no record. None of them were
visible because there was no boundary to put a postcondition on. There is one
now: each stage is a module with an entry point, and what it must be true of
when it returns can be stated there.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..train_progress import TrainProgress
    from ..core.corpus_index import CorpusIndex

from contextlib import contextmanager
from typing import Dict, List, Optional

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.assembly import (
    Assembly,
)
from neural_assemblies.assembly_calculus.readout import Lexicon

from ..core.areas import (
    ALL_AREAS, GROUNDING_TO_CORE, THEMATIC_AREAS, MUTUAL_INHIBITION_GROUPS,
)
from ..core.grounding import GroundingContext, VOCABULARY
from ..curriculum.data import GroundedSentence, create_training_sentences

from ._shared import (
    MERGE_ROUNDS,
    DistributionalStats,
    _MODALITY_FIELDS,
)
from .classify import CategoryClassificationMixin
from .gating import GatingMixin
from .lexicon import LexiconTrainingMixin
from .phrases import PhraseStructureMixin
from .roles import RoleBindingMixin

# Re-exported: importers outside this package take these from here, and
# `_parser_core` does `import *`. Moving them must not move their address.
__all__ = [
    "CoreParserMixin", "DistributionalStats", "MERGE_ROUNDS",
    "_MODALITY_FIELDS",
]


class CoreParserMixin(
    LexiconTrainingMixin,
    RoleBindingMixin,
    GatingMixin,
    PhraseStructureMixin,
    CategoryClassificationMixin,
):
    """Setup, training orchestration, and batch parsing.

    The stages themselves are the mixins above; what lives here is the
    order they run in and the state they share.
    """

    # Optional evidence stores shared by acquisition and wobbly parsing. They
    # are declared at the composition root so consumers agree on ownership;
    # the stores remain lazy because most parser instances never use them.
    _exposure_log: Optional[List[List[str]]]
    _wobbly_resolutions: Optional[Dict[str, Dict[str, object]]]

    # DEFAULTS phon_weight=6.0, beta=0.05 -- the Phase B pair, flipped on three
    # independent lines of evidence (deferred until all three were in):
    #   1. core-assembly duplicates 0.32 -> 0.06, role retrieval 0.77 -> 0.97
    #      (semantic drive share, research/notes phase_b);
    #   2. reconstruction-readout parsing 9/12 -> 12/12 BOTH voices, occupant
    #      gap 0.50 -> 0.95 (sentence_conditioned_readout.py);
    #   3. event-representation separation C2 0.9528 -> 0.0417 -- at the old
    #      default the substrate could not distinguish "child enters mouse"
    #      from "mouse enters child".
    # Known cost, stated: grounding-only recall 0.82 -> 0.67 (still 4x chance).
    def __init__(self, n: int = 10000, k: int = 100, p: float = 0.05,
                 beta: float = 0.05, seed: int = 42, rounds: int = 10,
                 phon_weight: float = 6.0,
                 engine: str = "auto",
                 inference_rounds: Optional[int] = None,
                 bridge_rounds: Optional[int] = None,
                 fast_training: Optional[bool] = None,
                 norm_init: Optional[bool] = None,
                 sampled_recurrence_policy: str = "warn",
                 vocabulary: Optional[Dict[str, GroundingContext]] = None,
                 synaptic_scaling=False,
                 synaptic_scaling_deferred: bool = False,
                 novelty_gain_max: float = 1.0,
                 novelty_gain_exp: float = 0.5,
                 split_feature_areas: bool = True):
        from ..training.perf import (
            budget_rounds,
            fast_training_enabled,
            resolve_engine,
        )

        self.fast_training = (
            fast_training_enabled() if fast_training is None else fast_training
        )
        train_r, infer_r, bridge_r = budget_rounds(
            rounds,
            inference_rounds=inference_rounds,
            bridge_rounds=bridge_rounds,
            fast=self.fast_training,
        )
        self.n = n
        self.k = k
        self.p = p
        self.beta = beta
        #: Word-form stimulus size as a multiple of k. 1.0 reproduces the
        #: historical behaviour exactly. See `add_phon_stimulus`.
        self.phon_weight = phon_weight
        self.seed = seed
        self.rounds = train_r
        self.inference_rounds = infer_r
        self.bridge_rounds = bridge_r
        resolved_engine = resolve_engine(engine, n_hint=n)

        brain_kwargs = dict(
            p=p, save_winners=True, seed=seed,
            engine=resolved_engine, n_hint=n,
            # Specification: neural_assemblies/ir/VERIFICATION.md#contract-sampled-recurrence
            sampled_recurrence_policy=sampled_recurrence_policy,
        )
        # Forward norm_init only when explicitly set so the Brain default (True)
        # is preserved otherwise. norm_init=False opts into the un-normalized
        # substrate -- used by the compiled-training-optimization tests, since
        # compiled training is disabled under norm_init (see
        # _compiled_training_enabled below).
        if norm_init is not None:
            brain_kwargs["norm_init"] = norm_init
        # Homeostatic scaling, forwarded verbatim: False (default), True
        # (every area -- carries the documented attractor-cancellation
        # hazard), or a collection of TARGET area names. The scoped form is
        # for stimulus-anchored feature areas, e.g. {TENSE, NUMBER} -- see
        # NumpySparseEngine._normalize_area_columns and task #130.
        if synaptic_scaling:
            brain_kwargs["synaptic_scaling"] = synaptic_scaling
            # SLOW HOMEOSTASIS (E9, #138): defer normalization to phase
            # boundaries (train_tense/train_number flush at their end) --
            # fast Hebbian inside a slowly renormalized envelope, the
            # timescale separation E8 measured the need for.
            if synaptic_scaling_deferred:
                brain_kwargs["synaptic_scaling_deferred"] = True
        # Surprise-modulated plasticity (E2, task #131): morph-feature
        # training episodes multiply the afferent fiber's beta by a novelty
        # gain derived from the learner's OWN exposure counts -- never from
        # a linguistic label (a gain keyed to "is plural" would be a corpus
        # knob in disguise). 1.0 disables (exact prior behavior). See
        # MorphosyntaxMixin._novelty_gain for the registered form.
        self.novelty_gain_max = float(novelty_gain_max)
        #: Epochs over the stage corpus for the morph-feature phases
        #: (train_tense/train_number). 1 = byte-identical prior behavior.
        #: E7 (#136) measured per-form exposure PINNED at ~1.5 by uniform
        #: sampling at every corpus size; repetition is how exposure rises
        #: at fixed diversity (Zipf's role in real corpora, repeated
        #: utterances in childhood). E8 (#137) sweeps it.
        self.morph_repetitions: int = 1
        #: E19 (#148): interim deferred-scaling flushes every K morph
        #: episodes (0 = phase-boundary only, the pre-E19 schedule).
        #: E18 measured the per-phase schedule as the 400-frame
        #: degradation channel (+0.19/+0.27 PL from 8 interim flushes):
        #: within-interval Hebbian mass concentrates multiplicatively and
        #: the eventual column normalization cannot undo the ratios. The
        #: slow loop must be slow relative to fast dynamics (E9) AND fast
        #: relative to accumulated mass (E18) -- a RATE, not a boundary.
        #: DEFAULT 40 (adopted #149): E19b at n=10 -- the right wall (mass
        #: concentration) is UNCONDITIONAL (K=40 and K=1 both beat
        #: per-phase by ~+0.09 paired at 400 frames), the left wall is
        #: conditional on repetition-style training (E9's per-update
        #: penalty appeared only under R4 repetition), and K=40 is the
        #: measured-best cell (0.727 +/- 0.037). INERT unless
        #: synaptic_scaling_deferred is on, so every non-homeostatic
        #: path is byte-identical.
        self.morph_flush_every: int = 40
        #: Exponent on (mean_count/count). 0.5 (sqrt) is E2's original form,
        #: measured VACUOUS on this corpus (max raw gain 1.215 at mean
        #: exposure 1.41 -- E3); 1.0 (linear) makes a once-seen form among
        #: 5x-seen neighbors write ~5x, so GAIN_MAX becomes a live cap.
        self.novelty_gain_exp = float(novelty_gain_exp)
        #: E15 (#144): one area PER FEATURE VALUE (NUMBER_SG/NUMBER_PL, ...)
        #: in a mutual-inhibition group, instead of two label values sharing
        #: one k-WTA area. E14 measured the one-area design's label images
        #: MERGING as total label projections grow (2.6 -> 17.8 shared cols
        #: of 30 at 200 frames) -- an architecture limit no corpus shape can
        #: fix. The value areas are created LAZILY by train_tense/train_number
        #: (never at construction), so this flag can be flipped on a parser
        #: restored from a pre-stage checkpoint.
        #: DEFAULT True (adopted #149 through an n=10 PAIRED gate at the
        #: default corpus, bars registered before data): balanced tense
        #: delta -0.039 +/- 0.058 (no measured harm; a single seed read
        #: -0.26 and two more read -0.26/+0.17 -- the ensemble lesson,
        #: both directions), SG 0.920, and PL 0.415 vs 0.085 shared --
        #: the split takes PL recall off the floor at the DEFAULT corpus,
        #: not only at the zipf-200 scale where E15/E19b measured 0.700/
        #: 0.727. False = the legacy shared-area path, byte-identical to
        #: pre-E15, kept reachable for parity reproductions.
        self.split_feature_areas = bool(split_feature_areas)
        #: #149: which readout recall_tense/recall_number RETURN under the
        #: split ("mi" | "overlap"). E15 measured the two CROSSING: MI
        #: (cross-area drive competition) wins at the 50-frame default
        #: budget (0.620 vs 0.540 on number), the within-area overlap
        #: readout wins at >=200 frames (0.700 vs 0.637; E19b's terminal
        #: 0.727 is overlap's) -- single-step drive comparison saturates
        #: early (#24's weak primitive). "mi" is measured-best at the
        #: default corpus; production at scale sets "overlap" -- see
        #: research/notes/language/production_configuration.md. Both answers always
        #: ride in diag regardless of this switch.
        self.morph_readout: str = "mi"
        #: #151 (paper-regime axes; what_the_papers_actually_prescribe.md).
        #: LABEL-STIMULUS SHARE: True fires number_SG/number_PL into the
        #: value area during training (the E-series protocol -- winners
        #: label-selected, images collapse to a class attractor, 0.998
        #: measured). False = ROUTING-ONLY, the papers' construction: the
        #: teacher still routes the projection to the correct value area
        #: but only the WORD drives it, so winners are word-selected and
        #: recall compares boosted-extreme vs unboosted-extreme (the
        #: label is WHERE, never WHAT -- acquisition 2025 p9, COLT22
        #: Alg. 1). Number-only until tense needs it.
        self.morph_label_stim: bool = True
        #: Fiber-beta gain on the core->value fiber during morph training,
        #: composed with the novelty gain. COLT22 Remark 2's margin
        #: (beta >~ sqrt(2 ln(n/k)/kp)) is a beta-vs-kp tradeoff; our
        #: kp=1.5 sits below margin at every measured n (extreme factor
        #: 2.48-2.78 vs effective boost ~2.1) and there is no per-fiber p
        #: lever, so beta is the margin's engine-supported knob:
        #: gain 4 -> beta_eff 0.2 -> boost (1.2)^15 ~ 15. 1.0 = exact
        #: prior behavior.
        self.morph_beta_gain: float = 1.0
        # Backing state for the `role_bind_gain` property (see its docstring
        # for the design). Written directly here because the setter touches
        # engine betas, and at this point in construction the role areas may
        # not exist yet.
        self._role_bind_gain: float = 1.0
        self._role_fiber_base_beta: Dict[tuple, float] = {}
        self._morph_exposure: Dict[str, int] = {}
        #: Label-image cache for the morph recall readout. A feature area's
        #: stimulus images are identical for every probed word on an
        #: unchanged substrate, and recomputing them per word was ~40% of
        #: readout time. INVALIDATION: train_tense/train_number clear it;
        #: any other path that trains into a feature area must too.
        self._feature_image_cache: Dict[tuple, object] = {}
        self.brain = Brain(**brain_kwargs)
        self.engine_name = self.brain.engine_name or resolved_engine

        self._bridge_topology_linked: bool = False
        self._role_topology_linked: bool = False
        self._lexicon_topology_linked: bool = False
        self._lexicon_linked_words_by_core: Dict[str, int] = {}
        # Compiled training (freeze connectome growth + ring-reuse of a bounded
        # pre-grown column pool, skipping fresh-candidate sampling) is a training
        # speed optimization -- but it is fundamentally incompatible with
        # norm_init. norm_init's anti-collapse depends on fresh-candidate
        # competition; compiled skips it AND caps the neuron pool (PREDICTION
        # grows to ~1322 columns under exact but is frozen at ~606 under
        # compiled), so under norm_init the CONTEXT/PREDICTION and role bridges
        # collapse to a single context-independent attractor (measured
        # exact-vs-compiled next-token top-1 agreement 0.05). The speedup and the
        # collapse are the SAME mechanism (freeze+ring), so no lightweight fix
        # keeps both -- verified exhaustively. Disable compiled when norm_init is
        # on (the Brain default); it stays enabled on the un-normalized substrate
        # where the exact reference also (correctly) does not diversify.
        self._compiled_training_enabled: bool = not getattr(
            self.brain, "norm_init", False,
        )

        # Per-core-area lexicons: {area_name: {word: Assembly}}
        self.core_lexicons: Dict[str, Lexicon] = {}

        # Role lexicons: {role_area: {word: Assembly}}
        self.role_lexicons: Dict[str, Lexicon] = {}

        # VP assemblies: {key: Assembly}
        self.vp_assemblies: Dict[str, Assembly] = {}

        # Word -> phon stimulus name
        self.stim_map: Dict[str, str] = {}

        # Word -> grounding context
        self.word_grounding: Dict[str, GroundingContext] = {}

        # Classification cache (cleared on register_word)
        self._category_cache: Dict[str, str] = {}

        # Incremental parse reuse
        self._incremental_circuit: Optional[object] = None

        # Set of all created grounding stimulus names (to avoid duplicates)
        self._grounding_stim_names_set: set = set()

        # Distributional statistics for category inference from raw text
        self.dist_stats = DistributionalStats()

        # Inferred constituent-order typology: one of the six labels in
        # ``core.word_order.WORD_ORDERS``, or None before any inference.
        # Set by ``train_word_order_typological`` and by ``train`` (via
        # ``_update_word_order_from_evidence``, from role annotations).
        # The six-label output space is BUILT IN; the choice within it is
        # learned -- see core/word_order.py.
        self.word_order_type: Optional[str] = None
        # Provenance of that choice: "roles", "transitions" or "none".
        self.word_order_evidence: str = "none"
        # False when only transition evidence was available, in which case the
        # subject/object ordering was a tie-break, not a finding.
        self.word_order_identifiable: bool = False
        self.word_order_confidence: float = 0.0

        # Learned gating patterns: function word sub-type → role expectations.
        # Populated during train_roles() by observing what role the next
        # noun receives after each function word type.
        # Format: {func_subcat: {"role_bias": {ROLE_AGENT: float, ROLE_PATIENT: float},
        #                        "clause_boundary": bool}}
        self.learned_gating: Dict[str, Dict] = {}
        #: Per-WORD gating for MARKER-class words -- the paper's control is
        #: per-word action programs, and subcategory grouping measurably
        #: cancels when two MARKER words have opposite effects ('by' reverses
        #: voice, 'to' marks a recipient: pooled, MARKER read conf 0.438 and
        #: passives died). Keys are words; values mirror `learned_gating`
        #: entries plus `goal_conf` (contrastive evidence the word marks a
        #: GOAL role).
        self.learned_word_gating: Dict[str, Dict] = {}

        self._setup_areas()
        self._register_vocabulary(vocabulary or VOCABULARY)

    # ==================================================================
    # Per-fiber plasticity policy
    # ==================================================================

    @contextmanager
    def _gain_on_fiber(self, target: str, source: str, gain: float):
        """Transiently multiply one fiber's beta through the engine's own
        set_beta/get_beta -- the authoritative per-fiber store (the #88
        lesson: writing any OTHER beta bookkeeping is a silent no-op).

        Lives on the core mixin because it is a property of FIBERS, not of
        any one training phase; morphosyntax and any future phase share
        this single bracket rather than growing siblings.
        """
        if gain == 1.0:
            yield
            return
        eng = self.brain._engine_for(self.brain.areas[target])
        base = eng.get_beta(target, source)
        self.brain.update_plasticity(source, target, base * gain)
        try:
            yield
        finally:
            self.brain.update_plasticity(source, target, base)

    @property
    def role_bind_gain(self) -> float:
        """Fiber-beta gain on every CORE->ROLE fiber (#52 recipe propagation).

        Same Remark-2 margin lever as `morph_beta_gain`, at role binding's
        operating point: the standing exam runs kp=1.5 with bind's T=2
        plasticity rounds, so one episode boosts (1.05)^2 ~ 1.10 against an
        extreme factor 2.48-3.29 (n=3e3-1e5) -- baseline needs E~9
        exposures where gain 4 needs E~3. Registered 2x2 decides adoption;
        1.0 = exact prior behavior.

        A PROPERTY WHOSE SETTER WRITES THE ENGINE'S PER-FIBER BETA STORE,
        not a bracket at one call site, deliberately: role_lexicons has
        FIVE writers (roles.train_roles, unsupervised/batch, consolidation,
        generation, constituent_order) and the curriculum path does NOT run
        train_roles -- a bracket at any one site is a dormant selector on
        the others ([[one-canonical-way]]). Setting the store once makes
        every present and future writer see the same gain, and makes the
        obvious idiom (`p.role_bind_gain = 4.0`) the correct one.
        """
        return self._role_bind_gain

    @role_bind_gain.setter
    def role_bind_gain(self, gain: float) -> None:
        gain = float(gain)
        if gain == self._role_bind_gain:
            return
        for core in sorted(set(GROUNDING_TO_CORE.values())):
            for role in THEMATIC_AREAS:
                key = (role, core)
                eng = self.brain._engine_for(self.brain.areas[role])
                base = self._role_fiber_base_beta.get(key)
                if base is None:
                    base = eng.get_beta(role, core)
                    self._role_fiber_base_beta[key] = base
                self.brain.update_plasticity(core, role, base * gain)
        self._role_bind_gain = gain

    def set_base_beta(self, beta: float) -> None:
        """THE ONE WRITER of the global area->area plasticity rate.

        The curriculum trainer's stage schedule used to loop over every
        fiber and write the engine's beta store directly
        (`CurriculumTrainer._set_global_beta`), which CLOBBERED any
        per-fiber policy: `role_bind_gain` set 0.2 on the core->role
        fibers and the next stage's schedule silently reset them to 0.1.
        Caught by test_role_bind_gain's liveness test -- trained mass was
        byte-identical between gain arms because the gain never survived
        into the SENTENCES stage.

        Two writers of one store compose only if one of them owns the
        policy. This method owns it: effective(fiber) = base * overlay,
        recomputed here for every registered overlay, so a stage schedule
        RE-PRICES the overlays instead of erasing them. The trainer
        delegates; new schedules must call this, never the engine.
        """
        beta = float(beta)
        brain = self.brain
        for area_name in brain.areas:
            area = brain.areas[area_name]
            for src in area.beta_by_area:
                # Brain owns the descriptor/backend synchronization.  Calling
                # the engine directly leaves explicit mirrors stale and makes
                # mixed-engine brains depend on which backend happens to be
                # primary.
                brain.update_plasticity(src, area_name, beta)
        # Re-apply overlays on the new base: the stage changed the price
        # level, not the policy.
        for core in sorted(set(GROUNDING_TO_CORE.values())):
            for role in THEMATIC_AREAS:
                self._role_fiber_base_beta[(role, core)] = beta
                brain.update_plasticity(core, role, beta * self._role_bind_gain)

    # ==================================================================
    # Setup
    # ==================================================================

    def add_phon_stimulus(self, word: str) -> str:
        """Register the word-form stimulus for *word*. THE ONLY WAY TO DO IT.

        WHY THIS EXISTS AS A METHOD. Three sites hand-rolled
        ``brain.add_stimulus(f"phon_{word}", self.k)`` -- here, and twice in
        `distributional.py` for corpus words. That is the shape this project
        keeps paying for: a change applied to one sibling and not the others
        ([[one-canonical-way]]). `phon_weight` has to reach all three or the
        drive share it controls would depend on which route registered a word.

        WHAT `phon_weight` CONTROLS, and why it is a drive share rather than a
        size. `apply_lexicon_word` fires phon PLUS one stimulus per grounding
        feature, simultaneously, all previously of size k. So word identity was
        1 of (1+F) equal drivers -- measured share 0.20-0.33 for F = 2..4 -- and
        two words with the same grounding had input overlap 2F/(2+2F) = 0.667,
        which the substrate law maps to assembly overlap ~1.0. That is the
        exact-duplicate clusters.

        With this weight, phon contributes ``phon_weight * k`` and its share
        becomes ``W / (W + F)``. Two words sharing all F features then have
        input overlap ``2F / (2W + 2F)``, so bounding that below the level where
        the substrate still preserves overlap is a matter of choosing W --
        W >= 6 puts the worst case under 0.25 at F = 2.

        THE COST IS REAL AND MUST BE MEASURED, not assumed away: grounding drive
        is what lets a word be placed from its features alone, which is the
        generalisation pathway. A W large enough to guarantee distinctness can
        make the representation phon-only, which scores beautifully on
        distinctness and has lost the thing the grounding was for.
        """
        phon = f"phon_{word}"
        size = max(1, int(round(float(getattr(self, "phon_weight", 1.0)) * self.k)))
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-phon-registration
        existing = self.brain.stimuli.get(phon)
        if existing is None:
            self.brain.add_stimulus(phon, size)
        elif existing.size != size:
            raise ValueError(f"phonological stimulus {phon!r} has size {existing.size}, requested {size}")
        self.stim_map[word] = phon
        return phon

    def _setup_areas(self):
        """Create the brain areas and register interarea inhibition."""
        for area_name in ALL_AREAS:
            self.brain.add_area(area_name, self.n, self.k, self.beta)

        # Wire the declared mutual-inhibition groups into the brain. These
        # were previously declared in core/areas.py and never registered, so
        # the one piece of interarea inhibition the model actually specifies
        # -- "the three ROLE areas are in mutual inhibition; this is the only
        # use of interarea inhibition in our model" (Mitropolsky &
        # Papadimitriou 2025, sec. 2.3) -- was inert, and role exclusivity was
        # enforced by a Python set instead.
        for group in MUTUAL_INHIBITION_GROUPS:
            present = [a for a in group if a in self.brain.areas]
            if len(present) > 1:
                self.brain.add_mutual_inhibition(present)

    def train(self, sentences: Optional[List[GroundedSentence]] = None,
              holdout_words: Optional[set] = None,
              train_prediction: bool = False,
              include_word_order: bool = True,
              include_morphology: bool = True,
              progress: Optional["TrainProgress"] = None):
        """Run all training phases.

        Args:
            sentences: Training sentences. If None, uses the default
                30-sentence corpus from training_data.py.
            holdout_words: Optional set of words to skip during lexicon
                training (for generalization testing).
            train_prediction: When True, run ``train_next_token`` after
                core grammar phases (required for dialogue / agent use).
            include_word_order: Run SEQ sequence-memorization phase.
            include_morphology: Run tense/mood/polarity/number/conjunctions.
            progress: Optional ``TrainProgress`` logger (streams to stderr).
        """
        from ..train_progress import current_progress

        prog = progress if progress is not None else current_progress()
        if sentences is None:
            sentences = create_training_sentences()

        from ..core.corpus_index import compile_corpus

        corpus_index = compile_corpus(self, sentences)

        raw_sents = [s.words for s in sentences]
        with prog.phase("ingest", f"{len(raw_sents)} sentences"):
            for sent in raw_sents:
                self.ingest_raw_sentence(sent)

        holdout = holdout_words or set()
        self.lexicon_holdouts = set(holdout)

        # Register any word the CORPUS grounds that the static vocabulary table
        # does not. Without this the lexicon is capped at
        # ``core.grounding.VOCABULARY`` -- 10 nouns and 8 verbs -- no matter how
        # rich the corpus is.
        #
        # The failure it caused was SILENT and produced plausible numbers. An
        # unregistered word still got a phon stimulus (``ingest_raw_sentence``)
        # and still classified correctly, but never entered ``core_lexicons``,
        # and ``NemoParser.parse`` skips a word with no core lexicon entry: it
        # returns None and the slot sequence never advances. Measured on a
        # rich_corpus run, `man`/`woman`/`child` were unparseable while `boy`
        # and `girl` worked, which silently invalidated the transfer arm of
        # `order_generalization.py` (task #34).
        #
        # Nothing new is needed to fix it -- GroundedSentence already carries a
        # GroundingContext per word, so corpus words have exactly the same
        # information the static table supplies. Registered BEFORE
        # ``train_lexicon`` so the compiled plan picks them up; holdouts are
        # excluded because their whole point is to be registered but untrained,
        # which ``train_lexicon`` handles.
        self._register_corpus_vocabulary(sentences)

        with prog.phase("lexicon"):
            self.train_lexicon(holdout_words=holdout, skip_known=True)
        if holdout_words:
            from ..acquisition.pos_inference import infer_holdout_categories

            infer_holdout_categories(self, set(holdout_words))
        with prog.phase("roles"):
            self.train_roles(sentences)
        # train_roles accumulated role-order evidence; commit the inferred
        # typology now so later phases and parse/generate use it. Guarded so a
        # corpus with no usable evidence leaves word_order_type untouched.
        self._update_word_order_from_evidence()
        with prog.phase("phrases"):
            self.train_phrases(sentences)
        if include_word_order:
            with prog.phase("word_order"):
                self.train_word_order(sentences)
        if include_morphology:
            with prog.phase("tense"):
                self.train_tense(raw_sents)
            with prog.phase("mood"):
                self.train_mood(raw_sents)
            with prog.phase("polarity"):
                self.train_polarity(raw_sents)
            with prog.phase("number"):
                self.train_number(raw_sents)
            with prog.phase("conjunctions"):
                self.train_conjunctions(raw_sents)

        if train_prediction:
            with prog.phase(
                "next_token",
                f"{len(corpus_index.transitions)} bridges",
            ):
                self.train_next_token(sentences, corpus_index=corpus_index)

    def compile_corpus(
        self,
        sentences: List[GroundedSentence],
    ) -> "CorpusIndex":
        """Compile sentences into a shared training index."""
        from ..core.corpus_index import compile_corpus
        return compile_corpus(self, sentences)

    def parse(self, words: List[str]) -> dict:
        """Parse a sentence through the full pipeline.

        1. Classify each word via differential readout
        2. Assign thematic roles via neural readout + mutual inhibition
        3. Identify phrase boundaries
        4. Detect tense, mood, polarity

        Args:
            words: List of word strings.

        Returns:
            {
                "categories": {word: "NOUN"/"VERB"/"ADJ"/...},
                "roles": {word: "AGENT"/"ACTION"/"PATIENT"/None},
                "phrases": {"NP": [...], "VP": [...], "PP": [...]},
                "tense": "PRESENT"/"PAST"/"FUTURE"/"PROGRESSIVE"/"PERFECT",
                "mood": "DECLARATIVE"/"INTERROGATIVE"/"IMPERATIVE",
                "polarity": "AFFIRMATIVE"/"NEGATIVE",
            }
        """
        result: dict = {"categories": {}, "roles": {}, "phrases": {}}

        # Step 1: Classify each word (with grounding for generalization)
        for word in words:
            grounding = self.word_grounding.get(word)
            cat, _ = self.classify_word_cached(word, grounding=grounding)
            result["categories"][word] = cat

        # Step 2: roles from THIS parse -- gate -> record -> recall.
        # `parse_roles_by_reconstruction` replaced `_assign_roles_neural` here
        # after the 2x2 measurement (40 held-out reversible probes, 5 seeds):
        # 1.0000 both voices against the margin route's 0.85 active / 0.50
        # passive, whose stored-lexicon term flips passives toward each word's
        # TRAINED majority role. The ERP runner deliberately still calls the
        # margin route: its thresholds are calibrated against it, and swapping
        # without re-calibration would silently shift every ERP number (#121).
        result["roles"], result["role_diagnostics"] = (
            self.parse_roles_by_reconstruction(words))

        # Step 3: Identify phrase boundaries
        result["phrases"] = self._identify_phrases(words, result["categories"])

        # Step 4: Detect tense, mood, polarity
        result["tense"] = self.detect_tense(words)
        result["mood"] = self.detect_mood(words)
        result["polarity"] = self.detect_polarity(words)

        return result

