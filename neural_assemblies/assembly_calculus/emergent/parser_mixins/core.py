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
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.assembly import (
    Assembly, overlap as assembly_overlap,
)
from neural_assemblies.assembly_calculus.ops import (
    activate_assembly, project, merge, sequence_memorize, _snap,
)
from neural_assemblies.assembly_calculus.readout import readout_all, Lexicon

from ..core.areas import (
    ALL_AREAS, CORE_AREAS, CORE_TO_CATEGORY, GROUNDING_TO_CORE, THEMATIC_AREAS, VERB_CORE, DET_CORE, ROLE_AGENT, ROLE_PATIENT, ROLE_ACTION,
    MUTUAL_INHIBITION_GROUPS,
    VP, SEQ,
    FUNC_AUX, FUNC_DET, FUNC_COMP, FUNC_MARKER,
)
from ..core.grounding import GroundingContext, VOCABULARY
from ..curriculum.data import GroundedSentence, create_training_sentences

#: Projection rounds for MERGE into a multi-assembly area.
#: Below `self.rounds` because deep reinforcement builds one strong attractor
#: and merges the rest. On the ISOLATED primitive this is decisive: 94 merges
#: into an n=1000 area give pairwise overlap 0.752 at rounds 10, 0.078 at 3,
#: 0.051 at 1.
#:
#: NOT load-bearing on the parser's own path, and it is not what collapsed VP
#: -- that was `reset_area_connections(VP)`, see `train_phrases`. Swept on the
#: real path after removing the reset (seed 42, 32 constituents), the value
#: barely matters and is not even monotonic:
#:
#:     rounds    1      2      3      5     10
#:     overlap   0.299  0.444  0.562  0.390  0.406
#:     rank-1    0.781  0.781  0.781  0.781  0.781   (subject cue; flat)
#:
#: Kept at 2 for consistency with `_ROLE_BINDING_ROUNDS`, which faced the same
#: shared-area problem. Do not read the spread above as tuning signal: it is
#: one seed, and the differences are within what seed variation covers.
MERGE_ROUNDS = 2


# Modality field names on GroundingContext (order matters for dominant_modality)
_MODALITY_FIELDS = (
    "visual", "motor", "properties", "spatial",
    "social", "temporal", "emotional",
)

# Role annotation string -> brain area
# Weight of the structural (word-order / gating) prior relative to lexical
# binding evidence, which is an overlap in [0, 1]. Above 1.0 structure wins
# whenever lexical evidence is absent or weak, so a filler never seen in any
# role is still assigned systematically; a strong stored binding can still
# overturn a lower-ranked structural preference.
_STRUCTURAL_PRIOR = 1.2

# Additive smoothing applied when the per-role lexical margins are normalised
# into a distribution over the competing roles (see _assign_roles_neural). It
# damps the case where two margins differ only by measurement jitter, which is
# what a corpus with no lexical role preference produces. Measured over 8 seeds
# on a 4-noun corpus (balanced SVO accuracy / object-initial accuracy when the
# corpus makes each noun exclusive to one role):
#     eps = 0.00   0.995 / 1.000
#     eps = 0.05   0.995 / 1.000
#     eps = 0.10   1.000 / 0.812
#     eps = 0.15   0.938 / 0.500
# Above ~0.05 the smoothing starts eating the evidence it is meant to protect,
# so it is kept small and deliberately does not carry the decision.
_LEXICAL_SMOOTHING = 0.05

# Steps used to bind a filler into a role area.
#
# This is a *traversal* of an already-stabilized lexical assembly, not the
# formation of a new one. The two are different constants in the literature:
# Papadimitriou et al. (PNAS 2020) report "a stable assembly is formed after
# about T = 10 steps", and Mitropolsky & Collins & Papadimitriou (TACL 2021)
# converge project* in 10-20 firing epochs -- both for formation. Mitropolsky &
# Papadimitriou (2025) fire a word for tau = 2 steps to traverse a trained
# pathway, and 2 is also what measures best here: binding quality degrades
# monotonically with added recurrence (pairwise overlap between fillers 0.875
# at T=2 rising to 0.997 at T=20, with retrieval falling from 6/6 to 2/6),
# because many fillers share one role area and recurrence merges them.
_ROLE_BINDING_ROUNDS = 2

_ROLE_MAP = {
    "agent": ROLE_AGENT,
    "action": ROLE_ACTION,
    "patient": ROLE_PATIENT,
}

# Role brain area -> human-readable label
_ROLE_LABEL = {
    ROLE_AGENT: "AGENT",
    ROLE_ACTION: "ACTION",
    ROLE_PATIENT: "PATIENT",
}


def _int_defaultdict() -> "defaultdict":
    """Picklable factory for a nested ``defaultdict(int)``.

    A nested ``lambda: defaultdict(int)`` is stored on the instance as the
    outer defaultdict's ``default_factory``, and local lambdas cannot be
    pickled -- which broke checkpoint/disk-cache round-trips of any parser
    carrying dist_stats. A module-level function pickles by reference.
    """
    return defaultdict(int)


@dataclass
class DistributionalStats:
    """Distributional statistics for category inference from raw text.

    Tracks position distributions, transitions, and co-occurrences to
    infer word categories without grounding information. Ported from
    the distributional tracking pattern in learner.py.
    """
    word_count: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    position_counts: Dict[str, Dict[int, int]] = field(
        default_factory=lambda: defaultdict(_int_defaultdict))
    transitions: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int))
    category_transitions: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int))
    word_cooccurrence: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(_int_defaultdict))
    word_as_pre_verb: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    word_as_post_verb: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    word_as_action: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    # Counts of complete S/V/O surface permutations observed in ROLE-ANNOTATED
    # sentences, keyed by an order label from ``core.word_order.WORD_ORDERS``.
    # This is the only evidence that can separate a subject-initial order from
    # its object-initial twin (SVO/OVS, SOV/OSV, VSO/VOS): those pairs have
    # identical part-of-speech transition statistics. See
    # ``core/word_order.py`` for the identifiability argument.
    role_order_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    sentences_seen: int = 0


class CoreParserMixin:
    """Core mixin: setup, training phases, classification, batch parsing."""

    def __init__(self, n: int = 10000, k: int = 100, p: float = 0.05,
                 beta: float = 0.1, seed: int = 42, rounds: int = 10,
                 engine: str = "auto",
                 inference_rounds: Optional[int] = None,
                 bridge_rounds: Optional[int] = None,
                 fast_training: Optional[bool] = None,
                 norm_init: Optional[bool] = None,
                 vocabulary: Optional[Dict[str, GroundingContext]] = None):
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
        self.seed = seed
        self.rounds = train_r
        self.inference_rounds = infer_r
        self.bridge_rounds = bridge_r
        resolved_engine = resolve_engine(engine, n_hint=n)

        brain_kwargs = dict(
            p=p, save_winners=True, seed=seed,
            engine=resolved_engine, n_hint=n,
        )
        # Forward norm_init only when explicitly set so the Brain default (True)
        # is preserved otherwise. norm_init=False opts into the un-normalized
        # substrate -- used by the compiled-training-optimization tests, since
        # compiled training is disabled under norm_init (see
        # _compiled_training_enabled below).
        if norm_init is not None:
            brain_kwargs["norm_init"] = norm_init
        self.brain = Brain(**brain_kwargs)
        self.engine_name = getattr(self.brain._engine, "name", resolved_engine)

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

        self._setup_areas()
        self._register_vocabulary(vocabulary or VOCABULARY)

    # ==================================================================
    # Setup
    # ==================================================================

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

    def _register_vocabulary(self, vocab: Dict[str, GroundingContext]):
        """Register all vocabulary words and their grounding stimuli."""
        for word, ctx in vocab.items():
            # Phonological stimulus
            phon = f"phon_{word}"
            self.brain.add_stimulus(phon, self.k)
            self.stim_map[word] = phon
            self.word_grounding[word] = ctx

            # Grounding feature stimuli
            for mod in _MODALITY_FIELDS:
                for feat in getattr(ctx, mod):
                    stim_name = f"{mod}_{feat}"
                    if stim_name not in self._grounding_stim_names_set:
                        self.brain.add_stimulus(stim_name, self.k)
                        self._grounding_stim_names_set.add(stim_name)

    def _register_corpus_vocabulary(
        self, sentences: List[GroundedSentence],
    ) -> Set[str]:
        """Register corpus words the static vocabulary table does not cover.

        Returns the set of newly registered words (empty when the corpus adds
        nothing), so callers and tests can assert on coverage rather than infer
        it.

        A ``GroundedSentence`` carries a ``GroundingContext`` per word in
        ``contexts``, which is the same thing ``VOCABULARY`` supplies, so a
        corpus word needs nothing extra to be trainable. Words already
        registered keep their EXISTING context: the static table is
        hand-authored and is treated as authoritative where the two disagree,
        and re-registering would also re-add stimuli.

        Sentences whose ``contexts`` are missing or the wrong length are
        skipped rather than guessed at -- a word with no grounding cannot be
        trained into a core lexicon, and silently inventing one is how the bug
        this fixes stayed hidden.
        """
        added: Set[str] = set()
        for sent in sentences:
            contexts = getattr(sent, "contexts", None)
            if not contexts or len(contexts) != len(sent.words):
                continue
            for word, ctx in zip(sent.words, contexts):
                if ctx is None or word in self.word_grounding:
                    continue
                self._register_vocabulary({word: ctx})
                added.add(word)
        return added

    def _grounding_stim_names(self, ctx: GroundingContext) -> List[str]:
        """Return stimulus names for all grounding features in a context."""
        names = []
        for mod in _MODALITY_FIELDS:
            for feat in getattr(ctx, mod):
                names.append(f"{mod}_{feat}")
        return names

    def _word_core_area(self, word: str) -> str:
        """Return the core area name for a word based on its grounding."""
        ctx = self.word_grounding.get(word)
        if ctx is None:
            return DET_CORE
        return GROUNDING_TO_CORE[ctx.dominant_modality]

    # ==================================================================
    # Training
    # ==================================================================

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

    def _clear_core_activity(self, core_area: str) -> None:
        """Clear core winners without wiping learned connectomes or ID mapping."""
        self.brain.inhibit_areas([core_area])
        if self.brain._engine.is_fixed(core_area):
            self.brain._engine.unfix_assembly(core_area)

    def train_lexicon(
        self,
        holdout_words: Optional[set] = None,
        *,
        skip_known: bool = True,
        words: Optional[Set[str]] = None,
    ):
        """Phase 1: Grounded word learning.

        For each word, project phon + grounding features simultaneously
        into the appropriate core area with recurrence. The assembly that
        forms represents the word in its grammatical category area.

        SIMULTANEITY IS THE MECHANISM.  The word form and its grounding are
        projected in the SAME time step, so both drive the winner selection
        and the resulting assembly is potentiated from both.  That single
        assembly is what later lets the phon stimulus alone retrieve the word
        (the grounded half of the input is no longer needed once the synapses
        exist) and lets grounding alone generalise to a word never heard.
        Presenting them in separate steps would build two assemblies and no
        link.

        Pattern follows readout.py:build_lexicon() — clear core winners
        between words without wiping learned connectomes.  The distinction
        matters: winners must be cleared so the next word does not inherit
        this one's activity, but the connectome must NOT be reset, because
        every word in a category shares one core area and the whole point is
        that they accumulate there.

        Holdout words have their grounding stimuli registered but never
        trained.  That is what makes the generalisation test meaningful: the
        features exist and can drive the area, but nothing about the specific
        word was ever written into the connectome.

        Args:
            holdout_words: Optional set of words to skip during training.
                These words' grounding stimuli are still registered in the
                brain, so their features can drive generalization.
            skip_known: Skip words already present in ``core_lexicons``.
            words: When set, only train words in this subset (e.g. corpus vocab).
        """
        from ..core.areas import CORE_AREAS, CORE_TO_CATEGORY
        from ..training.batch import BatchProjector
        from ..training.compiled import compiled_topology, lexicon_topology_spec
        from ..training.compiler import compile_lexicon_plan
        from ..training.linker import link_lexicon_topology

        holdout = holdout_words or set()
        batch = BatchProjector(self)

        for core_area in CORE_AREAS:
            if core_area not in self.core_lexicons:
                self.core_lexicons[core_area] = {}

        compiled_enabled = getattr(self, "_compiled_training_enabled", True)
        plan = compile_lexicon_plan(
            self,
            holdout_words=holdout,
            skip_known=skip_known,
            words=words,
        )
        if not plan.lexicon_ops:
            return

        if compiled_enabled:
            link_lexicon_topology(self, plan)
            plan.topology = lexicon_topology_spec(self, plan.core_areas)
            use_compiled = plan.topology.ready
            prev_fidelity = self.brain.projection_fidelity
            if self.fast_training and use_compiled:
                self.brain.projection_fidelity = "compiled"
            try:
                with compiled_topology(self, plan.topology):
                    for op in plan.lexicon_ops:
                        ctx = self.word_grounding[op.word]
                        batch.apply_lexicon_word(
                            op.word, ctx, op.core_area, rounds=op.rounds,
                        )
                        self.core_lexicons[op.core_area][op.word] = _snap(
                            self.brain, op.core_area,
                        )
                        self._category_cache[op.word] = CORE_TO_CATEGORY[
                            op.core_area
                        ]
            finally:
                self.brain.projection_fidelity = prev_fidelity
            return

        pending: Dict[str, List[tuple]] = {c: [] for c in CORE_AREAS}
        for op in plan.lexicon_ops:
            ctx = self.word_grounding[op.word]
            pending[op.core_area].append((op.word, ctx))

        for core_area in CORE_AREAS:
            for word, ctx in pending[core_area]:
                batch.apply_lexicon_word(
                    word, ctx, core_area, rounds=self.rounds,
                )
                self.core_lexicons[core_area][word] = _snap(self.brain, core_area)
                self._category_cache[word] = CORE_TO_CATEGORY[core_area]

    def train_roles(self, sentences: List[GroundedSentence]):
        """Phase 2: Role binding from annotated sentences.

        For each word with a role annotation (agent/patient):
        1. Activate word in its core area (project phon -> core)
        2. Fix the core assembly
        3. Project core -> role_area with recurrence
        4. Snapshot the role assembly

        Pattern follows parser.py:train_roles().

        Side effect: each annotated sentence also contributes one observation to
        ``dist_stats.role_order_counts`` (see ``record_role_order_evidence``).
        That is the evidence ``infer_word_order`` needs to tell an
        object-initial typology from its subject-initial twin; part-of-speech
        transitions cannot.
        """
        for role_area in THEMATIC_AREAS:
            if role_area not in self.role_lexicons:
                self.role_lexicons[role_area] = {}

        for sent in sentences:
            self.record_role_order_evidence(sent.words, sent.roles)
            for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
                # The verb binds into ROLE_ACTION like any other constituent.
                # Skipping it here left the verb outside the role system, so
                # its position could not be learned.
                if role is None:
                    continue
                role_area = _ROLE_MAP.get(role)
                if role_area is None:
                    continue
                if word not in self.stim_map:
                    continue

                core_area = GROUNDING_TO_CORE[ctx.dominant_modality]
                phon = self.stim_map[word]

                # Activate the word in its core area. Prefer the stabilized
                # lexicon assembly: re-projecting phon -> core carries
                # plasticity, so the core assembly drifts between the moment a
                # role binding is stored and the moment it is read back, and
                # retrieval then misses. train_lexicon has already converged
                # these, so replay the snapshot instead.
                stored_core = self.core_lexicons.get(core_area, {}).get(word)
                if stored_core is not None:
                    activate_assembly(self.brain, stored_core)
                else:
                    project(self.brain, phon, core_area, rounds=self.rounds)
                self.brain.areas[core_area].fix_assembly()

                # Canonical projection: round 1 is input-driven, so the role
                # assembly is a function of the filler, then a short recurrent
                # tail stabilizes it (see _ROLE_BINDING_ROUNDS).
                #
                # The previous code recurred from round 1 for self.rounds
                # steps and then called reset_area_connections(role_area).
                # Zeroing the connectome left every candidate neuron at equal
                # input, so the deterministic index tie-break in winner
                # selection picked the *same* k neurons for every word -- all
                # stored role assemblies were literally the identical winner
                # set (pairwise overlap 1.000). Readout could not discriminate,
                # and role assignment silently fell through to word order.
                self.brain.project({}, {core_area: [role_area]})
                for _ in range(_ROLE_BINDING_ROUNDS - 1):
                    self.brain.project(
                        {}, {core_area: [role_area], role_area: [role_area]},
                    )

                asm = _snap(self.brain, role_area)
                self.role_lexicons[role_area][word] = asm

                self.brain.areas[core_area].unfix_assembly()

        # Learn gating patterns from function word → role co-occurrences
        self._learn_gating_patterns(sentences)

    def _func_subcat_of(self, word: str) -> Optional[str]:
        """Function-word subcategory for `word`, or None if it has none.

        Prefers the distributionally learned frame subcategory. Falls back to
        the grounding signature so a role marker such as "by" -- which carries
        spatial grounding and therefore is not an ungrounded function word --
        is still recognised as a MARKER rather than dropping to None.
        """
        sc = (self.get_func_subcategory(word)
              if hasattr(self, "get_func_subcategory") else None)
        if sc is not None:
            return sc
        ctx = self.word_grounding.get(word)
        if ctx is None:
            return None
        if ctx.spatial:
            return FUNC_MARKER
        if not ctx.is_grounded:
            return FUNC_DET
        return None

    def _learn_gating_patterns(self, sentences: List[GroundedSentence]):
        """Learn gating patterns from how role assignment ORDER changes
        when different function word types are present in a sentence.

        The key insight: function words don't just affect the next word —
        they change the entire role assignment pattern for the sentence.
        "was" reverses first-noun=AGENT to first-noun=PATIENT.
        "by" after passive marks upcoming-noun=AGENT.

        Method:
        1. For each sentence, identify which ungrounded words are present
        2. Extract the role assignment order (1st noun role, 2nd noun role)
        3. Group by function word sub-type
        4. The dominant role order for each sub-type becomes its gating pattern

        This implements Stage 2: learning processing mode shifts triggered
        by function words, analogous to Broca's area learning top-down
        gating through procedural memory (basal ganglia).
        """
        # A function word only earns a gating pattern if its PRESENCE changes
        # the role order relative to its ABSENCE. Scoring the marginal
        # distribution instead lets every function word in a passive sentence
        # -- including "the" -- inherit the passive role order, and once "the"
        # crosses the confidence threshold every sentence gets reversed.
        # So we track both conditions per subcategory and compare them.
        subcat_present: Dict[str, List[bool]] = defaultdict(list)
        subcat_absent: Dict[str, List[bool]] = defaultdict(list)
        all_subcats: Set[str] = set()

        _subcat_of = self._func_subcat_of

        # First pass: which subcategories exist at all, and per sentence,
        # whether the sentence is patient-first and which subcats it contains.
        sentence_facts: List[Tuple[bool, Set[str]]] = []
        for sent in sentences:
            role_order = []
            for word, role in zip(sent.words, sent.roles):
                if role == "agent":
                    role_order.append(ROLE_AGENT)
                elif role == "patient":
                    role_order.append(ROLE_PATIENT)
            if not role_order:
                continue
            patient_first = role_order[0] == ROLE_PATIENT
            present: Set[str] = set()
            for word in sent.words:
                ctx = self.word_grounding.get(word)
                if ctx is not None and not ctx.is_grounded:
                    sc = _subcat_of(word)
                    if sc is not None:
                        present.add(sc)
            all_subcats |= present
            sentence_facts.append((patient_first, present))

        for patient_first, present in sentence_facts:
            for sc in all_subcats:
                if sc in present:
                    subcat_present[sc].append(patient_first)
                else:
                    subcat_absent[sc].append(patient_first)

        for sc in all_subcats:
            pres = subcat_present.get(sc, [])
            absent = subcat_absent.get(sc, [])
            if not pres:
                continue
            p_present = sum(pres) / len(pres)
            if absent:
                p_absent = sum(absent) / len(absent)
                effect = p_present - p_absent
            else:
                # Present in every sentence, so it cannot explain any variation
                # in role order. "the" occurs in both voices and must score 0
                # here; otherwise a passive-heavy corpus teaches the determiner
                # to reverse roles and every active sentence parses backwards.
                p_absent = float("nan")
                effect = 0.0
            reverses_roles = effect > 0.5
            role_order = ([ROLE_PATIENT, ROLE_AGENT] if p_present > 0.5
                          else [ROLE_AGENT, ROLE_PATIENT])
            self.learned_gating[sc] = {
                "role_order": role_order,
                "reverses_roles": reverses_roles,
                # Confidence is the contrastive effect size, not the raw rate,
                # so a word that appears in both voices scores ~0.
                "confidence": float(max(0.0, effect)),
                "n_examples": len(pres),
                "n_contrast": len(absent),
                "p_present": float(p_present),
                "p_absent": float(p_absent),
                "clause_boundary": sc == FUNC_COMP,
            }


    def train_phrases(self, sentences: List[GroundedSentence]):
        """Phase 3: Phrase structure via merge operations.

        For transitive sentences, merge subject and verb core assemblies
        into the VP area using the merge() operation.
        """
        for sent in sentences:
            subj_word = None
            verb_word = None
            obj_word = None

            for word, role in zip(sent.words, sent.roles):
                if role == "agent":
                    subj_word = word
                elif role == "action":
                    verb_word = word
                elif role == "patient":
                    obj_word = word

            if subj_word and verb_word:
                subj_core = self._word_core_area(subj_word)

                # Activate both source assemblies
                project(
                    self.brain, self.stim_map[subj_word],
                    subj_core, rounds=self.rounds,
                )
                project(
                    self.brain, self.stim_map[verb_word],
                    VERB_CORE, rounds=self.rounds,
                )

                # Merge subject + verb into VP.
                #
                # Shallow rounds, matching `_ROLE_BINDING_ROUNDS` above and for
                # the same documented reason: recurrence merges assemblies that
                # share an area. On the ISOLATED primitive this is decisive --
                # 94 merges into an n=1000 area give mean pairwise overlap
                # 0.752 at rounds=10 but 0.051 at rounds=1 (area size is the
                # other lever: n=3000 -> 0.231, n=10000 -> 0.002).
                vp_asm = merge(
                    self.brain, subj_core, VERB_CORE, VP,
                    rounds=MERGE_ROUNDS,
                )
                vp_key = f"{subj_word}_{verb_word}"
                self.vp_assemblies[vp_key] = vp_asm

                if obj_word:
                    obj_core = self._word_core_area(obj_word)
                    project(
                        self.brain, self.stim_map[obj_word],
                        obj_core, rounds=self.rounds,
                    )
                    # Extend VP with object via additional projection
                    self.brain.areas[obj_core].fix_assembly()
                    for _ in range(self.rounds):
                        self.brain.project(
                            {},
                            {obj_core: [VP], VP: [VP]},
                        )
                    self.brain.areas[obj_core].unfix_assembly()

                    vp_key_full = f"{subj_word}_{verb_word}_{obj_word}"
                    self.vp_assemblies[vp_key_full] = _snap(self.brain, VP)

                # NO reset_area_connections(VP) between sentences.
                #
                # It used to be here, "to reset VP connections for the next
                # sentence", and it was the cause of the phrase-structure
                # collapse -- the SAME failure already documented and fixed for
                # role areas above (see the note by `self.brain.project({},
                # {core_area: [role_area]})`). Resetting empties the sparse
                # connectome, so on the next merge every candidate neuron in VP
                # has equal input and the deterministic index tie-break in
                # winner selection hands back the same k neurons regardless of
                # which words were merged.
                #
                # Measured paired over seeds 42/7/123, means; the arms differ
                # only in this one line, and every seed agrees:
                #
                #                    pairwise    VP    rank-1 retrieval
                #                     overlap     w      subj    verb
                #   with the reset      1.000    74     0.156   0.094
                #   without it          0.492   752     0.781   0.646
                #
                # w is how many VP neurons have EVER fired: with the reset only
                # ~74 of 1000 were ever recruited, so k=50 winners drawn from a
                # 74-neuron pool cannot be distinct -- every phrase was the same
                # assembly, pairwise overlap 1.000.
                #
                # Rank-1 cues VP with ONE parent and asks whether the top-
                # scoring stored constituent contains that parent; chance is
                # 0.031 over the 32-constituent pool. The reset arm's 0.156 is
                # not partial credit, it is a tie-break artifact: when every
                # assembly is identical, `max` returns whichever key it sees
                # first. This is the property merge exists to provide (ops.merge,
                # [PNAS20] sec 3 -- the merged assembly responds to EITHER
                # source alone); without it a constituent is not retrievable
                # from its parts and nothing can be composed further from it.
                #
                # Carrying VP's connectome across sentences is not a leak to be
                # cleaned up; it is where the phrase lexicon LIVES. The role
                # areas reached the same conclusion for the same reason.

    def train_word_order(
        self,
        sentences: List[GroundedSentence],
        *,
        repetitions: Optional[int] = None,
    ):
        """Phase 4: Word order via sequence memorization in SEQ area."""
        from ..training.perf import sequence_rounds_per_step

        reps = repetitions
        if reps is None:
            reps = 1 if self.fast_training else 2
        seq_rounds = sequence_rounds_per_step(
            self.rounds, self.inference_rounds, fast=self.fast_training,
        )
        for sent in sentences:
            stim_seq = [
                self.stim_map[w] for w in sent.words
                if w in self.stim_map
            ]
            if stim_seq:
                sequence_memorize(
                    self.brain, stim_seq, SEQ,
                    rounds_per_step=seq_rounds,
                    repetitions=reps,
                    phase_b_ratio=0.5,
                    beta_boost=0.5,
                )

    def classify_word_cached(
        self,
        word: str,
        grounding: Optional[GroundingContext] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Fast classification with lexicon-grounding-distributional cache."""
        cached = self._category_cache.get(word)
        if cached is not None:
            return cached, {}

        if hasattr(self, "_bootstrap_categories") and word in self._bootstrap_categories:
            cat = self._bootstrap_categories[word]
            self._category_cache[word] = cat
            return cat, {}

        if hasattr(self, "_dist_categories") and word in self._dist_categories:
            cat = self._dist_categories[word]
            self._category_cache[word] = cat
            return cat, {}

        from ..acquisition.pos_inference import (
            classify_word_bootstrapped,
            is_word_in_lexicon,
        )

        ctx = grounding if grounding is not None else self.word_grounding.get(word)
        if not is_word_in_lexicon(self, word):
            cat, scores = classify_word_bootstrapped(self, word, ctx)
            if cat != "UNKNOWN":
                self._category_cache[word] = cat
            return cat, scores

        if self.dist_stats.word_count.get(word, 0) > 0:
            cat, scores = self.classify_distributional(word)
            if cat != "UNKNOWN":
                self._category_cache[word] = cat
                return cat, scores

        cat, scores = self.classify_word(word, grounding=grounding)
        if cat != "UNKNOWN":
            self._category_cache[word] = cat
        return cat, scores

    def _invalidate_category_cache(self, word: Optional[str] = None) -> None:
        if word is None:
            self._category_cache.clear()
        else:
            self._category_cache.pop(word, None)

    # ==================================================================
    # Classification
    # ==================================================================

    def classify_word(
        self,
        word: str,
        grounding: Optional[GroundingContext] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Classify a word by differential readout across all core areas.

        Projects available stimuli (phon and/or grounding features) into
        each core area independently and measures overlap against that
        area's lexicon.  The core area with the highest top-1 readout
        score determines the word's category.

        WHY "DIFFERENTIAL".  There is no classifier and no decision boundary.
        The same input is offered to all eight core areas, each of which was
        shaped by a different grounding modality during training, and the
        category is simply whichever area RECOGNISES it -- produces an
        assembly overlapping something it already stores.  Categorisation is
        thus a competition between populations, which is why it degrades
        gracefully (scores tail off) rather than flipping at a threshold.

        Two details that look like implementation noise but are not:

        * ``reset_area_connections`` is called on each core area before its
          probe.  Without it, whichever area was probed first would still be
          holding its recurrent attractor and would answer to anything;
          resetting makes the eight probes independent and therefore
          comparable.  It also means classification is DESTRUCTIVE of
          recurrent weights in every core area -- do not interleave it with
          training and expect training to be unaffected.
        * The score compared is ``overlaps[0][1]``, the single best matching
          word in that area, not a mean over the lexicon.  The question being
          asked is "does this area contain something this looks like?", and
          an average would be dominated by the many unrelated words each area
          holds.

        A zero across all areas is not "no category" but "no evidence" --
        hence the fall back to distributional classification, which asks a
        different question (what does this word's CONTEXT look like) and can
        answer for words that were never grounded at all.

        When grounding is provided, grounding feature stimuli are projected
        alongside phon.  This enables generalization: even for unseen words
        whose phon stimulus was never trained, shared grounding features
        (e.g. "visual_ANIMAL") drive assemblies toward the correct core area.

        Args:
            word: Word string to classify.
            grounding: Optional grounding context.  If None and the word is
                in word_grounding, no grounding features are used (phon-only,
                backward-compatible).  Pass explicitly for generalization.

        Returns:
            (category_label, {core_area: best_overlap_score})
        """
        phon = self.stim_map.get(word)
        if phon is None and grounding is None:
            # Fall back to distributional classification if available
            if self.dist_stats.word_count.get(word, 0) > 0:
                return self.classify_distributional(word)
            return "UNKNOWN", {}

        scores: Dict[str, float] = {}

        for core_area in CORE_AREAS:
            lexicon = self.core_lexicons.get(core_area, {})
            if not lexicon:
                scores[core_area] = 0.0
                continue

            self.brain._engine.reset_area_connections(core_area)

            # Build stimulus dict: phon + grounding features
            stim_dict: Dict[str, List[str]] = {}
            if phon:
                stim_dict[phon] = [core_area]
            if grounding:
                for gs in self._grounding_stim_names(grounding):
                    if gs in self._grounding_stim_names_set:
                        stim_dict[gs] = [core_area]

            if not stim_dict:
                scores[core_area] = 0.0
                continue

            # Project stimuli into core area with recurrence.
            # Use project_rounds fast path (saves one winner set, not one
            # per round) to avoid exhausting the area's neuron pool.
            self.brain.project(stim_dict, {})
            if self.rounds > 1:
                self.brain.project_rounds(
                    target=core_area,
                    areas_by_stim=stim_dict,
                    dst_areas_by_src_area={core_area: [core_area]},
                    rounds=self.rounds - 1,
                )

            asm = _snap(self.brain, core_area)

            # Readout against this area's lexicon
            overlaps = readout_all(asm, lexicon)
            scores[core_area] = overlaps[0][1] if overlaps else 0.0

        if not scores or max(scores.values()) == 0.0:
            # Fall back to distributional classification
            if self.dist_stats.word_count.get(word, 0) > 0:
                return self.classify_distributional(word)
            return "UNKNOWN", scores

        best_area = max(scores, key=scores.get)
        return CORE_TO_CATEGORY[best_area], scores

    # ==================================================================
    # Parsing
    # ==================================================================

    def _detect_passive(self, words: List[str],
                        categories: Dict[str, str]) -> bool:
        """Detect passive voice: 'was/were' before the main verb."""
        for i, word in enumerate(words):
            if word in ("was", "were"):
                # Check if next content word is a verb
                for j in range(i + 1, len(words)):
                    if categories.get(words[j]) == "VERB":
                        return True
                    if categories.get(words[j]) in ("NOUN", "PRON"):
                        break
        return False

    def constituent_role_order(self) -> List[str]:
        """Default role ranking implied by the inferred CONSTITUENT ORDER.

        This is the *unmarked* ranking for the language: which thematic role
        the first available noun takes when nothing else intervenes.  It is
        derived from ``self.word_order_type``, which ``infer_word_order``
        learned from the corpus, via the definitional mapping in
        ``core.word_order`` -- object-initial typologies (OSV, OVS, VOS) rank
        PATIENT first, the other three rank AGENT first.

        CONSTITUENT ORDER IS NOT VOICE.  This method answers "in this language,
        does the object normally come first?".  It says nothing about whether a
        particular sentence is passive; that is the separate, learned-gating
        mechanism in ``_determine_role_order``, which flips whatever ranking
        this returns.  Keeping the two apart matters because they compose: a
        passive clause in an OVS language is agent-initial.

        When no typology has been inferred the AGENT-first ranking is returned.
        That is a fallback prior, not a finding.
        """
        from ..core.word_order import is_object_initial, WORD_ORDERS

        order = getattr(self, "word_order_type", None)
        if order in WORD_ORDERS and is_object_initial(order):
            return [ROLE_PATIENT, ROLE_AGENT]
        return [ROLE_AGENT, ROLE_PATIENT]

    def _determine_role_order(self, words: List[str],
                              categories: Dict[str, str],
                              ) -> Tuple[List[str], bool]:
        """Determine role assignment order for one sentence.

        Two independent mechanisms, composed in this order:

        Stage 0 (constituent order): the unmarked ranking for the language,
        from the inferred typology -- see ``constituent_role_order``.  This is
        a property of the CORPUS, constant across its sentences.

        Stage 1 (voice gating, ELAN ~180ms): if a function-word sub-type
        present in THIS sentence has learned gating that reverses roles, the
        Stage 0 ranking is FLIPPED.  This is a property of the SENTENCE.

        The flip is deliberately expressed as a reversal of the Stage 0 result
        rather than a hardcoded ``[PATIENT, AGENT]``: a passive in an
        object-initial language must come out agent-first, and hardcoding the
        pair would silently give the wrong answer there.

        There is deliberately no spelling-based passive fallback: if the corpus
        contained no voice alternation, the parser has not learned one and
        should not pretend otherwise.  ``_detect_passive()`` remains available
        for diagnostics only.

        Args:
            words: Sentence word list.
            categories: Pre-classified {word: category}.

        Returns:
            (role_order_default, is_passive).
        """
        # Stage 0: unmarked ranking from the inferred constituent order.
        base_order = self.constituent_role_order()

        # Stage 1: voice gating learned from function-word sub-categories.
        if self.learned_gating:
            for word in words:
                # Get sub-category (from frame analysis or cache)
                subcat = (self.get_func_subcategory(word)
                          if hasattr(self, 'get_func_subcategory') else None)
                if subcat is None:
                    continue

                gating = self.learned_gating.get(subcat)
                if gating is None:
                    continue

                # If this sub-type reverses roles with high confidence,
                # flip the constituent-order ranking.
                if (gating.get("reverses_roles", False)
                        and gating.get("confidence", 0) > 0.5):
                    return list(reversed(base_order)), True

        return base_order, False

    def _role_binding_margin(self, word: str, core_area: str,
                             role_area: str) -> float:
        """Lexical evidence that `word` was bound into `role_area`, in [0, 1].

        Requires the core assembly for `word` to be active and FIXED.

        Why this is not simply ``overlap(projection, stored[word])``.  Every
        filler that is bound into a role area lands in that area's dominant
        attractor: measured on a 4-noun corpus, two DIFFERENT words' stored
        AGENT assemblies overlap 0.70-0.85, while the same word's AGENT and
        PATIENT assemblies overlap 0.00 -- but that last number is separation
        between two different brain areas' neuron index sets, which is true by
        construction and carries no learned information.  The consequence is
        that ``overlap(projection, stored[word])`` returns ~0.9 for EVERY
        candidate role, so the term cannot discriminate; it also survives
        ``reset_area_connections`` almost unchanged, i.e. it is nearly
        independent of what was learned.

        The discriminable part is the residual above the area-generic
        attractor: how much better the projection matches THIS word's stored
        binding than it matches the other fillers stored in the same area.
        That residual does collapse when the learned weights are wiped, which
        is the property a read-out of a learned binding must have.

        The projection uses ``_ROLE_BINDING_ROUNDS`` -- the same protocol that
        wrote the binding -- and runs with plasticity disabled so that reading
        a role does not rewrite it.  (The previous code ran one round, then
        called ``reset_area_connections(role_area)``, which destroyed the
        learned core->role weights as a side effect of parsing.)
        """
        lex = self.role_lexicons.get(role_area, {})
        stored = lex.get(word)
        if stored is None:
            return 0.0

        with self.brain.frozen():
            self.brain.project({}, {core_area: [role_area]})
            for _ in range(_ROLE_BINDING_ROUNDS - 1):
                self.brain.project(
                    {}, {core_area: [role_area], role_area: [role_area]},
                )
            asm = _snap(self.brain, role_area)

        own = assembly_overlap(asm, stored)
        others = [assembly_overlap(asm, a)
                  for other, a in lex.items() if other != word]
        if not others:
            return own
        base = sum(others) / len(others)
        return max(0.0, (own - base) / max(1e-6, 1.0 - base))

    def _assign_roles_neural(self, words: List[str],
                             categories: Dict[str, str],
                             filler_word: Optional[str] = None,
                             filler_role: Optional[str] = None,
                             ) -> Dict[str, Optional[str]]:
        """Neural role assignment via learned projections + mutual inhibition.

        Two-stage function word processing:
        1. ELAN-like rapid sub-categorization determines role order
           (learned gating from training, or hardcoded passive detection)
        2. Left-to-right role assignment with mutual inhibition

        For each NOUN/PRON, projects the word's core assembly into each
        uninhibited role area with recurrence, reads out against
        role_lexicons. The best-scoring uninhibited role wins, then
        that role is inhibited (mutual exclusion).

        Args:
            words: Sentence word list (ordered).
            categories: Pre-classified {word: category} from classify_word.
            filler_word: Optional word displaced from canonical position
                (e.g., antecedent of a relative clause).
            filler_role: Role to assign to filler_word ("AGENT" or "PATIENT").

        Returns:
            {word: "AGENT"/"ACTION"/"PATIENT"/None} for each word.
        """
        roles: Dict[str, Optional[str]] = {}
        inhibited: set = set()

        # Pre-assign filler if provided (filler-gap binding)
        if filler_word and filler_role:
            roles[filler_word] = filler_role
            role_area = (ROLE_AGENT if filler_role == "AGENT"
                         else ROLE_PATIENT)
            inhibited.add(role_area)

        # Determine role order: learned gating (Stage 1) or rules (Stage 2)
        role_order_default, is_passive = self._determine_role_order(
            words, categories)

        # A role MARKER (the "by" of a passive) signals that the noun after it
        # takes the reversed role. The marker is identified by its learned
        # distributional subcategory, not by spelling.
        after_marker = False

        for word in words:
            cat = categories.get(word)
            subcat = self._func_subcat_of(word)

            if is_passive and subcat == FUNC_MARKER:
                after_marker = True
                roles[word] = None
                continue

            # Function words carry no thematic role. Identified by learned
            # subcategory (AUX/COMP/DET), not by a literal word list.
            if subcat in (FUNC_AUX, FUNC_COMP, FUNC_DET):
                roles[word] = None
                continue

            if cat == "VERB":
                roles[word] = "ACTION"
                continue

            if cat not in ("NOUN", "PRON"):
                roles[word] = None
                continue

            phon = self.stim_map.get(word)
            if phon is None:
                roles[word] = None
                continue

            # After the role marker the voice flip is undone, so the language's
            # unmarked constituent-order ranking applies again (NOT a hardcoded
            # AGENT-first pair -- in an object-initial language the unmarked
            # ranking is PATIENT-first).
            if is_passive and after_marker:
                role_order = self.constituent_role_order()
            else:
                role_order = list(role_order_default)

            # Activate word in its core area
            core_area = self._word_core_area(word)
            infer_r = self.inference_rounds
            stored_core = self.core_lexicons.get(core_area, {}).get(word)
            if stored_core is not None:
                activate_assembly(self.brain, stored_core)
            else:
                project(self.brain, phon, core_area, rounds=infer_r)
            self.brain.areas[core_area].fix_assembly()

            best_role_area: Optional[str] = None
            best_score = -1.0

            candidates = [ra for ra in role_order if ra not in inhibited]
            margins = {
                ra: self._role_binding_margin(word, core_area, ra)
                for ra in candidates
            }
            # Turn the per-area margins into a distribution over the competing
            # roles. A filler bound in exactly one candidate role gets nearly
            # the whole mass there and can overturn the structural prior; a
            # filler bound in both (a corpus with no lexical role preference)
            # splits the mass and correctly leaves the decision to the prior.
            # The additive smoothing pulls the split toward uniform, so only a
            # clear asymmetry -- not the seed-to-seed jitter in the margins --
            # is worth more than the prior's rank gap.
            eps = _LEXICAL_SMOOTHING
            total = sum(margins.values()) + eps * len(margins)

            # Structural prior from word order / learned gating. It decays with
            # position in role_order so the preferred role wins when no lexical
            # binding exists -- this is what lets a filler never seen in any
            # role still be assigned systematically.
            for rank, role_area in enumerate(role_order):
                if role_area in inhibited:
                    continue

                prior = _STRUCTURAL_PRIOR * (0.5 ** rank)
                lexical = (((margins[role_area] + eps) / total)
                           if any(margins.values()) else 0.0)

                score = prior + lexical
                if score > best_score:
                    best_score = score
                    best_role_area = role_area

            self.brain.areas[core_area].unfix_assembly()

            if best_role_area is not None:
                roles[word] = _ROLE_LABEL[best_role_area]
                inhibited.add(best_role_area)
            else:
                roles[word] = None

        return roles

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

        # Step 2: Neural role assignment via learned projections
        result["roles"] = self._assign_roles_neural(words, result["categories"])

        # Step 3: Identify phrase boundaries
        result["phrases"] = self._identify_phrases(words, result["categories"])

        # Step 4: Detect tense, mood, polarity
        result["tense"] = self.detect_tense(words)
        result["mood"] = self.detect_mood(words)
        result["polarity"] = self.detect_polarity(words)

        return result

    def _identify_phrases(self, words: List[str],
                          categories: Dict[str, str]) -> dict:
        """Identify NP, VP, PP phrase boundaries from category sequence."""
        phrases: dict = {"NP": [], "VP": [], "PP": []}
        current_np: List[str] = []

        for word in words:
            cat = categories.get(word, "UNKNOWN")
            if cat in ("DET", "ADJ", "NOUN", "PRON"):
                current_np.append(word)
            else:
                if current_np:
                    phrases["NP"].append(current_np[:])
                    current_np = []
                if cat == "VERB":
                    phrases["VP"].append([word])
                elif cat == "PREP":
                    phrases["PP"].append([word])

        if current_np:
            phrases["NP"].append(current_np)

        return phrases
