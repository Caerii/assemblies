"""
CoreParserMixin -- core setup, training, classification, and batch parsing.

Extracted from the 44-area emergent NEMO parser (Mitropolsky & Papadimitriou
2025) on numpy_sparse.  Categories emerge from grounding patterns: a word
presented with VISUAL grounding forms its assembly in NOUN_CORE, a word with
MOTOR grounding forms in VERB_CORE, etc.

All operations compose existing assembly_calculus primitives:
    project, reciprocal_project, merge, sequence_memorize,
    readout_all, _snap

Architecture:
    Input:   PHON stimuli + grounding stimuli (VISUAL, MOTOR, PROPERTY, ...)
    Layer 1: 8 CORE areas + 2 LEX areas
    Layer 2: 6 ROLE areas + 3 SYNTACTIC areas
    Layer 3: 5 PHRASE areas + 3 VP_COMPONENT areas
    Control: SEQ, MOOD, TENSE, POLARITY, ERROR
    Advanced: CONTEXT, PRODUCTION, PREDICTION, DEP_CLAUSE

References:
    Mitropolsky, D. & Papadimitriou, C. H. (2025).
    "Simulated Language Acquisition with Neural Assemblies."
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.core.brain import Brain
from src.assembly_calculus.assembly import Assembly
from src.assembly_calculus.ops import (
    project, reciprocal_project, merge, sequence_memorize, _snap,
)
from src.assembly_calculus.readout import readout_all, Lexicon

from .areas import (
    ALL_AREAS, CORE_AREAS, CORE_TO_CATEGORY, CATEGORY_TO_CORE,
    GROUNDING_TO_CORE, THEMATIC_AREAS, PHRASE_AREAS,
    NOUN_CORE, VERB_CORE, ADJ_CORE, ADV_CORE,
    PREP_CORE, DET_CORE, PRON_CORE, CONJ_CORE,
    LEX_CONTENT, LEX_FUNCTION,
    ROLE_AGENT, ROLE_PATIENT,
    NP, VP, PP, ADJP, SENT, SEQ,
    SUBJ, OBJ,
    TENSE, MOOD, POLARITY,
    CONTEXT,
)
from .grounding import GroundingContext, VOCABULARY
from .training_data import GroundedSentence, create_training_sentences


# Modality field names on GroundingContext (order matters for dominant_modality)
_MODALITY_FIELDS = (
    "visual", "motor", "properties", "spatial",
    "social", "temporal", "emotional",
)

# Role annotation string -> brain area
_ROLE_MAP = {
    "agent": ROLE_AGENT,
    "patient": ROLE_PATIENT,
}

# Role brain area -> human-readable label
_ROLE_LABEL = {ROLE_AGENT: "AGENT", ROLE_PATIENT: "PATIENT"}


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
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    transitions: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int))
    category_transitions: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int))
    word_cooccurrence: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    word_as_pre_verb: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    word_as_post_verb: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    word_as_action: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    sentences_seen: int = 0


class CoreParserMixin:
    """Core mixin: setup, training phases, classification, batch parsing."""

    def __init__(self, n: int = 10000, k: int = 100, p: float = 0.05,
                 beta: float = 0.1, seed: int = 42, rounds: int = 10,
                 engine: str = "numpy_sparse",
                 vocabulary: Optional[Dict[str, GroundingContext]] = None):
        self.n = n
        self.k = k
        self.p = p
        self.beta = beta
        self.seed = seed
        self.rounds = rounds

        self.brain = Brain(
            p=p, save_winners=True, seed=seed, engine=engine,
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

        # Set of all created grounding stimulus names (to avoid duplicates)
        self._grounding_stim_names_set: set = set()

        # Distributional statistics for category inference from raw text
        self.dist_stats = DistributionalStats()

        # Inferred word order typology (set by train_word_order_typological)
        self.word_order_type: Optional[str] = None

        self._setup_areas()
        self._register_vocabulary(vocabulary or VOCABULARY)

    # ==================================================================
    # Setup
    # ==================================================================

    def _setup_areas(self):
        """Create all 44 brain areas."""
        for area_name in ALL_AREAS:
            self.brain.add_area(area_name, self.n, self.k, self.beta)

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
              holdout_words: Optional[set] = None):
        """Run all training phases.

        Args:
            sentences: Training sentences. If None, uses the default
                30-sentence corpus from training_data.py.
            holdout_words: Optional set of words to skip during lexicon
                training (for generalization testing).
        """
        if sentences is None:
            sentences = create_training_sentences()
        self.train_lexicon(holdout_words=holdout_words)
        self.train_roles(sentences)
        self.train_phrases(sentences)
        self.train_word_order(sentences)

        # Phase 5: Tense, mood, polarity, conjunctions
        raw_sents = [s.words for s in sentences]
        self.train_tense(raw_sents)
        self.train_mood(raw_sents)
        self.train_polarity(raw_sents)
        self.train_conjunctions(raw_sents)

    def train_lexicon(self, holdout_words: Optional[set] = None):
        """Phase 1: Grounded word learning.

        For each word, project phon + grounding features simultaneously
        into the appropriate core area with recurrence. The assembly that
        forms represents the word in its grammatical category area.


        Pattern follows readout.py:build_lexicon() — reset recurrent
        connections between words to prevent attractor carryover.

        Args:
            holdout_words: Optional set of words to skip during training.
                These words' grounding stimuli are still registered in the
                brain, so their features can drive generalization.
        """
        holdout = holdout_words or set()

        # Initialize lexicons for each core area
        for core_area in CORE_AREAS:
            self.core_lexicons[core_area] = {}

        for word, ctx in self.word_grounding.items():
            if word in holdout:
                continue

            core = GROUNDING_TO_CORE[ctx.dominant_modality]
            phon = self.stim_map[word]
            grounding_stims = self._grounding_stim_names(ctx)

            # Simultaneous projection: phon + grounding -> core area
            stim_dict = {phon: [core]}
            for gs in grounding_stims:
                stim_dict[gs] = [core]
            self.brain.project(stim_dict, {})
            if self.rounds > 1:
                self.brain.project_rounds(
                    target=core,
                    areas_by_stim=stim_dict,
                    dst_areas_by_src_area={core: [core]},
                    rounds=self.rounds - 1,
                )

            self.core_lexicons[core][word] = _snap(self.brain, core)

            # Reset recurrent connections for next word
            self.brain._engine.reset_area_connections(core)

    def train_roles(self, sentences: List[GroundedSentence]):
        """Phase 2: Role binding from annotated sentences.

        For each word with a role annotation (agent/patient):
        1. Activate word in its core area (project phon -> core)
        2. Fix the core assembly
        3. Project core -> role_area with recurrence
        4. Snapshot the role assembly

        Pattern follows parser.py:train_roles().
        """
        for role_area in THEMATIC_AREAS:
            if role_area not in self.role_lexicons:
                self.role_lexicons[role_area] = {}

        for sent in sentences:
            for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
                if role is None or role == "action":
                    continue
                role_area = _ROLE_MAP.get(role)
                if role_area is None:
                    continue
                if word not in self.stim_map:
                    continue

                core_area = GROUNDING_TO_CORE[ctx.dominant_modality]
                phon = self.stim_map[word]

                # Activate word in core area
                project(self.brain, phon, core_area, rounds=self.rounds)
                self.brain.areas[core_area].fix_assembly()

                # Project core -> role with recurrence
                for _ in range(self.rounds):
                    self.brain.project(
                        {},
                        {core_area: [role_area], role_area: [role_area]},
                    )

                asm = _snap(self.brain, role_area)
                self.role_lexicons[role_area][word] = asm

                self.brain.areas[core_area].unfix_assembly()
                self.brain._engine.reset_area_connections(role_area)

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

                # Merge subject + verb into VP
                vp_asm = merge(
                    self.brain, subj_core, VERB_CORE, VP,
                    rounds=self.rounds,
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

                # Reset VP connections for next sentence
                self.brain._engine.reset_area_connections(VP)

    def train_word_order(self, sentences: List[GroundedSentence]):
        """Phase 4: Word order via sequence memorization in SEQ area."""
        for sent in sentences:
            stim_seq = [
                self.stim_map[w] for w in sent.words
                if w in self.stim_map
            ]
            if stim_seq:
                sequence_memorize(
                    self.brain, stim_seq, SEQ,
                    rounds_per_step=self.rounds,
                    repetitions=5,
                    phase_b_ratio=0.5,
                    beta_boost=0.5,
                )

    # ==================================================================
    # Classification
    # ==================================================================

    def classify_word(self, word: str,
                      grounding: Optional[GroundingContext] = None,
                      ) -> Tuple[str, Dict[str, float]]:
        """Classify a word by differential readout across all core areas.

        Projects available stimuli (phon and/or grounding features) into
        each core area independently and measures overlap against that
        area's lexicon.  The core area with the highest top-1 readout
        score determines the word's category.

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

    def _assign_roles_neural(self, words: List[str],
                             categories: Dict[str, str]) -> Dict[str, Optional[str]]:
        """Neural role assignment via learned projections + mutual inhibition.

        Processes words left-to-right.  For each NOUN/PRON, projects the
        word's core assembly into each uninhibited role area with recurrence,
        then reads out against role_lexicons.  The best-scoring uninhibited
        role wins, and that role is then inhibited (mutual exclusion).

        This replaces the position-based heuristic with genuine neural
        readout against the role assemblies learned during training.

        Args:
            words: Sentence word list (ordered).
            categories: Pre-classified {word: category} from classify_word.

        Returns:
            {word: "AGENT"/"ACTION"/"PATIENT"/None} for each word.
        """
        roles: Dict[str, Optional[str]] = {}
        inhibited: set = set()

        for word in words:
            cat = categories.get(word)

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

            # Activate word in its core area
            core_area = self._word_core_area(word)
            project(self.brain, phon, core_area, rounds=self.rounds)
            self.brain.areas[core_area].fix_assembly()

            best_role_area: Optional[str] = None
            best_score = -1.0

            for role_area in [ROLE_AGENT, ROLE_PATIENT]:
                if role_area in inhibited:
                    continue

                lex = self.role_lexicons.get(role_area, {})
                if not lex:
                    continue

                # Project core -> role area with recurrence
                self.brain.project(
                    {}, {core_area: [role_area]},
                )
                if self.rounds > 1:
                    self.brain.project_rounds(
                        target=role_area,
                        areas_by_stim={},
                        dst_areas_by_src_area={
                            core_area: [role_area],
                            role_area: [role_area],
                        },
                        rounds=self.rounds - 1,
                    )

                asm = _snap(self.brain, role_area)
                overlaps = readout_all(asm, lex)
                score = overlaps[0][1] if overlaps else 0.0

                if score > best_score:
                    best_score = score
                    best_role_area = role_area

                self.brain._engine.reset_area_connections(role_area)

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
            cat, _ = self.classify_word(word, grounding=grounding)
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
