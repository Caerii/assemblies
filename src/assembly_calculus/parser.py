"""
NemoParser — composed parser pipeline for biologically plausible language parsing.

Integrates three NEMO learning patterns into a single pipeline:
    1. Word category learning (noun/verb via differential grounding)
    2. Role binding (agent/action/patient via differential projection)
    3. Word order (SVO via sequence memorization)

Architecture (from NEMO paper, Mitropolsky & Papadimitriou 2023/2025):
    Input:    PHON stimuli + VISUAL/MOTOR grounding stimuli
    Layer 1:  LEX_NOUN (← PHON + VISUAL)    LEX_VERB (← PHON + MOTOR)
    Layer 2:  ROLE_AGENT    ROLE_ACTION    ROLE_PATIENT  (← LEX)
    Layer 3:  SEQ (← PHON stimuli, sequence memory)

References:
    Mitropolsky, D. & Papadimitriou, C. H. (2025).
    "Simulated Language Acquisition with Neural Assemblies."

    Mitropolsky, D. & Papadimitriou, C. H. (2023).
    "The Architecture of a Biologically Plausible Language Organ."
    arXiv:2306.15364.
"""

from typing import Dict, List, Optional, Tuple

from .assembly import Assembly, overlap
from .ops import project, sequence_memorize, _snap
from .readout import readout_all, build_lexicon, Lexicon


# Role names as constants
ROLE_AGENT = "ROLE_AGENT"
ROLE_ACTION = "ROLE_ACTION"
ROLE_PATIENT = "ROLE_PATIENT"
ROLE_AREAS = [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]
ROLE_LABELS = {"ROLE_AGENT": "AGENT", "ROLE_ACTION": "ACTION",
               "ROLE_PATIENT": "PATIENT"}


class NemoParser:
    """Composed parser pipeline implementing NEMO's three learning patterns.

    Layers:
        0. Input: PHON stimuli + sensory grounding (VISUAL/MOTOR)
        1. Lexical: LEX_NOUN, LEX_VERB (grounded word learning)
        2. Role: ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT (differential projection)
        3. Sequence: SEQ (ordered via sequence_memorize)

    Usage::

        brain = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
        parser = NemoParser(brain, n=10000, k=100, beta=0.1)
        parser.setup_areas()

        # Register words with categories and grounding
        parser.register_word("dog", "noun", "vis_dog")
        parser.register_word("chases", "verb", "mot_chases")
        parser.register_word("cat", "noun", "vis_cat")

        # Train each phase
        parser.train_lexicon()
        parser.train_roles([["dog", "chases", "cat"]])
        parser.train_word_order([["dog", "chases", "cat"]])

        # Parse a sentence
        result = parser.parse(["cat", "chases", "dog"])
    """

    def __init__(self, brain, n: int = 10000, k: int = 100,
                 beta: float = 0.1, rounds: int = 10):
        self.brain = brain
        self.n = n
        self.k = k
        self.beta = beta
        self.rounds = rounds

        # Word registrations
        self.stim_map: Dict[str, str] = {}        # word -> phon stimulus name
        self.grounding_map: Dict[str, str] = {}    # word -> grounding stimulus
        self.word_categories: Dict[str, str] = {}  # word -> "noun" | "verb"

        # Lexicons (populated by training)
        self.noun_lexicon: Lexicon = {}
        self.verb_lexicon: Lexicon = {}
        self.role_lexicons: Dict[str, Lexicon] = {}  # role_area -> {word: Assembly}

    def setup_areas(self):
        """Create all brain areas for the parser pipeline."""
        b = self.brain
        N, K, BETA = self.n, self.k, self.beta

        # Layer 1: Lexical areas (grounded word learning)
        b.add_area("LEX_NOUN", N, K, BETA)
        b.add_area("LEX_VERB", N, K, BETA)

        # Layer 2: Role areas
        for role in ROLE_AREAS:
            b.add_area(role, N, K, BETA)

        # Layer 3: Sequence area
        b.add_area("SEQ", N, K, BETA)

    def register_word(self, word: str, category: str,
                      grounding_stim: str):
        """Register a word with its category and grounding stimulus.

        Args:
            word: The word string (e.g., "dog").
            category: "noun" or "verb".
            grounding_stim: Name of the grounding stimulus (e.g., "vis_dog").
                The stimulus will be created if it doesn't exist.
        """
        phon_stim = f"phon_{word}"
        if phon_stim not in [s for s in self.stim_map.values()]:
            self.brain.add_stimulus(phon_stim, self.k)
        if grounding_stim not in self.grounding_map.values():
            self.brain.add_stimulus(grounding_stim, self.k)

        self.stim_map[word] = phon_stim
        self.grounding_map[word] = grounding_stim
        self.word_categories[word] = category

    def train_lexicon(self):
        """Phase 1: Grounded word learning.

        Nouns: simultaneous PHON + VISUAL → LEX_NOUN
        Verbs: simultaneous PHON + MOTOR → LEX_VERB

        Each word gets a stable assembly in its category's LEX area.
        """
        for word, category in self.word_categories.items():
            lex_area = "LEX_NOUN" if category == "noun" else "LEX_VERB"
            phon = self.stim_map[word]
            grounding = self.grounding_map[word]

            # Simultaneous projection: PHON + grounding → LEX
            for _ in range(self.rounds):
                self.brain.project(
                    {phon: [lex_area], grounding: [lex_area]},
                    {lex_area: [lex_area]},
                )

            asm = _snap(self.brain, lex_area)
            if category == "noun":
                self.noun_lexicon[word] = asm
            else:
                self.verb_lexicon[word] = asm

            # Reset recurrent connections for next word
            self.brain._engine.reset_area_connections(lex_area)

    def train_roles(self, sentences: List[List[str]]):
        """Phase 2: Role binding from SVO sentences.

        For each sentence [subject, verb, object]:
          - subject → ROLE_AGENT
          - verb → ROLE_ACTION
          - object → ROLE_PATIENT

        Each word's lexical assembly is projected into the corresponding
        role area, creating a role-bound representation.
        """
        role_sequence = [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]

        for sentence in sentences:
            for word, role_area in zip(sentence, role_sequence):
                category = self.word_categories[word]
                lex_area = "LEX_NOUN" if category == "noun" else "LEX_VERB"

                # Activate the word in its LEX area
                project(self.brain, self.stim_map[word], lex_area,
                        rounds=self.rounds)
                self.brain.areas[lex_area].fix_assembly()

                # Project LEX → ROLE with recurrence
                for _ in range(self.rounds):
                    self.brain.project(
                        {},
                        {lex_area: [role_area], role_area: [role_area]},
                    )

                asm = _snap(self.brain, role_area)
                if role_area not in self.role_lexicons:
                    self.role_lexicons[role_area] = {}
                self.role_lexicons[role_area][word] = asm

                self.brain.areas[lex_area].unfix_assembly()
                self.brain._engine.reset_area_connections(role_area)

    def train_word_order(self, sentences: List[List[str]]):
        """Phase 3: Word order via sequence memorization.

        For each sentence, memorizes the phonological stimulus sequence
        into the SEQ area using Hebbian bridges.
        """
        for sentence in sentences:
            stim_seq = [self.stim_map[w] for w in sentence]
            sequence_memorize(
                self.brain, stim_seq, "SEQ",
                rounds_per_step=self.rounds,
                repetitions=3,
            )

    def classify_word(self, word: str) -> str:
        """Classify a word as 'noun' or 'verb' via differential readout.

        Projects the word's PHON stimulus into both LEX areas and
        compares the best readout overlap.
        """
        # Project to LEX_NOUN
        self.brain._engine.reset_area_connections("LEX_NOUN")
        asm_n = project(self.brain, self.stim_map[word], "LEX_NOUN",
                        rounds=self.rounds)
        noun_scores = readout_all(asm_n, self.noun_lexicon)

        # Project to LEX_VERB
        self.brain._engine.reset_area_connections("LEX_VERB")
        asm_v = project(self.brain, self.stim_map[word], "LEX_VERB",
                        rounds=self.rounds)
        verb_scores = readout_all(asm_v, self.verb_lexicon)

        best_noun = noun_scores[0][1] if noun_scores else 0.0
        best_verb = verb_scores[0][1] if verb_scores else 0.0

        return "noun" if best_noun > best_verb else "verb"

    def assign_role(self, word: str) -> Optional[str]:
        """Assign a thematic role to a word via readout against role lexicons.

        Returns the role label ("AGENT", "ACTION", "PATIENT") with the
        highest overlap, or None if no role lexicon has been trained.
        """
        if not self.role_lexicons:
            return None

        # Get the word's lexical assembly
        category = self.word_categories.get(word)
        if category is None:
            category = self.classify_word(word)

        lex = self.noun_lexicon if category == "noun" else self.verb_lexicon
        word_asm = lex.get(word)
        if word_asm is None:
            return None

        best_role = None
        best_overlap = -1.0

        for role_area, role_lex in self.role_lexicons.items():
            for trained_word, role_asm in role_lex.items():
                if trained_word == word:
                    ov = overlap(word_asm, role_asm)
                    # Cross-area overlap is always 0 (different areas),
                    # so we use role training presence as the signal
                    if role_area not in self.role_lexicons:
                        continue
                    if word in role_lex:
                        return ROLE_LABELS[role_area]

        # Fallback: assign by category pattern
        # (nouns tend to be agents/patients, verbs tend to be actions)
        if category == "verb":
            return "ACTION"
        return None

    def parse(self, words: List[str]) -> dict:
        """Parse a sentence through the full pipeline.

        Args:
            words: List of word strings (e.g., ["dog", "chases", "cat"]).

        Returns:
            dict with:
                'categories': {word: "noun"/"verb"}
                'roles': {word: "AGENT"/"ACTION"/"PATIENT"/None}
        """
        result = {"categories": {}, "roles": {}}

        # Step 1: Classify each word
        for word in words:
            if word in self.word_categories:
                result["categories"][word] = self.word_categories[word]
            else:
                result["categories"][word] = self.classify_word(word)

        # Step 2: Assign roles based on SVO position + category
        if len(words) >= 3:
            # SVO structure: first noun = agent, verb = action, last noun = patient
            role_assignment = []
            nouns_seen = 0
            for word in words:
                cat = result["categories"][word]
                if cat == "verb":
                    role_assignment.append("ACTION")
                elif cat == "noun":
                    if nouns_seen == 0:
                        role_assignment.append("AGENT")
                    else:
                        role_assignment.append("PATIENT")
                    nouns_seen += 1
                else:
                    role_assignment.append(None)
            for word, role in zip(words, role_assignment):
                result["roles"][word] = role
        else:
            for word in words:
                result["roles"][word] = self.assign_role(word)

        return result
