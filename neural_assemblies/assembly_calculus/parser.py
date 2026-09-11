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

The scientific content of Layer 1 is DIFFERENTIAL GROUNDING: nouns and verbs
are not told apart by anything phonological, but by the fact that a noun's
PHON stimulus co-fires with a visual stimulus while a verb's co-fires with a
motor one.  Two sensory streams, two target areas, and the category falls out
of which area develops a stable assembly for the word.  This is the model's
answer to how a learner could acquire lexical categories without labels.

HOW MUCH OF THIS PIPELINE IS NEURAL.  Layer 1 is: ``classify_word`` decides
by projecting and reading out.  Layer 3 is: ``train_word_order`` really does
memorize the sequence into SEQ.  Layer 2 as consumed by :meth:`NemoParser.parse`
is NOT -- for sentences of three or more words, roles come from a hardcoded
SVO template, and the trained role areas and the SEQ area are not consulted at
parse time at all.  See :meth:`NemoParser.parse` and
:meth:`NemoParser.assign_role` for the details.  ``EmergentParser`` in
``assembly_calculus.emergent`` is the version where structure is learned
rather than assumed.

References:
    Mitropolsky, D. & Papadimitriou, C. H. (2025).
    "Simulated Language Acquisition with Neural Assemblies."

    Mitropolsky, D. & Papadimitriou, C. H. (2023).
    "The Architecture of a Biologically Plausible Language Organ."
    arXiv:2306.15364.
"""

from typing import Dict, List, Optional

from .ops import project, sequence_memorize, _snap
from .readout import readout_all, Lexicon


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

            # Reset recurrent connections for next word.
            #
            # LOAD-BEARING, not housekeeping. The loop above runs
            # `{lex_area: [lex_area]}` through the SLOW path, so unlike
            # training/batch.py it is not saved by `project_rounds`'s
            # `a != target` filter, and self-recurrence really does apply here.
            # Without this reset the first word's potentiated self-connections
            # win the k-WTA against every later word's stimulus and the whole
            # lexicon converges. Measured, n=1000 k=50 beta=0.1, nouns only:
            #
            #     M=16  with reset  spread 0.0460, 16 distinct assemblies
            #     M=16  without     spread 0.7632,  5 distinct
            #     M=64  with reset  spread 0.0499, 64 distinct
            #     M=64  without     spread 0.9153,  7 distinct
            #
            # (floor = k/n = 0.0500, so "with reset" is at chance separation --
            # exactly what a lexicon needs.)
            #
            # This is the same accumulated-potentiation collapse documented at
            # core/brain.py:project_rounds, and the general fix there is to
            # train shared areas FEED-FORWARD rather than to reset after every
            # item -- resetting also discards the recurrent structure, so these
            # assemblies have none. See
            # research/experiments/norm_init_recurrence_limit.py.
            #
            # Note this is NOT the failure mode where a mid-learning reset
            # zeroes the connectome and the index tie-break returns identical
            # winners: each word here has its own grounding stimulus, so the
            # tie-break is never reached.
            self.brain.reset_area_connections(lex_area)

    def train_roles(self, sentences: List[List[str]]):
        """Phase 2: Role binding from SVO sentences.

        For each sentence [subject, verb, object]:
          - subject → ROLE_AGENT
          - verb → ROLE_ACTION
          - object → ROLE_PATIENT

        Each word's lexical assembly is projected into the corresponding
        role area, creating a role-bound representation.

        BROUGHT IN LINE WITH ``emergent/parser_mixins/roles.py``, which had
        already fixed all three of the problems below; this method is the older
        copy of the same pattern and had drifted.

        1. NO ``reset_area_connections(role_area)``.  It used to run after every
           word, and it destroyed exactly what had just been learned.  Measured
           by ``research/experiments/fiber_audit.py`` on a 40-sentence corpus:
           the reset fired 138 times and **all 138 zeroed a pathway that was
           carrying weight at that instant**, leaving ``SEQ -> SEQ`` as the only
           live area→area pathway in the whole brain afterwards.  So at parse
           time every role area received exactly zero drive.

           With the connectome zeroed, every candidate neuron has equal input
           and the deterministic index tie-break in winner selection returns
           the SAME k neurons for every word -- all stored role assemblies are
           literally the identical winner set.  That is why role retrieval read
           exactly chance with a unit margin, and why it was invariant to beta:
           no gain can separate assemblies that are bit-identical.

        2. The stabilized lexicon assembly is REPLAYED rather than re-projected.
           ``project(phon, lex_area)`` carries plasticity, so the LEX assembly
           drifts between storing a binding and reading it back and retrieval
           then misses the target it was trained on.  ``train_lexicon`` has
           already converged these, so ``activate_assembly`` replays the
           snapshot exactly.

        3. Round 1 is FEED-FORWARD, with only a short recurrent tail.
           Self-recurrence in a shared area is this project's documented
           collapse channel: the first item's self-connections potentiate until
           they beat every later item's input.  Round 1 input-driven makes the
           role assembly a function of the filler; ``_ROLE_BINDING_ROUNDS - 1``
           further rounds stabilize it.

        Measured after the change, n=1000 k=50 p=0.2 beta=0.0718, 40 nouns and
        20 verbs with each (word, role) binding trained once: distinctness
        1.000 and role retrieval 1.000 against a chance level of 0.025-0.062,
        versus exactly chance before.

        The protocol itself lives in :func:`ops.bind`, which is the single
        implementation shared with ``emergent.parser_mixins.roles`` and
        ``emergent.parser_mixins.generation``. It used to be hand-rolled in all
        three, and the three drifted into different states with only one
        correct -- which is precisely how the bug survived.
        """
        from .ops import bind

        role_sequence = [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]

        for sentence in sentences:
            for word, role_area in zip(sentence, role_sequence):
                category = self.word_categories[word]
                lex_area = "LEX_NOUN" if category == "noun" else "LEX_VERB"

                stored_lex = (self.noun_lexicon if category == "noun"
                              else self.verb_lexicon).get(word)
                if stored_lex is None:
                    # No stabilized snapshot yet (train_lexicon not run):
                    # fall back to driving the stimulus, accepting the drift
                    # that `bind`'s docstring warns about.
                    project(self.brain, self.stim_map[word], lex_area,
                            rounds=self.rounds)

                asm = bind(self.brain, lex_area, role_area, stored_lex)
                if role_area not in self.role_lexicons:
                    self.role_lexicons[role_area] = {}
                self.role_lexicons[role_area][word] = asm

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
        self.brain.reset_area_connections("LEX_NOUN")
        asm_n = project(self.brain, self.stim_map[word], "LEX_NOUN",
                        rounds=self.rounds)
        noun_scores = readout_all(asm_n, self.noun_lexicon)

        # Project to LEX_VERB
        self.brain.reset_area_connections("LEX_VERB")
        asm_v = project(self.brain, self.stim_map[word], "LEX_VERB",
                        rounds=self.rounds)
        verb_scores = readout_all(asm_v, self.verb_lexicon)

        best_noun = noun_scores[0][1] if noun_scores else 0.0
        best_verb = verb_scores[0][1] if verb_scores else 0.0

        return "noun" if best_noun > best_verb else "verb"

    def assign_role(self, word: str) -> Optional[str]:
        """Assign a thematic role to a word via readout against role lexicons.

        Returns the role label ("AGENT", "ACTION", "PATIENT") of the first
        role area this word was trained into, or None.

        WHAT THIS ACTUALLY MEASURES -- and it is not overlap.  Each role area
        has its own neuron population, and ``Assembly`` snapshots carry neuron
        IDs that are only meaningful within one area, so the overlap between a
        LEX assembly and a ROLE assembly is structurally ~0 regardless of how
        well the role was learned.  There is no signal to rank on.  The code
        therefore falls back to *membership*: it reports the role whose
        training lexicon contains this word.

        Consequences, stated plainly:

        * The answer is bookkeeping, not neural readout.  Nothing about the
          brain's current state is consulted.
        * A word trained in more than one role (e.g. a noun seen as both
          subject and object) resolves to whichever role area was inserted
          into ``role_lexicons`` first -- dict insertion order, i.e. the order
          roles appeared in the first training sentence.  There is no
          tie-break and no error.
        * Genuine cross-area role readout requires driving the LEX assembly
          into the role areas and comparing the resulting activity; see
          ``assembly_calculus.binding.input_drive``, which is the tool built
          for exactly this "which area responds" question.

        Left as-is because ``parse`` only reaches this path for sentences
        shorter than three words, and changing it would change results.
        """
        if not self.role_lexicons:
            return None

        # ``word_asm`` is looked up only to reject words with no lexical
        # assembly at all; it is not compared against anything, for the reason
        # given above.
        category = self.word_categories.get(word)
        if category is None:
            category = self.classify_word(word)

        lex = self.noun_lexicon if category == "noun" else self.verb_lexicon
        word_asm = lex.get(word)
        if word_asm is None:
            return None

        for role_area, role_lex in self.role_lexicons.items():
            if word in role_lex:
                return ROLE_LABELS[role_area]

        # Fallback: assign by category pattern
        # (nouns tend to be agents/patients, verbs tend to be actions).
        # Nouns get None rather than a guess, since agent-vs-patient is
        # genuinely undetermined without position information.
        if category == "verb":
            return "ACTION"
        return None

    def parse(self, words: List[str]) -> dict:
        """Parse a sentence through the full pipeline.

        Step 1 (categories) is neural where it matters: an unregistered word
        goes through ``classify_word``, which projects the word's PHON
        stimulus into both LEX areas and compares readout overlap.  A
        registered word short-circuits to its stored category.

        Step 2 (roles) is NOT neural for sentences of three or more words.
        It applies a hardcoded SVO template in Python -- first noun AGENT,
        verb ACTION, subsequent nouns PATIENT -- and consults neither the
        trained role areas nor the SEQ area that ``train_word_order``
        populated.  Consequences: role assignment is identical whether or not
        ``train_roles`` was ever called, it cannot be wrong about a
        well-formed SVO sentence, and it cannot be right about a non-SVO one.
        Do not cite ``parse`` role accuracy as evidence about role binding.
        The genuinely emergent role machinery is in
        ``assembly_calculus.emergent``; this class is a composition demo of
        the three training patterns.

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
