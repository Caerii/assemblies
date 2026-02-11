"""Incremental and recursive parsing mixin for EmergentParser.

Provides word-by-word incremental parsing with FiberCircuit gating
and recursive clause handling for embedded relative clauses.
"""

from typing import Dict, List, Optional

from src.assembly_calculus.ops import project, _snap
from src.assembly_calculus.readout import readout_all
from src.assembly_calculus.fiber import FiberCircuit

from .areas import (
    NOUN_CORE, VERB_CORE, ADJ_CORE, ADV_CORE,
    PREP_CORE, DET_CORE, PRON_CORE, CONJ_CORE,
    CORE_AREAS,
    SUBJ, OBJ, VP, PP,
    TENSE, MOOD, SENT,
    CONTEXT, DEP_CLAUSE,
    ALL_AREAS,
)


class IncrementalMixin:
    """Incremental word-by-word parsing and recursive clause handling."""

    def _build_circuit(self) -> FiberCircuit:
        """Build a FiberCircuit with all projection channels initially inhibited.

        Declares fibers for:
        - core → SUBJ, core → OBJ (syntactic role routing)
        - core → VP (verb phrase building)
        - core → CONTEXT (running context accumulation)
        - DET_CORE/ADJ_CORE → SUBJ/OBJ (modifier → syntactic role)
        - SUBJ → VP (subject-verb merge)

        All fibers start inhibited; pre-rules disinhibit them as needed.
        """
        circuit = FiberCircuit(self.brain)

        # Core areas → syntactic role areas
        for core in [NOUN_CORE, PRON_CORE]:
            circuit.add(core, SUBJ, active=False)
            circuit.add(core, OBJ, active=False)

        # Modifier cores → syntactic role areas (merge det/adj into NP)
        circuit.add(DET_CORE, SUBJ, active=False)
        circuit.add(DET_CORE, OBJ, active=False)
        circuit.add(ADJ_CORE, SUBJ, active=False)
        circuit.add(ADJ_CORE, OBJ, active=False)

        # Verb → VP
        circuit.add(VERB_CORE, VP, active=False)
        # Subject → VP merge
        circuit.add(SUBJ, VP, active=False)
        # Object → VP merge
        circuit.add(OBJ, VP, active=False)

        # Prep → PP
        circuit.add(PREP_CORE, PP, active=False)

        # Adverb → VP
        circuit.add(ADV_CORE, VP, active=False)

        # Verb → TENSE (tense binding during incremental parse)
        circuit.add(VERB_CORE, TENSE, active=False)

        # MOOD → SENT (mood feeds sentence-level representation)
        circuit.add(MOOD, SENT, active=False)

        # CONJ_CORE → SENT (conjunctions link sentence-level structures)
        circuit.add(CONJ_CORE, SENT, active=False)

        # All cores → CONTEXT (sequential context building)
        for core in CORE_AREAS:
            circuit.add(core, CONTEXT, active=False)
        # CONTEXT self-recurrence
        circuit.add(CONTEXT, CONTEXT, active=False)

        return circuit

    def _get_syntactic_target(self, verb_seen: bool,
                              noun_count: int) -> str:
        """Determine whether to route to SUBJ or OBJ based on word order.

        Args:
            verb_seen: Whether a verb has been processed.
            noun_count: Number of nouns seen so far.

        Returns:
            SUBJ or OBJ area constant.
        """
        order = self.word_order_type or "SVO"
        if order == "SVO":
            return OBJ if verb_seen else SUBJ
        elif order == "SOV":
            # SOV: first noun = SUBJ, subsequent nouns = OBJ (before verb)
            return SUBJ if noun_count == 0 else OBJ
        elif order == "VSO":
            # VSO: first noun after verb = SUBJ, second = OBJ
            return SUBJ if noun_count == 0 else OBJ
        return OBJ if verb_seen else SUBJ

    def _apply_pre_rules(self, circuit: FiberCircuit, category: str,
                         verb_seen: bool, noun_count: int = 0) -> None:
        """Disinhibit fibers before projecting the current word.

        Mirrors recursive_parser.py PRE_RULES: open channels that this
        word category needs for projection. Adapts routing based on
        learned word order typology.
        """
        target = self._get_syntactic_target(verb_seen, noun_count)

        if category == "DET":
            circuit.disinhibit(DET_CORE, target)

        elif category == "ADJ":
            circuit.disinhibit(ADJ_CORE, target)

        elif category in ("NOUN", "PRON"):
            core = NOUN_CORE if category == "NOUN" else PRON_CORE
            circuit.disinhibit(core, target)
            # Merge DET/ADJ into the same syntactic slot
            circuit.disinhibit(DET_CORE, target)
            circuit.disinhibit(ADJ_CORE, target)

        elif category == "VERB":
            circuit.disinhibit(VERB_CORE, VP)
            circuit.disinhibit(SUBJ, VP)
            # Activate verb → TENSE binding
            circuit.disinhibit(VERB_CORE, TENSE)

        elif category == "PREP":
            circuit.disinhibit(PREP_CORE, PP)

        elif category == "ADV":
            circuit.disinhibit(ADV_CORE, VP)

        elif category == "CONJ":
            circuit.disinhibit(CONJ_CORE, SENT)

    def _apply_post_rules(self, circuit: FiberCircuit, category: str,
                          verb_seen: bool, noun_count: int = 0) -> None:
        """Inhibit fibers after projecting, preparing for next word.

        Mirrors recursive_parser.py POST_RULES: close channels that
        were consumed by this word.
        """
        if category in ("NOUN", "PRON"):
            target = self._get_syntactic_target(verb_seen, noun_count)
            core = NOUN_CORE if category == "NOUN" else PRON_CORE
            # Close noun/det/adj → syntactic slot
            circuit.inhibit(core, target)
            circuit.inhibit(DET_CORE, target)
            circuit.inhibit(ADJ_CORE, target)

        elif category == "VERB":
            # Close subject → SUBJ routing (subject slot filled)
            for core in [NOUN_CORE, PRON_CORE]:
                if circuit.is_active(core, SUBJ):
                    circuit.inhibit(core, SUBJ)
            if circuit.is_active(DET_CORE, SUBJ):
                circuit.inhibit(DET_CORE, SUBJ)
            if circuit.is_active(ADJ_CORE, SUBJ):
                circuit.inhibit(ADJ_CORE, SUBJ)
            # Open object routing
            circuit.disinhibit(VERB_CORE, VP)
            circuit.disinhibit(OBJ, VP)

    def parse_incremental(self, words: List[str]) -> dict:
        """Parse a sentence word-by-word with FiberCircuit gating.

        For each word:
        1. Classify (phon + grounding → core area readout)
        2. Project phon → core area to activate word assembly
        3. Apply pre-rules (disinhibit appropriate channels)
        4. Project through active fibers (circuit.step × rounds)
        5. Project core → CONTEXT with recurrence (build context)
        6. Apply post-rules (inhibit consumed channels)
        7. Snapshot state

        Returns same format as parse() plus:
        - "steps": per-word snapshots with category and context assembly
        """
        result: dict = {
            "categories": {},
            "roles": {},
            "phrases": {},
            "steps": [],
        }

        # Clear CONTEXT winners for fresh context building (preserve weights)
        self.brain.areas[CONTEXT].winners = []

        circuit = self._build_circuit()
        verb_seen = False
        noun_count = 0
        steps: List[dict] = []

        for i, word in enumerate(words):
            # Step 1: Classify
            grounding = self.word_grounding.get(word)
            cat, scores = self.classify_word(word, grounding=grounding)
            result["categories"][word] = cat

            # Step 2: Activate word assembly in its core area
            phon = self.stim_map.get(word)
            core_area = self._word_core_area(word)
            if phon is not None:
                project(self.brain, phon, core_area, rounds=self.rounds)

            # Step 3: Apply pre-rules
            self._apply_pre_rules(circuit, cat, verb_seen, noun_count)

            # Step 4: Project through active fibers
            for _ in range(self.rounds):
                circuit.step()

            # Step 5: Build context assembly (core → CONTEXT with recurrence)
            # Use direct brain.project() instead of circuit.step() to avoid
            # firing unrelated fibers during context building.
            self.brain.project(
                {},
                {core_area: [CONTEXT], CONTEXT: [CONTEXT]},
            )
            if self.rounds > 1:
                self.brain.project_rounds(
                    target=CONTEXT,
                    areas_by_stim={},
                    dst_areas_by_src_area={
                        core_area: [CONTEXT],
                        CONTEXT: [CONTEXT],
                    },
                    rounds=self.rounds - 1,
                )

            # Step 6: Apply post-rules
            self._apply_post_rules(circuit, cat, verb_seen, noun_count)

            if cat == "VERB":
                verb_seen = True
            if cat in ("NOUN", "PRON"):
                noun_count += 1

            # Step 7: Snapshot
            ctx_assembly = _snap(self.brain, CONTEXT)
            steps.append({
                "word": word,
                "category": cat,
                "position": i,
                "context_assembly": ctx_assembly,
                "verb_seen": verb_seen,
            })

        result["steps"] = steps

        # Assign roles via neural readout (same as batch parse)
        result["roles"] = self._assign_roles_neural(
            words, result["categories"])

        # Identify phrases
        result["phrases"] = self._identify_phrases(
            words, result["categories"])

        # Detect tense, mood, polarity
        result["tense"] = self.detect_tense(words)
        result["mood"] = self.detect_mood(words)
        result["polarity"] = self.detect_polarity(words)

        return result

    @staticmethod
    def _detect_clause_boundary(word: str, prev_category: Optional[str]
                                ) -> bool:
        """Returns True if word is a relative pronoun after a NOUN."""
        return word in ("that", "which") and prev_category in ("NOUN", "PRON")

    def _save_outer_state(self) -> dict:
        """Save current winner assemblies in all active areas."""
        state = {}
        for area_name in ALL_AREAS:
            area = self.brain.areas[area_name]
            if hasattr(area, 'winners') and area.winners is not None:
                state[area_name] = list(area.winners)
            else:
                state[area_name] = []
        return state

    def _replay_without_plasticity(self, words: List[str],
                                   circuit: FiberCircuit) -> None:
        """Replay word sequence without plasticity to restore outer context.

        Mirrors recursive_parser.py:850-868: disable plasticity, replay
        each outer word (classify + pre-rules + project + post-rules),
        then re-enable plasticity.
        """
        self.brain.disable_plasticity = True

        verb_seen = False
        noun_count = 0
        for word in words:
            grounding = self.word_grounding.get(word)
            cat, _ = self.classify_word(word, grounding=grounding)

            phon = self.stim_map.get(word)
            core_area = self._word_core_area(word)
            if phon is not None:
                project(self.brain, phon, core_area, rounds=self.rounds)

            self._apply_pre_rules(circuit, cat, verb_seen, noun_count)
            circuit.step()
            self._apply_post_rules(circuit, cat, verb_seen, noun_count)

            if cat == "VERB":
                verb_seen = True
            if cat in ("NOUN", "PRON"):
                noun_count += 1

        self.brain.disable_plasticity = False

    def parse_recursive(self, words: List[str]) -> dict:
        """Parse a sentence that may contain embedded relative clauses.

        Extends incremental parsing with clause boundary detection:

        On "that"/"which" after a NOUN:
        1. Save outer context (fix all area assemblies)
        2. Disinhibit DEP_CLAUSE, project antecedent noun → DEP_CLAUSE
        3. Parse inner clause words incrementally

        On "," or sentence end after embedded clause:
        1. Snapshot DEP_CLAUSE assembly
        2. Replay outer words WITHOUT plasticity to restore context
        3. Inhibit DEP_CLAUSE, continue main parse

        Args:
            words: Full sentence including clause markers ("that", ",").

        Returns:
            Same format as parse() plus:
            - "clauses": {"main": [...], "embedded": [...]}
            - "dep_clause_assembly": Assembly of embedded clause (or None)
        """
        result: dict = {
            "categories": {},
            "roles": {},
            "phrases": {},
            "clauses": {"main": [], "embedded": []},
            "dep_clause_assembly": None,
        }

        circuit = self._build_circuit()
        verb_seen = False
        noun_count = 0
        in_clause = False
        clause_start_idx = None
        outer_words: List[str] = []
        inner_words: List[str] = []
        main_words: List[str] = []
        prev_category: Optional[str] = None

        antecedent_word: Optional[str] = None
        inner_verb_seen = False

        i = 0
        while i < len(words):
            word = words[i]

            # --- Classify early so we can use category for clause decisions ---
            grounding = self.word_grounding.get(word)
            cat, _ = self.classify_word(word, grounding=grounding)
            result["categories"][word] = cat

            # --- Check for clause entry ---
            if (not in_clause
                    and self._detect_clause_boundary(word, prev_category)):
                in_clause = True
                clause_start_idx = i
                outer_words = list(main_words)  # Words before "that"
                inner_verb_seen = False

                # Remember the antecedent noun for filler-gap binding
                if main_words:
                    antecedent_word = main_words[-1]
                    ant_phon = self.stim_map.get(antecedent_word)
                    if ant_phon:
                        ant_core = self._word_core_area(antecedent_word)
                        project(
                            self.brain, ant_phon, ant_core,
                            rounds=self.rounds,
                        )
                        for _ in range(self.rounds):
                            self.brain.project(
                                {},
                                {ant_core: [DEP_CLAUSE],
                                 DEP_CLAUSE: [DEP_CLAUSE]},
                            )

                # Reset core area connections for clean inner clause
                for core in CORE_AREAS:
                    self.brain._engine.reset_area_connections(core)

                # Build fresh circuit for inner clause
                circuit = self._build_circuit()

                prev_category = cat
                i += 1
                continue

            # --- Check for clause exit ---
            # Exit when: comma, or inner clause verb already seen and
            # current word is a verb (main clause verb), or end of sentence.
            is_clause_exit = False
            if in_clause:
                if word == ",":
                    is_clause_exit = True
                elif inner_verb_seen and cat == "VERB":
                    # Inner clause had its verb; this verb belongs to main
                    is_clause_exit = True
                elif i == len(words) - 1 and inner_verb_seen:
                    # End of sentence; last word goes to main clause
                    is_clause_exit = True

            if is_clause_exit:
                # Snapshot DEP_CLAUSE
                result["dep_clause_assembly"] = _snap(
                    self.brain, DEP_CLAUSE)
                result["clauses"]["embedded"] = list(inner_words)

                # Replay outer words without plasticity to restore context
                circuit = self._build_circuit()
                self._replay_without_plasticity(outer_words, circuit)

                in_clause = False
                verb_seen = any(
                    result["categories"].get(w) == "VERB"
                    for w in outer_words
                )
                noun_count = sum(
                    1 for w in outer_words
                    if result["categories"].get(w) in ("NOUN", "PRON")
                )

                if word == ",":
                    prev_category = cat
                    i += 1
                    continue
                # Otherwise fall through to process this word as main clause

            # --- Normal word processing ---
            if in_clause:
                inner_words.append(word)
                if cat == "VERB":
                    inner_verb_seen = True
            else:
                main_words.append(word)

            # Activate word assembly
            phon = self.stim_map.get(word)
            core_area = self._word_core_area(word)
            if phon is not None:
                project(self.brain, phon, core_area, rounds=self.rounds)

            # Apply gating rules and project
            self._apply_pre_rules(circuit, cat, verb_seen, noun_count)
            for _ in range(self.rounds):
                circuit.step()

            # Build context
            circuit.disinhibit(core_area, CONTEXT)
            circuit.disinhibit(CONTEXT, CONTEXT)
            for _ in range(self.rounds):
                circuit.step()
            circuit.inhibit(core_area, CONTEXT)

            self._apply_post_rules(circuit, cat, verb_seen, noun_count)

            if cat == "VERB":
                verb_seen = True
            if cat in ("NOUN", "PRON"):
                noun_count += 1

            prev_category = cat
            i += 1

        result["clauses"]["main"] = main_words

        # Assign roles for main clause words
        main_cats = {w: result["categories"][w]
                     for w in main_words
                     if w in result["categories"]}
        result["roles"] = self._assign_roles_neural(main_words, main_cats)

        # Assign roles for inner clause with filler-gap binding:
        # The antecedent noun is the "filler" — it was displaced from its
        # canonical position in the inner clause.  For SRCs the antecedent
        # is the agent; for ORCs it is the patient.
        if inner_words:
            inner_cats = {w: result["categories"][w]
                          for w in inner_words
                          if w in result["categories"]}

            # Determine filler role from inner clause structure:
            # - ORC active: "dog that THE CAT chased" → pre-verb noun → PATIENT
            # - SRC active: "dog that chased THE CAT" → no pre-verb noun → AGENT
            # - Passive RC:  "dog that was chased by ..." → passive → PATIENT
            filler_w = antecedent_word
            filler_r = None
            if filler_w:
                inner_is_passive = self._detect_passive(
                    inner_words, inner_cats)
                inner_has_pre_verb_noun = False
                for w in inner_words:
                    ic = inner_cats.get(w)
                    if ic == "VERB":
                        break
                    if ic in ("NOUN", "PRON"):
                        inner_has_pre_verb_noun = True
                        break
                if inner_is_passive or inner_has_pre_verb_noun:
                    filler_r = "PATIENT"
                else:
                    filler_r = "AGENT"

            inner_roles = self._assign_roles_neural(
                inner_words, inner_cats,
                filler_word=filler_w,
                filler_role=filler_r,
            )
            result["roles"].update(inner_roles)

        result["phrases"] = self._identify_phrases(
            main_words + inner_words, result["categories"])

        # Detect tense, mood, polarity from full word list
        result["tense"] = self.detect_tense(words)
        result["mood"] = self.detect_mood(words)
        result["polarity"] = self.detect_polarity(words)

        return result
