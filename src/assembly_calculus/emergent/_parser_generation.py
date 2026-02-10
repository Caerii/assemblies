"""GenerationMixin -- language production / generation from semantic representations."""

from typing import Dict, List, Optional

from src.assembly_calculus.ops import project, _snap
from src.assembly_calculus.readout import readout_all

from .areas import (
    NOUN_CORE, VERB_CORE, DET_CORE,
    ROLE_AGENT, ROLE_PATIENT,
    CORE_TO_CATEGORY, CATEGORY_TO_CORE,
)


class GenerationMixin:
    """Language production / generation from semantic representations."""

    def _decode_role_to_word(self, role_area: str,
                             core_area: str) -> Optional[str]:
        """Project from a role area to a core area and readout the best word.

        Activates the role assembly, projects to core area, reads out
        against core lexicon.
        """
        lex = self.core_lexicons.get(core_area, {})
        if not lex:
            return None

        self.brain._engine.reset_area_connections(core_area)

        # Project role -> core with recurrence
        self.brain.project({}, {role_area: [core_area]})
        if self.rounds > 1:
            self.brain.project_rounds(
                target=core_area,
                areas_by_stim={},
                dst_areas_by_src_area={
                    role_area: [core_area],
                    core_area: [core_area],
                },
                rounds=self.rounds - 1,
            )

        asm = _snap(self.brain, core_area)
        overlaps = readout_all(asm, lex)
        if overlaps and overlaps[0][1] > 0.0:
            return overlaps[0][0]
        return None

    def generate(self, semantics: dict) -> List[str]:
        """Generate a sentence from a semantic representation.

        Args:
            semantics: Dict with keys:
                "agent": word string (must be in vocabulary)
                "action": word string (must be in vocabulary)
                "patient": word string (optional)

        Process:
        1. For each role, activate the word's assembly in its core area
        2. Project core -> role area to form role assembly
        3. Readout role -> core to recover content words
        4. Assemble sentence in learned word order with determiners

        Returns:
            List of word strings in surface order.
        """
        agent_word = semantics.get("agent")
        action_word = semantics.get("action")
        patient_word = semantics.get("patient")

        # Build phrase for each role
        agent_phrase: List[str] = []
        verb_phrase: List[str] = []
        patient_phrase: List[str] = []

        # Activate agent in its core area, project to ROLE_AGENT
        if agent_word and agent_word in self.stim_map:
            agent_core = self._word_core_area(agent_word)
            phon = self.stim_map[agent_word]
            project(self.brain, phon, agent_core, rounds=self.rounds)
            self.brain.areas[agent_core].fix_assembly()

            # Project to ROLE_AGENT
            for _ in range(self.rounds):
                self.brain.project(
                    {},
                    {agent_core: [ROLE_AGENT], ROLE_AGENT: [ROLE_AGENT]},
                )
            self.brain.areas[agent_core].unfix_assembly()

            # Readout: role -> core -> word
            decoded = self._decode_role_to_word(ROLE_AGENT, agent_core)
            if decoded:
                ctx = self.word_grounding.get(decoded)
                if ctx and ctx.dominant_modality == "visual":
                    agent_phrase.append("the")
                agent_phrase.append(decoded)
            self.brain._engine.reset_area_connections(ROLE_AGENT)

        # Action word
        if action_word and action_word in self.stim_map:
            project(
                self.brain, self.stim_map[action_word],
                VERB_CORE, rounds=self.rounds,
            )
            asm = _snap(self.brain, VERB_CORE)
            lex = self.core_lexicons.get(VERB_CORE, {})
            if lex:
                overlaps = readout_all(asm, lex)
                if overlaps and overlaps[0][1] > 0.0:
                    verb_phrase.append(overlaps[0][0])

        # Patient
        if patient_word and patient_word in self.stim_map:
            patient_core = self._word_core_area(patient_word)
            phon = self.stim_map[patient_word]
            project(self.brain, phon, patient_core, rounds=self.rounds)
            self.brain.areas[patient_core].fix_assembly()

            for _ in range(self.rounds):
                self.brain.project(
                    {},
                    {patient_core: [ROLE_PATIENT],
                     ROLE_PATIENT: [ROLE_PATIENT]},
                )
            self.brain.areas[patient_core].unfix_assembly()

            decoded = self._decode_role_to_word(ROLE_PATIENT, patient_core)
            if decoded:
                ctx = self.word_grounding.get(decoded)
                if ctx and ctx.dominant_modality == "visual":
                    patient_phrase.append("the")
                patient_phrase.append(decoded)
            self.brain._engine.reset_area_connections(ROLE_PATIENT)

        # Assemble in learned word order
        order = self.word_order_type or "SVO"
        output: List[str] = []

        if order == "SVO":
            output.extend(agent_phrase)
            output.extend(verb_phrase)
            output.extend(patient_phrase)
        elif order == "SOV":
            output.extend(agent_phrase)
            output.extend(patient_phrase)
            output.extend(verb_phrase)
        elif order == "VSO":
            output.extend(verb_phrase)
            output.extend(agent_phrase)
            output.extend(patient_phrase)
        else:
            # Default to SVO
            output.extend(agent_phrase)
            output.extend(verb_phrase)
            output.extend(patient_phrase)

        return output
