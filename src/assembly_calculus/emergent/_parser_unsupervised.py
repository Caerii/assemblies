"""UnsupervisedMixin -- unsupervised thematic role learning from raw exposure."""

from typing import Dict, List, Optional

from src.assembly_calculus.ops import project, _snap
from src.assembly_calculus.readout import readout_all

from .areas import (
    NOUN_CORE, VERB_CORE, PRON_CORE,
    ROLE_AGENT, ROLE_PATIENT,
    CORE_TO_CATEGORY, GROUNDING_TO_CORE,
    CORE_AREAS, THEMATIC_AREAS,
)


class UnsupervisedMixin:
    """Unsupervised thematic role learning from raw exposure."""

    def train_unsupervised(self, sentences: List["GroundedSentence"],
                           repetitions: int = 3):
        """Learn role assignments from raw sentence exposure (no role labels).

        Uses distributional regularity based on learned word order:
        - SVO: noun before verb = agent, after verb = patient
        - SOV: first noun = agent, second noun = patient, verb at end
        - VSO: noun after verb = agent, second noun = patient

        Role structure emerges from Hebbian plasticity through repeated
        exposure, not from explicit labels.

        Args:
            sentences: Training sentences (role annotations are IGNORED).
            repetitions: Number of passes over the training data.
        """
        order = self.word_order_type or "SVO"

        for role_area in THEMATIC_AREAS:
            if role_area not in self.role_lexicons:
                self.role_lexicons[role_area] = {}

        for _rep in range(repetitions):
            for sent in sentences:
                # Classify each word to find verb position
                categories = []
                for word in sent.words:
                    grounding = self.word_grounding.get(word)
                    cat, _ = self.classify_word(word, grounding=grounding)
                    categories.append(cat)

                # Find verb position
                verb_pos = None
                for idx, cat in enumerate(categories):
                    if cat == "VERB":
                        verb_pos = idx
                        break

                if verb_pos is None:
                    continue

                # Collect noun/pron positions
                noun_positions = [
                    (idx, word) for idx, (word, cat)
                    in enumerate(zip(sent.words, categories))
                    if cat in ("NOUN", "PRON") and word in self.stim_map
                ]

                # Assign roles based on word order typology
                for idx, word in noun_positions:
                    if order == "SVO":
                        # Pre-verb = agent, post-verb = patient
                        role_area = ROLE_AGENT if idx < verb_pos \
                            else ROLE_PATIENT
                    elif order == "SOV":
                        # First noun = agent, second noun = patient
                        # (verb is at end)
                        first_noun_idx = noun_positions[0][0] \
                            if noun_positions else -1
                        role_area = ROLE_AGENT if idx == first_noun_idx \
                            else ROLE_PATIENT
                    elif order == "VSO":
                        # First noun after verb = agent,
                        # second noun after verb = patient
                        nouns_after_verb = [
                            (i, w) for i, w in noun_positions if i > verb_pos
                        ]
                        if nouns_after_verb and idx == nouns_after_verb[0][0]:
                            role_area = ROLE_AGENT
                        else:
                            role_area = ROLE_PATIENT
                    else:
                        role_area = ROLE_AGENT if idx < verb_pos \
                            else ROLE_PATIENT

                    core_area = self._word_core_area(word)
                    phon = self.stim_map[word]

                    # Activate word, project to role area
                    project(
                        self.brain, phon, core_area, rounds=self.rounds)
                    self.brain.areas[core_area].fix_assembly()

                    self.brain.project(
                        {},
                        {core_area: [role_area], role_area: [role_area]},
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

                    self.brain.areas[core_area].unfix_assembly()
                    self.brain._engine.reset_area_connections(role_area)

        # Extract role lexicons from the trained weights
        self._extract_role_lexicons_from_weights()

    def _extract_role_lexicons_from_weights(self):
        """Extract role lexicons by projecting each noun into role areas.

        After unsupervised training, the Hebbian weights encode which
        nouns belong to which roles. Snap the resulting assemblies as
        the role lexicon entries.
        """
        for role_area in [ROLE_AGENT, ROLE_PATIENT]:
            self.role_lexicons[role_area] = {}

        for word, ctx in self.word_grounding.items():
            if ctx.dominant_modality not in ("visual", "social"):
                continue  # Only nouns and pronouns get roles
            if word not in self.stim_map:
                continue

            core_area = self._word_core_area(word)
            phon = self.stim_map[word]

            project(self.brain, phon, core_area, rounds=self.rounds)
            self.brain.areas[core_area].fix_assembly()

            for role_area in [ROLE_AGENT, ROLE_PATIENT]:
                self.brain.project(
                    {},
                    {core_area: [role_area], role_area: [role_area]},
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
                self.role_lexicons[role_area][word] = asm
                self.brain._engine.reset_area_connections(role_area)

            self.brain.areas[core_area].unfix_assembly()
