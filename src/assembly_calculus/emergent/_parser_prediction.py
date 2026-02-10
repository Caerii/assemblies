"""PredictionMixin -- structural next-token prediction using context assemblies."""

from typing import Dict, List, Optional, Tuple

from src.assembly_calculus.ops import project, _snap
from src.assembly_calculus.readout import readout_all

from .areas import (
    CORE_AREAS, CORE_TO_CATEGORY,
    PREDICTION, CONTEXT,
)


class PredictionMixin:
    """Structural next-token prediction using context assemblies."""

    def train_next_token(self, sentences: List["GroundedSentence"]):
        """Train next-token prediction using structural context.

        Two phases:
        1. Build a prediction lexicon: for each vocabulary word, project
           its phon into PREDICTION and snap the reference assembly.
        2. Train context bridges: for each sentence position, project
           context + next_word_phon into PREDICTION simultaneously,
           building Hebbian bridges from context to next-word assemblies.

        Args:
            sentences: Training sentences.
        """
        # Phase 1: Build prediction lexicon (word -> Assembly in PREDICTION)
        self.prediction_lexicon: dict = {}
        for word, phon in self.stim_map.items():
            self.brain._engine.reset_area_connections(PREDICTION)
            project(self.brain, phon, PREDICTION, rounds=self.rounds)
            self.prediction_lexicon[word] = _snap(self.brain, PREDICTION)

        # Phase 2: Train context -> PREDICTION bridges
        for sent in sentences:
            words = [w for w in sent.words if w in self.stim_map]
            if len(words) < 2:
                continue

            # Build context assemblies incrementally
            circuit = self._build_circuit()
            verb_seen = False
            noun_count = 0

            for i in range(len(words) - 1):
                word = words[i]
                grounding = self.word_grounding.get(word)
                cat, _ = self.classify_word(word, grounding=grounding)

                # Activate word
                phon = self.stim_map[word]
                core_area = self._word_core_area(word)
                project(self.brain, phon, core_area, rounds=self.rounds)

                # Build context via direct projection
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

                self._apply_pre_rules(circuit, cat, verb_seen, noun_count)
                for _ in range(self.rounds):
                    circuit.step()
                self._apply_post_rules(circuit, cat, verb_seen, noun_count)
                if cat == "VERB":
                    verb_seen = True
                if cat in ("NOUN", "PRON"):
                    noun_count += 1

                # Train: context -> PREDICTION, next_phon -> PREDICTION
                next_phon = self.stim_map[words[i + 1]]
                self.brain.project(
                    {next_phon: [PREDICTION]},
                    {CONTEXT: [PREDICTION]},
                )
                if self.rounds > 1:
                    self.brain.project_rounds(
                        target=PREDICTION,
                        areas_by_stim={next_phon: [PREDICTION]},
                        dst_areas_by_src_area={
                            CONTEXT: [PREDICTION],
                            PREDICTION: [PREDICTION],
                        },
                        rounds=self.rounds - 1,
                    )
                self.brain._engine.reset_area_connections(PREDICTION)

    def predict_next(self, words: List[str]) -> List[Tuple[str, float]]:
        """Predict the next word given a partial sentence.

        1. Run incremental processing on words to build context assembly
        2. Project context -> PREDICTION area with recurrence
        3. Readout PREDICTION against the pre-built prediction lexicon

        Args:
            words: Context words (partial sentence).

        Returns:
            List of (word, overlap) sorted by overlap descending.
        """
        if not words:
            return []

        if not hasattr(self, 'prediction_lexicon') or not self.prediction_lexicon:
            return []

        # Build context via incremental processing
        self.parse_incremental(words)

        # Project context -> PREDICTION
        self.brain._engine.reset_area_connections(PREDICTION)
        self.brain.project(
            {},
            {CONTEXT: [PREDICTION]},
        )
        if self.rounds > 1:
            self.brain.project_rounds(
                target=PREDICTION,
                areas_by_stim={},
                dst_areas_by_src_area={
                    CONTEXT: [PREDICTION],
                    PREDICTION: [PREDICTION],
                },
                rounds=self.rounds - 1,
            )

        pred_assembly = _snap(self.brain, PREDICTION)
        return readout_all(pred_assembly, self.prediction_lexicon)
