"""Category queries with an explicit neural observation boundary."""


from typing import Dict, Optional, Tuple
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.assembly_calculus.readout import readout_all

from ..core.areas import CORE_AREAS, CORE_TO_CATEGORY
from ..core.grounding import GroundingContext
from ..core.classification import ClassificationEvidence


class CategoryClassificationMixin:
    """Which category area holds a word's assembly."""

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

    def classify_word(self, word: str, grounding: Optional[GroundingContext] = None
                      ) -> Tuple[str, Dict[str, float]]:
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-word-classification

        Legacy tuple view; use classify_word_evidence to retain score provenance.
        """
        return self.classify_word_evidence(word, grounding).as_legacy_tuple()

    def classify_word_evidence(
        self,
        word: str,
        grounding: Optional[GroundingContext] = None,
    ) -> ClassificationEvidence:
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-word-classification

        Read existing core lexicons without neural learning or construction.

        Phon and explicitly supplied, registered grounding cues drive a
        stimulus-only schedule for self.rounds steps. Each target is temporarily
        cleared and unclamped inside read_only; fibers, activity and RNG are
        preserved. A nonempty lexicon requires a population usable for probing.

        Neural scores are maximum stored-word overlaps keyed by core area,
        not probabilities. CORE_AREAS order resolves positive ties. No positive
        neural evidence falls back to distributional classification when corpus
        statistics exist, otherwise UNKNOWN. Distributional scores have category
        keys and may update parser subcategory metadata, not neural weights.
        """
        phon = self.stim_map.get(word)
        if phon is None and grounding is None:
            if self.dist_stats.word_count.get(word, 0) > 0:
                category, scores = self.classify_distributional(word)
                return ClassificationEvidence(category, "distributional", scores)
            return ClassificationEvidence("UNKNOWN", "none", {})

        cues = [phon] if phon is not None else []
        if grounding is not None:
            cues.extend(gs for gs in self._grounding_stim_names(grounding)
                        if gs in self._grounding_stim_names_set)
        scores: Dict[str, float] = {}
        brain = self.brain
        with brain.read_only():
            for core_area in CORE_AREAS:
                lexicon = self.core_lexicons.get(core_area, {})
                if not lexicon or not cues:
                    scores[core_area] = 0.0
                    continue
                brain.clear_activity([core_area])
                brain.areas[core_area].unfix_assembly()
                brain.project_rounds(
                    core_area, {cue: [core_area] for cue in cues}, {}, self.rounds)
                overlaps = readout_all(_snap(brain, core_area), lexicon)
                scores[core_area] = overlaps[0][1] if overlaps else 0.0

        if not scores or max(scores.values()) == 0.0:
            # Fall back to distributional classification
            if self.dist_stats.word_count.get(word, 0) > 0:
                category, scores = self.classify_distributional(word)
                return ClassificationEvidence(category, "distributional", scores)
            return ClassificationEvidence("UNKNOWN", "neural", scores)

        best_area = max(scores, key=scores.get)
        return ClassificationEvidence(CORE_TO_CATEGORY[best_area], "neural", scores)

