"""Category queries with an explicit neural observation boundary."""


from typing import Dict, Optional, Tuple
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.assembly_calculus.readout import readout_all
from neural_assemblies.core.brain import Brain

from ..core.areas import CORE_AREAS, CORE_TO_CATEGORY
from ..core.grounding import GroundingContext
from ..core.classification import ClassificationEvidence
from ._shared import DistributionalStats


class CategoryClassificationMixin:
    """Which category area holds a word's assembly."""

    # Shared parser state used by classification. These fields are initialized
    # by the core/lexicon/distributional mixins; declaring them here makes the
    # read-only observation boundary explicit for composed implementations.
    brain: Brain
    stim_map: Dict[str, str]
    rounds: int
    word_grounding: Dict[str, GroundingContext]
    core_lexicons: Dict[str, Dict]
    dist_stats: DistributionalStats
    _category_cache: Dict[str, str]
    _bootstrap_categories: Dict[str, str]
    _dist_categories: Dict[str, str]

    def classify_word_cached(
        self,
        word: str,
        grounding: Optional[GroundingContext] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Fast default-context classification; alternate grounding is uncached.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-classification-cache-context
        """
        if grounding is not None and grounding != self.word_grounding.get(word):
            from ..acquisition.pos_inference import classify_word_bootstrapped
            return classify_word_bootstrapped(self, word, grounding)
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

    def classify_word(self, word: str, grounding: Optional[GroundingContext] = None,
                      *, cue_mode: str = "combined") -> Tuple[str, Dict[str, float]]:
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-word-classification

        Legacy tuple view; use classify_word_evidence to retain score provenance.
        """
        return self.classify_word_evidence(word, grounding, cue_mode=cue_mode).as_legacy_tuple()

    def classify_word_evidence(
        self,
        word: str,
        grounding: Optional[GroundingContext] = None,
        *, cue_mode: str = "combined",
    ) -> ClassificationEvidence:
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-word-classification

        Read existing core lexicons without neural learning or construction.

        cue_mode selects combined (default), phon_only or grounding_only inputs.
        Only registered phon and explicitly supplied grounding cues are resolved;
        the evidence retains the mode and resolved stimulus names. These drive a
        stimulus-only schedule for self.rounds steps. Each target is temporarily
        cleared and unclamped inside read_only; fibers, activity and RNG are
        preserved. A nonempty lexicon requires a population usable for probing.

        Neural scores are maximum stored-word overlaps keyed by core area,
        not probabilities. CORE_AREAS order resolves positive ties. No positive
        neural evidence falls back to distributional classification when corpus
        statistics exist, otherwise UNKNOWN. Distributional scores have category
        keys and may update parser subcategory metadata, not neural weights.
        """
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-classification-cues
        if cue_mode not in ("combined", "phon_only", "grounding_only"):
            raise ValueError("cue_mode must be combined, phon_only or grounding_only")
        phon = self.stim_map.get(word) if cue_mode != "grounding_only" else None
        supplied_grounding = grounding if cue_mode != "phon_only" else None
        cues = [phon] if phon is not None else []
        if supplied_grounding is not None:
            cues.extend(gs for gs in self._grounding_stim_names(supplied_grounding)
                        if gs in self._grounding_stim_names_set)
        cues = tuple(dict.fromkeys(cues))

        def evidence(category, source, scores):
            return ClassificationEvidence(category, source, scores, cue_mode=cue_mode, cues=cues)

        if phon is None and supplied_grounding is None:
            if self.dist_stats.word_count.get(word, 0) > 0:
                category, scores = self.classify_distributional(word)
                return evidence(category, "distributional", scores)
            return evidence("UNKNOWN", "none", {})

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
                return evidence(category, "distributional", scores)
            return evidence("UNKNOWN", "neural", scores)

        best_area = max(scores, key=lambda area: scores[area])
        return evidence(CORE_TO_CATEGORY[best_area], "neural", scores)

