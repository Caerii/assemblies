"""CategoryClassificationMixin -- Which category area holds a word's assembly.

Split out of the CoreParserMixin monolith. Bodies are unchanged;
only their address is.
"""


from typing import Dict, List, Optional, Tuple
from neural_assemblies.assembly_calculus.ops import project, _snap
from neural_assemblies.assembly_calculus.readout import readout_all

from ..core.areas import CORE_AREAS, CORE_TO_CATEGORY
from ..core.grounding import GroundingContext


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

