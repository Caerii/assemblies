"""Distributional learning mixin for EmergentParser.

THE SECOND ROUTE TO CATEGORY.  ``CoreParserMixin`` derives a word's part of
speech from its GROUNDING -- which sensory modality accompanied it.  That
route is unavailable for exactly the words a learner meets most often: "the",
"of", "that" are not accompanied by anything visual or motor, and the
grounding table lumps all of them into DET_CORE.  It is also unavailable for
any word encountered in text without a scene attached.

This mixin supplies the complementary route, and it is the one distributional
linguistics has always pointed at: a word's category is predicted by the
COMPANY IT KEEPS.  Three signals are tracked as sentences are ingested:

    position       where in the sentence the word tends to appear
    transitions    which words directly precede and follow it
    co-occurrence  which words appear within a +/-2 window

Words that share a distributional profile get classified alike, which is how
function words are separated into DET / AUX / COMP / CONJ / MARKER
sub-categories even though all of them are ungrounded and all of them route
through the same core area.

The two routes are not redundant and their precedence is deliberate.  Where
grounding exists it wins, because it is evidence about what the word MEANS;
distribution is the fallback and is consulted when grounding is absent or
produces no signal (see ``CoreParserMixin.classify_word``, which falls
through to ``classify_distributional``).

Provides raw-text ingestion, distributional category inference,
word-order typology detection, and the full train-from-text pipeline.

ON THE POSITION PROFILES.  Position scoring uses ``position_profiles()``,
which has three tiers and reports which one it used:

    "learned"    measured from THIS corpus -- for each category, the mean and
                 spread of the normalised sentence positions of the words
                 already known to belong to it.  No typology assumed.
    "typology"   derived definitionally from the inferred order label by
                 ``core.word_order.position_profiles_for_order``.  A PRIOR.
    "svo-prior"  the hand-authored English table below.  A PRIOR, and used
                 only when there is neither corpus evidence nor an inferred
                 typology.

The previous behaviour was tier 3 unconditionally: an English SVO prior was
imposed even on a corpus inferred to be verb-final or object-initial, and the
failure was silent -- scores shifted rather than an error being raised.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, cast

from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.assembly import Assembly
from ..core.areas import (
    CORE_TO_CATEGORY, CATEGORY_TO_CORE, GROUNDING_TO_CORE,
    FUNC_DET, FUNC_AUX, FUNC_COMP, FUNC_CONJ, FUNC_MARKER, FUNC_SUBCAT_TO_CORE,
)
from ..core.grounding import GroundingContext
from ..core.word_order import (
    WORD_ORDERS,
    canonical_for_class,
    position_profiles_for_order,
)
from ..curriculum.data import GroundedSentence
from ..parser_mixins.core import _MODALITY_FIELDS
from ._shared import DistributionalStats

if TYPE_CHECKING:
    from ..core.corpus_index import CorpusIndex
    from .core import CoreParserMixin

# HAND-AUTHORED ENGLISH-SVO PRIOR.  Category -> typical normalised position
# range.  This is NOT learned and is NOT typology-neutral: it encodes an SVO,
# determiner-initial language.  It is now the LAST resort in
# ``position_profiles()`` -- used only when the corpus yields no measured
# positions and no typology has been inferred.  Do not reintroduce it as an
# unconditional default; that was the defect.
_SVO_POSITION_PROFILES = {
    "DET": (0.0, 0.25),    # Early in phrases
    "ADJ": (0.15, 0.35),   # After determiners, before nouns
    "NOUN": (0.2, 0.5),    # Middle positions
    "VERB": (0.3, 0.6),    # After subject, before object
    "ADV": (0.5, 0.8),     # Late in sentences
    "PREP": (0.5, 0.75),   # Between phrases
    "PRON": (0.0, 0.3),    # Early (subject pronouns)
}

# Back-compatible alias. Kept so existing importers do not break, but reading
# it gives the English prior only -- call ``position_profiles()`` instead.
_CATEGORY_POSITION_PROFILES = _SVO_POSITION_PROFILES

# Minimum number of distinct words, and minimum occurrences per word, before a
# category's position profile is considered measured rather than assumed.
_PROFILE_MIN_WORDS = 2
_PROFILE_MIN_OCCURRENCES = 3


class DistributionalMixin:
    """Distributional learning, raw text pipeline, and word order typology."""

    brain: Brain
    n: int
    k: int
    rounds: int
    stim_map: Dict[str, str]
    word_grounding: Dict[str, GroundingContext]
    dist_stats: DistributionalStats
    core_lexicons: Dict[str, Dict[str, Assembly]]
    _category_cache: Dict[str, str]
    _grounding_stim_names_set: Set[str]

    if TYPE_CHECKING:
        def add_phon_stimulus(self, word: str) -> str: ...
        def _invalidate_category_cache(self, word: Optional[str] = None) -> None: ...
        def _clear_core_activity(self, core_area: Optional[str] = None) -> None: ...
        def train_lexicon(self, *args: Any, **kwargs: Any) -> None: ...
        def train_unsupervised(self, *args: Any, **kwargs: Any) -> None: ...
        def train_phrases(self, *args: Any, **kwargs: Any) -> None: ...
        def train_word_order(self, *args: Any, **kwargs: Any) -> None: ...
        def train_tense(self, *args: Any, **kwargs: Any) -> None: ...
        def train_mood(self, *args: Any, **kwargs: Any) -> None: ...
        def train_polarity(self, *args: Any, **kwargs: Any) -> None: ...
        def train_conjunctions(self, *args: Any, **kwargs: Any) -> None: ...
        def classify_word_cached(self, word: str, grounding: Optional[GroundingContext] = None) -> Tuple[str, Any]: ...

    def ingest_raw_sentence(self, words: List[str]) -> None:
        """Ingest a raw sentence (no grounding) to build distributional stats.

        Registers unknown words as phon stimuli, updates position counts,
        transitions, co-occurrence, and verb-relative position stats.

        Args:
            words: List of word strings (plain text tokens).
        """
        stats = self.dist_stats
        stats.sentences_seen += 1
        n = len(words)

        from ..acquisition.pos_inference import record_exposure_sentence
        record_exposure_sentence(cast("CoreParserMixin", self), words)

        # Find verb position (using known categories or distributional)
        verb_pos = None
        for idx, word in enumerate(words):
            cat = self._quick_category(word)
            if cat == "VERB":
                verb_pos = idx
                break

        for idx, word in enumerate(words):
            stats.word_count[word] += 1
            stats.position_counts[word][idx] += 1

            # Register unknown words
            if word not in self.stim_map:
                self.add_phon_stimulus(word)

            # Transition to next word
            if idx + 1 < n:
                stats.transitions[(word, words[idx + 1])] += 1

            # Co-occurrence within window of 2
            for j in range(max(0, idx - 2), min(n, idx + 3)):
                if j != idx:
                    stats.word_cooccurrence[word][words[j]] += 1

            # Verb-relative position
            if verb_pos is not None:
                if idx < verb_pos:
                    stats.word_as_pre_verb[word] += 1
                elif idx > verb_pos:
                    stats.word_as_post_verb[word] += 1
                elif idx == verb_pos:
                    stats.word_as_action[word] += 1

        # Update category transitions for known words
        cats = [self._quick_category(w) for w in words]
        for idx in range(len(cats) - 1):
            left, right = cats[idx], cats[idx + 1]
            if left is not None and right is not None:
                stats.category_transitions[(left, right)] += 1

    def _quick_category(self, word: str) -> Optional[str]:
        """Return known category for a word (from grounding or lexicon), or None."""
        ctx = self.word_grounding.get(word)
        if ctx is not None:
            modality = ctx.dominant_modality
            if modality is None:
                return None
            core = GROUNDING_TO_CORE.get(cast(str, modality))
            if core is None:
                return None
            return CORE_TO_CATEGORY.get(core)
        # Check if word has a distributional classification cached
        if hasattr(self, '_dist_categories') and word in self._dist_categories:
            return self._dist_categories[word]
        return None

    def _reingest_verb_positions(self, words: List[str]) -> None:
        """Re-analyze verb-relative positions using preliminary categories.

        Called during the bootstrapping iteration of train_distributional
        to update verb-relative stats once some words have been classified.
        """
        stats = self.dist_stats

        # Find verb position using all available categories
        verb_pos = None
        for idx, word in enumerate(words):
            cat = self._quick_category(word)
            if cat == "VERB":
                verb_pos = idx
                break

        if verb_pos is None:
            return

        for idx, word in enumerate(words):
            if idx < verb_pos:
                stats.word_as_pre_verb[word] += 1
            elif idx > verb_pos:
                stats.word_as_post_verb[word] += 1
            elif idx == verb_pos:
                stats.word_as_action[word] += 1

        # Update category transitions
        cats = [self._quick_category(w) for w in words]
        for idx in range(len(cats) - 1):
            left, right = cats[idx], cats[idx + 1]
            if left is not None and right is not None:
                stats.category_transitions[(left, right)] += 1

    def classify_by_frame(self, word: str
                          ) -> Tuple[Optional[str], float]:
        """Classify a word by its bigram frame (left/right context categories).

        Implements Stage 1 of the two-stage function word model:
        rapid sub-categorization from distributional frames, analogous
        to the ELAN response (~180ms) in human language processing.

        The frame is the pair (dominant_left_category, dominant_right_category).
        Different function word sub-types occupy distinct frames:
          DET:    (*, NOUN/ADJ)   — precedes nominal content
          AUX:    (NOUN, VERB)    — between subject NP and main verb
          COMP:   (NOUN, DET/NOUN) — after NP, introduces new clause
          CONJ:   (NOUN, DET)     — between parallel NPs
          MARKER: (VERB, DET)     — after verb, before new NP (agent marker)

        Args:
            word: Word string to classify.

        Returns:
            (sub_category_or_None, confidence) where sub_category is one of
            FUNC_DET/FUNC_AUX/FUNC_COMP/FUNC_CONJ/FUNC_MARKER, or the
            standard POS category if the word is not a function word.
        """
        stats = self.dist_stats
        count = stats.word_count.get(word, 0)
        if count == 0:
            return None, 0.0

        # Build left/right context category distributions
        left_cats: Dict[str, int] = defaultdict(int)
        right_cats: Dict[str, int] = defaultdict(int)
        for (w1, w2), c in stats.transitions.items():
            if w2 == word:
                cat_left = self._quick_category(w1)
                if cat_left:
                    left_cats[cat_left] += c
            if w1 == word:
                cat_right = self._quick_category(w2)
                if cat_right:
                    right_cats[cat_right] += c

        if not left_cats and not right_cats:
            return None, 0.0

        # Dominant left/right categories
        left_total = sum(left_cats.values()) or 1
        right_total = sum(right_cats.values()) or 1

        # Compute frame-match scores for each function word sub-type
        frame_scores: Dict[str, float] = {}

        # DET: primarily precedes NOUN or ADJ (right context is nominal)
        noun_adj_right = (right_cats.get("NOUN", 0)
                          + right_cats.get("ADJ", 0)) / right_total
        frame_scores[FUNC_DET] = noun_adj_right

        # AUX: left=NOUN/PRON, right=VERB (between NP and VP)
        np_left = (left_cats.get("NOUN", 0)
                   + left_cats.get("PRON", 0)) / left_total
        verb_right = right_cats.get("VERB", 0) / right_total
        frame_scores[FUNC_AUX] = min(np_left, verb_right) * 1.5

        # COMP: left=NOUN, right=DET/NOUN/PRON (new clause opener)
        noun_left = left_cats.get("NOUN", 0) / left_total
        clause_right = (right_cats.get("DET", 0) + right_cats.get("NOUN", 0)
                        + right_cats.get("PRON", 0)
                        + right_cats.get("VERB", 0)) / right_total
        # COMP must have both left=NOUN and right=new-clause material
        frame_scores[FUNC_COMP] = min(noun_left, clause_right) * 1.3

        # CONJ: left=NOUN, right=DET (between parallel NPs)
        det_right = right_cats.get("DET", 0) / right_total
        frame_scores[FUNC_CONJ] = min(noun_left, det_right) * 1.2

        # MARKER: left=VERB, right=DET (after verb, before agent NP)
        verb_left = left_cats.get("VERB", 0) / left_total
        frame_scores[FUNC_MARKER] = min(verb_left, det_right) * 1.4

        # Also score content word categories from frames
        # Content words have DISTINCTIVE frame patterns:
        # NOUN: left=DET/ADJ, right=VERB (position after determiner)
        det_adj_left = (left_cats.get("DET", 0)
                        + left_cats.get("ADJ", 0)) / left_total
        frame_scores["NOUN"] = min(det_adj_left, 0.5) + verb_right * 0.5

        # VERB: left=NOUN/PRON, right=DET/NOUN/ADV
        content_right = (right_cats.get("DET", 0)
                         + right_cats.get("NOUN", 0)
                         + right_cats.get("ADV", 0)) / right_total
        frame_scores["VERB"] = min(np_left, content_right)

        # ADJ: left=DET, right=NOUN
        det_left = left_cats.get("DET", 0) / left_total
        noun_right = right_cats.get("NOUN", 0) / right_total
        frame_scores["ADJ"] = min(det_left, noun_right) * 1.1

        best = max(frame_scores, key=lambda key: frame_scores[key])
        confidence = frame_scores[best]

        return best, confidence

    def position_profiles(
        self,
    ) -> Tuple[Dict[str, Tuple[float, float]], str]:
        """Per-category normalised-position ranges, plus their provenance.

        Returns ``(profiles, source)`` where ``source`` is one of:

        ``"learned"``
            Measured from this corpus.  For every category with at least
            ``_PROFILE_MIN_WORDS`` known words totalling at least
            ``_PROFILE_MIN_OCCURRENCES`` tokens, the range is
            ``mean +/- sd`` of the normalised position of those tokens.  This
            assumes no typology; it reads one off the data.
        ``"typology"``
            Derived definitionally from ``self.word_order_type`` (see
            ``core.word_order.position_profiles_for_order``).  A PRIOR selected
            by the inferred typology -- not a learned result.
        ``"svo-prior"``
            The hand-authored English SVO table.  A PRIOR, and the only tier
            that is typology-blind; reached only when there is neither corpus
            evidence nor an inferred order.

        Categories with corpus evidence are taken from tier 1 and the rest are
        filled from the best available prior, so the returned dict can be
        mixed; ``source`` names the tier that supplied the majority.
        """
        stats = self.dist_stats
        cache_key = (
            stats.sentences_seen,
            getattr(self, "word_order_type", None),
            len(getattr(self, "_dist_categories", {}) or {}),
        )
        cached = getattr(self, "_position_profile_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2]

        # Tier 2/3: the prior we will fall back to for unmeasured categories.
        order = getattr(self, "word_order_type", None)
        if order in WORD_ORDERS:
            prior = position_profiles_for_order(order)
            prior_source = "typology"
        else:
            prior = dict(_SVO_POSITION_PROFILES)
            prior_source = "svo-prior"

        # Tier 1: measure from the corpus.
        by_cat: Dict[str, List[float]] = defaultdict(list)
        words_per_cat: Dict[str, Set[str]] = defaultdict(set)
        for word, positions in stats.position_counts.items():
            cat = self._quick_category(word)
            if cat is None:
                continue
            for pos, count in positions.items():
                # Normalise by the longest sentence this word was seen in.
                # position_counts stores absolute indices, so the best
                # available length estimate is the corpus-wide max index.
                by_cat[cat].extend([float(pos)] * count)
            words_per_cat[cat].add(word)

        max_index = 0
        for positions in stats.position_counts.values():
            if positions:
                max_index = max(max_index, max(positions.keys()))
        span = float(max_index + 1)

        profiles: Dict[str, Tuple[float, float]] = dict(prior)
        measured = 0
        if span > 1.0:
            for cat, raw in by_cat.items():
                if (len(words_per_cat[cat]) < _PROFILE_MIN_WORDS
                        or len(raw) < _PROFILE_MIN_OCCURRENCES):
                    continue
                norm = [p / span for p in raw]
                mean = sum(norm) / len(norm)
                var = sum((x - mean) ** 2 for x in norm) / len(norm)
                sd = var ** 0.5
                profiles[cat] = (max(0.0, mean - sd), min(1.0, mean + sd))
                measured += 1

        source = "learned" if measured >= len(prior) / 2 else prior_source
        self._position_profile_cache = (cache_key, profiles, source)
        return profiles, source

    def classify_distributional(self, word: str
                                ) -> Tuple[str, Dict[str, float]]:
        """Infer word category from distributional statistics.

        Uses a two-pass approach:
        1. Frame-based classification (bigram context patterns)
        2. Position profile and verb-relative position (fallback features)

        For ungrounded words (function words), frame-based classification
        dominates. For content words, all features contribute.

        Args:
            word: Word string to classify.

        Returns:
            (category_label, {category: confidence_score})
        """
        stats = self.dist_stats
        count = stats.word_count.get(word, 0)
        if count == 0:
            return "UNKNOWN", {}

        # Check if word is ungrounded (potential function word)
        ctx = self.word_grounding.get(word)
        is_ungrounded = (ctx is None or not ctx.is_grounded)

        # Stage 1: Frame-based classification (always computed)
        frame_cat, frame_conf = self.classify_by_frame(word)
        if frame_cat is None:
            pos_cat = None
        else:
            subcore = FUNC_SUBCAT_TO_CORE.get(cast(str, frame_cat))
            mapped_cat = CORE_TO_CATEGORY.get(subcore) if subcore else frame_cat
            pos_cat = mapped_cat if mapped_cat is not None else frame_cat

        # For ungrounded words, frame classification is authoritative
        # (analogous to ELAN rapid categorization)
        if is_ungrounded and frame_cat is not None and frame_conf > 0.3:
            # Store the sub-category for gating purposes
            if not hasattr(self, '_func_subcategories'):
                self._func_subcategories: Dict[str, str] = {}
            self._func_subcategories[word] = frame_cat

            category = cast(str, pos_cat)
            return category, {category: frame_conf}

        scores: Dict[str, float] = {}

        # Seed from frame scores (all categories including content)
        if frame_cat is not None:
            # Get all frame scores by re-running frame analysis
            # and using the confidence as a feature
            scores[cast(str, pos_cat)] = frame_conf * 2.0

        # 2. Verb-relative position scores
        pre = stats.word_as_pre_verb.get(word, 0)
        post = stats.word_as_post_verb.get(word, 0)
        action = stats.word_as_action.get(word, 0)
        total_rel = pre + post + action

        if total_rel > 0:
            pre_ratio = pre / total_rel
            post_ratio = post / total_rel
            action_ratio = action / total_rel

            scores["VERB"] = scores.get("VERB", 0.0) + action_ratio * 3.0
            scores["NOUN"] = scores.get("NOUN", 0.0) + max(pre_ratio, post_ratio) * 2.0
            scores["PRON"] = scores.get("PRON", 0.0) + (pre_ratio * 1.5 if post_ratio < 0.2 else 0.0)
            scores["ADJ"] = scores.get("ADJ", 0.0) + (pre_ratio * 1.0 if action_ratio < 0.1 else 0.0)
            scores["DET"] = scores.get("DET", 0.0) + (pre_ratio * 1.0 if action_ratio < 0.1 else 0.0)
            scores["PREP"] = scores.get("PREP", 0.0) + (post_ratio * 1.0 if action_ratio < 0.1 else 0.0)
            scores["ADV"] = scores.get("ADV", 0.0) + (post_ratio * 1.0 if action_ratio < 0.1 else 0.0)

        # 3. Position profile scoring
        positions = stats.position_counts.get(word, {})
        if positions:
            total_pos = sum(positions.values())
            avg_pos = sum(p * c for p, c in positions.items()) / total_pos
            max_pos = max(positions.keys()) + 1
            norm_pos = avg_pos / max(max_pos, 1)

            profiles, _profile_source = self.position_profiles()
            for cat, (lo, hi) in profiles.items():
                mid = (lo + hi) / 2
                dist = abs(norm_pos - mid)
                pos_score = max(0, 1.0 - dist * 3)
                scores[cat] = scores.get(cat, 0.0) + pos_score * 0.5

        # 4. Transition-based scoring (bigram context)
        left_cats: Dict[str, int] = defaultdict(int)
        right_cats: Dict[str, int] = defaultdict(int)
        for (w1, w2), c in stats.transitions.items():
            if w2 == word:
                cat_left = self._quick_category(w1)
                if cat_left:
                    left_cats[cat_left] += c
            if w1 == word:
                cat_right = self._quick_category(w2)
                if cat_right:
                    right_cats[cat_right] += c

        if left_cats.get("DET", 0) > 0 or left_cats.get("ADJ", 0) > 0:
            scores["NOUN"] = scores.get("NOUN", 0.0) + 1.0
        if right_cats.get("VERB", 0) > 0:
            scores["NOUN"] = scores.get("NOUN", 0.0) + 0.5
        if (left_cats.get("NOUN", 0) > 0 or left_cats.get("PRON", 0) > 0):
            if (right_cats.get("DET", 0) > 0 or right_cats.get("NOUN", 0) > 0
                    or right_cats.get("ADV", 0) > 0):
                scores["VERB"] = scores.get("VERB", 0.0) + 2.0
            else:
                scores["VERB"] = scores.get("VERB", 0.0) + 1.0
        if left_cats.get("DET", 0) > 0 and right_cats.get("NOUN", 0) > 0:
            scores["ADJ"] = scores.get("ADJ", 0.0) + 1.5
        if right_cats.get("NOUN", 0) > 0 or right_cats.get("ADJ", 0) > 0:
            scores["DET"] = scores.get("DET", 0.0) + 1.0
        if (left_cats.get("VERB", 0) > 0 and right_cats.get("DET", 0) > 0):
            scores["PREP"] = scores.get("PREP", 0.0) + 1.5

        if not scores:
            return "UNKNOWN", scores

        best = max(scores, key=lambda key: scores[key])
        return best, scores

    def get_func_subcategory(self, word: str) -> Optional[str]:
        """Return the function word sub-category for a word, or None.

        Returns one of FUNC_DET, FUNC_AUX, FUNC_COMP, FUNC_CONJ,
        FUNC_MARKER if the word was sub-categorized by frame analysis,
        or None if the word is a content word or hasn't been analyzed.
        """
        if hasattr(self, '_func_subcategories'):
            cached = self._func_subcategories.get(word)
            if cached is not None:
                return cached

        # Try frame classification on the fly
        ctx = self.word_grounding.get(word)
        is_ungrounded = (ctx is None or not ctx.is_grounded)
        if not is_ungrounded:
            return None

        frame_cat, frame_conf = self.classify_by_frame(word)
        if frame_cat in (FUNC_DET, FUNC_AUX, FUNC_COMP,
                         FUNC_CONJ, FUNC_MARKER):
            if not hasattr(self, '_func_subcategories'):
                self._func_subcategories = {}
            self._func_subcategories[word] = frame_cat
            return frame_cat
        return None

    def train_distributional(self, sentences: List[List[str]],
                             repetitions: int = 3) -> None:
        """Learn word categories from raw text via distributional statistics.

        Phase 1: Ingest all sentences to build statistics.
        Phase 2: Infer preliminary categories, then re-ingest with those
                 categories available (bootstrapping: once some verbs are
                 identified, verb-relative positions for other words improve).
        Phase 3: Project inferred words into core areas with Hebbian learning.

        Args:
            sentences: List of token lists (no GroundingContext needed).
            repetitions: Number of passes over the data for statistics.
        """
        # Phase 1: Initial ingestion to build statistics
        for _rep in range(repetitions):
            for sent in sentences:
                self.ingest_raw_sentence(sent)

        # Phase 2: Iterative refinement — classify, then re-ingest to
        # update verb-relative positions with preliminary categories
        self._dist_categories: Dict[str, str] = {}
        min_count = max(2, self.dist_stats.sentences_seen // 10)
        holdout = self._lexicon_holdout_set()

        for _iteration in range(2):
            # Infer categories for words not yet in core lexicons
            for word, count in self.dist_stats.word_count.items():
                if count < min_count:
                    continue
                in_lex = any(
                    word in lex for lex in self.core_lexicons.values()
                )
                if in_lex:
                    continue
                if word in holdout:
                    ctx = self.word_grounding.get(word)
                    if ctx is not None and ctx.is_grounded:
                        from ..acquisition.pos_inference import (
                            classify_word_bootstrapped,
                        )

                        cat, _ = classify_word_bootstrapped(cast("CoreParserMixin", self), word, ctx)
                        if cat != "UNKNOWN":
                            if not hasattr(self, "_bootstrap_categories"):
                                self._bootstrap_categories = {}
                            self._bootstrap_categories[word] = cat
                            self._category_cache[word] = cat
                    continue
                ctx = self.word_grounding.get(word)
                if ctx is not None and ctx.is_grounded:
                    from ..acquisition.pos_inference import classify_word_bootstrapped

                    cat, _ = classify_word_bootstrapped(cast("CoreParserMixin", self), word, ctx)
                else:
                    cat, scores = self.classify_distributional(word)
                if cat != "UNKNOWN":
                    self._dist_categories[word] = cat

            if _iteration == 0:
                # Reset verb-relative stats and re-ingest with new categories
                self.dist_stats.word_as_pre_verb = defaultdict(int)
                self.dist_stats.word_as_post_verb = defaultdict(int)
                self.dist_stats.word_as_action = defaultdict(int)
                self.dist_stats.category_transitions = defaultdict(int)
                for sent in sentences:
                    self._reingest_verb_positions(sent)

        # Phase 3: Project inferred words to core areas
        self._project_distributional_lexicon(self._dist_categories)

    def train_distributional_from_index(
        self,
        corpus_index: "CorpusIndex",
        repetitions: int = 1,
    ) -> None:
        """Fast distributional path using precompiled ``CorpusIndex``.

        Skips redundant ingestion/classification for grounded preset vocab;
        stats and categories come from the shared index.
        """
        from ..core.corpus_index import category_oracle, ingest_index_stats

        for _rep in range(repetitions):
            ingest_index_stats(cast("CoreParserMixin", self), corpus_index)

        self._dist_categories = {}
        min_count = max(2, max(1, self.dist_stats.sentences_seen // 10))
        holdout = self._lexicon_holdout_set()

        for word in corpus_index.corpus_vocab:
            ctx = self.word_grounding.get(word)
            in_lex = any(
                word in lex for lex in self.core_lexicons.values()
            )

            if word in holdout:
                if ctx is not None and ctx.is_grounded and not in_lex:
                    from ..acquisition.pos_inference import infer_holdout_categories

                    infer_holdout_categories(cast("CoreParserMixin", self), {word}, min_count=0)
                continue

            if ctx is not None and ctx.is_grounded and in_lex:
                cat = category_oracle(cast("CoreParserMixin", self), word, ctx)
                self._category_cache[word] = cat
                continue

            count = self.dist_stats.word_count.get(word, 0)
            if count < min_count:
                if ctx is not None and ctx.is_grounded and not in_lex:
                    from ..acquisition.pos_inference import (
                        infer_holdout_categories,
                    )

                    infer_holdout_categories(cast("CoreParserMixin", self), {word}, min_count=0)
                continue

            if ctx is not None and ctx.is_grounded and not in_lex:
                from ..acquisition.pos_inference import classify_word_bootstrapped

                cat, _ = classify_word_bootstrapped(cast("CoreParserMixin", self), word, ctx)
            else:
                cat, _ = self.classify_distributional(word)

            if cat != "UNKNOWN":
                if not in_lex:
                    if not hasattr(self, "_bootstrap_categories"):
                        self._bootstrap_categories = {}
                    self._bootstrap_categories[word] = cat
                    self._category_cache[word] = cat
                else:
                    self._dist_categories[word] = cat

        self._project_distributional_lexicon(self._dist_categories)

    def _lexicon_holdout_set(self) -> Set[str]:
        return set(getattr(self, "lexicon_holdouts", None) or ())

    def _project_distributional_lexicon(
        self,
        dist_categories: Dict[str, str],
    ) -> None:
        """Project distributional-only words into core areas."""
        holdout = self._lexicon_holdout_set()
        for word, cat in dist_categories.items():
            if word in holdout:
                continue
            core_area = CATEGORY_TO_CORE.get(cat)
            if core_area is None:
                continue
            phon = self.stim_map.get(word)
            if phon is None:
                continue

            if core_area not in self.core_lexicons:
                self.core_lexicons[core_area] = {}

            self._clear_core_activity(core_area)
            stim_dict = {phon: [core_area]}
            self.brain.project(stim_dict, {})
            if self.rounds > 1:
                self.brain.project_rounds(
                    target=core_area,
                    areas_by_stim=stim_dict,
                    dst_areas_by_src_area={core_area: [core_area]},
                    rounds=self.rounds - 1,
                )

            self.core_lexicons[core_area][word] = _snap(self.brain, core_area)
            self._category_cache[word] = cat

    # ==================================================================
    # Raw Text Pipeline (Feature 7)
    # ==================================================================

    def auto_ground(self, word: str) -> Optional[GroundingContext]:
        """Look up a word in lexicon data and generate GroundingContext.

        Checks both lemmas and inflected forms (e.g., "runs" -> motor).
        Results are cached for repeated lookups.

        Homographs resolve to the category-priority winner (NOUN before
        VERB, ...): "loves" grounds as the noun. This is a deliberate
        single-grounding-per-surface choice -- grounding has one slot per
        word string -- not an index limitation; POS-aware readers use
        `lookup_lexicon_entries` to see the other readings.

        Args:
            word: Word string to look up.

        Returns:
            GroundingContext if word found in lexicon, else None.
        """
        if not hasattr(self, '_auto_ground_cache'):
            self._auto_ground_cache: Dict[str, Optional[GroundingContext]] = {}

        if word in self._auto_ground_cache:
            return self._auto_ground_cache[word]

        from ..vocabulary_builder import lookup_lexicon_entry, entry_to_grounding

        result = lookup_lexicon_entry(word)
        if result is not None:
            entry, pos = result
            ctx = entry_to_grounding(entry, pos)
            self._auto_ground_cache[word] = ctx
            return ctx

        self._auto_ground_cache[word] = None
        return None

    def register_word(self, word: str) -> None:
        """Register a word for use in the parser.

        Creates phon stimulus if needed. If the word is found in lexicon data,
        also registers grounding stimuli. Idempotent.

        Args:
            word: Word string to register.
        """
        if word in self.stim_map:
            return

        # Create phon stimulus
        self.add_phon_stimulus(word)
        if hasattr(self, "_invalidate_category_cache"):
            self._invalidate_category_cache(word)

        # Try to auto-ground from lexicon
        ctx = self.auto_ground(word)
        if ctx is not None:
            self.word_grounding[word] = ctx
            # Register grounding stimuli
            for mod in _MODALITY_FIELDS:
                features = getattr(ctx, mod, [])
                for feat in features:
                    stim_name = f"{mod}_{feat}"
                    if stim_name not in self._grounding_stim_names_set:
                        self.brain.add_stimulus(stim_name, self.k)
                        self._grounding_stim_names_set.add(stim_name)

    def register_fuzzy_surface(
        self,
        canonical: str,
        surface: str,
    ) -> None:
        """Register a child-like surface form mapped to a canonical lemma."""
        if not hasattr(self, "surface_to_canonical"):
            self.surface_to_canonical: Dict[str, str] = {}

        if surface == canonical:
            self.register_word(canonical)
            return

        self.register_word(canonical)
        self.register_word(surface)
        self.surface_to_canonical[surface] = canonical

        if canonical in self.word_grounding:
            self.word_grounding[surface] = self.word_grounding[canonical]

    def resolve_surface_word(self, word: str) -> str:
        """Normalize fuzzy surface to canonical lemma when registered."""
        if hasattr(self, "surface_to_canonical"):
            return self.surface_to_canonical.get(word, word)
        return word

    def ingest_text(self, text: str) -> List[List[str]]:
        """Tokenize raw text into sentences and register all words.

        Splits on sentence-ending punctuation (.!?), tokenizes by whitespace,
        lowercases, and registers each word.

        Args:
            text: Raw text string.

        Returns:
            List of token lists (one per sentence).
        """
        import re

        # Split on sentence boundaries
        raw_sents = re.split(r'[.!?]+', text)

        sentences: List[List[str]] = []
        for raw in raw_sents:
            raw = raw.strip()
            if not raw:
                continue
            tokens = raw.lower().split()
            if not tokens:
                continue
            for word in tokens:
                self.register_word(word)
            sentences.append(tokens)

        return sentences

    def train_from_text(self, text: str, use_grounding: bool = True) -> None:
        """Train the parser from raw text (no GroundedSentence needed).

        Full pipeline: tokenize -> auto-ground -> train lexicon -> train
        distributional -> train roles (unsupervised) -> train phrases +
        word order.

        Args:
            text: Raw text string containing sentences.
            use_grounding: If True, auto-ground words from lexicon data.
        """
        sentences = self.ingest_text(text)
        self.train_from_sentences(sentences, use_grounding=use_grounding)

    def train_from_sentences(self, sentences: List[List[str]],
                             use_grounding: bool = True) -> None:
        """Train from pre-tokenized sentences (no GroundingContext needed).

        Args:
            sentences: List of token lists.
            use_grounding: If True, auto-ground words from lexicon data.
        """
        # Register all words
        for sent in sentences:
            for word in sent:
                self.register_word(word)

        # Phase 1: Train lexicon for grounded words
        if use_grounding:
            grounded_sents = []
            for sent in sentences:
                contexts = []
                for word in sent:
                    ctx = self.word_grounding.get(word, GroundingContext())
                    contexts.append(ctx)
                grounded_sents.append(GroundedSentence(
                    words=sent,
                    contexts=contexts,
                    roles=[None] * len(sent),
                ))
            self.train_lexicon(grounded_sents)

        # Phase 2: Train distributional categories
        self.train_distributional(sentences, repetitions=3)

        # Phase 3: Train roles unsupervised
        grounded_sents_for_roles = []
        for sent in sentences:
            contexts = [self.word_grounding.get(w, GroundingContext())
                        for w in sent]
            grounded_sents_for_roles.append(GroundedSentence(
                words=sent, contexts=contexts,
                roles=[None] * len(sent),
            ))
        self.train_unsupervised(grounded_sents_for_roles, repetitions=3)

        # Phase 4: Train phrases and word order
        self.train_phrases(grounded_sents_for_roles)
        self.train_word_order(grounded_sents_for_roles)

        # Phase 5: Train tense, mood, polarity, conjunctions
        self.train_tense(sentences)
        self.train_mood(sentences)
        self.train_polarity(sentences)
        self.train_conjunctions(sentences)

    # ==================================================================
    # Word Order Typology Learning (Feature 8)
    # ==================================================================

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def record_role_order_evidence(
        self,
        words: List[str],
        roles: List[Optional[str]],
    ) -> Optional[str]:
        """Tally the S/V/O permutation of one ROLE-ANNOTATED sentence.

        Role annotations are the only evidence in this model that can tell a
        subject-initial order from its object-initial twin.  ``agent`` maps to
        slot ``S``, ``patient`` to ``O``, ``action`` to ``V``; adjacent repeats
        of a slot are collapsed (a multi-word NP still contributes one ``S``),
        and anything else is skipped.  A sentence contributes an observation
        only if the collapsed sequence is a full permutation of S, V and O --
        intransitives carry no information about object position.

        Returns the order label recorded, or None.
        """
        if len(words) != len(roles):
            raise ValueError(
                "words and roles must have the same length; role evidence "
                "must stay aligned with its sentence"
            )
        seq: List[str] = []
        slot_of = {"agent": "S", "action": "V", "patient": "O"}
        for role in roles:
            slot = slot_of.get(role) if role else None
            if slot is None:
                continue
            if seq and seq[-1] == slot:
                continue
            seq.append(slot)

        label = "".join(seq)
        if label not in WORD_ORDERS:
            return None
        self.dist_stats.role_order_counts[label] += 1
        return label

    # ------------------------------------------------------------------
    # Typology inference
    # ------------------------------------------------------------------

    def infer_word_order(
        self,
        *,
        use_role_evidence: bool = True,
    ) -> Tuple[str, float]:
        """Infer the constituent-order typology from corpus evidence.

        The output space is the SIX basic orders -- SVO, SOV, VSO, OSV, OVS,
        VOS.  That six-member label set is BUILT IN (``WORD_ORDERS``); the
        model is handed the hypothesis space and chooses within it.  Which
        label a corpus gets is learned.

        TWO EVIDENCE SOURCES, in precedence order.

        1. ROLE-ORDER COUNTS (``dist_stats.role_order_counts``), accumulated
           from role-annotated training sentences.  These observe the full
           permutation directly, so all six orders are separable.  When
           present this evidence is authoritative and the confidence returned
           is the purity of the winning permutation.

        2. CATEGORY TRANSITION STATISTICS, unsupervised.  These can only place
           the VERB: ``N->V`` without ``V->N`` means verb-final, ``V->N``
           without ``N->V`` means verb-initial, both means verb-medial.  Each
           class contains one subject-initial and one object-initial order
           with IDENTICAL transition statistics (SVO/OVS, SOV/OSV, VSO/VOS),
           so this source cannot choose between them.  The subject-initial
           member is returned as a tie-break and
           ``self.word_order_identifiable`` is set False.  A transitions-only
           "SVO" is NOT evidence against OVS.

        Side effects: sets ``self.word_order_evidence`` to ``"roles"``,
        ``"transitions"`` or ``"none"``, and ``self.word_order_identifiable``.

        Args:
            use_role_evidence: When False, ignore ``role_order_counts`` and
                use transitions only. ``role_order_counts`` accumulates across
                every corpus the parser has ever seen, so a caller that wants
                the typology of ONE specific unannotated corpus (e.g.
                ``train_word_order_typological`` on raw token lists) must not
                let annotations from an earlier corpus decide it.

        Returns:
            (typology_label, confidence) with confidence in [0, 1].
        """
        role_counts = (
            (getattr(self.dist_stats, "role_order_counts", None) or {})
            if use_role_evidence else {}
        )
        role_total = sum(role_counts.get(o, 0) for o in WORD_ORDERS)

        if role_total > 0:
            best = max(WORD_ORDERS, key=lambda o: role_counts.get(o, 0))
            self.word_order_evidence = "roles"
            self.word_order_identifiable = True
            return best, role_counts.get(best, 0) / role_total

        ct = self.dist_stats.category_transitions

        # Verb followed by nominal material (verb precedes a noun phrase).
        v_to_np = (ct.get(("VERB", "NOUN"), 0) +
                   ct.get(("VERB", "DET"), 0) +
                   ct.get(("VERB", "PRON"), 0))
        # Nominal material followed by verb.
        n_to_v = (ct.get(("NOUN", "VERB"), 0) +
                  ct.get(("PRON", "VERB"), 0) +
                  ct.get(("ADJ", "VERB"), 0))

        # Scores over the three VERB-POSITION classes, not over the six
        # orders. Preserved from the previous three-way implementation so a
        # subject-initial corpus scores exactly as it did before.
        class_scores = {
            "medial": min(n_to_v, v_to_np) + v_to_np * 0.5,
            "final": n_to_v * 2 if v_to_np == 0 else n_to_v * 0.5,
            "initial": v_to_np * 2 if n_to_v == 0 else 0.0,
        }
        total = max(sum(class_scores.values()), 1.0)

        if max(class_scores.values()) <= 0.0:
            self.word_order_evidence = "none"
            self.word_order_identifiable = False
            return "SVO", 0.0

        # Deterministic tie-break order matching the historical behaviour:
        # medial >= final >= initial.
        best_class = max(
            ("medial", "final", "initial"), key=lambda c: class_scores[c],
        )
        self.word_order_evidence = "transitions"
        # The class is identified; the S/O ordering within it is not.
        self.word_order_identifiable = False
        return (canonical_for_class(best_class),
                class_scores[best_class] / total)

    def _update_word_order_from_evidence(self) -> Optional[str]:
        """Commit ``infer_word_order`` to ``self.word_order_type`` if useful.

        Leaves the existing value alone when there is no evidence at all, so a
        typology set explicitly by a caller is not clobbered.
        """
        order, conf = self.infer_word_order()
        if getattr(self, "word_order_evidence", "none") == "none":
            return getattr(self, "word_order_type", None)
        self.word_order_type = order
        self.word_order_confidence = conf
        return order

    def train_word_order_typological(self, sentences) -> None:
        """Learn the constituent-order typology from training sentences.

        Accepts either raw token lists (``List[List[str]]``) or
        ``GroundedSentence`` objects.  Raw lists supply only category
        transitions, which fixes the verb's position but cannot separate a
        subject-initial order from its object-initial twin (see
        ``infer_word_order``).  Grounded sentences additionally supply role
        annotations, which can, so pass those when the object-initial orders
        are in play.

        Args:
            sentences: List of token lists, or list of GroundedSentence.
        """
        saw_roles = False
        for sent in sentences:
            words = getattr(sent, "words", sent)
            roles = getattr(sent, "roles", None)
            if roles is not None:
                if self.record_role_order_evidence(
                    list(words), list(roles),
                ) is not None:
                    saw_roles = True

            cats = []
            for word in words:
                grounding = self.word_grounding.get(word)
                cat, _ = self.classify_word_cached(word, grounding=grounding)
                cats.append(cat)
            for idx in range(len(cats) - 1):
                if cats[idx] != "UNKNOWN" and cats[idx + 1] != "UNKNOWN":
                    self.dist_stats.category_transitions[
                        (cats[idx], cats[idx + 1])] += 1

        # Infer word order from THIS call's sentences. If they carried no role
        # annotations, role counts left over from an earlier corpus must not
        # decide the answer -- otherwise a parser trained on English and then
        # shown SOV text would still report SVO.
        self.word_order_type, self.word_order_confidence = (
            self.infer_word_order(use_role_evidence=saw_roles)
        )
