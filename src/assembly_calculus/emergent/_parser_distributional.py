"""Distributional learning mixin for EmergentParser.

Provides raw-text ingestion, distributional category inference,
word-order typology detection, and the full train-from-text pipeline.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.assembly_calculus.ops import _snap
from .areas import (
    CORE_TO_CATEGORY, CATEGORY_TO_CORE, GROUNDING_TO_CORE,
    DET_CORE, CORE_AREAS,
)
from .grounding import GroundingContext
from .training_data import GroundedSentence
from ._parser_core import _MODALITY_FIELDS

# Category -> typical normalized position range in a sentence
# Used by distributional classification to score position profiles.
_CATEGORY_POSITION_PROFILES = {
    "DET": (0.0, 0.25),    # Early in phrases
    "ADJ": (0.15, 0.35),   # After determiners, before nouns
    "NOUN": (0.2, 0.5),    # Middle positions
    "VERB": (0.3, 0.6),    # After subject, before object
    "ADV": (0.5, 0.8),     # Late in sentences
    "PREP": (0.5, 0.75),   # Between phrases
    "PRON": (0.0, 0.3),    # Early (subject pronouns)
}


class DistributionalMixin:
    """Distributional learning, raw text pipeline, and word order typology."""

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
                phon = f"phon_{word}"
                self.brain.add_stimulus(phon, self.k)
                self.stim_map[word] = phon

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
            if cats[idx] and cats[idx + 1]:
                stats.category_transitions[(cats[idx], cats[idx + 1])] += 1

    def _quick_category(self, word: str) -> Optional[str]:
        """Return known category for a word (from grounding or lexicon), or None."""
        ctx = self.word_grounding.get(word)
        if ctx is not None:
            return CORE_TO_CATEGORY.get(GROUNDING_TO_CORE.get(
                ctx.dominant_modality))
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
            if cats[idx] and cats[idx + 1]:
                stats.category_transitions[(cats[idx], cats[idx + 1])] += 1

    def classify_distributional(self, word: str
                                ) -> Tuple[str, Dict[str, float]]:
        """Infer word category from distributional statistics.

        Scores each category based on:
        1. Position profile similarity (where the word appears in sentences)
        2. Verb-relative position (pre-verb -> NOUN, at-verb -> VERB, etc.)
        3. Co-occurrence with known-category words

        Args:
            word: Word string to classify.

        Returns:
            (category_label, {category: confidence_score})
        """
        stats = self.dist_stats
        count = stats.word_count.get(word, 0)
        if count == 0:
            return "UNKNOWN", {}

        scores: Dict[str, float] = {}

        # 1. Verb-relative position scores
        pre = stats.word_as_pre_verb.get(word, 0)
        post = stats.word_as_post_verb.get(word, 0)
        action = stats.word_as_action.get(word, 0)
        total_rel = pre + post + action

        if total_rel > 0:
            pre_ratio = pre / total_rel
            post_ratio = post / total_rel
            action_ratio = action / total_rel

            scores["VERB"] = action_ratio * 3.0
            scores["NOUN"] = max(pre_ratio, post_ratio) * 2.0
            scores["PRON"] = pre_ratio * 1.5 if post_ratio < 0.2 else 0.0
            scores["ADJ"] = pre_ratio * 1.0 if action_ratio < 0.1 else 0.0
            scores["DET"] = pre_ratio * 1.0 if action_ratio < 0.1 else 0.0
            scores["PREP"] = post_ratio * 1.0 if action_ratio < 0.1 else 0.0
            scores["ADV"] = post_ratio * 1.0 if action_ratio < 0.1 else 0.0

        # 2. Position profile scoring
        positions = stats.position_counts.get(word, {})
        if positions:
            total_pos = sum(positions.values())
            avg_pos = sum(p * c for p, c in positions.items()) / total_pos
            # Normalize by typical sentence length
            max_pos = max(positions.keys()) + 1
            norm_pos = avg_pos / max(max_pos, 1)

            for cat, (lo, hi) in _CATEGORY_POSITION_PROFILES.items():
                mid = (lo + hi) / 2
                dist = abs(norm_pos - mid)
                pos_score = max(0, 1.0 - dist * 3)
                scores[cat] = scores.get(cat, 0.0) + pos_score * 0.5

        # 3. Transition-based scoring (bigram context)
        # Check what categories typically precede/follow this word
        left_cats: Dict[str, int] = defaultdict(int)
        right_cats: Dict[str, int] = defaultdict(int)
        for (w1, w2), count in stats.transitions.items():
            if w2 == word:
                cat_left = self._quick_category(w1)
                if cat_left:
                    left_cats[cat_left] += count
            if w1 == word:
                cat_right = self._quick_category(w2)
                if cat_right:
                    right_cats[cat_right] += count

        # NOUN follows DET/ADJ, precedes VERB
        if left_cats.get("DET", 0) > 0 or left_cats.get("ADJ", 0) > 0:
            scores["NOUN"] = scores.get("NOUN", 0.0) + 1.0
        if right_cats.get("VERB", 0) > 0:
            scores["NOUN"] = scores.get("NOUN", 0.0) + 0.5

        # VERB follows NOUN/PRON, precedes DET/NOUN
        if (left_cats.get("NOUN", 0) > 0 or left_cats.get("PRON", 0) > 0):
            if (right_cats.get("DET", 0) > 0 or right_cats.get("NOUN", 0) > 0
                    or right_cats.get("ADV", 0) > 0):
                scores["VERB"] = scores.get("VERB", 0.0) + 2.0
            else:
                scores["VERB"] = scores.get("VERB", 0.0) + 1.0

        # ADJ follows DET, precedes NOUN
        if left_cats.get("DET", 0) > 0 and right_cats.get("NOUN", 0) > 0:
            scores["ADJ"] = scores.get("ADJ", 0.0) + 1.5

        # DET precedes NOUN/ADJ
        if right_cats.get("NOUN", 0) > 0 or right_cats.get("ADJ", 0) > 0:
            scores["DET"] = scores.get("DET", 0.0) + 1.0

        # PREP follows VERB/NOUN, precedes DET
        if (left_cats.get("VERB", 0) > 0 and right_cats.get("DET", 0) > 0):
            scores["PREP"] = scores.get("PREP", 0.0) + 1.5

        if not scores:
            return "UNKNOWN", scores

        best = max(scores, key=scores.get)
        return best, scores

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

        for _iteration in range(2):
            # Infer categories for ungrounded words
            for word, count in self.dist_stats.word_count.items():
                if count < min_count:
                    continue
                if word in self.word_grounding:
                    continue
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
        for word, cat in self._dist_categories.items():
            core_area = CATEGORY_TO_CORE.get(cat)
            if core_area is None:
                continue
            phon = self.stim_map.get(word)
            if phon is None:
                continue

            # Initialize lexicon if needed
            if core_area not in self.core_lexicons:
                self.core_lexicons[core_area] = {}

            # Project phon -> core area with recurrence
            self.brain._engine.reset_area_connections(core_area)
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
            self.brain._engine.reset_area_connections(core_area)

    # ==================================================================
    # Raw Text Pipeline (Feature 7)
    # ==================================================================

    def auto_ground(self, word: str) -> Optional[GroundingContext]:
        """Look up a word in lexicon data and generate GroundingContext.

        Checks both lemmas and inflected forms (e.g., "runs" -> motor).
        Results are cached for repeated lookups.

        Args:
            word: Word string to look up.

        Returns:
            GroundingContext if word found in lexicon, else None.
        """
        if not hasattr(self, '_auto_ground_cache'):
            self._auto_ground_cache: Dict[str, Optional[GroundingContext]] = {}

        if word in self._auto_ground_cache:
            return self._auto_ground_cache[word]

        from .vocabulary_builder import lookup_lexicon_entry, entry_to_grounding

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
        phon = f"phon_{word}"
        self.brain.add_stimulus(phon, self.k)
        self.stim_map[word] = phon

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

    def infer_word_order(self) -> Tuple[str, float]:
        """Infer word order typology (SVO/SOV/VSO) from category transitions.

        Key discriminating patterns:
        - SVO: NOUN->VERB (subject before verb) AND VERB->DET/VERB->NOUN
               (verb precedes object NP)
        - SOV: NOUN->VERB (end of sentence) but NO VERB->NOUN/VERB->DET
               (verb never precedes a noun)
        - VSO: VERB->NOUN (verb starts) AND DET->VERB or similar

        Returns:
            (typology_label, confidence) where confidence is in [0, 1].
        """
        ct = self.dist_stats.category_transitions

        # Verb-to-noun/det transitions (verb followed by object NP)
        v_to_np = (ct.get(("VERB", "NOUN"), 0) +
                   ct.get(("VERB", "DET"), 0) +
                   ct.get(("VERB", "PRON"), 0))

        # Noun-to-verb transitions (noun followed by verb)
        n_to_v = (ct.get(("NOUN", "VERB"), 0) +
                  ct.get(("PRON", "VERB"), 0))

        # Verb-first patterns (verb at start -> VERB->NOUN+NOUN)
        v_first = (ct.get(("VERB", "NOUN"), 0) +
                   ct.get(("VERB", "DET"), 0))

        # SVO: both NV and V->NP are present
        svo_score = min(n_to_v, v_to_np) + v_to_np * 0.5

        # SOV: NV high but V->NP absent (verb at end)
        sov_score = n_to_v * 2 if v_to_np == 0 else n_to_v * 0.5

        # VSO: V->NP much higher than NV (verb first)
        vso_score = v_to_np * 2 if n_to_v == 0 else 0

        total = max(svo_score + sov_score + vso_score, 1)

        if svo_score >= sov_score and svo_score >= vso_score:
            return "SVO", svo_score / total
        elif sov_score > svo_score and sov_score >= vso_score:
            return "SOV", sov_score / total
        else:
            return "VSO", vso_score / total

    def train_word_order_typological(self, sentences: List[List[str]]
                                     ) -> None:
        """Learn word order typology from raw sentence data.

        Classifies each word, builds category transitions, and infers
        the dominant word order pattern (SVO/SOV/VSO).

        Args:
            sentences: List of token lists.
        """
        # Build category transitions from these sentences
        for sent in sentences:
            cats = []
            for word in sent:
                grounding = self.word_grounding.get(word)
                cat, _ = self.classify_word(word, grounding=grounding)
                cats.append(cat)
            for idx in range(len(cats) - 1):
                if cats[idx] != "UNKNOWN" and cats[idx + 1] != "UNKNOWN":
                    self.dist_stats.category_transitions[
                        (cats[idx], cats[idx + 1])] += 1

        # Infer word order
        self.word_order_type, order_conf = self.infer_word_order()
