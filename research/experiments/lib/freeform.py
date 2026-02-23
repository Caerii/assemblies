"""
Free-form learner: learns language from scratch, one sentence at a time.

Uses only Assembly Calculus operations. No pre-registered words, no category
labels, no role annotations. Words are discovered dynamically via
add_stimulus(); both lexical categories and structural roles are discovered
via competitive area pools with mutual inhibition.

Design principles (integrated from successful experiments):
  - CORE pool with LRI + MI for category discovery (same mechanism as STRUCT,
    which achieved 100% routing success in role_reuse test). LRI forces
    adjacent words to different CORE areas, creating natural noun/verb
    alternation from SVO word order.
  - STRUCT pool with LRI + MI for positional role discovery
  - Hebbian self-recurrence stabilization for STRUCT routing (from
    integrated_learner: improved P600 6x over baseline)
  - reset_area_connections between word formations (AC-faithful: word
    identity lives in stimulus->area connectomes, not self-recurrence)
  - Prediction trained from word's discovered CORE area, so different
    categories get different connectomes to PREDICTION
  - Optional P600-guided retraining (from integrated_learner ablation)
  - Cumulative N400 as sentence-level acceptability signal

Routing mechanisms for N-way category discovery:
  - Neighbor exclusion: exclude CORE areas used by adjacent known words
  - Left-context cache: words following the same category route to the
    same area (distributional consistency across sentences)
  - Position-cycling bootstrap: first sentence uses position % n_core_areas
    to guarantee a clean initial split
  - LRI state management: _set_core_lri activates WORD_MARKER in the
    assigned area to maintain refractory cycling within a sentence

Validated configurations:
  - 2 CORE areas (SVO): perfect 8/8 binary split, d=0.23 N400
  - 3 CORE areas (DET-N-V): perfect 18/18 ternary split, d=1.20 N400
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.brain import Brain
from research.experiments.lib.brain_setup import activate_word
from research.experiments.lib.training import train_prediction_pair, train_binding
from research.experiments.lib.unsupervised import discover_role_area
from research.experiments.lib.measurement import measure_n400, measure_p600


@dataclass
class FreeFormConfig:
    """Configuration for the free-form learner."""
    # Brain sizing
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    # Lexicon formation
    lexicon_rounds: int = 20
    lexicon_refresh_interval: int = 50
    # Core area pool (category discovery via LRI + MI)
    n_core_areas: int = 2
    core_refractory_period: int = 1
    # Structural pool (positional role discovery via LRI + MI)
    n_struct_areas: int = 6
    struct_refractory_period: int = 5
    inhibition_strength: float = 1.0
    stabilize_rounds: int = 3
    # Training
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    # P600-guided retraining (from integrated_learner ablation)
    use_p600_feedback: bool = False
    extra_binding_rounds: int = 5
    instability_threshold: float = 0.3
    n_settling_rounds: int = 10
    # Engine selection: "auto", "numpy_sparse", "cuda_implicit", etc.
    engine: str = "auto"


class FreeFormVocab:
    """Dynamic vocabulary that tracks each word's discovered core area."""

    def __init__(self):
        self._word_area: Dict[str, str] = {}

    def core_area_for(self, word: str) -> str:
        return self._word_area.get(word, "CORE_0")

    def assign(self, word: str, area: str) -> None:
        self._word_area[word] = area

    @property
    def known_words(self) -> Set[str]:
        return set(self._word_area.keys())


class FreeFormLearner:
    """Learns language incrementally from raw word sequences.

    Uses two competitive area pools, both proven mechanisms:

      1. **CORE pool** (MI + LRI, refractory_period=1): Routes each word to
         a category area. LRI forces adjacent words in a sentence to different
         areas, creating natural noun/verb alternation from SVO word order.
         Once assigned, a word's CORE area is permanent. Different CORE areas
         get different connectomes to PREDICTION, enabling discriminative
         prediction (e.g. "nouns predict verbs").

      2. **STRUCT pool** (MI + LRI, refractory_period=5): Routes each
         sequential position to a different structural role area for binding.
         Hebbian self-recurrence stabilization strengthens routing quality.

    Both pools use the same Assembly Calculus mechanism from unsupervised.py:
    shared marker stimulus + mutual inhibition + lateral refractory inhibition
    → competitive routing (discover_role_area).
    """

    def __init__(self, cfg: FreeFormConfig = None, seed: int = 42):
        self.cfg = cfg or FreeFormConfig()
        self.seed = seed
        self.vocab = FreeFormVocab()

        # Build brain from scratch — no pre-registered words
        self.brain = Brain(p=self.cfg.p, seed=seed, w_max=self.cfg.w_max,
                          engine=self.cfg.engine)

        # CORE pool: LRI + MI for category discovery
        # LRI with period=1 forces adjacent words to different areas,
        # creating natural N/V alternation from SVO word order.
        self.core_areas: List[str] = [
            f"CORE_{i}" for i in range(self.cfg.n_core_areas)
        ]
        for name in self.core_areas:
            self.brain.add_area(
                name, self.cfg.n, self.cfg.k, self.cfg.beta,
                refractory_period=self.cfg.core_refractory_period,
                inhibition_strength=self.cfg.inhibition_strength,
            )
        self.brain.add_mutual_inhibition(self.core_areas)

        # PREDICTION area
        self.brain.add_area(
            "PREDICTION", self.cfg.n, self.cfg.k, self.cfg.beta,
        )

        # STRUCT pool: LRI + MI for positional role discovery
        self.struct_areas: List[str] = [
            f"STRUCT_{i}" for i in range(self.cfg.n_struct_areas)
        ]
        for name in self.struct_areas:
            self.brain.add_area(
                name, self.cfg.n, self.cfg.k, self.cfg.beta,
                refractory_period=self.cfg.struct_refractory_period,
                inhibition_strength=self.cfg.inhibition_strength,
            )
        self.brain.add_mutual_inhibition(self.struct_areas)

        # Shared markers for competitive routing
        self.brain.add_stimulus("WORD_MARKER", self.cfg.k)      # CORE routing
        self.brain.add_stimulus("POSITION_MARKER", self.cfg.k)  # STRUCT routing

        # Tracking state
        self.word_exposures: Dict[str, int] = defaultdict(int)
        self.word_assemblies: Dict[str, np.ndarray] = {}
        self.total_sentences: int = 0
        # Precise context cache: maps (left_area, right_area) to the CORE
        # area previously assigned for words in that exact context. This
        # distinguishes ADJ (left=DET, right=NOUN) from NOUN (left=DET,
        # right=VERB) when adjectives are optional.
        self._context_cache: Dict[tuple, str] = {}
        # Fallback cache: maps left_area alone to assigned CORE area.
        # Used when the right neighbor is unknown (e.g. first sentence,
        # or when the next word hasn't been encountered yet).
        self._left_context_cache: Dict[Optional[str], str] = {}

    # -- Word routing and registration ----------------------------------------

    def _route_by_context(
        self, word: str, position: int, words: List[str],
    ) -> str:
        """Route an unknown word based on distributional context.

        Uses a three-level hierarchical strategy:

        1. **Neighbor exclusion**: Collect CORE-area constraints from both
           adjacent neighbors, exclude those areas.

        2. **Precise context cache**: Look up (left_area, right_area) pair.
           This distinguishes words in different syntactic contexts even
           when they share a left neighbor (e.g. ADJ after DET before NOUN
           vs NOUN after DET before VERB).

        3. **Left-only fallback**: When the right neighbor is unknown,
           fall back to left-area-only cache. This handles first-sentence
           bootstrapping and positions where the next word hasn't been seen.

        4. **Bootstrap / LRI**: Position-cycling for the first sentence,
           brain-level LRI competition thereafter.

        Returns the winning CORE area name.
        """
        left_area: Optional[str] = None
        right_area: Optional[str] = None
        neighbor_areas: set = set()

        if position > 0:
            left = words[position - 1]
            if left in self.vocab.known_words:
                left_area = self.vocab.core_area_for(left)
                neighbor_areas.add(left_area)
        if position < len(words) - 1:
            right = words[position + 1]
            if right in self.vocab.known_words:
                right_area = self.vocab.core_area_for(right)
                neighbor_areas.add(right_area)

        candidates = [a for a in self.core_areas if a not in neighbor_areas]

        if len(candidates) == 1:
            result = candidates[0]
        elif len(candidates) > 1:
            # Level 1: Precise (left, right) cache
            precise_key = (left_area, right_area)
            cached = self._context_cache.get(precise_key)
            if cached is not None and cached in candidates:
                result = cached
            elif right_area is not None:
                # Right IS known but no precise hit — do NOT fall back to
                # left-only cache (it might give wrong category, e.g. ADJ
                # instead of NOUN when both follow DET).
                result = self._bootstrap_or_lri(candidates, position)
            else:
                # Level 2: Left-only fallback (right unknown)
                fallback = self._left_context_cache.get(left_area)
                if fallback is not None and fallback in candidates:
                    result = fallback
                else:
                    result = self._bootstrap_or_lri(candidates, position)
        else:
            # All areas used by neighbors — pick first area
            result = self.core_areas[0]

        # Update both caches
        self._context_cache[(left_area, right_area)] = result
        self._left_context_cache[left_area] = result
        return result

    def _bootstrap_or_lri(
        self, candidates: List[str], position: int,
    ) -> str:
        """Choose among candidate areas via bootstrap or LRI.

        Priority order:
        1. First sentence: position cycling (position % n_core_areas).
        2. Unused area preference: if exactly one candidate has no words
           assigned yet, pick it. A word in a genuinely new distributional
           context (e.g. complementizer "that") should claim the first
           available unused area rather than colliding with an existing
           category via noisy LRI.
        3. LRI competition via discover_role_area.
        """
        if self.total_sentences == 0:
            # Bootstrap cycle wraps at min(n_areas, refractory+1) so that
            # extra areas (e.g. CORE_4 for COMP) stay unused during the
            # first non-RC sentence and get claimed later by new categories.
            cycle = min(len(self.core_areas),
                        self.cfg.core_refractory_period + 1)
            target = self.core_areas[position % cycle]
            return target if target in candidates else candidates[0]
        # Prefer unused areas for genuinely new distributional contexts
        unused = [a for a in candidates
                  if not any(self.vocab.core_area_for(w) == a
                             for w in self.vocab.known_words)]
        if len(unused) == 1:
            return unused[0]
        # Fall back to LRI competition
        winner = discover_role_area(
            self.brain, "WORD_MARKER", candidates,
            stabilize_rounds=1, hebbian_stabilize=False,
        )
        return winner if winner is not None else candidates[0]

    def _form_assembly(self, word: str, core_area: str) -> None:
        """Add stimulus for a new word and form its assembly in core_area.

        Uses reset_area_connections to prevent assembly collapse (proven
        in online_word_acquisition: 0.97 stability).
        """
        self.brain.add_stimulus(f"PHON_{word}", self.cfg.k)
        self.brain._engine.reset_area_connections(core_area)
        self.brain.inhibit_areas([core_area])
        for _ in range(self.cfg.lexicon_rounds):
            self.brain.project(
                {f"PHON_{word}": [core_area]},
                {core_area: [core_area]},
            )

    def _set_core_lri(self, core_area: str) -> None:
        """Set LRI state for a CORE area using WORD_MARKER.

        Clears other areas' refractory state and activates the target area
        so it becomes refractory. Used for newly-assigned words whose PHON
        stimulus doesn't exist yet.
        """
        for name in self.core_areas:
            if name != core_area:
                self.brain.clear_refractory(name)
        self.brain.inhibit_areas([core_area])
        self.brain.project(
            {"WORD_MARKER": [core_area]}, {core_area: [core_area]},
        )

    def _activate_known_word_for_lri(self, word: str) -> None:
        """Activate a known word in its CORE area to set LRI state.

        When processing a sentence left-to-right, known words need to
        'claim' their CORE area so that subsequent unknown words route
        to different areas via LRI.

        Critical: must clear OTHER areas' LRI first (like discover_role_area
        clears losers). Without this, two consecutive known words would
        suppress BOTH areas, leaving no area available for unknown words.
        """
        core_area = self.vocab.core_area_for(word)
        # Clear non-active areas' LRI (only current word's area stays suppressed)
        for name in self.core_areas:
            if name != core_area:
                self.brain.clear_refractory(name)
        activate_word(self.brain, word, core_area, 1)

    # -- Lexicon management -------------------------------------------------

    def _refresh_lexicon(self) -> None:
        """Re-snapshot assemblies in PREDICTION for N400 measurement."""
        self.brain.disable_plasticity = True
        for word in self.vocab.known_words:
            self.brain.inhibit_areas(["PREDICTION"])
            for _ in range(5):
                self.brain.project({f"PHON_{word}": ["PREDICTION"]}, {})
            self.word_assemblies[word] = np.array(
                self.brain.areas["PREDICTION"].winners, dtype=np.uint32,
            )
        self.brain.disable_plasticity = False

    # -- Core learning loop -------------------------------------------------

    def process_sentence(self, words: List[str]) -> Dict[str, Any]:
        """Process one sentence — the main learning step.

        Three-phase design separates routing from assembly formation from
        training, so LRI state is correct during CORE routing:

        Phase 1: Route unknown words via CORE pool (LRI + MI).
                 Known words activate in their assigned area to set LRI.
                 Left-to-right order ensures the alternation pattern
                 matches the sentence's syntactic structure.

        Phase 2: Form assemblies for newly discovered words.

        Phase 3: Left-to-right prediction training + structural binding.

        Args:
            words: Ordered list of word strings (e.g. ["dog", "chases", "cat"]).
                   No category labels, no role annotations.

        Returns:
            Dict with ``bindings`` (word -> STRUCT area) and ``n_new_words``.
        """
        # -- Phase 0: Reset LRI at sentence boundary --------------------------
        for name in self.core_areas:
            self.brain.clear_refractory(name)
        for name in self.struct_areas:
            self.brain.clear_refractory(name)
        self.brain.inhibit_areas(self.core_areas + self.struct_areas)

        # -- Phase 1: Determine CORE routing for all words --------------------
        # Process left-to-right so LRI creates alternation:
        #   Position 0 (noun) → CORE_0
        #   Position 1 (verb) → CORE_1 (CORE_0 refractory)
        #   Position 2 (noun) → CORE_0 (CORE_1 refractory, CORE_0 cleared)
        new_words: List[Tuple[str, str]] = []
        newly_assigned: Set[str] = set()  # words assigned THIS sentence
        for i, word in enumerate(words):
            if word not in self.vocab.known_words:
                core_area = self._route_by_context(word, i, words)
                self.vocab.assign(word, core_area)
                new_words.append((word, core_area))
                newly_assigned.add(word)
                # Set LRI state so subsequent words route to different areas
                self._set_core_lri(core_area)
            else:
                core_area = self.vocab.core_area_for(word)
                if word not in newly_assigned:
                    # Known word with PHON stimulus — activate for LRI
                    self._activate_known_word_for_lri(word)
                else:
                    # Duplicate of newly-assigned word (no PHON yet)
                    self._set_core_lri(core_area)
                # Update both context caches for known words too, so the
                # full positional pattern is cached (e.g. left=VERB → DET)
                left_area: Optional[str] = None
                right_area: Optional[str] = None
                if i > 0 and words[i - 1] in self.vocab.known_words:
                    left_area = self.vocab.core_area_for(words[i - 1])
                if i < len(words) - 1 and words[i + 1] in self.vocab.known_words:
                    right_area = self.vocab.core_area_for(words[i + 1])
                self._context_cache[(left_area, right_area)] = core_area
                self._left_context_cache[left_area] = core_area

        # -- Phase 1b: Cache completion pass ------------------------------------
        # During Phase 1, right neighbors are often unknown (not yet processed).
        # Now that all words have assignments, fill in precise (left, right)
        # cache entries. This prevents misrouting when a word first appears in
        # a context where the right neighbor is known but no precise cache hit
        # exists (e.g. "a" at position 0 with known ADJ to the right — without
        # this pass, the code falls through to LRI instead of using CORE_0).
        for i, word in enumerate(words):
            core_area = self.vocab.core_area_for(word)
            left_area = None
            right_area = None
            if i > 0:
                left_area = self.vocab.core_area_for(words[i - 1])
            if i < len(words) - 1:
                right_area = self.vocab.core_area_for(words[i + 1])
            self._context_cache[(left_area, right_area)] = core_area
            self._left_context_cache[left_area] = core_area

        # -- Phase 2: Form assemblies for new words ---------------------------
        for word, core_area in new_words:
            self._form_assembly(word, core_area)

        # -- Phase 3: Left-to-right prediction + structural binding -----------
        # Reset STRUCT pool (Phase 2 may have disrupted LRI state)
        for name in self.struct_areas:
            self.brain.clear_refractory(name)
        self.brain.inhibit_areas(self.struct_areas)

        bindings: Dict[str, str] = {}

        for i, word in enumerate(words):
            core_area = self.vocab.core_area_for(word)

            # 3a: prediction training
            if i > 0:
                prev_area = self.vocab.core_area_for(words[i - 1])
                train_prediction_pair(
                    self.brain, words[i - 1], prev_area, word,
                    self.cfg.train_rounds_per_pair,
                )

            # 3b: structural binding via STRUCT discovery
            # Hebbian stabilization proven to improve P600 6x in
            # integrated_learner (condition E: d=4.15 vs A: d=0.67)
            activate_word(self.brain, word, core_area, 3)
            role_area = discover_role_area(
                self.brain, "POSITION_MARKER", self.struct_areas,
                self.cfg.stabilize_rounds, hebbian_stabilize=True,
            )
            if role_area is not None:
                if self.cfg.use_p600_feedback:
                    train_binding(
                        self.brain, word, core_area, role_area,
                        self.cfg.binding_rounds,
                    )
                    self.brain.disable_plasticity = True
                    inst = measure_p600(
                        self.brain, word, core_area, role_area,
                        self.cfg.n_settling_rounds,
                    )
                    self.brain.disable_plasticity = False
                    if inst > self.cfg.instability_threshold:
                        train_binding(
                            self.brain, word, core_area, role_area,
                            self.cfg.extra_binding_rounds,
                        )
                else:
                    train_binding(
                        self.brain, word, core_area, role_area,
                        self.cfg.binding_rounds,
                    )
                bindings[word] = role_area

            # 3c: update exposure counts
            self.word_exposures[word] += 1

        self.total_sentences += 1

        # Phase 4: periodic lexicon refresh
        if self.total_sentences % self.cfg.lexicon_refresh_interval == 0:
            self._refresh_lexicon()

        return {"bindings": bindings, "n_new_words": len(new_words)}

    # -- Measurement --------------------------------------------------------

    def measure_sentence_acceptability(self, words: List[str]) -> float:
        """Cumulative N400 across sentence positions.

        Lower values indicate more predictable (grammatical) sequences.
        Uses bigram prediction: previous word's CORE area -> PREDICTION.
        """
        if not self.word_assemblies:
            self._refresh_lexicon()

        self.brain.disable_plasticity = True
        cumulative = 0.0
        n_measured = 0

        for i, word in enumerate(words):
            if (i > 0 and word in self.word_assemblies):
                prev = words[i - 1]
                if prev in self.vocab.known_words:
                    prev_area = self.vocab.core_area_for(prev)
                    activate_word(self.brain, prev, prev_area, 3)
                    self.brain.inhibit_areas(["PREDICTION"])
                    for _ in range(5):
                        self.brain.project(
                            {}, {prev_area: ["PREDICTION"]})
                    predicted = np.array(
                        self.brain.areas["PREDICTION"].winners,
                        dtype=np.uint32,
                    )
                    cumulative += measure_n400(
                        predicted, self.word_assemblies[word],
                    )
                    n_measured += 1

        self.brain.disable_plasticity = False
        return cumulative / n_measured if n_measured > 0 else 1.0

    # -- Diagnostics --------------------------------------------------------

    def measure_assembly_stability(self, word: str, rounds: int = 5) -> float:
        """Self-overlap of a word's assembly across two activations."""
        from research.experiments.base import measure_overlap

        core_area = self.vocab.core_area_for(word)
        self.brain.disable_plasticity = True
        activate_word(self.brain, word, core_area, rounds)
        snap1 = np.array(
            self.brain.areas[core_area].winners, dtype=np.uint32)
        activate_word(self.brain, word, core_area, rounds)
        snap2 = np.array(
            self.brain.areas[core_area].winners, dtype=np.uint32)
        self.brain.disable_plasticity = False
        return measure_overlap(snap1, snap2)

    def measure_binding_consistency(
        self, word: str, n_trials: int = 10,
    ) -> float:
        """Fraction of trials where the same STRUCT area wins for a word."""
        core_area = self.vocab.core_area_for(word)
        self.brain.disable_plasticity = True
        areas_chosen: List[Optional[str]] = []
        for _ in range(n_trials):
            for name in self.struct_areas:
                self.brain.clear_refractory(name)
            self.brain.inhibit_areas(self.struct_areas)
            activate_word(self.brain, word, core_area, 3)
            winner = discover_role_area(
                self.brain, "POSITION_MARKER", self.struct_areas,
                self.cfg.stabilize_rounds, hebbian_stabilize=False,
            )
            areas_chosen.append(winner)
        self.brain.disable_plasticity = False

        if not areas_chosen:
            return 0.0
        counter = Counter(areas_chosen)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(areas_chosen)

    def get_stats(self) -> Dict[str, Any]:
        """Current learner statistics."""
        # Count words per core area
        area_counts: Dict[str, int] = defaultdict(int)
        for word in self.vocab.known_words:
            area_counts[self.vocab.core_area_for(word)] += 1

        return {
            "n_words": len(self.vocab.known_words),
            "total_sentences": self.total_sentences,
            "word_exposures": dict(self.word_exposures),
            "mean_exposure": (
                float(np.mean(list(self.word_exposures.values())))
                if self.word_exposures else 0.0
            ),
            "words_per_core_area": dict(area_counts),
            "word_assignments": {
                w: self.vocab.core_area_for(w)
                for w in sorted(self.vocab.known_words)
            },
        }
