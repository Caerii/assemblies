"""GenerationMixin -- language production / generation from semantic representations.

Generation here runs comprehension backwards.  Comprehension goes word ->
core -> role; production goes role -> core -> word, and the SAME synapses
carry both directions, which is the point: a bidirectionally trained pathway
gives production for free rather than needing a separate decoder.

The round trip in :meth:`GenerationMixin.generate` is deliberately not a
lookup.  For each role the intended word is driven into its core area,
projected into the role area to form a role assembly, and then that role
assembly is projected BACK to core and read out
(``_decode_role_to_word``).  What comes back need not be what went in -- a
role area holds the superposition of every filler ever bound to it, so a
weakly-bound word can be overwritten by a strongly-bound one.  That the
returned word usually matches is the empirical result being demonstrated;
treating it as an identity function would miss what is being tested.

WHAT IS NOT NEURAL IN THIS FILE.  Two steps are Python:

* Surface ORDER.  The role phrases are concatenated according to
  ``self.word_order_type`` in an if/elif ladder.  The typology itself was
  inferred from the corpus, but the ordering at generation time is applied
  symbolically.  ``ConstituentOrderMixin`` implements the paper's actual
  mechanism, where order lives in SYN[i] -> ROLE[i+1] synapses; prefer it
  when order is the claim being made.
* Determiner insertion.  "the" is prepended when the decoded word's dominant
  modality is visual.  This is a hardcoded heuristic standing in for a
  determiner-selection process, not something the model learned.

Surface order covers all six basic orders (SVO, SOV, VSO, OSV, OVS, VOS) by
walking the slot sequence of ``self.word_order_type``; see
``core.word_order``.  The six-member label set is built in, the choice among
them is inferred from the corpus, and the concatenation itself is symbolic.
"""

from typing import Dict, List, Optional, Set

from neural_assemblies.assembly_calculus.ops import project, _snap
from neural_assemblies.assembly_calculus.readout import readout_all

from ..core.areas import (
    VERB_CORE, ROLE_AGENT, ROLE_PATIENT,
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

        # Assemble in the learned word order. All six basic orders are
        # produced by walking the slot sequence of the inferred typology, so
        # there is no if/elif ladder and no order that silently falls through
        # to SVO. An unrecognised label falls back to SVO explicitly.
        from ..core.word_order import WORD_ORDERS, order_slots

        order = self.word_order_type
        if order not in WORD_ORDERS:
            order = "SVO"

        phrase_for = {
            "S": agent_phrase,
            "V": verb_phrase,
            "O": patient_phrase,
        }
        output: List[str] = []
        for slot in order_slots(order):
            output.extend(phrase_for[slot])

        return output

    def continue_sentence(
        self,
        prefix: List[str],
        *,
        max_words: int = 12,
        min_words: int = 4,
        temperature: float = 0.0,
    ) -> List[str]:
        """Extend *prefix* word-by-word via ``predict_next`` (bridge readout).

        Stops at a plausible clause boundary or ``max_words``.
        """
        words = [w for w in prefix if w in self.stim_map]
        if not words:
            return list(prefix)

        self._ensure_prediction_lexicon()
        used: Set[str] = set()

        for _ in range(max_words):
            preds = self.predict_next(words)
            if not preds:
                break

            if temperature <= 0.0:
                next_w = preds[0][0]
            else:
                import random
                pool = [w for w, _ in preds[:5]]
                next_w = random.choice(pool) if pool else preds[0][0]

            if not next_w or next_w in used:
                break
            if next_w in (".", "?", "!"):
                break

            words.append(next_w)
            used.add(next_w)

            if len(words) >= min_words and self._clause_boundary(words):
                break

        return words

    def generate_novel_sentence(
        self,
        *,
        seed_prefix: Optional[List[str]] = None,
        max_attempts: int = 5,
        min_words: int = 4,
        max_words: int = 10,
    ) -> List[str]:
        """Sample a sentence likely absent from the training corpus."""
        import random

        if seed_prefix is None:
            starters = [
                ["the"], ["a"], ["she"], ["he"], ["they"],
            ]
            dets = [w for w, c in self.word_grounding.items()
                    if c.dominant_modality == "none" and w in ("the", "a")]
            if dets:
                starters = [[random.choice(dets)]]
            seed_prefix = random.choice(starters)

        best: List[str] = []
        best_score = -1.0
        for _ in range(max_attempts):
            candidate = self.continue_sentence(
                seed_prefix,
                max_words=max_words,
                min_words=min_words,
                temperature=0.15,
            )
            score = self.sentence_novelty(candidate)
            if score > best_score and len(candidate) >= min_words:
                best_score = score
                best = candidate

        return best if best else self.continue_sentence(
            seed_prefix, max_words=max_words, min_words=min_words,
        )

    def sentence_novelty(self, words: List[str]) -> float:
        """Score in [0, 1]: 1 = fully novel vs stored training corpus."""
        if not words:
            return 0.0
        key = tuple(words)
        if getattr(self, "_corpus_sentence_set", None) and key in self._corpus_sentence_set:
            return 0.0

        bigrams = getattr(self, "_corpus_bigram_set", None)
        if not bigrams:
            return 1.0

        pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        if not pairs:
            return 1.0
        novel = sum(1 for p in pairs if p not in bigrams)
        return novel / len(pairs)

    def _clause_boundary(self, words: List[str]) -> bool:
        """Heuristic: stop after verb + optional object."""
        if len(words) < 3:
            return False

        cats: Dict[str, str] = {}
        for w in words:
            cat, _ = self.classify_word_cached(w)
            cats[w] = cat

        verb_idx = -1
        for i, w in enumerate(words):
            if cats.get(w) == "VERB":
                verb_idx = i

        if verb_idx < 0:
            return len(words) >= 6

        after = len(words) - verb_idx - 1
        if after >= 2:
            return True
        if after == 1 and cats.get(words[-1]) in ("NOUN", "PRON", "ADV", "ADJ"):
            return True
        if after == 0 and len(words) >= 4:
            return True
        return False
