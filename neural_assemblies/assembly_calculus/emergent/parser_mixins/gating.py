"""GatingMixin -- Word order as a positional template over role areas.

Split out of the CoreParserMixin monolith. Bodies are unchanged;
only their address is.
"""


from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from neural_assemblies.assembly_calculus.ops import sequence_memorize

from ..core.areas import ROLE_AGENT, ROLE_PATIENT, SEQ, FUNC_COMP
from ..curriculum.data import GroundedSentence


class GatingMixin:
    """Word order as a positional template over role areas."""

    def _learn_gating_patterns(self, sentences: List[GroundedSentence]):
        """Learn gating patterns from how role assignment ORDER changes
        when different function word types are present in a sentence.

        The key insight: function words don't just affect the next word —
        they change the entire role assignment pattern for the sentence.
        "was" reverses first-noun=AGENT to first-noun=PATIENT.
        "by" after passive marks upcoming-noun=AGENT.

        Method:
        1. For each sentence, identify which ungrounded words are present
        2. Extract the role assignment order (1st noun role, 2nd noun role)
        3. Group by function word sub-type
        4. The dominant role order for each sub-type becomes its gating pattern

        This implements Stage 2: learning processing mode shifts triggered
        by function words, analogous to Broca's area learning top-down
        gating through procedural memory (basal ganglia).
        """
        # A function word only earns a gating pattern if its PRESENCE changes
        # the role order relative to its ABSENCE. Scoring the marginal
        # distribution instead lets every function word in a passive sentence
        # -- including "the" -- inherit the passive role order, and once "the"
        # crosses the confidence threshold every sentence gets reversed.
        # So we track both conditions per subcategory and compare them.
        subcat_present: Dict[str, List[bool]] = defaultdict(list)
        subcat_absent: Dict[str, List[bool]] = defaultdict(list)
        all_subcats: Set[str] = set()

        _subcat_of = self._func_subcat_of

        # First pass: which subcategories exist at all, and per sentence,
        # whether the sentence is patient-first and which subcats it contains.
        sentence_facts: List[Tuple[bool, Set[str]]] = []
        for sent in sentences:
            role_order = []
            for word, role in zip(sent.words, sent.roles):
                if role == "agent":
                    role_order.append(ROLE_AGENT)
                elif role == "patient":
                    role_order.append(ROLE_PATIENT)
            if not role_order:
                continue
            patient_first = role_order[0] == ROLE_PATIENT
            present: Set[str] = set()
            for word in sent.words:
                ctx = self.word_grounding.get(word)
                if ctx is not None and not ctx.is_grounded:
                    sc = _subcat_of(word)
                    if sc is not None:
                        present.add(sc)
            all_subcats |= present
            sentence_facts.append((patient_first, present))

        for patient_first, present in sentence_facts:
            for sc in all_subcats:
                if sc in present:
                    subcat_present[sc].append(patient_first)
                else:
                    subcat_absent[sc].append(patient_first)

        for sc in all_subcats:
            pres = subcat_present.get(sc, [])
            absent = subcat_absent.get(sc, [])
            if not pres:
                continue
            p_present = sum(pres) / len(pres)
            if absent:
                p_absent = sum(absent) / len(absent)
                effect = p_present - p_absent
            else:
                # Present in every sentence, so it cannot explain any variation
                # in role order. "the" occurs in both voices and must score 0
                # here; otherwise a passive-heavy corpus teaches the determiner
                # to reverse roles and every active sentence parses backwards.
                p_absent = float("nan")
                effect = 0.0
            reverses_roles = effect > 0.5
            role_order = ([ROLE_PATIENT, ROLE_AGENT] if p_present > 0.5
                          else [ROLE_AGENT, ROLE_PATIENT])
            self.learned_gating[sc] = {
                "role_order": role_order,
                "reverses_roles": reverses_roles,
                # Confidence is the contrastive effect size, not the raw rate,
                # so a word that appears in both voices scores ~0.
                "confidence": float(max(0.0, effect)),
                "n_examples": len(pres),
                "n_contrast": len(absent),
                "p_present": float(p_present),
                "p_absent": float(p_absent),
                "clause_boundary": sc == FUNC_COMP,
            }


    def train_word_order(
        self,
        sentences: List[GroundedSentence],
        *,
        repetitions: Optional[int] = None,
    ):
        """Phase 4: Word order via sequence memorization in SEQ area."""
        from ..training.perf import sequence_rounds_per_step

        reps = repetitions
        if reps is None:
            reps = 1 if self.fast_training else 2
        seq_rounds = sequence_rounds_per_step(
            self.rounds, self.inference_rounds, fast=self.fast_training,
        )
        for sent in sentences:
            stim_seq = [
                self.stim_map[w] for w in sent.words
                if w in self.stim_map
            ]
            if stim_seq:
                sequence_memorize(
                    self.brain, stim_seq, SEQ,
                    rounds_per_step=seq_rounds,
                    repetitions=reps,
                    phase_b_ratio=0.5,
                    beta_boost=0.5,
                )

    def _detect_passive(self, words: List[str],
                        categories: Dict[str, str]) -> bool:
        """Detect passive voice: 'was/were' before the main verb."""
        for i, word in enumerate(words):
            if word in ("was", "were"):
                # Check if next content word is a verb
                for j in range(i + 1, len(words)):
                    if categories.get(words[j]) == "VERB":
                        return True
                    if categories.get(words[j]) in ("NOUN", "PRON"):
                        break
        return False

    def constituent_role_order(self) -> List[str]:
        """Default role ranking implied by the inferred CONSTITUENT ORDER.

        This is the *unmarked* ranking for the language: which thematic role
        the first available noun takes when nothing else intervenes.  It is
        derived from ``self.word_order_type``, which ``infer_word_order``
        learned from the corpus, via the definitional mapping in
        ``core.word_order`` -- object-initial typologies (OSV, OVS, VOS) rank
        PATIENT first, the other three rank AGENT first.

        CONSTITUENT ORDER IS NOT VOICE.  This method answers "in this language,
        does the object normally come first?".  It says nothing about whether a
        particular sentence is passive; that is the separate, learned-gating
        mechanism in ``_determine_role_order``, which flips whatever ranking
        this returns.  Keeping the two apart matters because they compose: a
        passive clause in an OVS language is agent-initial.

        When no typology has been inferred the AGENT-first ranking is returned.
        That is a fallback prior, not a finding.
        """
        from ..core.word_order import is_object_initial, WORD_ORDERS

        order = getattr(self, "word_order_type", None)
        if order in WORD_ORDERS and is_object_initial(order):
            return [ROLE_PATIENT, ROLE_AGENT]
        return [ROLE_AGENT, ROLE_PATIENT]

    def _determine_role_order(self, words: List[str],
                              categories: Dict[str, str],
                              ) -> Tuple[List[str], bool]:
        """Determine role assignment order for one sentence.

        Two independent mechanisms, composed in this order:

        Stage 0 (constituent order): the unmarked ranking for the language,
        from the inferred typology -- see ``constituent_role_order``.  This is
        a property of the CORPUS, constant across its sentences.

        Stage 1 (voice gating, ELAN ~180ms): if a function-word sub-type
        present in THIS sentence has learned gating that reverses roles, the
        Stage 0 ranking is FLIPPED.  This is a property of the SENTENCE.

        The flip is deliberately expressed as a reversal of the Stage 0 result
        rather than a hardcoded ``[PATIENT, AGENT]``: a passive in an
        object-initial language must come out agent-first, and hardcoding the
        pair would silently give the wrong answer there.

        There is deliberately no spelling-based passive fallback: if the corpus
        contained no voice alternation, the parser has not learned one and
        should not pretend otherwise.  ``_detect_passive()`` remains available
        for diagnostics only.

        Args:
            words: Sentence word list.
            categories: Pre-classified {word: category}.

        Returns:
            (role_order_default, is_passive).
        """
        # Stage 0: unmarked ranking from the inferred constituent order.
        base_order = self.constituent_role_order()

        # Stage 1: voice gating learned from function-word sub-categories.
        if self.learned_gating:
            for word in words:
                # Get sub-category (from frame analysis or cache)
                subcat = (self.get_func_subcategory(word)
                          if hasattr(self, 'get_func_subcategory') else None)
                if subcat is None:
                    continue

                gating = self.learned_gating.get(subcat)
                if gating is None:
                    continue

                # If this sub-type reverses roles with high confidence,
                # flip the constituent-order ranking.
                if (gating.get("reverses_roles", False)
                        and gating.get("confidence", 0) > 0.5):
                    return list(reversed(base_order)), True

        return base_order, False

