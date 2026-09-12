"""GatingMixin -- Word order as a positional template over role areas.

Split out of the CoreParserMixin monolith. Bodies are unchanged;
only their address is.
"""


from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from neural_assemblies.assembly_calculus.ops import sequence_memorize

from ..core.areas import ROLE_AGENT, ROLE_PATIENT, SEQ, FUNC_COMP, FUNC_MARKER
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
        # Per-WORD tracking for the MARKER class only. Sub-category pooling
        # is a compression, and it measurably FAILS exactly there: 'by'
        # reverses voice, 'to' marks a recipient, and pooled they cancelled to
        # conf 0.438 -- passives died the moment ditransitives entered the
        # corpus. The paper's control is per-word action programs; this learns
        # them, contrastively, for the one class where words disagree.
        word_present: Dict[str, List[bool]] = defaultdict(list)
        word_absent: Dict[str, List[bool]] = defaultdict(list)
        word_goal_present: Dict[str, List[bool]] = defaultdict(list)
        word_goal_absent: Dict[str, List[bool]] = defaultdict(list)
        marker_words: Set[str] = set()

        sentence_facts: List[Tuple[bool, Set[str]]] = []
        marker_facts: List[Tuple[bool, bool, Set[str]]] = []
        for sent in sentences:
            role_order = []
            for _word, role in zip(sent.words, sent.roles, strict=True):
                if role == "agent":
                    role_order.append(ROLE_AGENT)
                elif role == "patient":
                    role_order.append(ROLE_PATIENT)
            if not role_order:
                continue
            patient_first = role_order[0] == ROLE_PATIENT
            has_goal = any(r == "goal" for r in sent.roles)
            present: Set[str] = set()
            markers_here: Set[str] = set()
            for word in sent.words:
                # NOT `if not ctx.is_grounded`. That filter made the ROLE
                # MARKER unreachable, which is the one subcategory this whole
                # function exists to find: "by" carries spatial grounding, so
                # it IS grounded and was skipped before `_func_subcat_of` ever
                # saw it -- while that method's own docstring says it "falls
                # back to the grounding signature so a role marker such as
                # 'by' ... is still recognised as a MARKER rather than dropping
                # to None". The fallback was written and then made unreachable
                # from here. Measured: with the filter, a passive-bearing
                # corpus taught only DET and `is_passive` fired 0/2.
                #
                # Prepositions are admitted by CATEGORY rather than by dropping
                # the guard entirely: `_func_subcat_of` calls anything with
                # spatial grounding a MARKER, which would sweep in ordinary
                # nouns like "beach" and let a content word gate the voice.
                ctx = self.word_grounding.get(word)
                if ctx is None:
                    continue
                is_function_word = not ctx.is_grounded
                if not is_function_word:
                    cat, _conf = self.classify_word_cached(word)
                    is_function_word = cat == "PREP"
                if is_function_word:
                    sc = _subcat_of(word)
                    if sc is not None:
                        present.add(sc)
                        if sc == FUNC_MARKER:
                            markers_here.add(word)
            all_subcats |= present
            marker_words |= markers_here
            sentence_facts.append((patient_first, present))
            marker_facts.append((patient_first, has_goal, markers_here))

        for patient_first, has_goal, markers_here in marker_facts:
            for w in marker_words:
                if w in markers_here:
                    word_present[w].append(patient_first)
                    word_goal_present[w].append(has_goal)
                else:
                    word_absent[w].append(patient_first)
                    word_goal_absent[w].append(has_goal)

        for w in marker_words:
            pres = word_present.get(w, [])
            absent = word_absent.get(w, [])
            if not pres:
                continue
            p_present = sum(pres) / len(pres)
            effect = (p_present - sum(absent) / len(absent)) if absent else 0.0
            gp = word_goal_present.get(w, [])
            ga = word_goal_absent.get(w, [])
            g_present = sum(gp) / len(gp) if gp else 0.0
            g_effect = (g_present - sum(ga) / len(ga)) if ga else 0.0
            self.learned_word_gating[w] = {
                "reverses_roles": effect > 0.5,
                "confidence": float(max(0.0, effect)),
                "goal_conf": float(max(0.0, g_effect)),
                "n_examples": len(pres),
                "n_contrast": len(absent),
            }

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
        Returns:
            (role_order_default, is_passive).
        """
        missing = [word for word in words if word not in categories]
        if missing:
            raise ValueError(
                "categories must classify every word before role ordering; "
                f"missing {sorted(set(missing))}"
            )
        # Stage 0: unmarked ranking from the inferred constituent order.
        base_order = self.constituent_role_order()

        # Stage 1: voice gating learned from function-word sub-categories.
        if self.learned_gating or getattr(self, "learned_word_gating", None):
            for word in words:
                # `_func_subcat_of`, NOT raw `get_func_subcategory` -- the same
                # lookup `_learn_gating_patterns` uses to WRITE these entries.
                # They were two spellings of one question: the raw form returns
                # only what frame analysis learned, so "by" (whose MARKER
                # subcategory comes from the grounding fallback) resolved to
                # None here and the gating entry keyed on MARKER could never be
                # found. Measured: MARKER learned at confidence 0.960 and
                # `is_passive` still fired 0/2 until the reader was pointed at
                # the same function as the writer.
                subcat = self._func_subcat_of(word)
                if subcat is None:
                    continue

                # WORD-LEVEL gating outranks the subcategory pool for MARKER
                # words: 'by' reverses voice and 'to' marks a recipient, and
                # pooling them cancelled both (MARKER conf 0.438, passives
                # dead). A word with its own learned entry answers for itself.
                wg = getattr(self, "learned_word_gating", {}).get(word)
                if wg is not None:
                    if (wg.get("reverses_roles", False)
                            and wg.get("confidence", 0) > 0.5):
                        return list(reversed(base_order)), True
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

