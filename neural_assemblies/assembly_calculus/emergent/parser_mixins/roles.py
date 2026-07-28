"""RoleBindingMixin -- Binding fillers into thematic role areas, and reading them back.

Split out of the CoreParserMixin monolith. Bodies are unchanged;
only their address is.
"""


from typing import Dict, List, Optional
from neural_assemblies.assembly_calculus.assembly import (
    overlap as assembly_overlap,
)
from neural_assemblies.assembly_calculus.ops import (
    activate_assembly,
    project,
    _snap,
)

from ..core.areas import (
    GROUNDING_TO_CORE,
    THEMATIC_AREAS,
    ROLE_AGENT,
    ROLE_PATIENT,
    ROLE_ACTION,
    FUNC_AUX,
    FUNC_DET,
    FUNC_COMP,
    FUNC_MARKER,
)
from ..curriculum.data import GroundedSentence
from ._shared import _STRUCTURAL_PRIOR, _LEXICAL_SMOOTHING, _ROLE_BINDING_ROUNDS, _ROLE_MAP, _ROLE_LABEL


class RoleBindingMixin:
    """Binding fillers into thematic role areas, and reading them back."""

    def train_roles(self, sentences: List[GroundedSentence]):
        """Phase 2: Role binding from annotated sentences.

        For each word with a role annotation (agent/patient):
        1. Activate word in its core area (project phon -> core)
        2. Fix the core assembly
        3. Project core -> role_area with recurrence
        4. Snapshot the role assembly

        Pattern follows parser.py:train_roles().

        Side effect: each annotated sentence also contributes one observation to
        ``dist_stats.role_order_counts`` (see ``record_role_order_evidence``).
        That is the evidence ``infer_word_order`` needs to tell an
        object-initial typology from its subject-initial twin; part-of-speech
        transitions cannot.
        """
        for role_area in THEMATIC_AREAS:
            if role_area not in self.role_lexicons:
                self.role_lexicons[role_area] = {}

        for sent in sentences:
            self.record_role_order_evidence(sent.words, sent.roles)
            for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
                # The verb binds into ROLE_ACTION like any other constituent.
                # Skipping it here left the verb outside the role system, so
                # its position could not be learned.
                if role is None:
                    continue
                role_area = _ROLE_MAP.get(role)
                if role_area is None:
                    continue
                if word not in self.stim_map:
                    continue

                core_area = GROUNDING_TO_CORE[ctx.dominant_modality]
                phon = self.stim_map[word]

                # Activate the word in its core area. Prefer the stabilized
                # lexicon assembly: re-projecting phon -> core carries
                # plasticity, so the core assembly drifts between the moment a
                # role binding is stored and the moment it is read back, and
                # retrieval then misses. train_lexicon has already converged
                # these, so replay the snapshot instead.
                stored_core = self.core_lexicons.get(core_area, {}).get(word)
                if stored_core is not None:
                    activate_assembly(self.brain, stored_core)
                else:
                    project(self.brain, phon, core_area, rounds=self.rounds)
                self.brain.areas[core_area].fix_assembly()

                # Canonical projection: round 1 is input-driven, so the role
                # assembly is a function of the filler, then a short recurrent
                # tail stabilizes it (see _ROLE_BINDING_ROUNDS).
                #
                # The previous code recurred from round 1 for self.rounds
                # steps and then called reset_area_connections(role_area).
                # Zeroing the connectome left every candidate neuron at equal
                # input, so the deterministic index tie-break in winner
                # selection picked the *same* k neurons for every word -- all
                # stored role assemblies were literally the identical winner
                # set (pairwise overlap 1.000). Readout could not discriminate,
                # and role assignment silently fell through to word order.
                self.brain.project({}, {core_area: [role_area]})
                for _ in range(_ROLE_BINDING_ROUNDS - 1):
                    self.brain.project(
                        {}, {core_area: [role_area], role_area: [role_area]},
                    )

                asm = _snap(self.brain, role_area)
                self.role_lexicons[role_area][word] = asm

                self.brain.areas[core_area].unfix_assembly()

        # Learn gating patterns from function word → role co-occurrences
        self._learn_gating_patterns(sentences)

    def _func_subcat_of(self, word: str) -> Optional[str]:
        """Function-word subcategory for `word`, or None if it has none.

        Prefers the distributionally learned frame subcategory. Falls back to
        the grounding signature so a role marker such as "by" -- which carries
        spatial grounding and therefore is not an ungrounded function word --
        is still recognised as a MARKER rather than dropping to None.
        """
        sc = (self.get_func_subcategory(word)
              if hasattr(self, "get_func_subcategory") else None)
        if sc is not None:
            return sc
        ctx = self.word_grounding.get(word)
        if ctx is None:
            return None
        if ctx.spatial:
            return FUNC_MARKER
        if not ctx.is_grounded:
            return FUNC_DET
        return None

    def _role_binding_margin(self, word: str, core_area: str,
                             role_area: str) -> float:
        """Lexical evidence that `word` was bound into `role_area`, in [0, 1].

        Requires the core assembly for `word` to be active and FIXED.

        Why this is not simply ``overlap(projection, stored[word])``.  Every
        filler that is bound into a role area lands in that area's dominant
        attractor: measured on a 4-noun corpus, two DIFFERENT words' stored
        AGENT assemblies overlap 0.70-0.85, while the same word's AGENT and
        PATIENT assemblies overlap 0.00 -- but that last number is separation
        between two different brain areas' neuron index sets, which is true by
        construction and carries no learned information.  The consequence is
        that ``overlap(projection, stored[word])`` returns ~0.9 for EVERY
        candidate role, so the term cannot discriminate; it also survives
        ``reset_area_connections`` almost unchanged, i.e. it is nearly
        independent of what was learned.

        The discriminable part is the residual above the area-generic
        attractor: how much better the projection matches THIS word's stored
        binding than it matches the other fillers stored in the same area.
        That residual does collapse when the learned weights are wiped, which
        is the property a read-out of a learned binding must have.

        The projection uses ``_ROLE_BINDING_ROUNDS`` -- the same protocol that
        wrote the binding -- and runs with plasticity disabled so that reading
        a role does not rewrite it.  (The previous code ran one round, then
        called ``reset_area_connections(role_area)``, which destroyed the
        learned core->role weights as a side effect of parsing.)
        """
        lex = self.role_lexicons.get(role_area, {})
        stored = lex.get(word)
        if stored is None:
            return 0.0

        with self.brain.frozen():
            self.brain.project({}, {core_area: [role_area]})
            for _ in range(_ROLE_BINDING_ROUNDS - 1):
                self.brain.project(
                    {}, {core_area: [role_area], role_area: [role_area]},
                )
            asm = _snap(self.brain, role_area)

        own = assembly_overlap(asm, stored)
        others = [assembly_overlap(asm, a)
                  for other, a in lex.items() if other != word]
        if not others:
            return own
        base = sum(others) / len(others)
        return max(0.0, (own - base) / max(1e-6, 1.0 - base))

    def _assign_roles_neural(self, words: List[str],
                             categories: Dict[str, str],
                             filler_word: Optional[str] = None,
                             filler_role: Optional[str] = None,
                             ) -> Dict[str, Optional[str]]:
        """Neural role assignment via learned projections + mutual inhibition.

        Two-stage function word processing:
        1. ELAN-like rapid sub-categorization determines role order
           (learned gating from training, or hardcoded passive detection)
        2. Left-to-right role assignment with mutual inhibition

        For each NOUN/PRON, projects the word's core assembly into each
        uninhibited role area with recurrence, reads out against
        role_lexicons. The best-scoring uninhibited role wins, then
        that role is inhibited (mutual exclusion).

        Args:
            words: Sentence word list (ordered).
            categories: Pre-classified {word: category} from classify_word.
            filler_word: Optional word displaced from canonical position
                (e.g., antecedent of a relative clause).
            filler_role: Role to assign to filler_word ("AGENT" or "PATIENT").

        Returns:
            {word: "AGENT"/"ACTION"/"PATIENT"/None} for each word.
        """
        roles: Dict[str, Optional[str]] = {}
        inhibited: set = set()

        # Pre-assign filler if provided (filler-gap binding)
        if filler_word and filler_role:
            roles[filler_word] = filler_role
            role_area = (ROLE_AGENT if filler_role == "AGENT"
                         else ROLE_PATIENT)
            inhibited.add(role_area)

        # Determine role order: learned gating (Stage 1) or rules (Stage 2)
        role_order_default, is_passive = self._determine_role_order(
            words, categories)

        # A role MARKER (the "by" of a passive) signals that the noun after it
        # takes the reversed role. The marker is identified by its learned
        # distributional subcategory, not by spelling.
        after_marker = False

        for word in words:
            cat = categories.get(word)
            subcat = self._func_subcat_of(word)

            if is_passive and subcat == FUNC_MARKER:
                after_marker = True
                roles[word] = None
                continue

            # Function words carry no thematic role. Identified by learned
            # subcategory (AUX/COMP/DET), not by a literal word list.
            if subcat in (FUNC_AUX, FUNC_COMP, FUNC_DET):
                roles[word] = None
                continue

            if cat == "VERB":
                roles[word] = "ACTION"
                continue

            if cat not in ("NOUN", "PRON"):
                roles[word] = None
                continue

            phon = self.stim_map.get(word)
            if phon is None:
                roles[word] = None
                continue

            # After the role marker the voice flip is undone, so the language's
            # unmarked constituent-order ranking applies again (NOT a hardcoded
            # AGENT-first pair -- in an object-initial language the unmarked
            # ranking is PATIENT-first).
            if is_passive and after_marker:
                role_order = self.constituent_role_order()
            else:
                role_order = list(role_order_default)

            # Activate word in its core area
            core_area = self._word_core_area(word)
            infer_r = self.inference_rounds
            stored_core = self.core_lexicons.get(core_area, {}).get(word)
            if stored_core is not None:
                activate_assembly(self.brain, stored_core)
            else:
                project(self.brain, phon, core_area, rounds=infer_r)
            self.brain.areas[core_area].fix_assembly()

            best_role_area: Optional[str] = None
            best_score = -1.0

            candidates = [ra for ra in role_order if ra not in inhibited]
            margins = {
                ra: self._role_binding_margin(word, core_area, ra)
                for ra in candidates
            }
            # Turn the per-area margins into a distribution over the competing
            # roles. A filler bound in exactly one candidate role gets nearly
            # the whole mass there and can overturn the structural prior; a
            # filler bound in both (a corpus with no lexical role preference)
            # splits the mass and correctly leaves the decision to the prior.
            # The additive smoothing pulls the split toward uniform, so only a
            # clear asymmetry -- not the seed-to-seed jitter in the margins --
            # is worth more than the prior's rank gap.
            eps = _LEXICAL_SMOOTHING
            total = sum(margins.values()) + eps * len(margins)

            # Structural prior from word order / learned gating. It decays with
            # position in role_order so the preferred role wins when no lexical
            # binding exists -- this is what lets a filler never seen in any
            # role still be assigned systematically.
            for rank, role_area in enumerate(role_order):
                if role_area in inhibited:
                    continue

                prior = _STRUCTURAL_PRIOR * (0.5 ** rank)
                lexical = (((margins[role_area] + eps) / total)
                           if any(margins.values()) else 0.0)

                score = prior + lexical
                if score > best_score:
                    best_score = score
                    best_role_area = role_area

            self.brain.areas[core_area].unfix_assembly()

            if best_role_area is not None:
                roles[word] = _ROLE_LABEL[best_role_area]
                inhibited.add(best_role_area)
            else:
                roles[word] = None

        return roles

