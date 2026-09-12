"""RoleBindingMixin -- Binding fillers into thematic role areas, and reading them back.

Split out of the CoreParserMixin monolith. Bodies are unchanged;
only their address is.
"""


from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from neural_assemblies.core.measurement import Measured
from neural_assemblies.assembly_calculus.assembly import (
    Assembly,
    overlap as assembly_overlap,
)
from neural_assemblies.assembly_calculus.ops import (
    activate_assembly,
    bind,
    project,
    _snap,
)

from ..core.areas import (
    GROUNDING_TO_CORE,
    ROLE_LABEL_TO_AREA,
    THEMATIC_AREAS,
    ROLE_AGENT,
    ROLE_PATIENT,
    ROLE_ACTION,
    ROLE_GOAL,
    FUNC_AUX,
    FUNC_DET,
    FUNC_COMP,
    FUNC_MARKER,
)
from ..curriculum.data import GroundedSentence
from ..core.grounding import GroundingContext
from ._shared import _STRUCTURAL_PRIOR, _LEXICAL_SMOOTHING, _ROLE_BINDING_ROUNDS, _ROLE_LABEL

if TYPE_CHECKING:
    from neural_assemblies.core.brain import Brain
    from ..acquisition.pos_inference import BootstrapScores


class RoleBindingMixin:
    """Binding fillers into thematic role areas, and reading them back."""

    brain: "Brain"
    stim_map: Dict[str, str]
    rounds: int
    inference_rounds: int
    word_grounding: Dict[str, GroundingContext]
    core_lexicons: Dict[str, Dict[str, Assembly]]
    role_lexicons: Dict[str, Dict[str, Assembly]]

    if TYPE_CHECKING:
        def record_role_order_evidence(self, words: List[str], roles: List[Optional[str]]) -> Optional[str]: ...
        def _learn_gating_patterns(self, sentences: List[GroundedSentence]) -> None: ...
        def get_func_subcategory(self, word: str) -> Optional[str]: ...
        def _word_core_area(self, word: str) -> str: ...
        def classify_word_cached(self, word: str, grounding: Optional[GroundingContext] = None) -> Tuple[str, "BootstrapScores"]: ...
        def _determine_role_order(self, words: List[str], categories: Dict[str, str]) -> Tuple[List[str], bool]: ...
        def constituent_role_order(self) -> List[str]: ...

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
            for word, ctx, role in zip(sent.words, sent.contexts, sent.roles, strict=True):
                # The verb binds into ROLE_ACTION like any other constituent.
                # Skipping it here left the verb outside the role system, so
                # its position could not be learned.
                if role is None:
                    continue
                role_area = ROLE_LABEL_TO_AREA.get(role)
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
                if stored_core is None:
                    project(self.brain, phon, core_area, rounds=self.rounds)

                # ONE SHARED IMPLEMENTATION -- see `ops.bind`, which carries the
                # rationale for all three of its properties (no reset, replay
                # the stabilized snapshot, feed-forward round 1 + short tail).
                #
                # This code used to be the only correct copy of that protocol,
                # while `assembly_calculus.parser.train_roles` and
                # `generation.py` had drifted to older, broken versions. The
                # drift is what let the bug survive: fixing it here fixed
                # nothing there, and nothing connected them. Calling one
                # function is the fix for the class.
                asm = bind(self.brain, core_area, role_area, stored_core,
                           tail_rounds=_ROLE_BINDING_ROUNDS - 1)
                self.role_lexicons[role_area][word] = asm

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

    def parse_roles_by_reconstruction(
            self, words: List[str],
            filler_word: Optional[str] = None,
            filler_role: Optional[str] = None,
    ) -> "tuple[Dict[str, Optional[str]], dict]":
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-role-reconstruction

        Gate -> record -> recall using existing role populations.

        The 2021 parser paper's division of labor, from parts this parser
        already had (promoted here from
        `research/experiments/sentence_conditioned_readout.py`, which measured
        it before it was wired):

          GATE    `_determine_role_order` (learned voice gating) selects the
                  filler sequence -- agent-first active, patient-first passive.
          RECORD  each content word's core assembly bind-traverses (T=2) into
                  the open slot; the parse's state is the role areas' WINNERS.
          RECALL  reconstruction: occupant(R) = argmax_w overlap(image(w->R),
                  winners(R)). No `role_lexicons` anywhere, so a word never
                  TRAINED in a role still reads out of it -- the exact case
                  where `_role_binding_margin` measurably fails (it flips
                  passives toward each word's trained majority role).

        Measured at the phon_weight=6/beta=0.05 defaults, 40 held-out
        reversible probes x 5 seeds: 1.0000 both voices, against the margin
        route's 0.85 active / 0.50 passive. With an active-only corpus the
        same readout reads passives at 0.0000 -- perfectly inverted -- because
        no MARKER was learned and the gate never selects patient-first: the
        corpus effect flows entirely through the learned control, and the
        substrate supplies identity.

        Traversal and recall run under ``brain.read_only()`` and restore their
        dynamical state on exit. Preparatory neural classification has its own
        read-only scope; parser classification/subcategory caches may still be
        populated. The answer rests on the substrate's
        image separation (occupant gap 0.87+ at current defaults; at
        phon_weight=1 it collapsed to ties), which is what makes parsing
        accuracy a measurement OF the substrate.

        Returns ({word: role_label_or_None}, diagnostics). Diagnostics carry
        `is_passive`, per-area winners, and (role, occupant, top, runner, gap)
        tuples -- the GAP is the substrate-dependence metric and callers
        asserting only the labels are measuring the gate. `unavailable_areas`
        records missing areas/populations; an unavailable traversal returns no
        role evidence and never initializes a population during observation.
        """
        from neural_assemblies.assembly_calculus.ops import (
            project as _ops_project,
        )

        brain = self.brain
        cats = {w: self.classify_word_cached(w)[0] for w in words}
        _order, is_passive = self._determine_role_order(words, cats)
        sequence = ((ROLE_PATIENT, ROLE_AGENT) if is_passive
                    else (ROLE_AGENT, ROLE_PATIENT))

        diag: dict = {"is_passive": bool(is_passive), "gaps": [],
                      "winners": {}, "unavailable_areas": {}}
        out: Dict[str, Optional[str]] = {w: None for w in words}

        # FILLER-GAP (relative clauses): the antecedent has already claimed a
        # role from OUTSIDE this clause ("the dog that ___ chased the cat" --
        # `dog` is the inner clause's agent but is not among its words). Same
        # contract as the margin route: pre-assign the filler, and remove its
        # role from the sequence so an inner noun cannot take it.
        if filler_word and filler_role:
            out[filler_word] = filler_role
            taken = {"AGENT": ROLE_AGENT, "PATIENT": ROLE_PATIENT}.get(
                filler_role)
            sequence = tuple(r for r in sequence if r != taken)

        def _traverse(word: str, role: str) -> bool:
            core = self._word_core_area(word)
            if core is None or core not in brain.areas:
                return False
            for name in (core, role):
                area = brain.areas.get(name)
                if area is None:
                    diag["unavailable_areas"][name] = "area_missing"
                    return False
                if not brain._engine_for(area).probe_target_ready(name):
                    diag["unavailable_areas"][name] = "population_not_materialized"
                    return False
            stored = self.core_lexicons.get(core, {}).get(word)
            if stored is not None:
                activate_assembly(brain, stored)
            else:
                phon = self.stim_map.get(word)
                if phon is None:
                    return False
                _ops_project(brain, phon, core, rounds=self.rounds)
            brain.areas[core].fix_assembly()
            try:
                brain.project({}, {core: [role]})
                for _ in range(_ROLE_BINDING_ROUNDS - 1):
                    brain.project({}, {core: [role], role: [role]})
            finally:
                brain.areas[core].unfix_assembly()
            return True

        with brain.read_only():
            # A parse must not inherit residue: training leaves the LAST
            # trained sentence's winners in the role areas (measured:
            # deterministic 4-of-24 failures until cleared).
            role_areas = [a for a in (ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT)
                          if a in brain.areas]
            for area in role_areas:
                brain.areas[area].unfix_assembly()
            brain.inhibit_areas(role_areas)

            # RECORD. Nouns consume the voice-ordered filler sequence; the
            # first verb-classified token takes ACTION independently (a noun
            # queued behind an ACTION pivot deadlocks on any unclassifiable
            # verb form). A slot is consumed even when the word cannot be
            # recorded -- an unknown word must not misalign the tail.
            #
            # PREPOSITIONS GOVERN THE NOUN THAT FOLLOWS, and the governor
            # decides its slot:
            #   * the passive MARKER's noun IS a filler (the by-phrase agent);
            #   * an UNGROUNDED goal preposition ("to" -- no grounding entry,
            #     unlike every locative) routes its noun to ROLE_GOAL;
            #   * any other preposition's noun fills NOTHING.
            # The third case also closes a latent defect: before this, ANY
            # third noun consumed the next filler slot, so "the boy sleeps in
            # the house" read house=PATIENT -- the PP noun mistaken for a
            # participant.
            slot_idx = 0
            action_taken = False
            pending: Optional[str] = None
            fillers: list = []
            for w in words:
                subcat = self._func_subcat_of(w)
                cat = cats.get(w)
                if cat == "PREP" or subcat == FUNC_MARKER:
                    # The governor's slot comes from the LEARNED per-word
                    # gating, not from groundedness -- 'to' ACQUIRES grounding
                    # during training, so any groundedness test rots. A word
                    # that reverses voice governs a filler (the by-phrase
                    # agent); one that marks goal roles governs ROLE_GOAL;
                    # everything else (locatives) governs nothing.
                    wg = getattr(self, "learned_word_gating", {}).get(w, {})
                    if (is_passive and wg.get("reverses_roles")
                            and wg.get("confidence", 0) > 0.5):
                        pending = "filler"
                    elif wg.get("goal_conf", 0) > 0.5:
                        pending = "goal"
                    elif is_passive and subcat == FUNC_MARKER and not wg:
                        # No word statistics (unseen marker): fall back to the
                        # subcategory, preserving pre-word-gating behavior.
                        pending = "filler"
                    else:
                        pending = "skip"
                    continue
                if subcat is not None:
                    continue  # function word: control, not a filler
                if cat in ("NOUN", "PRON"):
                    governor, pending = pending, None
                    if governor == "skip":
                        continue
                    if governor == "goal":
                        if _traverse(w, ROLE_GOAL):
                            fillers.append((w, ROLE_GOAL))
                        continue
                    if slot_idx < len(sequence):
                        if _traverse(w, sequence[slot_idx]):
                            fillers.append((w, sequence[slot_idx]))
                        slot_idx += 1
                elif cat == "VERB" and not action_taken:
                    if _traverse(w, ROLE_ACTION):
                        out[w] = "ACTION"
                    action_taken = True

            # Capture the parse state BEFORE replays disturb it.
            snaps = {role: _snap(brain, role) for _w, role in fillers}
            diag["winners"] = {
                role: tuple(int(x) for x in snap.winners)
                for role, snap in snaps.items()
            }

            # RECALL: which candidate's image reproduces each area's winners?
            nouns = [w for w in words if cats.get(w) in ("NOUN", "PRON")
                     and self._func_subcat_of(w) is None]
            for role, snap in snaps.items():
                scores = {}
                for w in nouns:
                    if not _traverse(w, role):
                        continue
                    scores[w] = float(
                        assembly_overlap(_snap(brain, role), snap))
                if not scores:
                    continue
                ranked = sorted(scores.items(), key=lambda kv: -kv[1])
                occupant, top = ranked[0]
                runner = ranked[1][1] if len(ranked) > 1 else 0.0
                diag["gaps"].append((role, occupant, top, runner,
                                     top - runner))
                if top > runner:  # a tie is a failure to read, not a guess
                    out[occupant] = _ROLE_LABEL[role]
        return out, diag

    def _role_binding_margin(self, word: str, core_area: str,
                             role_area: str) -> Measured:
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

        UNDEFINED, NOT ZERO, when the residual cannot be formed. There are two
        such cases, and returning a number for either used to make this function
        report TWO DIFFERENT QUANTITIES under one name:

        * the word has no stored binding here -- no evidence, rather than
          evidence of zero;
        * the word is the ONLY filler in this role's lexicon, so there is no
          baseline to subtract. The old code returned the raw ``own`` overlap
          for this case, which is the quantity this whole docstring exists to
          reject: ~0.9 for every candidate role, carrying no learned
          information. Because ``_assign_roles_neural`` NORMALIZES these
          margins against each other, that raw value took probability mass away
          from properly-baselined competitors -- making LEXICON SIZE move the
          role decision.

        Measured before changing it (`research/experiments/
        role_margin_branch_census.py`, seeds 11/12/42): the single-filler branch
        fires **0 times** in a real parse, so this costs nothing today. It is
        typed so it cannot start costing something silently.
        """
        lex = self.role_lexicons.get(role_area, {})
        stored = lex.get(word)
        if stored is None:
            return Measured.undefined(
                f"{word!r} has no stored binding in {role_area}, so there is "
                f"no lexical evidence either way",
                word=word, role_area=role_area, lexicon_size=len(lex))

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
            return Measured.undefined(
                f"{word!r} is the only filler stored in {role_area}, so there "
                f"is no baseline to subtract and the residual is not defined",
                word=word, role_area=role_area, own=own)
        base = sum(others) / len(others)
        return Measured.of(max(0.0, (own - base) / max(1e-6, 1.0 - base)))

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
        role_order_default, is_passive = self._determine_role_order(words, categories)

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
            # `.or_else(0.0)` is EXPLICIT, and it is the right default here:
            # an undefined margin means there is no lexical evidence for this
            # role, so it should contribute nothing and let the structural prior
            # decide. The old code produced the same 0.0 for the no-binding case
            # and a near-1.0 RAW OVERLAP for the single-filler case, which is
            # the confound this replaces (see `_role_binding_margin`).
            margins = {
                ra: self._role_binding_margin(word, core_area, ra).or_else(0.0)
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

