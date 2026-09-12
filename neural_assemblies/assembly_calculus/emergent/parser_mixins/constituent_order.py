"""ConstituentOrderMixin -- word order as ROLE<->SYNTACTIC synapses.

This implements the constituent-order mechanism of Mitropolsky &
Papadimitriou (2025), "Simulated Language Acquisition in a Biologically
Realistic Model of the Brain", section 2.4, rather than an accumulated
context buffer.

The paper's learning rule
-------------------------
    "if the first constituent is the subject, then the firing of the noun's
    assembly in PHON propagates through LEX1, to ROLE_agent, and finally into
    SUBJ. This chain of assemblies continues to fire until the next word, say
    the verb, is input. Again starting from PHON, the firing of the verb's
    phonological assembly will propagate through firing into ROLE_action; at
    this point, neurons from SUBJ (from the previous word) also fire into
    ROLE_action, and this is how the fact that verb comes after subject is
    recorded in the synapses between SUBJ and ROLE_action."

So word order lives in ``SYN[i] -> ROLE[i+1]`` synapses. Nothing accumulates:
the state is "which syntactic area is currently firing", which is bounded no
matter how long the sentence is. This is why the model needs no CONTEXT area,
and why a fixed-capacity context buffer was the wrong shape for the problem.

The paper's generation rule (the trigger)
-----------------------------------------
    "we assume that there is an assembly, in a new area, which, by firing,
    inhibits all three ROLE role areas for a single step. We call this
    assembly the trigger."

and, on why transient inhibition is required at all:

    "the current constituent will continue to receive the most input from its
    own recurrent firing, and because the role areas are in mutual inhibition,
    will continue to fire until the next time all the role areas are
    inhibited."

That is the published account of the same attractor-lock this codebase hits
elsewhere: recurrent self-input makes the active assembly win indefinitely,
and a one-step global inhibition is the sanctioned escape. Role areas are in
mutual inhibition, which the paper notes is "the only use of interarea
inhibition in our model".

Per-word timing is tau = 2 steps (paper Fig. 3a), not the ~10 steps needed to
*form* a new assembly (Papadimitriou et al., PNAS 2020: "a stable assembly is
formed after about T = 10 steps").
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.assembly_calculus.binding import (
    bind, bind_strength, materialize_fiber,
)
from neural_assemblies.assembly_calculus.ops import (
    activate_assembly, project, _snap,
)
from neural_assemblies.assembly_calculus.readout import readout_all

from ..core.areas import (
    MOOD, OBJ, ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT, ROLE_SCENE,
    SUBJ, SYN_VERB,
)

if TYPE_CHECKING:
    from ..curriculum.data import GroundedSentence

# Per-word propagation steps. Paper Fig. 3a: tau = 2.
TAU = 2

# Thematic role area <-> the syntactic area it feeds.
_ROLE_TO_SYN = {
    ROLE_AGENT: SUBJ,
    ROLE_ACTION: SYN_VERB,
    ROLE_PATIENT: OBJ,
}
# All three constituents of a transitive clause. The verb must be here or its
# position is unrepresentable and OSV/OVS become indistinguishable.
_ROLE_ORDER = (ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT)


class ConstituentOrderMixin:
    """Learn and use word order via SYN -> ROLE transition synapses."""

    _order_paths_bootstrapped: bool = False

    def _bootstrap_order_paths(self) -> None:
        """Materialize SYN->ROLE and MOOD->ROLE fibers once, plasticity off.

        A sparse projection can only strengthen synapses that already exist,
        so the transition fibers must be materialized before any Hebbian
        pairing can be written into them.
        """
        if self._order_paths_bootstrapped:
            return
        brain = self.brain

        # Every fiber is materialized with the source actually firing. A
        # projection from an empty area allocates no columns, so a later
        # Hebbian pairing would have nothing to write to -- see
        # assembly_calculus.binding for the full failure mode.
        seeds: Dict[str, Assembly] = {}
        for role in _ROLE_ORDER:
            lex = self.role_lexicons.get(role, {})
            if lex:
                seeds[role] = next(iter(lex.values()))

        for role, seed in seeds.items():
            materialize_fiber(brain, role, _ROLE_TO_SYN[role], src_assembly=seed)
            materialize_fiber(brain, role, ROLE_SCENE, src_assembly=seed)

        # The syntactic areas and SCENE now hold activity, so their outgoing
        # fibers can be materialized in turn. Every syntactic slot is covered,
        # including the verb's -- a missing fiber here is permanently dead.
        # dict.fromkeys, not set(): this loop ALLOCATES neurons, so its order is
        # load-bearing, and set-of-str iteration order varies with PYTHONHASHSEED
        # from one process to the next. That made identical seeds give different
        # parses across runs while being perfectly stable within a run.
        for syn in dict.fromkeys(_ROLE_TO_SYN.values()):
            for role in _ROLE_ORDER:
                materialize_fiber(brain, syn, role)
        for role in _ROLE_ORDER:
            materialize_fiber(brain, ROLE_SCENE, role)

        for n in (3, 2):
            fm = self._frame_assembly(n)
            if fm is not None:
                for role in _ROLE_ORDER:
                    materialize_fiber(brain, MOOD, role, src_assembly=fm)
        mood = self._ensure_mood_assembly()
        if mood is not None:
            for role in _ROLE_ORDER:
                materialize_fiber(brain, MOOD, role, src_assembly=mood)

        self._order_paths_bootstrapped = True

    def prepare_scene(self, fillers: Dict[str, str]) -> None:
        """Bind a scene into the role areas and build its SCENE assembly.

        `fillers` maps role area -> word, e.g. {ROLE_AGENT: "dog"}.

        This is the paper's precondition for generation: assemblies sit in the
        role areas, and ROLE_SCENE holds one assembly for the whole scene,
        synaptically connected to them. Without it the trigger would clear the
        role areas with nothing to restore the fillers, and generation would
        stop after the first constituent.
        """
        self._bootstrap_order_paths()

        bound: List[str] = []
        # The scene fixes WHICH filler occupies each role. Generation then
        # decides only the ORDER in which those roles are realized -- it must
        # not re-derive the filler by readout, which would let a lexicon
        # neighbour outscore the participant actually in the scene.
        self._scene_fillers = {}
        for role_area, word in fillers.items():
            stored = self.role_lexicons.get(role_area, {}).get(word)
            if stored is None:
                continue
            self._scene_fillers[role_area] = word
            activate_assembly(self.brain, stored)
            self.brain.areas[role_area].fix_assembly()
            bound.append(role_area)

        if not bound:
            return
        try:
            # All bound roles co-fire into SCENE, so the scene assembly is a
            # conjunction of the participants.
            self.brain.project({}, {r: [ROLE_SCENE] for r in bound})
            for _ in range(TAU - 1):
                self.brain.project(
                    {},
                    {**{r: [ROLE_SCENE] for r in bound},
                     ROLE_SCENE: [ROLE_SCENE]},
                )
            self._scene_assembly = _snap(self.brain, ROLE_SCENE)
            # Reciprocal SCENE -> role, so the scene can restore each filler.
            for role_area in bound:
                self.brain.project({}, {ROLE_SCENE: [role_area]})
        finally:
            for role_area in bound:
                self.brain.areas[role_area].unfix_assembly()

    # ------------------------------------------------------------------

    def _fire_constituent(
        self, word: str, role_area: str,
        *, mood_assembly: Optional[Assembly] = None,
    ) -> Optional[str]:
        """Fire one constituent: filler -> ROLE -> SYN, for tau steps.

        When ``mood_assembly`` is given, MOOD co-fires into the syntactic area,
        so the assembly that lands in SUBJ/VERB/OBJ is MOOD-SPECIFIC. That is
        what lets one brain hold several moods with different word orders: the
        ``SYN[i] -> ROLE[i+1]`` synapses are keyed by a syntactic assembly that
        already differs per mood, so two moods that disagree about what follows
        the subject write into disjoint synapses instead of competing for the
        same ones. It mirrors the reference implementation, which projects MOOD
        into the syntactic area on every step
        (``project_map[MOOD] = [SYNTAX_area]``).

        Returns the syntactic area now holding the constituent, or None.
        """
        stored = self.role_lexicons.get(role_area, {}).get(word)
        if stored is None:
            return None
        syn = _ROLE_TO_SYN.get(role_area)
        if syn is None:
            return None

        # Drive the syntactic slot from the filler-INDEPENDENT role code, not
        # from this filler. A syntactic area represents "subject position",
        # not "dog-as-subject": if the filler drives it, SUBJ holds a
        # different assembly for every agent, so the SYN -> ROLE order
        # pairings scatter across as many disjoint source patterns as there
        # are fillers, and at generation SUBJ matches only the fraction of
        # training that used that one filler. Measured: the whole cue -> role
        # transition matrix sat at ~0.02 despite each pairing individually
        # reaching 0.85. The filler stays in the role area and in SCENE.
        use_mood = mood_assembly is not None and MOOD in self.brain.areas
        code = self._role_identity(role_area)
        activate_assembly(self.brain, code if code is not None else stored)
        # ROLE (+ MOOD, tonic) -> SYN, tau steps.
        srcs = {role_area: [syn]}
        if use_mood:
            activate_assembly(self.brain, mood_assembly)
            srcs[MOOD] = [syn]
        self.brain.project({}, srcs)
        for _ in range(TAU - 1):
            recur = dict(srcs)
            recur[syn] = [syn]
            self.brain.project({}, recur)
        return syn

    def _role_identity(self, role_area: str) -> Optional[Assembly]:
        """The filler-independent 'role code' for `role_area`.

        Word order carries one bit -- which role opens the clause -- but the
        stored role assemblies are filler-specific by construction. Binding a
        single MOOD assembly to a different filler in every sentence spreads
        the strengthening over largely disjoint neuron sets, so no coherent
        per-area preference accumulates and every role area ends up receiving
        about the same drive. That was measured: the MOOD->role margin sat at
        0.2-0.4% and did not grow with training.

        The role code is the component the fillers already share. Binding into
        one role area leaves a common core plus a filler-specific remainder --
        measured here at 0.78-0.88 pairwise overlap between different fillers
        in the same role, against 0.000 for the same filler across roles. So
        the shared neurons are recovered by frequency across the role's stored
        bindings; nothing is hand-installed.
        """
        cached = getattr(self, "_role_identity_cache", None)
        if cached is None:
            cached = {}
            self._role_identity_cache = cached
        if role_area in cached:
            return cached[role_area]

        lex = self.role_lexicons.get(role_area, {})
        if not lex:
            cached[role_area] = None
            return None
        # With a single exemplar the shared core is trivially that exemplar --
        # correct, and the common case for ROLE_ACTION in a one-verb corpus.

        counts: Counter = Counter()
        for asm in lex.values():
            counts.update(int(x) for x in asm.winners)
        top = [n for n, _ in counts.most_common(self.k)]
        if not top:
            cached[role_area] = None
            return None

        # `counts` is accumulated from snapped assemblies, so `top` is already
        # in NEURON-ID space; the wrapper states that rather than implying it.
        identity = Assembly(
            area=role_area,
            winners=NeuronIds(np.array(top, dtype=np.uint32)),
        )
        cached[role_area] = identity
        return identity

    def _activate_filler_core(self, word: str) -> None:
        """Put `word`'s stabilized core assembly into its core area."""
        core = self._word_core_area(word)
        stored = self.core_lexicons.get(core, {}).get(word)
        if stored is not None:
            activate_assembly(self.brain, stored)

    # -- mood ------------------------------------------------------------

    DEFAULT_MOOD = "declarative"

    @property
    def mood(self) -> str:
        """The mood currently being trained / generated in."""
        return getattr(self, "_mood", self.DEFAULT_MOOD)

    def set_mood(self, mood: str) -> None:
        """Select the mood (a language register with its own word order).

        The paper sweeps the NUMBER OF MOODS as one of its two axes: each mood
        of a language may impose a different constituent order, and the model
        must learn to condition on mood rather than memorize a single global
        order. With one mood the task is degenerate -- there is exactly one
        correct order, so a learner can ignore everything else and still be
        right.
        """
        self._mood = str(mood)

    def _frame_assembly(self, n_constituents: int) -> Optional[Assembly]:
        """A distinct MOOD assembly per (mood, clause frame).

        The paper gives each mood "a distinct chain of assemblies between ROLE
        and SUBJ, VERB, OBJ". Clause frame behaves the same way: what follows
        the subject is the object in a transitive clause and the verb in an
        intransitive one, so a single shared context assembly makes the two
        chains compete for the same synapses. Measured, that competition is
        fatal for every order whose intransitive chain is not a prefix of its
        transitive chain (SOV, OSV, OVS, VOS), and more input does not help --
        it strengthens both transitions equally.

        MOOD is the same argument one level up: two moods with different orders
        disagree about what follows the subject, so they must not share a
        context assembly either. Keying on (mood, frame) gives each mood its
        own chain, which is exactly what the paper specifies.
        """
        cache = getattr(self, "_frame_assemblies", None)
        if cache is None:
            cache = {}
            self._frame_assemblies = cache
        frame = "transitive" if n_constituents >= 3 else "intransitive"
        key = (self.mood, frame)
        if key in cache:
            return cache[key]
        if MOOD not in self.brain.areas:
            return None
        stim = f"mood_{self.mood}_frame_{frame}"
        if stim not in self.brain.stimuli:
            self.brain.add_stimulus(stim, self.k)
        project(self.brain, stim, MOOD, rounds=self.rounds)
        cache[key] = _snap(self.brain, MOOD)
        return cache[key]

    def _ensure_mood_assembly(self) -> Optional[Assembly]:
        """A stable assembly in MOOD for the default (declarative) mood.

        The paper drives the FIRST constituent from MOOD; without an assembly
        there, generation has no cue to open a sentence with.
        """
        cached = getattr(self, "_mood_assembly", None)
        if cached is not None:
            return cached
        if MOOD not in self.brain.areas:
            return None
        stim = "mood_declarative"
        if stim not in self.brain.stimuli:
            self.brain.add_stimulus(stim, self.k)
        project(self.brain, stim, MOOD, rounds=self.rounds)
        self._mood_assembly = _snap(self.brain, MOOD)
        return self._mood_assembly

    def train_constituent_order(
        self,
        sentences: List["GroundedSentence"],
        *,
        repetitions: int = 1,
    ) -> None:
        """Record which role opens a sentence, and ``SYN[i] -> ROLE[i+1]``.

        Two pairings per sentence type, matching the paper: after the trigger
        the role areas "receive synaptic input from either (1) mood, (in the
        case of the first word) or (2) one of the syntactic areas". Training
        only (2) leaves generation with no way to choose the opening
        constituent, so the first step falls back to an arbitrary tie-break.
        """
        # Role codes are derived from the current bindings, so drop any cache
        # from an earlier lexicon state.
        self._role_identity_cache = {}
        self._bootstrap_order_paths()
        mood = self._ensure_mood_assembly()

        for _ in range(max(1, repetitions)):
            for sent in sentences:
                seq = self._constituent_sequence(sent)
                if len(seq) < 2:
                    continue

                # Order is bound to the filler-independent ROLE CODE, not to
                # this sentence's filler. Order carries one bit -- which role
                # opens the clause -- and binding it to a different filler
                # each sentence scatters the strengthening over disjoint
                # neuron sets, leaving no per-area preference at all.
                #
                # (1) MOOD -> opening constituent.
                _first_word, first_role = seq[0]
                first_core = self._word_core_area(_first_word)
                mood = self._frame_assembly(len(seq)) or mood
                if mood is not None and first_core is not None:
                    # The filler's core area is the TEACHER: it drives the
                    # role area to the assembly being taught while MOOD fires
                    # alongside, and plasticity records MOOD -> that assembly.
                    self._activate_filler_core(_first_word)
                    bind(
                        self.brain,
                        sources=[MOOD],
                        target_area=first_role,
                        teachers=[first_core],
                        source_assemblies={MOOD: mood},
                    )

                # (2) SYN[i] -> ROLE[i+1].
                #
                # The MOOD conditioning is carried by the SYNTACTIC assembly,
                # not by adding MOOD to this cue: `_fire_constituent` co-fires
                # MOOD into the syntactic area, so SUBJ-under-mood-A is a
                # different assembly from SUBJ-under-mood-B and these synapses
                # are already mood-specific. That is the reference
                # implementation's arrangement (MOOD projects into the SYNTAX
                # area at every step) and it keeps mood out of the role
                # competition, where it would otherwise dominate.
                prev_syn: Optional[str] = None
                for word, role_area in seq:
                    if prev_syn is not None:
                        core = self._word_core_area(word)
                        if core is not None:
                            self._activate_filler_core(word)
                            bind(
                                self.brain,
                                sources=[prev_syn],
                                target_area=role_area,
                                teachers=[core],
                            )
                    prev_syn = self._fire_constituent(
                        word, role_area, mood_assembly=mood) or prev_syn

    def _constituent_sequence(
        self, sent: "GroundedSentence",
    ) -> List[Tuple[str, str]]:
        """Annotated (word, role_area) pairs in surface order."""
        out: List[Tuple[str, str]] = []
        for word, role in zip(sent.words, sent.roles, strict=True):
            if role == "agent":
                out.append((word, ROLE_AGENT))
            elif role == "action":
                out.append((word, ROLE_ACTION))
            elif role == "patient":
                out.append((word, ROLE_PATIENT))
        return out

    # ------------------------------------------------------------------

    def _compete(
        self,
        cue_area: str,
        cue_assembly: Optional[Assembly] = None,
        *,
        exclude: Optional[set] = None,
    ) -> Optional[Tuple[str, str, float]]:
        """Trigger, then let the role areas compete for the cue.

        `exclude` holds roles already realized in this clause. A thematic role
        is expressed once per clause, and the constituent that just fired
        still has the strongest trace, so without excluding it the competition
        simply re-selects it and generation cannot advance.

        Returns (role_area, word, score) for the winner.
        """
        exclude = exclude or set()
        scene_fillers = getattr(self, "_scene_fillers", {}) or {}
        candidates = [
            r for r in _ROLE_ORDER
            if r not in exclude
            and r in scene_fillers
            and scene_fillers[r] in self.role_lexicons.get(r, {})
        ]
        if not candidates:
            return None
        # The trigger: one step of inhibition across the role areas. Without
        # it the currently firing constituent keeps winning on its own
        # recurrence, which is exactly the lock the paper describes.
        self.brain.inhibit_areas(list(_ROLE_ORDER))

        # Score by total synaptic DRIVE from the cue, which is what the paper
        # selects on ("the role area with the most synaptic input will be
        # selected") and what Brain._apply_mutual_inhibition itself compares.
        #
        # Scoring by overlap with a particular filler assembly does not work
        # here: one MOOD assembly is paired over training with a different
        # filler in every sentence, so it cannot reproduce any single one of
        # them and every candidate saturates at ~1.0. Drive still differs.
        #
        # SCENE is excluded from the cue: it contains every participant, so it
        # drives all role areas about equally and washes out the ordering
        # signal. Restoring filler content is _fire_constituent's job.
        best: Optional[Tuple[str, str, float]] = None
        for role in candidates:
            code = self._role_identity(role)
            if code is None:
                continue
            # How strongly does the cue reproduce THIS role's code? The code
            # is the same target across every sentence, so the pairings add
            # coherently and the score reflects learned order rather than
            # which filler happens to be in the scene.
            # Scored on the cue ALONE. Adding SCENE drive here was
            # tested and is a regression (SVO 1.00 -> 0.00): the scene
            # contains every participant, so it drives all role areas and
            # swamps the ordering signal.
            # NOTE: MOOD is deliberately NOT added to this cue. Mood
            # conditioning happens upstream, in the SYNTACTIC assembly (see
            # _fire_constituent): the syntactic area is what differs per mood,
            # so SYN[i] -> ROLE[i+1] is already mood-specific. Adding MOOD here
            # instead was measured to destroy the ordering signal outright --
            # MOOD carries the "which role OPENS the clause" pairing, so it
            # dominates the weaker syntactic drive and every order collapses to
            # the same answer (SVO and SOV both produced 'OSV').
            score = bind_strength(
                self.brain,
                sources=[cue_area],
                target_area=role,
                target_assembly=code,
                source_assemblies=(
                    {cue_area: cue_assembly}
                    if cue_assembly is not None else None
                ),
            )
            if best is None or score > best[2]:
                best = (role, scene_fillers[role], score)
        return best

    def next_constituent(
        self, current_syn: Optional[str],
    ) -> Optional[Tuple[str, str]]:
        """Fire the trigger and return the next (role_area, word).

        The trigger inhibits the role areas for one step; on release, the role
        area receiving the most input from the current syntactic area wins the
        mutual-inhibition competition. Its assembly is then read out against
        the role lexicon to recover the filler.
        """
        self._bootstrap_order_paths()
        if current_syn is None:
            return None

        # Trigger: one step of inhibition across the role areas.
        self.brain.inhibit_areas(list(_ROLE_ORDER))

        with self.brain.frozen():
            scores: Dict[str, float] = {}
            reads: Dict[str, Tuple[str, float]] = {}
            for role in _ROLE_ORDER:
                lex = self.role_lexicons.get(role, {})
                if not lex:
                    continue
                # SCENE (tonic) re-supplies the filler content that the
                # trigger just cleared; the syntactic area supplies the order.
                # The role area with the greatest combined input wins the
                # mutual-inhibition competition.
                sources = {current_syn: [role]}
                if ROLE_SCENE in self.brain.areas and getattr(
                    self, "_scene_assembly", None,
                ) is not None:
                    activate_assembly(self.brain, self._scene_assembly)
                    sources[ROLE_SCENE] = [role]
                self.brain.project({}, sources)
                asm = _snap(self.brain, role)
                ranked = readout_all(asm, lex)
                if ranked:
                    reads[role] = (ranked[0][0], ranked[0][1])
                    scores[role] = ranked[0][1]

        if not scores:
            return None
        # Mutual inhibition: the role area with the greatest input fires.
        winner = max(scores, key=lambda r: scores[r])
        return winner, reads[winner][0]

    def generate_from_roles(self, max_len: int = 6) -> List[str]:
        """Generate a word sequence by repeatedly firing the trigger.

        Assumes the role areas have been prepared with a scene (fillers bound
        into ROLE_AGENT / ROLE_PATIENT). Mirrors the paper's loop:
        ``repeat: fire the trigger, until the sentence is generated``.
        """
        self._bootstrap_order_paths()
        out: List[str] = []
        used: set = set()
        # First constituent is cued by MOOD; subsequent ones by the syntactic
        # area the previous constituent left firing.
        cue_area: Optional[str] = MOOD
        mood_assembly = (self._frame_assembly(
            len(getattr(self, "_scene_fillers", {}) or {}))
            or self._ensure_mood_assembly())
        cue_assembly = mood_assembly

        for _ in range(max_len):
            if cue_area is None:
                break
            step = self._compete(
                cue_area, cue_assembly, exclude=used,
            )
            if step is None:
                break
            role_area, word, _score = step
            used.add(role_area)
            out.append(word)
            # MOOD is tonic -- the paper has it "firing at every step
            # throughout generation". It must co-fire here for the same reason
            # it did in training: the syntactic assembly this leaves behind is
            # the cue for the next constituent, and it has to be the
            # mood-specific one the transitions were trained against.
            cue_area = self._fire_constituent(
                word, role_area, mood_assembly=mood_assembly)
            cue_assembly = None
        return out
