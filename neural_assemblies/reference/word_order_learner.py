"""Faithful port of the reference word-order learner (helper-area architecture).

Source: ``.reference/dmitropolsky-assemblies/word_order_int.py`` -- "word order
learner with 'intermediate' helper TPJ areas" -- the author's own implementation
of Mitropolsky & Papadimitriou (2025) Sec. 2.4-2.5.

WHY A SEPARATE PORT RATHER THAN AN EDIT TO ConstituentOrderMixin.  The two
designs differ in more than one area:

* the mixin drives the syntactic area from a filler-INDEPENDENT "role code"
  (``_role_identity``), an abstraction the reference does not have; the
  reference drives ``PHON -> TPJ -> helper -> SYNTAX`` from real word
  assemblies;
* the mixin scores the generation competition with ``bind_strength`` -- overlap
  against a stored identity assembly -- while the reference scores TOTAL
  SYNAPTIC INPUT between live assemblies (``get_total_input``), which is also
  what the paper specifies: "the role area with the most synaptic input will be
  selected".

Grafting only the helper areas onto the mixin was tried and destroyed the order
signal outright (trained SVO produced VSO, SOV produced VSO -- outputs constant
and uncorrelated with training). The chain and the scoring have to move
together, so they are ported together here, standalone and independently
testable, leaving the working mixin untouched.

Architecture (identical to the reference)::

    PHON  ->  TPJ_x  <->  TPJ_x_helper  ->  SYNTAX_x
    MOOD  ->  SYNTAX_x                       (every step)
    MOOD  ->  TPJ_x_helper                   (first word only: which role opens)
    SYNTAX_i -> TPJ_x_helper                 (what follows constituent i)

Word order lives in the ``SYNTAX_i -> helper`` synapses, and the mood
conditioning rides in because ``MOOD -> SYNTAX`` fires at every step, making the
syntactic assemblies themselves mood-specific.

One deliberate improvement over the reference: cross-area drive comparison uses
``input_drive(..., metric="pre_kwta")``, which normalizes per candidate neuron.
The reference sums raw weights over winners, which biases the comparison toward
whichever area happens to have recruited more neurons (this repository measured
two role areas differing 391 vs 449, enough to reverse a ranking on size alone).
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Sequence

from neural_assemblies.assembly_calculus.binding import input_drive
from neural_assemblies.core.brain import Brain

PHON = "PHON"
MOOD = "MOOD"

TPJ = {"S": "TPJ_agent", "V": "TPJ_action", "O": "TPJ_patient"}
HELPER = {
    "S": "TPJ_agent_helper",
    "V": "TPJ_action_helper",
    "O": "TPJ_patient_helper",
}
SYNTAX = {"S": "SYNTAX_subject", "V": "SYNTAX_verb", "O": "SYNTAX_object"}

# The FSM arc area (`conjunctive_arc`). Not in the reference.
ARC = "ARC"

CONSTITUENTS = ("S", "V", "O")


class WordOrderLearner:
    """Reference word-order learner: scene -> sentence with learned order."""

    def __init__(
        self,
        *,
        num_nouns: int = 4,
        num_verbs: int = 2,
        mood_orders: Optional[Dict[int, Sequence[str]]] = None,
        n: int = 1000,
        k: int = 50,
        p: float = 0.05,
        beta: float = 0.1,
        seed: int = 0,
        training_fire_rounds: int = 10,
        previous_constituent_fire_rounds: int = 2,
        norm_init: bool = False,
        per_mood_syntax: bool = False,
        conjunctive_arc: bool = False,
    ):
        self.num_nouns = num_nouns
        self.num_verbs = num_verbs
        self.num_words = num_nouns + num_verbs
        self.mood_orders = dict(mood_orders or {0: ("S", "V", "O")})
        self.num_moods = len(self.mood_orders)
        self.k = k
        self.training_fire_rounds = training_fire_rounds
        self.previous_constituent_fire_rounds = previous_constituent_fire_rounds
        self._rng = random.Random(seed)

        # norm_init is off by default: this is a literature reproduction, and
        # the reference substrate is un-normalized (see the repo's
        # norm_init-substrate-vs-reference convention).
        self.brain = Brain(p=p, seed=seed, norm_init=norm_init)
        # PHON and MOOD are EXPLICIT: each word / mood is a fixed, addressable
        # assembly, activated by index exactly as the reference does.
        self.brain.add_explicit_area(PHON, self.num_words * k, k, beta)
        self.brain.add_explicit_area(MOOD, max(self.num_moods, 1) * k, k, beta)
        for c in CONSTITUENTS:
            self.brain.add_area(TPJ[c], n, k, beta)
            self.brain.add_area(HELPER[c], n, k, beta)
            self.brain.add_area(SYNTAX[c], n, k, beta)

        # per_mood_syntax: give each mood its OWN syntactic areas.
        #
        # This is a DELIBERATE DEVIATION from the paper, which keeps one
        # SUBJ/VERB/OBJ and expects the per-mood "distinct chain of assemblies"
        # to be emergent -- MOOD fires tonically into the syntactic areas and is
        # supposed to select different winners per mood. Measured here, that
        # emergent version forms the distinct chains correctly at initialization
        # (SYNTAX overlap 0.04 between two moods) and then LOSES them to
        # training (-> 1.00 within ~20 sentences): the helper is shared between
        # moods, so helper->SYN potentiates toward whichever assembly won in
        # BOTH moods' sentences and they merge. That collapse survives 100x more
        # capacity (n=1e5), the paper's beta=0.06, norm_init, and raising MOOD's
        # plasticity -- see the module tests for the numbers.
        #
        # Making the distinctness STRUCTURAL instead removes the shared target.
        # Measured: SYNTAX separation 0.02, and multi-mood generation 24/24
        # across 4 mood pairs x 3 seeds x 2 moods, against 17/24 for the
        # emergent version (whose failures are concentrated in the pairs that
        # need to diverge after a shared opening constituent: 3/6 and 2/6).
        #
        # Off by default so the class stays a faithful reproduction; turn it on
        # to actually learn several moods.
        # conjunctive_arc: make HELPER the FSM ARC ASSEMBLY.
        #
        # Multi-mood word order is a finite state machine -- state = the last
        # constituent emitted, symbol = the mood, transition = the next
        # constituent -- and Dabagia/Papadimitriou/Vempala (2025) Thm 4 shows
        # NEMO learns an arbitrary FSM. Their architecture is explicit about
        # what carries a transition: "each pair of state and symbol assemblies
        # projects to the associated ARC ASSEMBLY, which in turn projects back
        # to the assembly corresponding to the state the FSM would switch to."
        # The arc A_{q,sigma} is a CONJUNCTION of (state, symbol) holding its
        # own identity.
        #
        # Our HELPER areas are already positioned to be that arc: they receive
        # SYNTAX_prev (the state) and MOOD (the symbol). But the reference
        # fires MOOD into the helper on the FIRST WORD ONLY, so every
        # non-initial transition is mood-blind by construction -- which is
        # exactly the recorded failure profile, with failures concentrated in
        # the mood pairs that must diverge AFTER a shared opening constituent.
        #
        # FIRST ATTEMPT, MEASURED AND REJECTED: simply co-fire MOOD into the
        # current constituent's helper at every step, making the EXISTING
        # helper the arc. It changes nothing -- helper mood-separation stays at
        # 1.00 and generation is unchanged. The reason is legible in the
        # architecture: `_activate_role` drives TPJ -> helper for ten rounds,
        # so a single MOOD co-fire is a few percent of the helper's drive and
        # cannot move its winners. That is [[mood-collapse-is-a-drive-ratio]]
        # again, one layer down.
        #
        # And that is precisely where the reference architecture departs from
        # Theorem 4: the theorem's arc A_{q,sigma} receives the state and the
        # symbol AND NOTHING ELSE. Our helper also receives the word, which
        # dominates it. A conjunction cannot form in an area whose winners are
        # already decided by a third input.
        #
        # SO THE ARC GETS ITS OWN AREA. `ARC` receives only SYNTAX_prev (the
        # state q) and MOOD (the symbol sigma); its assembly IS A_{q,sigma}.
        # The transition is then learned as ARC -> helper[next], and generation
        # picks the next constituent by drive from ARC alone. Both ends of
        # Theorem 4's "each pair of state and symbol assemblies projects to the
        # associated arc assembly, which in turn projects back to the ... state
        # the FSM would switch to".
        #
        # The first word uses MOOD alone into ARC -- the start state q0 -- which
        # is what the reference already did with MOOD -> helper, now routed
        # through the same area as every other transition instead of being a
        # special case.
        #
        # Off by default: the reference does not do this, and this class is a
        # faithful port. Unlike `per_mood_syntax` the distinctness stays
        # LEARNED -- one shared area, no per-mood areas, and the number of arc
        # assemblies it must hold is |states| x |moods|, which Thm 4 sizes at
        # n >= |Q|^2|Sigma|^2 (a few hundred neurons here).
        self.conjunctive_arc = conjunctive_arc
        if conjunctive_arc:
            self.brain.add_area(ARC, n, k, beta)
        self.per_mood_syntax = per_mood_syntax
        self._mood_now = 0
        self._syn_areas: Dict[tuple, str] = {}
        if per_mood_syntax:
            for mi in self.mood_orders:
                for c in CONSTITUENTS:
                    name = f"{SYNTAX[c]}_mood{mi}"
                    self.brain.add_area(name, n, k, beta)
                    self._syn_areas[(mi, c)] = name

    def _syn(self, c: str) -> str:
        """The syntactic area for constituent `c` under the current mood."""
        if self.per_mood_syntax:
            return self._syn_areas[(self._mood_now, c)]
        return SYNTAX[c]

    # -- training ---------------------------------------------------------

    def _activate_role(self, phon_index: int, c: str, firings: int = 10) -> None:
        """Drive a word into its role area and its helper (reference
        ``activate_role``)."""
        tpj, helper = TPJ[c], HELPER[c]
        self.brain.activate(PHON, phon_index)
        self.brain.project({}, {PHON: [tpj]})
        self.brain.project({}, {PHON: [tpj], tpj: [helper, tpj]})
        for _ in range(firings):
            self.brain.project({}, {
                PHON: [tpj],
                tpj: [helper, tpj],
                helper: [tpj, helper],
            })

    def _project_training(
        self, c: str, t: int, *, first_word: bool, previous: Optional[str],
    ) -> None:
        """One training step for constituent `c` (reference
        ``project_training``)."""
        tpj, helper, syn = TPJ[c], HELPER[c], self._syn(c)
        # NOTE: an earlier version primed SYN from MOOD alone at t == 0, on the
        # theory that MOOD should establish the syntactic frame before the
        # constituent filled it. It is removed: it deviates from the reference
        # for no measured benefit (mood separation in SYNTAX moved 0.955 ->
        # 0.952), and the one multi-mood pair it appeared to fix (SVO+VSO) was
        # in fact passing on unbounded weights, which the w_max clamp on the
        # dense bridge later exposed. MOOD co-fires with the helper below, as
        # in the reference.
        pmap: Dict[str, List[str]] = {
            PHON: [tpj],
            tpj: [helper, tpj],
            helper: [helper, tpj, syn],
            MOOD: [syn],          # tonic: keeps the syntactic assembly
        }                         # mood-specific, which is what keys the order
        if t > 0:
            pmap[syn] = [syn]
        if self.conjunctive_arc:
            # The order synapse runs through the ARC instead: ARC already
            # holds A_{q,sigma} (formed in `_form_arc` before these rounds), so
            # ARC -> helper records "c follows q IN THIS MOOD". Fired for the
            # same number of rounds as the direct synapse it replaces, so the
            # two arms differ in the SOURCE of the order signal and not in how
            # much drive it gets.
            if t <= self.previous_constituent_fire_rounds:
                pmap[ARC] = [helper]
        else:
            if first_word:
                # Which constituent OPENS the clause is learned from MOOD.
                pmap[MOOD] = pmap[MOOD] + [helper]
            if (t <= self.previous_constituent_fire_rounds
                    and previous is not None):
                # The order synapse: the PREVIOUS constituent's syntactic area
                # fires into THIS constituent's helper, recording "c follows
                # previous". Mood-blind -- SYNTAX_prev is the same assembly in
                # every mood, which is the whole problem.
                pmap[self._syn(previous)] = [helper]
        self.brain.project({}, pmap)

    def _form_arc(self, previous: Optional[str], rounds: int = 2) -> None:
        """Settle A_{q,sigma} in ARC from (SYNTAX_prev, MOOD).

        `previous is None` is the start state q0, where the arc is MOOD alone
        -- the same signal the reference uses to choose the opening
        constituent, routed through the same area as every other transition
        rather than kept as a special case.
        """
        src: Dict[str, List[str]] = {MOOD: [ARC]}
        if previous is not None:
            src[self._syn(previous)] = [ARC]
        for _ in range(rounds):
            self.brain.project({}, src)

    def train_sentence(self, mood_index: int = 0) -> None:
        """Present one random transitive sentence in `mood_index`'s order."""
        self._mood_now = mood_index
        subj = self._rng.randrange(self.num_nouns)
        obj = self._rng.randrange(self.num_nouns)
        verb = self._rng.randrange(self.num_nouns, self.num_words)
        self.brain.activate(MOOD, mood_index)
        for c, idx in (("S", subj), ("O", obj), ("V", verb)):
            self._activate_role(idx, c)

        order = list(self.mood_orders[mood_index])
        previous: Optional[str] = None
        for pos, c in enumerate(order):
            if self.conjunctive_arc:
                self._form_arc(previous)
            for t in range(self.training_fire_rounds):
                self._project_training(
                    c, t, first_word=(pos == 0), previous=previous)
            previous = c

    def train(self, num_sentences: int, mood_index: Optional[int] = None) -> None:
        for _ in range(num_sentences):
            m = (mood_index if mood_index is not None
                 else self._rng.randrange(self.num_moods))
            self.train_sentence(m)

    # -- generation -------------------------------------------------------

    def _strongest(self, source: str | Sequence[str],
                   candidates: Sequence[str]) -> str:
        """The candidate helper receiving the most synaptic input from `source`.

        This is the paper's selection rule ("the role area with the most
        synaptic input will be selected") and the reference's
        ``get_biggest_input_TPJ_from_*``, but normalized per candidate neuron so
        the comparison is not decided by which area recruited more neurons.

        `source` may be several areas. Under `conjunctive_arc` the transition
        is scored on (SYNTAX_prev, MOOD) together, because that pair -- not
        either alone -- is what identifies the arc.
        """
        sources = [source] if isinstance(source, str) else list(source)
        drives = input_drive(
            self.brain,
            sources=sources,
            target_areas=[HELPER[c] for c in candidates],
            metric="pre_kwta",
        )
        best = max(candidates, key=lambda c: drives.get(HELPER[c], 0.0))
        return best

    def generate(self, mood_index: int = 0, firings: int = 3) -> List[str]:
        """Generate the constituent order for a scene (reference
        ``generate_random_sentence``). Returns e.g. ``['S', 'V', 'O']``."""
        self._mood_now = mood_index
        subj = self._rng.randrange(self.num_nouns)
        obj = self._rng.randrange(self.num_nouns)
        verb = self._rng.randrange(self.num_nouns, self.num_words)

        with self.brain.frozen():          # reference: no_plasticity = True
            self.brain.activate(MOOD, mood_index)
            for c, idx in (("S", subj), ("O", obj), ("V", verb)):
                self._activate_role(idx, c, firings=firings)

            # First constituent: the start arc q0. Without the arc this is
            # MOOD -> helper directly, exactly as the reference does it.
            if self.conjunctive_arc:
                self._form_arc(None)
                cue: str = ARC
            else:
                cue = MOOD
            current = self._strongest(cue, CONSTITUENTS)
            self.brain.project({}, {cue: [HELPER[current]]})
            order = [current]

            for _ in range(len(CONSTITUENTS) - 1):
                syn = self._syn(current)
                self.brain.project({}, {HELPER[current]: [syn], MOOD: [syn]})
                remaining = [c for c in CONSTITUENTS if c not in order]
                if self.conjunctive_arc:
                    # Re-form the arc for the transition out of `current`,
                    # then let it -- not the mood-blind syntactic area -- pick
                    # the next constituent.
                    self._form_arc(current)
                    cue = ARC
                else:
                    cue = syn
                # Refresh every role area's helper, then let the cue pick.
                self.brain.project({}, {
                    cue: [HELPER[c] for c in remaining],
                    **{TPJ[c]: [HELPER[c]] for c in CONSTITUENTS},
                })
                current = self._strongest(cue, remaining)
                order.append(current)
        return order

    # -- diagnostics ------------------------------------------------------

    def mood_separation(self, which: str = "syntax") -> Dict[str, float]:
        """Overlap of each constituent's assembly between moods; 1.0 = merged.

        The quantity the whole multi-mood question turns on, exposed as a
        method so experiments stop reaching into ``brain.areas``. `which` is
        ``"syntax"`` or ``"helper"`` -- the latter looks at the arc itself.

        Reference points already measured for SYNTAX: ~0.04 at initialization,
        ~1.00 after ~20 training sentences in the emergent configuration.

        HOW IT DRIVES THE AREA, AND WHY IT MATTERS.  The first version of this
        method fired ``MOOD -> SYNTAX`` alone and read 1.000 at initialization,
        where the true value is 0.04. That is not a measurement: on an untrained
        fiber every weight is 1, every candidate ties, and the deterministic
        index tie-break hands back the same winners for every mood -- failure
        mode 4 in ``assembly_calculus.binding``. A probe that cannot distinguish
        "merged" from "never driven" is worthless here, since those are exactly
        the two hypotheses.

        So the area is driven through the REAL chain, the same one ``generate``
        uses: PHON -> TPJ -> helper -> SYNTAX with MOOD co-firing. The scene is
        held FIXED across moods (same word indices) so that only the mood
        differs; otherwise word identity confounds the comparison.

        AND IT USES ``frozen()``, NOT ``read_only()``.  The second version of
        this method used ``read_only()`` for isolation and ALSO read 1.000 at
        initialization -- for a different reason, which is that ``read_only()``
        freezes the winners. The probe was returning whatever was already in
        the area, identically for every mood. Two different degenerate probes,
        both reading exactly 1.000, neither of them a measurement.

        ``frozen()`` disables plasticity but lets winners move, which is what
        ``generate`` itself runs under, so this reads the same state generation
        reads. ``frozen()`` alone does NOT isolate, though: recruitment is not
        rolled back (see [[probe-isolation-required]]), and measured here,
        calling this method between ``train`` and ``generate`` CHANGED the
        generated order. So the whole probe runs against a deepcopy and the
        trained brain is never touched. That costs one copy of a small brain
        and removes the entire class of "the measurement moved the result".
        """
        if self.num_moods < 2:
            return {c: float("nan") for c in CONSTITUENTS}
        fixed = {"S": 0, "O": min(1, self.num_nouns - 1), "V": self.num_nouns}
        out: Dict[str, float] = {}
        live, saved_mood = self.brain, self._mood_now
        try:
            for c in CONSTITUENTS:
                snaps: List[set] = []
                for mi in self.mood_orders:
                    self.brain = copy.deepcopy(live)
                    with self.brain.frozen():
                        self._mood_now = mi
                        self.brain.activate(MOOD, mi)
                        self._activate_role(fixed[c], c, firings=3)
                        syn = self._syn(c)
                        self.brain.project(
                            {}, {HELPER[c]: [syn], MOOD: [syn]})
                        name = syn if which == "syntax" else HELPER[c]
                        snaps.append(set(self.brain.areas[name].winners))
                pairs = [(a, b) for i, a in enumerate(snaps)
                         for b in snaps[i + 1:]]
                out[c] = (
                    sum(len(a & b) / max(min(len(a), len(b)), 1)
                        for a, b in pairs) / max(len(pairs), 1)
                )
        finally:
            self.brain, self._mood_now = live, saved_mood
        return out

    def arc_separation(self) -> Dict[str, float]:
        """Overlap of the ARC assembly between moods, per state q; 1.0 = merged.

        The quantity `conjunctive_arc` exists to move. A_{q,sigma} must differ
        across sigma for a fixed q, or the arc is not a conjunction and the
        transition it drives cannot be mood-specific. Key ``"q0"`` is the start
        state (MOOD alone); the others are keyed by the previous constituent.

        Returns NaN everywhere when the arc is off -- there is no such area,
        and returning zeros would read as a passing result.
        """
        if not self.conjunctive_arc or self.num_moods < 2:
            return {q: float("nan") for q in ("q0",) + CONSTITUENTS}
        out: Dict[str, float] = {}
        live, saved_mood = self.brain, self._mood_now
        try:
            for q in ("q0",) + CONSTITUENTS:
                snaps: List[set] = []
                for mi in self.mood_orders:
                    self.brain = copy.deepcopy(live)
                    with self.brain.frozen():
                        self._mood_now = mi
                        self.brain.activate(MOOD, mi)
                        if q != "q0":
                            # Put the state assembly up before reading the arc.
                            self._activate_role(0, q, firings=3)
                            self.brain.project(
                                {}, {HELPER[q]: [self._syn(q)], MOOD: [self._syn(q)]})
                        self._form_arc(None if q == "q0" else q)
                        snaps.append(set(self.brain.areas[ARC].winners))
                pairs = [(a, b) for i, a in enumerate(snaps)
                         for b in snaps[i + 1:]]
                out[q] = (
                    sum(len(a & b) / max(min(len(a), len(b)), 1)
                        for a, b in pairs) / max(len(pairs), 1)
                )
        finally:
            self.brain, self._mood_now = live, saved_mood
        return out
