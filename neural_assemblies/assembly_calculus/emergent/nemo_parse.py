"""Parse by gating: the NEMO control loop, as a path parallel to the existing one.

Step 2 of the wiring in `research/PRIMITIVES_AUDIT.md`. This deliberately does
NOT touch `_assign_roles_neural`. It exists to be MEASURED against it
(`research/experiments/nemo_vs_symbolic.py`); only if it matches the current
0.930 +/-0.043 role accuracy is replacing anything worth discussing.

THE LOOP, from `parser.py::parseHelper`:

    for word in sentence:
        activate the word in its CORE area and fix it
        apply PRE_RULES                     # mutate fiber/area state
        prepare_targets(...)                # fix unreached areas, clear reached ones
        proj = state.project_map(...)       # targets are DERIVED, never named
        project `project_rounds` times
        apply POST_RULES                    # advance the slot

HOW THE ROLE IS READ OFF
------------------------
It is not read off. It falls out. When the word's core area is projected, the
derived map contains at most one role target -- the "war of fibers" invariant
guarantees it -- and THAT is the role assignment. There is nothing to score,
rank, or compare. The contrast with `_assign_roles_neural`, which computes a
margin per candidate role and then blocks used roles with a Python set, is the
whole point of the exercise.

WHAT IS STILL NOT NEURAL HERE, stated plainly
---------------------------------------------
* The word ORDER is a learned discrete parameter selecting a gating program
  (SVO only so far), not an emergent property.
* Word CATEGORY comes from `classify_word`, which is this repo's learned
  contribution -- but it is consulted as a Python value to pick the program.
* TRANSITIVITY is inferred distributionally by `infer_transitive_verbs` below
  (a verb is transitive if the corpus ever gives it a patient). That is
  learned from the same annotation training already uses, not hand-listed.

So this replaces the symbolic ROLE SELECTION with gating. It does not claim to
make the whole pipeline neural, and the audit doc should not be read as saying
otherwise.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from neural_assemblies.core.inhibition import (
    InhibitionState, apply_rule, prepare_targets,
)

from .core.areas import ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT
from .nemo_rules import (
    CONTENT_CATEGORIES, all_areas, category_addresses_open_slot,
    initial_open_areas, program_for_category,
    sequential_initial_open_areas, sequential_verb_program, slot_sequence,
)

_ROLE_LABEL = {
    ROLE_AGENT: "AGENT",
    ROLE_ACTION: "ACTION",
    ROLE_PATIENT: "PATIENT",
}
_ROLE_AREAS = tuple(_ROLE_LABEL)


def infer_transitive_verbs(sentences: Iterable) -> Set[str]:
    """Verbs the corpus ever gives a patient. Distributional, not hand-listed.

    Uses the same role annotation `train_roles` consumes, so it adds no
    supervision the parser did not already have.
    """
    transitive: Set[str] = set()
    for sent in sentences:
        roles = list(getattr(sent, "roles", ()) or ())
        words = list(getattr(sent, "words", ()) or ())
        if "patient" not in roles:
            continue
        for w, r in zip(words, roles):
            if r == "action":
                transitive.add(w)
    return transitive


class NemoParser:
    """Gating-based role assignment. One instance per sentence (state is stateful).

    `use_lexical` selects between two models, and the contrast is the point:

    * False (default) -- PURE GATING. Measured 1.000 on reversible items and
      0.000 on irreversible ones at n=1000/3000/10000 with zero variance: a
      perfect positional mechanism with no way to let what the corpus taught
      about a word override where the word order puts it.
    * True -- GATING AS THE STRUCTURAL PRIOR, with the lexical margin able to
      override it. This is the minimal composition, not a new mechanism:
      `_assign_roles_neural` already blends a structural prior with a lexical
      margin, and the only change is WHERE THE PRIOR COMES FROM -- the open
      fiber instead of `self.word_order_type`, a stored Python string.

    That swap is the whole point of the exercise. It is what would make the
    lesion study's positional arm a genuine structural lesion (close a fiber)
    rather than a symbolic one (corrupt an attribute).
    """

    def __init__(self, parser, *, word_order_type: str = "SVO",
                 transitive_verbs: Optional[Set[str]] = None,
                 use_lexical: bool = False,
                 competitive: bool = False,
                 sequential: bool = False,
                 head_start: int = 0,
                 lexical_weight: float = 1.0,
                 rounds: Optional[int] = None) -> None:
        self.parser = parser
        self.brain = parser.brain
        self.transitive_verbs = transitive_verbs or set()
        self.use_lexical = use_lexical
        # Competitive mode: leave both role slots open and let MUTUAL
        # INHIBITION arbitrate on learned weight. MI only fires when >=2 group
        # areas are targets of the SAME project() call, which the derived map
        # now produces because both fibers stay open.
        self.competitive = competitive
        # Rounds ROLE_PATIENT's fiber stays shut while ROLE_AGENT accumulates.
        # 0 = pure competition (lexical statistics decide, no positional
        # information at all); large = the SVO template, since PATIENT never
        # opens in time to compete. Only meaningful with `competitive`.
        self.head_start = head_start
        if competitive:
            present = [a for a in (ROLE_AGENT, ROLE_PATIENT)
                       if a in parser.brain.areas]
            if len(present) > 1:
                parser.brain.add_mutual_inhibition(present)
        self.lexical_weight = lexical_weight
        self.rounds = rounds if rounds is not None else max(
            1, int(getattr(parser, "rounds", 5)) // 2)
        self._areas = [a for a in all_areas() if a in self.brain.areas]
        # SEQUENTIAL gating generalizes past SVO. `initial_open_areas` raises
        # for any other order, because the SVO rule table hard-codes the verb
        # as the word that advances the slot. Here the order is a SEQUENCE of
        # slots, one open at a time, and each CONTENT word takes the next -- so
        # all six orders run the same machinery. Opt in explicitly for SVO so
        # the existing measured path is untouched by default.
        self.word_order_type = word_order_type or "SVO"
        self.sequential = bool(sequential) or (
            not competitive and self.word_order_type != "SVO")
        self._sequence = (slot_sequence(self.word_order_type)
                          if self.sequential else ())
        #: [(word, addressed_open_slot?)] from the last parse. None means no
        #: slot was open to address. Recorded live during parsing.
        self.last_mismatches: List[tuple] = []
        if competitive:
            from .nemo_rules import competitive_initial_open_areas
            open_areas = competitive_initial_open_areas()
        elif self.sequential:
            open_areas = sequential_initial_open_areas(self.word_order_type)
        else:
            open_areas = initial_open_areas(word_order_type)
        self.state = InhibitionState(
            self._areas, [a for a in open_areas if a in self.brain.areas])

    # ------------------------------------------------------------------
    def _open_slot(self) -> Optional[str]:
        """The single slot this order currently has open, if any.

        Sequential gating keeps exactly one open, which is what makes the
        readout unambiguous -- there is nothing to score or compare.
        """
        live = [s for s in self._sequence
                if s in self.brain.areas and self.state.area_open(s)]
        return live[0] if len(live) == 1 else None

    def _next_slot(self, current: str) -> Optional[str]:
        """The slot this order fills after `current`; None at the end."""
        seq = self._sequence
        if current not in seq:
            return None
        idx = seq.index(current) + 1
        return seq[idx] if idx < len(seq) else None

    # ------------------------------------------------------------------
    def parse(self, words: List[str]) -> Dict[str, Optional[str]]:
        """Return {word: role}. Roles are a CONSEQUENCE of which fibers opened."""
        from neural_assemblies.assembly_calculus.ops import project

        out: Dict[str, Optional[str]] = {}

        # A parse must not inherit training residue. Straight after training the
        # role areas still hold the LAST training sentence: measured on seed 42,
        # ROLE_AGENT matched `boy` at 1.0 and ROLE_PATIENT matched `girl` at 1.0.
        # The readout then found a perfect match for `girl` in the WRONG area
        # whenever `girl` was the subject -- exactly the deterministic 4-of-24
        # failures the substrate sweep found at every n (0.833, zero variance).
        #
        # The symbolic path is immune because `_score_role_binding` scores a
        # RELATIVE margin (own overlap minus the mean over other stored
        # fillers), which cancels a constant residue. An absolute overlap does
        # not, so the residue has to actually be gone.
        for area in _ROLE_AREAS:
            if area in self.brain.areas:
                self.brain.areas[area].unfix_assembly()
        self.brain.inhibit_areas([a for a in _ROLE_AREAS
                                  if a in self.brain.areas])

        for word in words:
            category = self._category(word)
            core = self.parser._word_core_area(word)
            if core is None or core not in self.brain.areas:
                out[word] = None
                continue
            program = program_for_category(
                category,
                transitive=word in self.transitive_verbs,
                core_area=core,
            )
            if self.competitive and category == "VERB" and                     word in self.transitive_verbs:
                from .nemo_rules import competitive_verb_program
                program = competitive_verb_program(core)
            elif self.sequential and category == "VERB":
                # The sequencer owns the advance, so the verb must NOT also
                # carry it -- `trans_verb_program`'s POST area rules would
                # double-step and skip a slot.
                program = sequential_verb_program(core)
            if program is None:
                out[word] = None
                continue

            # Activate the word in its core area, and hold it there while the
            # role areas settle -- the reference's activateWord + fix_assembly.
            phon = self.parser.stim_map.get(word)
            if phon is None:
                out[word] = None
                continue
            with self.brain.frozen():
                project(self.brain, phon, core, rounds=self.parser.rounds)
                self.brain.areas[core].fix_assembly()
                try:
                    for rule in program.pre:
                        apply_rule(self.state, rule)

                    prepare_targets(self.brain, self.state, lex_area=core)
                    proj = self.state.project_map(self.brain, lex_area=core)
                    # Captured BEFORE the word's own post rules or the
                    # sequencer run, so it is the slot this word arrived at.
                    open_slot = self._open_slot() if self.sequential else None
                    if self.sequential:
                        # The ELAN signal, recorded from the LIVE gating state
                        # rather than re-simulated afterwards. `slots` is the
                        # order's full sequence, not just the filler slots, so
                        # a noun meeting the ACTION slot counts as a mismatch --
                        # which is exactly what a wrong VERB POSITION produces.
                        self.last_mismatches.append((
                            word,
                            category_addresses_open_slot(
                                self.state, category,
                                transitive=word in self.transitive_verbs,
                                core_area=core, slots=self._sequence),
                        ))
                    if not self.competitive:
                        # In the SVO program two open slots is a MALFORMED rule
                        # set. In competitive mode offering two slots IS the
                        # design, so the invariant is deliberately relaxed
                        # rather than silently violated.
                        self.state.check_war_of_fibers(proj, core)

                    if (self.head_start and self.competitive
                            and category in ("NOUN", "PRON")
                            and ROLE_PATIENT in proj.get(core, ())):
                        # THE POSITIONAL PRIOR, PAID IN TIME RATHER THAN WEIGHT.
                        # Inhibition is binary, so a graded "subject slot is
                        # favoured at sentence start" cannot be expressed by
                        # opening or closing a slot -- that is the all-or-nothing
                        # SVO program. What IS gradable in assembly calculus is
                        # WHEN a fiber opens: hold core->ROLE_PATIENT shut for
                        # `head_start` rounds so the agent slot accumulates drive
                        # first, then open it and let MI compare. A sentence-
                        # initial subject expectation becomes anticipatory
                        # processing, and its strength is a number of rounds
                        # rather than a tuned coefficient.
                        #
                        # `prepare_targets` has ALREADY run with both fibers
                        # open, so both role areas are cleared. Do not call it
                        # again after opening PATIENT -- it would wipe the very
                        # head start this is trying to give.
                        #
                        # Channel 2: channel 0 belongs to the word programs and
                        # channel 1 holds the winner-protection, so the head
                        # start needs its own or releasing it would stamp on
                        # theirs.
                        #
                        # Close the AREA, not just the core->PATIENT fiber.
                        # MEASURED: with only that fiber shut, ROLE_PATIENT was
                        # still reaching `activation_scores` from its other
                        # sources, so MI fired DURING the head start, compared a
                        # live AGENT against an empty PATIENT, and killed
                        # PATIENT destructively (w=0) before it could ever
                        # compete. Every word then read P=0.0 at the real
                        # decision point and AGENT won unconditionally -- the
                        # head start PRE-EMPTED the competition instead of
                        # biasing it. A closed area is absent from the derived
                        # map entirely, which is the only way to keep a
                        # destructive WTA from running early. It is also how the
                        # SVO program expresses slot state (`area_rule`).
                        self.state.inhibit_area(ROLE_PATIENT, 2)
                        head_proj = self.state.project_map(
                            self.brain, lex_area=core)
                        for _ in range(self.head_start):
                            if head_proj:
                                self.brain.project({}, head_proj)
                        self.state.disinhibit_area(ROLE_PATIENT, 2)
                        proj = self.state.project_map(self.brain, lex_area=core)
                        for _ in range(max(0, self.rounds - self.head_start)):
                            if proj:
                                self.brain.project({}, proj)
                    elif proj:
                        for _ in range(self.rounds):
                            self.brain.project({}, proj)
                    # Readout is deliberately NOT `self._role_from(proj, core)`.
                    # That reads the GATE, so the answer would depend only on
                    # which fiber the rules opened and not at all on the brain
                    # -- a symbolic readout that makes canonical SVO items
                    # trivially correct and measures nothing. The reference
                    # reads out neurally (`getWord`: match an area's actual
                    # winners against stored assemblies), so this does too.
                    if self.competitive and category in ("NOUN", "PRON"):
                        # MI leaves exactly one area alive; that survivor IS the
                        # assignment. No overlap matching, no scoring.
                        # OPEN, not merely non-empty. Closing a slot does not
                        # clear it, so a slot won by an EARLIER noun still has
                        # winners; counting those made `alive` len 2 and the
                        # second noun read out None on 8 of 12 reversible items.
                        # A closed slot is out of the competition by definition
                        # -- that is what closing it means.
                        alive = [a for a in _ROLE_AREAS
                                 if a in self.brain.areas
                                 and self.state.area_open(a)
                                 and len(self.brain.areas[a].winners) > 0
                                 and a != ROLE_ACTION]
                        out[word] = (_ROLE_LABEL[alive[0]]
                                     if len(alive) == 1 else None)
                        # CLOSE THE SLOT THAT WON. Without this the next noun
                        # overwrites the binding: `prepare_targets` CLEARS every
                        # area the core still reaches, so an open ROLE_AGENT
                        # holding `ball` is wiped when `dog` arrives. That is
                        # what `INHIBIT ROLE_AGENT` does in the SVO program --
                        # it is not only a word-order rule, it is what protects
                        # a completed binding. Closing the WINNER instead of a
                        # fixed slot keeps that protection while letting
                        # lexical preference choose which slot is taken.
                        #
                        # INDEX 1, NOT 0, and this is what index channels are
                        # for. Word programs operate on channel 0, and the
                        # verb's POST does `DISINHIBIT ROLE_PATIENT, 0` -- on a
                        # shared channel that REOPENS a slot a noun just won,
                        # and `prepare_targets` then wipes the binding. Holding
                        # the protection on its own channel means the area stays
                        # closed until the binder releases it, which a boolean
                        # inhibited-flag could not express.
                        if len(alive) == 1:
                            self.state.inhibit_area(alive[0], 1)
                    elif self.competitive:
                        # Only NOUNS compete for role slots. Letting the verb
                        # run the same readout made it report PATIENT -- it was
                        # seeing the previous noun's surviving binding -- and it
                        # would then have closed that slot spuriously.
                        out[word] = self._role_from(proj, core)
                    elif self.sequential:
                        # Exactly one slot is open, so the slot the word landed
                        # in IS the open one -- no overlap matching needed, and
                        # no ambiguity to resolve. Then ADVANCE: close what was
                        # filled, open the next slot of this order. Only
                        # CONTENT words consume a slot; a determiner that
                        # advanced would put "the dog" in the object slot.
                        out[word] = _ROLE_LABEL.get(open_slot) if open_slot else None
                        if category in CONTENT_CATEGORIES and open_slot:
                            self.state.inhibit_area(open_slot, 0)
                            nxt = self._next_slot(open_slot)
                            if nxt is not None:
                                self.state.disinhibit_area(nxt, 0)
                    elif self.use_lexical:
                        out[word] = self._blend(
                            word, core, self._role_from(proj, core))
                    else:
                        out[word] = self._role_by_readout(word)
                finally:
                    self.brain.areas[core].unfix_assembly()
                    for rule in program.post:
                        apply_rule(self.state, rule)
        return out

    # ------------------------------------------------------------------
    def _blend(self, word: str, core: str, gated: Optional[str]
               ) -> Optional[str]:
        """Structural prior from the OPEN FIBER, lexical margin able to override.

        Deliberately reuses the parser's own `_role_binding_margin`, so this is
        not a second lexical mechanism -- it is the existing one, reading a
        prior that now comes from the gating instead of from a stored word-order
        string. Keeping the blend identical is what makes the comparison a test
        of the PRIOR'S SOURCE rather than of two different scoring schemes.
        """
        margin_fn = getattr(self.parser, "_role_binding_margin", None)
        if margin_fn is None:
            return gated
        scores = {}
        for area, label in _ROLE_LABEL.items():
            if area not in self.brain.areas or label == "ACTION":
                continue
            try:
                lex = float(margin_fn(word, core, area))
            except Exception:
                lex = 0.0
            prior = 1.0 if label == gated else 0.0
            scores[label] = prior + self.lexical_weight * lex
        if not scores:
            return gated
        best = max(scores, key=scores.get)
        return best if scores[best] > 0.0 else gated

    def _role_by_readout(self, word: str, min_overlap: float = 0.25
                         ) -> Optional[str]:
        """Which role area actually holds this word, by assembly overlap.

        The analogue of the reference's `getWord`, which matches an area's
        WINNERS against stored assemblies rather than consulting the control
        state. The distinction is not cosmetic: reading the gate instead makes
        every canonical SVO item correct by construction, because the gate was
        configured from the word order.

        Returns None when no role area holds the word above threshold -- a
        failure to bind is reported as a failure, not silently filled in from
        the rules.
        """
        from neural_assemblies.assembly_calculus.assembly import (
            overlap as assembly_overlap,
        )
        from neural_assemblies.assembly_calculus.ops import _snap

        best_role, best = None, min_overlap
        for area, label in _ROLE_LABEL.items():
            if area not in self.brain.areas:
                continue
            w = self.brain.areas[area].winners
            if w is None or len(w) == 0:
                continue
            stored = (getattr(self.parser, "role_lexicons", {}) or {}
                      ).get(area, {}).get(word)
            if stored is None:
                continue
            score = float(assembly_overlap(_snap(self.brain, area), stored))
            if score > best:
                best, best_role = score, label
        return best_role

    def _role_from(self, proj: Dict[str, List[str]], core: str) -> Optional[str]:
        """The single role target the gating left open, if any.

        `check_war_of_fibers` has already guaranteed at most one, so this is a
        lookup rather than a decision.
        """
        for target in proj.get(core, ()):
            if target in _ROLE_LABEL:
                return _ROLE_LABEL[target]
        return None

    def _category(self, word: str) -> str:
        cached = getattr(self.parser, "_category_cache", {}).get(word)
        if cached:
            return cached
        classify = getattr(self.parser, "classify_word", None)
        return classify(word) if classify else "NOUN"
