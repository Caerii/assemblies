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
    all_areas, initial_open_areas, program_for_category,
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
    """Gating-based role assignment. One instance per sentence (state is stateful)."""

    def __init__(self, parser, *, word_order_type: str = "SVO",
                 transitive_verbs: Optional[Set[str]] = None,
                 rounds: Optional[int] = None) -> None:
        self.parser = parser
        self.brain = parser.brain
        self.transitive_verbs = transitive_verbs or set()
        self.rounds = rounds if rounds is not None else max(
            1, int(getattr(parser, "rounds", 5)) // 2)
        self._areas = [a for a in all_areas() if a in self.brain.areas]
        self.state = InhibitionState(
            self._areas,
            [a for a in initial_open_areas(word_order_type) if a in self.brain.areas],
        )

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
                    self.state.check_war_of_fibers(proj, core)

                    if proj:
                        for _ in range(self.rounds):
                            self.brain.project({}, proj)
                    # Readout is deliberately NOT `self._role_from(proj, core)`.
                    # That reads the GATE, so the answer would depend only on
                    # which fiber the rules opened and not at all on the brain
                    # -- a symbolic readout that makes canonical SVO items
                    # trivially correct and measures nothing. The reference
                    # reads out neurally (`getWord`: match an area's actual
                    # winners against stored assemblies), so this does too.
                    out[word] = self._role_by_readout(word)
                finally:
                    self.brain.areas[core].unfix_assembly()
                    for rule in program.post:
                        apply_rule(self.state, rule)
        return out

    # ------------------------------------------------------------------
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
