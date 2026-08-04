"""Fiber- and area-level inhibition, and projection-map derivation (NEMO).

This is a faithful port of the control layer in
``.reference/dmitropolsky-assemblies/parser.py`` (``ParserBrain``). It is
deliberately NOT a ``Brain`` method: in NEMO inhibition is not a neural
operation. ``grep -c inhibit`` on the reference ``brain.py`` returns 0.
Inhibition is a state machine that decides WHICH PROJECTIONS HAPPEN, and the
brain below it only ever executes a projection map it is handed.

That distinction is the whole point, and getting it wrong is what
``research/PRIMITIVES_AUDIT.md`` documents: this repo previously had only
``Brain.add_mutual_inhibition``, a winner-take-all applied AFTER projecting,
which is post-hoc suppression rather than gating. Gating needs no cross-area
comparison -- exclusivity holds because only one fiber was ever open -- whereas
the WTA needs a reliable comparison between areas, which is measurably not
available on this substrate (a 7x pre-k-WTA separation collapses to a ~7%
margin).

THE TWO STATE MAPS
------------------
``fiber_states[src][dst]`` and ``area_states[area]`` each hold a SET OF
INDICES. Open means the set is EMPTY. ``inhibit`` adds an index, ``disinhibit``
discards one.

The index is not decoration. It gives independent rules independent channels on
the same fiber, so a fiber reopens only when every holder has released it. The
reference relies on this: ``generic_trans_verb`` does
``AreaRule(DISINHIBIT, ADVERB, 1)`` while ``generic_adverb`` does
``AreaRule(INHIBIT, ADVERB, 1)`` -- index 1 is a separate channel from the
index-0 default, and a boolean flag cannot express it. That is precisely why
the bookkeeping ended up in Python here.

FIBER RULES ARE SYMMETRIC
-------------------------
``applyFiberRule`` writes BOTH directions. Inhibiting LEX->SUBJ also inhibits
SUBJ->LEX. ``project_map`` then reads the state directionally. Preserved here
because rule sets are written assuming it.

INITIAL STATE
-------------
Everything starts inhibited at index 0; only ``initial_areas`` are opened. So a
fresh state machine projects nothing at all, which is the correct default -- a
fiber that nobody has opened must not carry signal.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set

INHIBIT = "INHIBIT"
DISINHIBIT = "DISINHIBIT"


class InhibitionState:
    """Fiber/area inhibition state plus the projection map it implies."""

    def __init__(self, areas: Iterable[str],
                 initial_areas: Optional[Iterable[str]] = None) -> None:
        self.areas: List[str] = list(areas)
        self.fiber_states: Dict[str, Dict[str, Set[int]]] = {
            a: defaultdict(set) for a in self.areas
        }
        self.area_states: Dict[str, Set[int]] = {a: set() for a in self.areas}
        # Everything closed at index 0; only the initial areas are opened.
        for src in self.areas:
            for dst in self.areas:
                self.fiber_states[src][dst].add(0)
        for a in self.areas:
            self.area_states[a].add(0)
        for a in (initial_areas or ()):
            if a in self.area_states:
                self.area_states[a].discard(0)

    @classmethod
    def all_open(cls, areas: Iterable[str]) -> "InhibitionState":
        """Every area AND every fiber open. The only safe default for a Brain.

        The normal constructor closes everything, which is right for a parser
        whose rule set opens exactly what each word needs. It is wrong as a
        Brain default: a Brain that gated projections shut by default would
        silently stop projecting, and on this substrate "nothing happened"
        still returns k winners, so it would look like a result rather than an
        error ([[silent-no-op-dead-fibers]]).
        """
        state = cls(areas, initial_areas=areas)
        for src in state.areas:
            for dst in state.areas:
                state.fiber_states[src][dst].clear()
        return state

    def ensure_areas(self, areas: Iterable[str], *, open_new: bool) -> None:
        """Register areas created after this state was built.

        A Brain can ``add_area`` at any time, and an area the state has never
        heard of would raise on lookup. ``open_new`` must be stated by the
        caller rather than defaulted: for a Brain the answer is True (a new
        area is not gated until someone gates it) and for a parser rule set it
        is False (nothing is open until a rule opens it), and picking one
        silently would be wrong half the time.
        """
        for a in areas:
            if a in self.area_states:
                continue
            self.areas.append(a)
            self.fiber_states.setdefault(a, defaultdict(set))
            self.area_states[a] = set() if open_new else {0}
            for other in self.areas:
                if open_new:
                    self.fiber_states[a][other].clear()
                    self.fiber_states[other][a].clear()
                else:
                    self.fiber_states[a][other].add(0)
                    self.fiber_states[other][a].add(0)

    def any_closed(self) -> bool:
        """True if anything at all is inhibited. Lets callers skip the filter."""
        if any(self.area_states.values()):
            return True
        return any(idx for dsts in self.fiber_states.values()
                   for idx in dsts.values())

    # ---- state mutation ------------------------------------------------
    def inhibit_fiber(self, a1: str, a2: str, index: int = 0) -> None:
        """Close fiber a1<->a2 on `index`. Symmetric, matching the reference."""
        self.fiber_states[a1][a2].add(index)
        self.fiber_states[a2][a1].add(index)

    def disinhibit_fiber(self, a1: str, a2: str, index: int = 0) -> None:
        """Release this holder's claim on a1<->a2. Open only when ALL released."""
        self.fiber_states[a1][a2].discard(index)
        self.fiber_states[a2][a1].discard(index)

    def inhibit_area(self, area: str, index: int = 0) -> None:
        self.area_states[area].add(index)

    def disinhibit_area(self, area: str, index: int = 0) -> None:
        self.area_states[area].discard(index)

    # ---- queries -------------------------------------------------------
    def fiber_open(self, a1: str, a2: str) -> bool:
        return not self.fiber_states[a1][a2]

    def area_open(self, area: str) -> bool:
        return not self.area_states[area]

    # ---- the composition ----------------------------------------------
    def project_map(self, brain, lex_area: Optional[str] = None
                    ) -> Dict[str, List[str]]:
        """Derive what projects where. Targets are NEVER named by the caller.

        Mirrors ``ParserBrain.getProjectMap``: for each ordered pair, project
        ``src -> dst`` when both areas are open, the fiber between them is open,
        and the source actually has winners to send. An area that is itself open
        and active also projects into itself (the reference's recurrence).

        `lex_area` reproduces the reference's one special case, skipping
        LEX -> LEX.
        """
        proj: Dict[str, Set[str]] = defaultdict(set)
        for a1 in self.areas:
            if not self.area_open(a1):
                continue
            for a2 in self.areas:
                if lex_area is not None and a1 == lex_area and a2 == lex_area:
                    continue
                if not self.area_open(a2):
                    continue
                if not self.fiber_open(a1, a2):
                    continue
                if self._has_winners(brain, a1):
                    proj[a1].add(a2)
                if self._has_winners(brain, a2):
                    proj[a2].add(a2)
        return {src: sorted(dsts) for src, dsts in proj.items()}

    def check_war_of_fibers(self, proj_map: Dict[str, List[str]],
                            lex_area: str) -> None:
        """The reference's "war of fibers" invariant. Do not drop it.

        `EnglishParserBrain.getProjectMap` raises when the lexical area projects
        into more than TWO areas, and its comment -- ``# because LEX->LEX`` --
        records why the bound is 2 rather than 1: the self-projection is
        expected, so 2 means "itself plus one real target".

        This is the guard for exactly the failure this primitive exists to
        prevent. If LEX reaches two role slots at once, the word is being
        offered to several roles simultaneously and the binding is ambiguous --
        which is the situation a winner-take-all would then have to arbitrate,
        badly. Under correct rules it never arises, so raising is right:
        it catches a malformed rule set at the point of the mistake.
        """
        targets = proj_map.get(lex_area)
        if targets is not None and len(targets) > 2:
            raise ValueError(
                f"war of fibers: {lex_area} projects into {targets}. Rules "
                f"opened more than one role slot at once, so the binding is "
                f"ambiguous. (Two is the legal maximum: {lex_area}->{lex_area} "
                f"plus a single target.)"
            )

    @staticmethod
    def _has_winners(brain, area: str) -> bool:
        a = brain.areas.get(area) if hasattr(brain.areas, "get") else None
        if a is None:
            return False
        w = getattr(a, "winners", None)
        return w is not None and len(w) > 0


class Rule:
    """A single state mutation. Mirrors the reference's rule namedtuples."""

    __slots__ = ("kind", "action", "a1", "a2", "index")

    def __init__(self, kind: str, action: str, a1: str,
                 a2: Optional[str] = None, index: int = 0) -> None:
        self.kind, self.action = kind, action
        self.a1, self.a2, self.index = a1, a2, index

    def __repr__(self) -> str:
        tgt = f"{self.a1}<->{self.a2}" if self.kind == "fiber" else self.a1
        return f"{self.action} {self.kind} {tgt} @{self.index}"


def fiber_rule(action: str, a1: str, a2: str, index: int = 0) -> Rule:
    return Rule("fiber", action, a1, a2, index)


def area_rule(action: str, area: str, index: int = 0) -> Rule:
    return Rule("area", action, area, None, index)


def apply_rule(state: InhibitionState, rule: Rule) -> None:
    if rule.kind == "fiber":
        assert rule.a2 is not None
        if rule.action == INHIBIT:
            state.inhibit_fiber(rule.a1, rule.a2, rule.index)
        elif rule.action == DISINHIBIT:
            state.disinhibit_fiber(rule.a1, rule.a2, rule.index)
    elif rule.kind == "area":
        if rule.action == INHIBIT:
            state.inhibit_area(rule.a1, rule.index)
        elif rule.action == DISINHIBIT:
            state.disinhibit_area(rule.a1, rule.index)


def prepare_targets(brain, state: InhibitionState, lex_area: str) -> None:
    """Fix areas the word does NOT reach; erase the ones it does.

    Easy to omit, and the parse is wrong without it. Before projecting a new
    word the reference walks the derived map: an area the lexical area does NOT
    project into is ``fix_assembly()``-ed, so whatever is already bound there is
    preserved while the new word settles; an area it DOES project into is
    unfixed and its winners CLEARED, so the incoming word is not blended with
    the previous occupant of that slot.

    This is what makes role binding sequential and non-destructive: "dogs" stays
    put in SUBJ while "chase" binds into VERB, because LEX no longer reaches
    SUBJ once the noun's POST_RULES closed that fiber.
    """
    proj = state.project_map(brain, lex_area=lex_area)
    from_lex = set(proj.get(lex_area, ()))
    for area in list(proj.keys()):
        if area not in from_lex:
            brain.areas[area].fix_assembly()
        elif area != lex_area:
            brain.areas[area].unfix_assembly()
            brain.areas[area].winners = brain.areas[area].winners[:0]
