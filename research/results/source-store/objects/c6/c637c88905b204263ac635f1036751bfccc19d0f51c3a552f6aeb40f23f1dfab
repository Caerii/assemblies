"""Finite-group word problems as FSMs. [[SEQ-FSM]]

The word problem for a finite group G: given a sequence of generators, decide
which element their product is. As a machine this is an FSM whose states are
the elements of G, whose symbols are the generators, and whose transition is
right multiplication -- exactly the shape `NemoArcFSM` already takes, so no new
mechanism is needed to run one.

WHY THIS BENCHMARK. Barrington (1989): the word problem of a finite
NON-SOLVABLE group is NC1-complete. Merrill, Petty & Sabharwal
(arXiv:2404.08819) prove S4 and Mamba-style SSMs lie in L-uniform TC0, because
their recurrence is a prefix scan over constant matrix powers or over DIAGONAL
(scalar) products -- so under TC0 != NC1 they cannot solve one, and their
Figure 3 measures the consequence as depth growing with sequence length.

The groups here are chosen to make SOLVABILITY the variable and size a
constant:

    Z60       order  60   abelian               word problem in TC0
    A4 x Z5   order  60   solvable, non-abelian word problem in TC0
    A5        order  60   NON-SOLVABLE          NC1-complete
    S5        order 120   NON-SOLVABLE          NC1-complete

Three at identical order, so an arm that degrades on A5 but not on Z60 is
degrading on solvability rather than on the number of states it has to hold.

Elements are built by CLOSURE from the generators and the order is asserted
against the known value, rather than enumerated by a formula that could
silently produce a subgroup -- a generating set that fails to generate would
otherwise show up as an easier task reported under a harder name.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from numbers import Integral
from typing import Callable, Dict, Hashable, List, Sequence, Tuple


@dataclass(frozen=True)
class Group:
    """A finite group given by its elements, generators and product."""

    name: str
    elements: Tuple[Hashable, ...]
    generators: Tuple[Hashable, ...]
    compose: Callable[[Hashable, Hashable], Hashable]
    identity: Hashable
    solvable: bool

    @property
    def order(self) -> int:
        return len(self.elements)

    def label(self, element: Hashable) -> str:
        return str(self._index()[element])

    def _index(self) -> Dict[Hashable, int]:
        return {e: i for i, e in enumerate(self.elements)}


def closure(generators: Sequence[Hashable], compose, identity):
    """Every element reachable from *identity* by right-multiplying generators."""
    seen = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for g in generators:
            nxt = compose(current, g)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return tuple(sorted(seen, key=repr))


def _perm_compose(a, b):
    """``(a . b)(i) = a(b(i))``. One convention, used everywhere here."""
    return tuple(a[b[i]] for i in range(len(a)))


def _cycle(n: int, cycle: Sequence[int]):
    """The permutation of ``range(n)`` sending cycle[i] -> cycle[i+1]."""
    p = list(range(n))
    for i, x in enumerate(cycle):
        p[x] = cycle[(i + 1) % len(cycle)]
    return tuple(p)


def symmetric_group_5() -> Group:
    """S5 = <(0 1), (0 1 2 3 4)>. Order 120, non-solvable."""
    gens = (_cycle(5, [0, 1]), _cycle(5, [0, 1, 2, 3, 4]))
    ident = tuple(range(5))
    elements = closure(gens, _perm_compose, ident)
    assert len(elements) == 120, f"S5 closure gave {len(elements)}"
    return Group("S5", elements, gens, _perm_compose, ident, solvable=False)


def alternating_group_5() -> Group:
    """A5 = <(0 1 2), (0 1 2 3 4)>. Order 60, non-solvable -- the paper's arm."""
    gens = (_cycle(5, [0, 1, 2]), _cycle(5, [0, 1, 2, 3, 4]))
    ident = tuple(range(5))
    elements = closure(gens, _perm_compose, ident)
    assert len(elements) == 60, f"A5 closure gave {len(elements)}"
    return Group("A5", elements, gens, _perm_compose, ident, solvable=False)


def cyclic_group(order: int, generators: Sequence[int] = (1, 7)) -> Group:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-cyclic-group

    Additive residues modulo order, with a generating alphabet chosen explicitly.
    Reject a proper subgroup before constructing a benchmark under the wrong name.
    """
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError("cyclic group order must be a positive integer")
    order = int(order)
    generators = tuple(generators)
    if any(isinstance(g, bool) or not isinstance(g, Integral) for g in generators):
        raise ValueError("cyclic generators must be integer residues")
    gens = tuple(int(g) % order for g in generators)
    if gcd(order, *gens) != 1:
        raise ValueError("cyclic generators span a proper subgroup of the requested order")

    def compose(a, b):
        return (a + b) % order

    elements = closure(gens, compose, 0)
    if len(elements) != order:
        raise ValueError("cyclic closure does not match the requested order")
    return Group(f"Z{order}", elements, gens, compose, 0, solvable=True)


def cyclic_group_60() -> Group:
    """Z60 = <1, 7>, the order-60 abelian control."""
    return cyclic_group(60)


def cyclic_group_120() -> Group:
    """Z120 = <1, 7>, S5's size control (cliff-anatomy Addendum 4)."""
    return cyclic_group(120)


def a4_times_z5() -> Group:
    """A4 x Z5. Order 60, SOLVABLE but non-abelian -- the middle arm.

    Generators pair a generator of A4 with a step in Z5, so the projection onto
    each factor is onto; closure asserts the product actually has order 60
    rather than some proper subgroup.
    """
    a, b = _cycle(4, [0, 1, 2]), _cycle(4, [1, 2, 3])

    def compose(x, y):
        return (_perm_compose(x[0], y[0]), (x[1] + y[1]) % 5)

    gens = ((a, 1), (b, 0))
    ident = (tuple(range(4)), 0)
    elements = closure(gens, compose, ident)
    assert len(elements) == 60, f"A4xZ5 closure gave {len(elements)}"
    return Group("A4xZ5", elements, gens, compose, ident, solvable=True)


GROUPS = {
    "Z60": cyclic_group_60,
    "A4xZ5": a4_times_z5,
    "A5": alternating_group_5,
    "S5": symmetric_group_5,
    "Z120": cyclic_group_120,
}


def word_problem_fsm(group: Group):
    """``(states, symbols, transitions)`` for `NemoArcFSM`.

    Transitions are ``(from_state, symbol, to_state)`` with ``to = from . g``,
    matching `mod3_transition_table`'s ordering.
    """
    index = {e: str(i) for i, e in enumerate(group.elements)}
    states = [index[e] for e in group.elements]
    symbols = [f"g{j}" for j in range(len(group.generators))]
    transitions: List[Tuple[str, str, str]] = []
    for e in group.elements:
        for j, g in enumerate(group.generators):
            transitions.append((index[e], f"g{j}", index[group.compose(e, g)]))
    return states, symbols, transitions


def true_trajectory(group: Group, symbol_sequence: Sequence[str],
                    start: Hashable = None) -> List[str]:
    """Ground truth: the state after each symbol, as `NemoArcFSM` labels."""
    index = {e: str(i) for i, e in enumerate(group.elements)}
    current = group.identity if start is None else start
    out = []
    for s in symbol_sequence:
        current = group.compose(current, group.generators[int(s[1:])])
        out.append(index[current])
    return out
