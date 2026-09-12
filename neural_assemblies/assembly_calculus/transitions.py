"""
Transition objects for assembly-based state machines.

This module provides a typed transition contract that sits between
learned or hand-authored transition sources and the current FSM/PFA
implementations. The initial goal is validation and normalization, not
to replace the existing runtime mechanics.

It is deliberately PURE SYMBOLIC BOOKKEEPING -- no Brain, no assemblies, no
projections.  That separation is the point: FSMNetwork and PFANetwork both
need to know what transitions exist and whether they are well-formed, and
mixing that check into code that is also running neural dynamics makes it
impossible to tell a malformed automaton from a failed simulation.  Validation
here fails loudly at construction, before any neurons are involved.

The two validators encode the FSM/PFA distinction:

    deterministic_table()       one target per (state, symbol), probability 1.
        This is what FSMNetwork consumes; ambiguity is an error, not a
        coin-flip.
    validate_probability_mass() several targets per key summing to 1.  This is
        what PFANetwork consumes as target weights. The sum condition validates
        the symbolic table; it does not calibrate the neural branch selector or
        establish that observed transition frequencies match these weights.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from numbers import Real
from typing import DefaultDict, Iterable, Iterator, cast


@dataclass(frozen=True)
class Transition:
    """A normalized state transition."""

    from_state: str
    symbol: str
    to_state: str
    probability: float = 1.0

    def __post_init__(self):
        for name in ('from_state', 'symbol', 'to_state'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'Transition {name} must be a nonempty string')
        if (isinstance(self.probability, bool) or not isinstance(self.probability, Real)
                or not 0.0 < self.probability <= 1.0):
            raise ValueError("Transition probability must be a real number in (0, 1].")
        probability = float(self.probability)
        if probability == 0.0:
            raise ValueError('Transition probability underflows binary64')
        object.__setattr__(self, 'probability', probability)

    @property
    def key(self) -> tuple[str, str]:
        return (self.from_state, self.symbol)

    @classmethod
    def from_value(cls, value: "TransitionLike") -> "Transition":
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, bytes)):
            raise TypeError(
                "Transitions must be Transition objects or 3/4-tuples, "
                "not plain strings or bytes."
            )
        try:
            raw = tuple(cast(Iterable[object], value))
        except TypeError as exc:
            raise TypeError(
                "Transitions must be Transition objects or 3/4-tuples "
                "of (from_state, symbol, to_state[, probability])."
            ) from exc
        if len(raw) == 3:
            from_state, symbol, to_state = cast(tuple[str, str, str], raw)
            return cls(from_state, symbol, to_state)
        if len(raw) == 4:
            from_state, symbol, to_state, probability = cast(
                tuple[str, str, str, float], raw
            )
            return cls(from_state, symbol, to_state, probability)
        raise TypeError(
            "Transitions must be Transition objects or 3/4-tuples "
            "of (from_state, symbol, to_state[, probability])."
        )

    def as_tuple(self, include_probability: bool = False) -> tuple:
        if include_probability:
            return (
                self.from_state,
                self.symbol,
                self.to_state,
                self.probability,
            )
        return (self.from_state, self.symbol, self.to_state)


TransitionLike = Transition | tuple[str, str, str] | tuple[str, str, str, float]


def normalize_transitions(
    transitions: Iterable[TransitionLike],
) -> list[Transition]:
    """Convert raw transition values into validated Transition objects."""
    return [Transition.from_value(transition) for transition in transitions]


class TransitionMap:
    """Normalized transition storage shared by FSM and PFA helpers.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-transition-domain
    """

    def __init__(self, transitions: Iterable[TransitionLike]):
        self._transitions = tuple(normalize_transitions(transitions))
        self._by_key: DefaultDict[tuple[str, str], list[Transition]] = defaultdict(list)

        edges = set()
        for transition in self._transitions:
            edge = transition.as_tuple()
            if edge in edges:
                raise ValueError(f'Duplicate transition edge: {edge!r}')
            edges.add(edge)
            self._by_key[transition.key].append(transition)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._transitions)

    def keys(self) -> list[tuple[str, str]]:
        return list(self._by_key.keys())

    def targets(self, from_state: str, symbol: str) -> tuple[Transition, ...]:
        return tuple(self._by_key.get((from_state, symbol), ()))

    def as_tuples(self, include_probability: bool = False) -> list[tuple]:
        return [
            transition.as_tuple(include_probability=include_probability)
            for transition in self._transitions
        ]

    def validate_domain(self, states: Iterable[str], symbols: Iterable[str],
                        initial_state: str) -> "TransitionMap":
        """Reject invalid declarations and dangling edges before brain allocation."""
        domains = {}
        for name, values in (('states', states), ('symbols', symbols)):
            if isinstance(values, (str, bytes)):
                raise ValueError(f'{name} must be a collection of names')
            values = tuple(values)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f'{name} must contain nonempty strings')
            if len(values) != len(set(values)):
                raise ValueError(f'{name} must contain unique names')
            domains[name] = set(values)
        if not isinstance(initial_state, str) or initial_state not in domains['states']:
            raise ValueError('initial_state must belong to the declared states')
        for transition in self:
            if (transition.from_state not in domains['states']
                    or transition.to_state not in domains['states']
                    or transition.symbol not in domains['symbols']):
                raise ValueError(f'Transition outside declared domain: {transition.as_tuple()!r}')
        return self

    def branch_schedule(self, from_state: str, symbol: str) -> tuple[tuple[str, float], ...]:
        """Ordered conditional weights; the last target is the fallback (weight 1).

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-branch-schedule
        This symbolic factorization does not calibrate a neural seed mixture.
        Sum remaining positive weights directly, avoiding 1-minus-prefix cancellation.
        """
        targets = self.targets(from_state, symbol)
        if not targets:
            raise KeyError((from_state, symbol))
        return tuple((item.to_state, item.probability / math.fsum(
            target.probability for target in targets[index:]))
            for index, item in enumerate(targets))

    def validate_probability_mass(self, tol: float = 1e-9) -> "TransitionMap":
        """Validate that each transition key has a coherent probability mass."""
        if (isinstance(tol, bool) or not isinstance(tol, Real)
                or not math.isfinite(tol) or tol < 0):
            raise ValueError('probability tolerance must be finite and nonnegative')
        for key, transitions in self._by_key.items():
            total = math.fsum(transition.probability for transition in transitions)
            if len(transitions) == 1:
                if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tol):
                    raise ValueError(
                        f"Transition {key!r} must have probability 1.0 when it "
                        "has a single target."
                    )
                continue

            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tol):
                raise ValueError(
                    f"Transition probabilities for {key!r} must sum to 1.0, "
                    f"got {total:.6f}."
                )

        return self

    def deterministic_table(self) -> dict[tuple[str, str], str]:
        """Return a deterministic transition table or raise if ambiguous."""
        table: dict[tuple[str, str], str] = {}
        for key, transitions in self._by_key.items():
            if len(transitions) != 1:
                raise ValueError(
                    f"Transition {key!r} is not deterministic: "
                    f"{len(transitions)} targets supplied."
                )

            transition = transitions[0]
            if not math.isclose(transition.probability, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"Transition {key!r} is not deterministic: "
                    f"probability={transition.probability}."
                )

            table[key] = transition.to_state

        return table
