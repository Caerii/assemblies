"""
Transition objects for assembly-based state machines.

This module provides a typed transition contract that sits between
learned or hand-authored transition sources and the current FSM/PFA
implementations. The initial goal is validation and normalization, not
to replace the existing runtime mechanics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import DefaultDict, Iterable, Iterator


@dataclass(frozen=True)
class Transition:
    """A normalized state transition."""

    from_state: str
    symbol: str
    to_state: str
    probability: float = 1.0

    def __post_init__(self):
        if not 0.0 < self.probability <= 1.0:
            raise ValueError("Transition probability must be in (0, 1].")

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
        if len(value) == 3:
            from_state, symbol, to_state = value
            return cls(from_state, symbol, to_state)
        if len(value) == 4:
            from_state, symbol, to_state, probability = value
            return cls(from_state, symbol, to_state, float(probability))
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
    """Normalized transition storage shared by FSM and PFA helpers."""

    def __init__(self, transitions: Iterable[TransitionLike]):
        self._transitions = tuple(normalize_transitions(transitions))
        self._by_key: DefaultDict[tuple[str, str], list[Transition]] = defaultdict(list)

        for transition in self._transitions:
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

    def validate_probability_mass(self, tol: float = 1e-9) -> "TransitionMap":
        """Validate that each transition key has a coherent probability mass."""
        for key, transitions in self._by_key.items():
            total = sum(transition.probability for transition in transitions)
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
