"""
Infer FSM structure from observed state/symbol traces (sequences paper protocol).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from neural_assemblies.assembly_calculus.fsm import FSMNetwork


TransitionTrace = Tuple[str, str, str]


def learn_fsm_from_sequences(
    traces: Sequence[TransitionTrace],
) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
    """Build a deterministic transition table from observations.

    Args:
        traces: ``(from_state, symbol, to_state)`` tuples.

    Returns:
        (states, symbols, transitions) suitable for ``FSMNetwork``.
    """
    table: Dict[Tuple[str, str], str] = {}
    for from_state, symbol, to_state in traces:
        key = (from_state, symbol)
        if key in table and table[key] != to_state:
            raise ValueError(
                f"Conflicting transitions for {key}: "
                f"{table[key]!r} vs {to_state!r}"
            )
        table[key] = to_state

    states = sorted({s for fr, _, to in traces for s in (fr, to)})
    symbols = sorted({sym for _, sym, _ in traces})
    transitions = [
        (fr, sym, table[(fr, sym)])
        for fr, sym in sorted(table)
    ]
    return states, symbols, transitions


def build_fsm_from_traces(
    brain,
    traces: Sequence[TransitionTrace],
    initial_state: str,
    **fsm_kwargs,
) -> FSMNetwork:
    """Infer and instantiate an ``FSMNetwork`` from transition traces."""
    states, symbols, transitions = learn_fsm_from_sequences(traces)
    if initial_state not in states:
        raise ValueError(f"initial_state {initial_state!r} not in {states}")
    return FSMNetwork(
        brain, states, symbols, transitions, initial_state, **fsm_kwargs
    )
