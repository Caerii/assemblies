"""
Mod-3 digit-sum FSM (dabagia.org/nemo/sequences/ fsm_0modthree demo).

Uses ``NemoArcFSM`` with 3 residue states plus accept/reject after the
end symbol (digit 10 in the reference notebook).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .nemo_fsm import NemoArcFSM

RESIDUE_STATES = ("0", "1", "2")
TERMINAL_STATES = ("accept", "reject")
ALL_STATES = RESIDUE_STATES + TERMINAL_STATES
DIGIT_SYMBOLS = tuple(str(d) for d in range(10))
END_SYMBOL = "end"
ALL_SYMBOLS = DIGIT_SYMBOLS + (END_SYMBOL,)


def mod3_transition_table() -> List[Tuple[str, str, str]]:
    """Reference ``nemo-demo.ipynb`` transitions as ``(from_state, symbol, to_state)``."""
    transitions: List[Tuple[str, str, str]] = []
    for mod in range(3):
        for digit in range(10):
            transitions.append((str(mod), str(digit), str((mod + digit) % 3)))
    transitions.extend([
        ("0", END_SYMBOL, "accept"),
        ("1", END_SYMBOL, "reject"),
        ("2", END_SYMBOL, "reject"),
    ])
    return transitions


@dataclass
class Mod3FsmResult:
    final_state: str
    trajectory: List[str]
    positive_accepted: bool
    negative_rejected: bool
    parameters: dict


def build_mod3_fsm(
    brain,
    *,
    n: int = 2000,
    k: int = 40,
    beta: float = 0.1,
    rounds: int = 6,
    refracted_strength: float = 0.1,
    prefix: str = "_mod3",
) -> NemoArcFSM:
    return NemoArcFSM(
        brain,
        states=list(ALL_STATES),
        symbols=list(ALL_SYMBOLS),
        transitions=mod3_transition_table(),
        n=n,
        k=k,
        beta=beta,
        rounds=rounds,
        refracted_strength=refracted_strength,
        prefix=prefix,
    )


def train_mod3_fsm(
    fsm: NemoArcFSM,
    *,
    presentations: int = 15,
) -> None:
    fsm.train_from_list(
        [(sym, fr, to) for fr, sym, to in mod3_transition_table()],
        presentations=presentations,
    )


def run_digit_sequence(fsm: NemoArcFSM, digits: Sequence[int]) -> Tuple[str, List[str]]:
    """Simulate digit string; ``10`` denotes the end symbol."""
    state = "0"
    trajectory = [state]
    for d in digits:
        sym = END_SYMBOL if d == 10 else str(d)
        state = fsm.step_symbol(sym, state)
        trajectory.append(state)
    return state, trajectory


def run_mod3_fsm_demo(
    *,
    seed: int = 42,
    n: int = 2000,
    k: int = 40,
    beta: float = 0.1,
    rounds: int = 6,
    presentations: int = 15,
    positive_sequence: Iterable[int] = (3, 0, 4, 7, 1, 10),
    negative_sequence: Iterable[int] = (6, 7, 3, 10),
) -> Mod3FsmResult:
    """Train mod-3 FSM and evaluate reference positive/negative digit strings."""
    from neural_assemblies.core.brain import Brain

    pos = list(positive_sequence)
    neg = list(negative_sequence)
    brain = Brain(p=0.05, save_winners=True, seed=seed, engine="numpy_sparse")
    fsm = build_mod3_fsm(brain, n=n, k=k, beta=beta, rounds=rounds)
    train_mod3_fsm(fsm, presentations=presentations)

    pos_final, pos_traj = run_digit_sequence(fsm, pos)
    neg_final, neg_traj = run_digit_sequence(fsm, neg)

    return Mod3FsmResult(
        final_state=pos_final,
        trajectory=pos_traj,
        positive_accepted=pos_final == "accept",
        negative_rejected=neg_final == "reject",
        parameters={
            "seed": seed,
            "n": n,
            "k": k,
            "beta": beta,
            "rounds": rounds,
            "presentations": presentations,
            "positive_sequence": pos,
            "negative_sequence": neg,
        },
    )
