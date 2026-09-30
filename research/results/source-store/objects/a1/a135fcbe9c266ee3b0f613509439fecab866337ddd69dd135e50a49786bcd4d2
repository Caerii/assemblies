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
    n_state: int | None = None,
    beta: float = 0.1,
    organ_p: float | None = None,
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
        n_state=n_state,
        beta=beta,
        organ_p=organ_p,
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
    """Simulate digit string; ``10`` denotes the end symbol.

    Every state after the first is read out of the state assembly. This used
    to advance by table lookup, which made the accept/reject verdict below
    independent of the network -- it was returned correctly by an untrained
    brain and by beta=0.
    """
    symbols = [END_SYMBOL if d == 10 else str(d) for d in digits]
    trajectory = ["0"] + fsm.run(symbols, start_state="0")
    return trajectory[-1], trajectory


def run_mod3_fsm_demo(
    *,
    seed: int = 42,
    n: int = 5000,
    k: int = 70,
    n_state: int = 500,
    p: float = 0.2,
    beta: float = 0.1,
    presentations: int = 15,
    positive_sequence: Iterable[int] = (3, 0, 4, 7, 1, 10),
    negative_sequence: Iterable[int] = (6, 7, 3, 10),
) -> Mod3FsmResult:
    """Train mod-3 FSM and evaluate reference positive/negative digit strings.

    Defaults are the reference's regime, not the ones this demo shipped with.
    It ran at n=2000, k=40, p=0.05, giving the arc kp = 2 against a floor of
    3 ln 2000 = 22.8 -- eleven times below what every theorem in the sequences
    paper requires. That was survivable only because the readout was a table
    lookup and the dynamics could not affect the answer. At the reference's
    n=5000, k=70, p=0.2 the arc sits at kp = 28 against 25.6 and is a clean
    conjunction (`research/notes/sequence/the_arc_is_a_conjunction_and_the_state_drifts.md`).

    `norm_init=False` pins the reference substrate; neither area here has a
    self fiber, which is the only thing norm_init exists to stabilise.

    THIS DOES NOT RELIABLY DECIDE YET. A1 measured 5/10 seeds at these
    parameters, and only 4/10 with a fully correct trajectory: single
    transitions are perfect (330/330) but the state assembly drifts along a
    sequence. At seed 42 the trajectory tracks ground truth through all five
    digit steps and misses only the final `end` transition.
    """
    from neural_assemblies.core.brain import Brain

    pos = list(positive_sequence)
    neg = list(negative_sequence)
    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse",
                  norm_init=False)
    fsm = build_mod3_fsm(brain, n=n, k=k, n_state=n_state, beta=beta)
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
            "presentations": presentations,
            "positive_sequence": pos,
            "negative_sequence": neg,
        },
    )
