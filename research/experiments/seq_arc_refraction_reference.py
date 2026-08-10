"""Characterization of the VENDORED reference FSM, ahead of building our own.

This measures `neural_assemblies/reference/nemo_numpy` only -- no `Brain` is
involved -- because before porting Theorem 4's three-area FSM we need to know
which of the reference's mechanisms are load-bearing and which are decoration.
It also audits the FSM programs we already ship.

QUESTIONS AND DECISION RULES (stated before the numbers were read):

  Q1 GOLDEN. Does `run_mod3_fsm_numpy` decide correctly? If it does not, there
     is no target and every downstream parity claim is vacuous.

  Q2 WHICH WAY DOES THE ARC COLLAPSE? Measure BOTH degeneracies:
        across-STATE  overlap at fixed symbol  -> high means A_sigma (state-blind)
        across-SYMBOL overlap at fixed state   -> high means A_q     (symbol-blind)
     Task #92 measured 0.90-0.99 on the first and concluded the conjunctive arc
     "has no operating point". Measuring only one direction cannot distinguish a
     conjunction from a collapse onto the OTHER conjunct, so both are required.

  Q3 IS REFRACTION LOAD-BEARING? Ablate it. If the task still passes without it,
     refraction is tiling cosmetics and #92's failure lies elsewhere.

  Q4 IS OUR ENGINES' RULE EQUIVALENT? The reference accumulates
        bias[winner] += raw_drive[winner] * plasticity        (drive-proportional)
     while every engine in this repo accumulates
        bias[winner] += refracted_strength                    (constant)
     Swap in the constant rule and sweep it. Constant refraction is an
     acceptable substitute ONLY if some fixed value works AND keeps working when
     the drive scale changes; Hebbian growth multiplies drive by (1+beta) per
     presentation, so a constant increment falls behind by construction unless
     the sweep says otherwise. Test by varying presentation count: if the
     winning constant MOVES, the parameter is unnormalized and the engine rule
     has to change rather than be tuned.

  Q5 FAKE-PERFECT AUDIT of the shipped programs. `NemoArcFSM.step_symbol`
     returns `self._table[(from_state, symbol)]`. Any arm that scores perfectly
     with no training refutes every FSM claim the repo currently makes.
"""
from __future__ import annotations

import sys
import types

import numpy as np

from neural_assemblies.reference.nemo_numpy.areas import FFArea, RefractedArea
from neural_assemblies.reference.nemo_numpy.fsm_network import (
    FSMNetwork, build_mod3_symbols_states, mod3_transition_list,
)

SEEDS = (42, 7, 13)
CAP = 70
N_SYMBOL, N_STATE, N_ARC = 1000, 500, 5000
DENSITY, PLASTICITY = 0.2, 0.1
LABELS = {0: "0", 1: "1", 2: "2", 3: "accept", 4: "reject"}
POSITIVE = (3, 0, 4, 7, 1, 10)      # digits sum to 15, 15 % 3 == 0 -> accept
NEGATIVE = (6, 7, 3, 10)            # digits sum to 16, 16 % 3 == 1 -> reject
CHANCE_OVERLAP = CAP / N_ARC


class ConstantRefractedArea(RefractedArea):
    """This repo's engine rule: a fixed increment per firing, blind to drive."""

    def __init__(self, *a, const_strength: float = 0.1, **kw):
        self.const_strength = const_strength
        super().__init__(*a, **kw)

    def update(self, new_activations):
        self.bias[new_activations] += self.const_strength
        FFArea.update(self, new_activations)


def train(seed, *, rule="proportional", strength=0.0, presentations=15):
    """Train the reference FSM under one refraction rule.

    `rule` is one of "proportional" (the reference), "constant" (this repo's
    engines), or "off" (no bias accumulation at all -- the ablation).
    """
    rng = np.random.default_rng(seed)
    fsm = FSMNetwork(N_SYMBOL, N_STATE, N_ARC, CAP, DENSITY, PLASTICITY, rng)
    if rule == "constant":
        fsm.arc_area = ConstantRefractedArea(
            [N_SYMBOL, N_STATE], N_ARC, CAP, DENSITY, PLASTICITY, rng,
            const_strength=strength,
        )
    elif rule == "off":
        # FFArea.update never touches `bias`, which RefractedArea then subtracts
        # as a vector of zeros -- refraction removed without touching anything else.
        fsm.arc_area.update = types.MethodType(FFArea.update, fsm.arc_area)

    symbols, states = build_mod3_symbols_states(CAP, rng)
    for _ in range(presentations):
        for from_state, symbol, to_state in mod3_transition_list():
            fsm.train(symbols[symbol], states[from_state], states[to_state])

    # Manipulation check: an arm that was supposed to change the bias must have.
    peak = float(fsm.arc_area.bias.max())
    if rule == "off":
        assert peak == 0.0, "ablation arm accumulated bias anyway"
    elif presentations > 0 and (rule == "proportional" or strength > 0.0):
        assert peak > 0.0, "refraction arm accumulated no bias"
    return fsm, symbols, states


def arc_assembly(fsm, symbol, state):
    """Arc winners for one (state, symbol), with no plasticity and no bias update."""
    fsm.inhibit()
    fsm.arc_area.forward([symbol, state], update=False)
    return fsm.arc_area.read()


def collapse_overlaps(fsm, symbols, states):
    """(across-state, across-symbol) mean pairwise arc overlap."""
    across_state = []
    for digit in range(10):
        a = [arc_assembly(fsm, symbols[digit], states[q]) for q in range(3)]
        across_state += [len(np.intersect1d(a[i], a[j])) / CAP
                         for i in range(3) for j in range(i + 1, 3)]
    across_symbol = []
    for q in range(3):
        a = [arc_assembly(fsm, symbols[d], states[q]) for d in range(10)]
        across_symbol += [len(np.intersect1d(a[i], a[j])) / CAP
                          for i in range(10) for j in range(i + 1, 10)]
    return float(np.mean(across_state)), float(np.mean(across_symbol))


def decide(fsm, symbols, states, sequence):
    """Run a digit string from state 0 and label the final state assembly."""
    fsm.inhibit()
    fsm.state_area.fire(states[0], update=False)
    for symbol in sequence:
        fsm.forward(symbols[symbol], update=False)
    read = fsm.read()
    best = max(range(len(states)),
               key=lambda i: len(np.intersect1d(read, states[i])) / max(len(read), 1))
    return LABELS[best]


def evaluate(rule="proportional", strength=0.0, presentations=15):
    across_state, across_symbol, passed = [], [], 0
    for seed in SEEDS:
        fsm, symbols, states = train(
            seed, rule=rule, strength=strength, presentations=presentations)
        a, b = collapse_overlaps(fsm, symbols, states)
        across_state.append(a)
        across_symbol.append(b)
        passed += int(decide(fsm, symbols, states, POSITIVE) == "accept"
                      and decide(fsm, symbols, states, NEGATIVE) == "reject")
    return float(np.mean(across_state)), float(np.mean(across_symbol)), passed


def q1_q3_ablation():
    print("=== Q1/Q2/Q3: golden, both collapse directions, refraction ablation ===")
    print(f"    chance overlap = k/n = {CHANCE_OVERLAP:.3f}; "
          f"task = seeds decided correctly of {len(SEEDS)}")
    for rule in ("proportional", "off"):
        a, b, t = evaluate(rule=rule)
        print(f"  refraction {rule:>12s}: across-state {a:.3f}  "
              f"across-symbol {b:.3f}  task {t}/{len(SEEDS)}")


def q4_constant_rule():
    print("\n=== Q4a: does the constant rule have an operating point at all? ===")
    for s in (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        a, b, t = evaluate(rule="constant", strength=s)
        print(f"  strength {s:7.1f}: across-state {a:.3f}  across-symbol {b:.3f}  "
              f"task {t}/{len(SEEDS)}")


def q4_scale_dependence():
    print("\n=== Q4b: does that operating point MOVE with the drive scale? ===")
    strengths = (1.0, 3.0, 10.0, 30.0, 100.0)
    header = "  presentations |" + "".join(f"  const={s:<6g}" for s in strengths) \
             + "   proportional"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in (5, 15, 30):
        row = f"  {p:>13d} |"
        for s in strengths:
            row += f"      {evaluate('constant', s, p)[2]}/3    "
        row += f"       {evaluate('proportional', 0.0, p)[2]}/3"
        print(row)


def q5_fake_perfect_audit():
    print("\n=== Q5: fake-perfect audit of the SHIPPED programs ===")
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.programs.nemo_fsm import NemoArcFSM
    from neural_assemblies.programs.mod3_fsm import (
        build_mod3_fsm, run_digit_sequence, train_mod3_fsm,
    )

    b = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
    fsm = NemoArcFSM(b, states=["q0", "q1"], symbols=["a"],
                     transitions=[("q0", "a", "q1"), ("q1", "a", "q0")],
                     n=2000, k=40, beta=0.1, rounds=6)
    nxt = fsm.step_symbol("a", "q0")   # test_deterministic_transition's assertion
    print(f"  untrained step_symbol('a','q0') -> {nxt!r}  "
          f"(shipped test asserts 'q1': {'PASSES' if nxt == 'q1' else 'fails'})")

    for label, presentations, beta in (("zero presentations", 0, 0.1),
                                       ("beta = 0", 15, 0.0)):
        brain = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
        demo = build_mod3_fsm(brain, n=2000, k=40, beta=beta, rounds=6)
        train_mod3_fsm(demo, presentations=presentations)
        pos, _ = run_digit_sequence(demo, list(POSITIVE))
        neg, _ = run_digit_sequence(demo, list(NEGATIVE))
        print(f"  {label:<20s}: positive -> {pos!r} ({pos == 'accept'}), "
              f"negative -> {neg!r} ({neg == 'reject'})")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ablation"):
        q1_q3_ablation()
    if which in ("all", "constant"):
        q4_constant_rule()
        q4_scale_dependence()
    if which in ("all", "audit"):
        q5_fake_perfect_audit()
