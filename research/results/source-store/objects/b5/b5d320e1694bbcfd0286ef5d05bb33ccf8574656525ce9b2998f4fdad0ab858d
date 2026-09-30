"""#93: can p, beta or homeostasis move the arc's 2:1 trade?

WHERE THIS COMES FROM
---------------------
`arc_conjunction_has_no_operating_point.md`: the ARC area encodes the mood and
drops the state, and sweeping the MOOD->ARC gain does not fix it -- it is a
LOSING trade, 0.38 of state-separation spent to buy 0.19 of mood-separation,
with no interior optimum. Mechanism: MOOD fires into ARC at every one of the
three constituents while any given SYNTAX_prev fires only when it precedes, so
MOOD accumulates ~3x the potentiation and takes the k-WTA.

Sweeping the gain again would only move the split. This file tries to move the
RATIO, along the three axes where our setup violates the theorem that promises
the arc should work (Dabagia/Papadimitriou/Vempala 2025, Thm 2 and Thm 4).

    p            their sequence experiments run p = 0.2; we run 0.05, the
                 bottom of their own Fig. 8 sweep, where they report exactly
                 our failure mode -- high overlap between the assemblies for
                 distinct sequence elements.
    beta         "the plasticity cannot be too high", and the number of
                 presentations needed grows as it falls. We run 0.06.
    homeostasis  Thm 2 assumes incoming weights are renormalised EVERY round.
                 We apply norm_init once.

`synaptic_scaling` supplies the third. Note what it is and is not: it holds each
FIBER into a winner to a setpoint of `rows * p` after every projection, whereas
the theorem normalises JOINTLY across all fibers into a neuron, and stimulus
fibers do not participate at all. Both inputs to ARC are areas, so both are
covered here. Its own docstring judges it wrong for the recurrence problem it
was written for -- a per-fiber setpoint cancels the gain a self-sustaining
attractor needs -- but that objection does not apply to the arc, which has no
self-recurrence and whose failure IS one fiber out-accumulating another.

A LIMITATION THAT CANNOT BE ENGINEERED AWAY TODAY, stated because it bounds
every reading below: `p` here is a Brain-level constant, so raising it raises
density on EVERY fiber, not only the two into ARC. Per-fiber `p` exists only on
`numpy_exact`; `synaptic_scaling` exists only on `numpy_sparse`. They cannot be
combined, so "raise p on the arc alone" is not currently expressible. The p
factor below is therefore a whole-model manipulation, exactly as it is in the
paper's own sweep -- which is the right comparison, but it is not a targeted one.

MEASURED
--------
On NEURON IDS (`diagnostics.read_assembly`), never `area.winners` -- the compact
index space is what produced the retracted "conserved budget".

    across-STATE   ARC overlap for different previous constituents, one mood.
                   Must be LOW: the arc has to distinguish q.
    across-MOOD    ARC overlap for one previous constituent, different moods.
                   Must be LOW: the arc has to distinguish sigma.

A real conjunction needs BOTH low. The baseline (p=0.05, no scaling) reads
0.98 / 0.81.

PRE-REGISTERED
--------------
P1 Raising p LOWERS across-STATE. This is the paper's Fig. 8 direction and the
   cheapest of the three. If p does nothing here, [[kp-decides-whether-beta-helps]]
   gets a counterexample worth having.
P2 `synaptic_scaling` lowers across-STATE WITHOUT raising across-MOOD -- i.e.
   it moves the ratio rather than the split. This is the one with a mechanism
   behind it: capping each fiber's contribution is precisely what stops MOOD
   out-accumulating SYNTAX_prev.
P3 Some cell reaches BOTH below 0.5. If none does, the failure is architectural
   rather than parametric, and the answer is a different architecture (separate
   arc area per state, or the paper's mutual inhibition) rather than more
   sweeping. Stated in advance so a null is a finding rather than a to-do.

CONTROL: the single-mood floor must be 4/4 in any cell claimed as an
improvement. An arm that cannot generate one fixed order is not a better
conjunction, it is a broken model -- this is the check that caught the arc arms
in #95, where they read 1/4 and 0/4.
"""

from __future__ import annotations

import copy
import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.diagnostics import (  # noqa: E402
    assembly_overlap, read_assembly)
from neural_assemblies.reference.word_order_learner import (  # noqa: E402
    ARC, HELPER, MOOD, WordOrderLearner)

ORDERS = {"SVO": ("S", "V", "O"), "SOV": ("S", "O", "V"),
          "VSO": ("V", "S", "O"), "OVS": ("O", "V", "S")}
PAIRS = [("SVO", "SOV"), ("SVO", "VSO"), ("SOV", "OVS"), ("VSO", "OVS")]
SEEDS = (1, 2)
N, K, SENTENCES = 1000, 50, 60
CONSTITUENTS = ("S", "V", "O")

# (label, p, beta, synaptic_scaling)
CELLS = [
    ("baseline",        0.05, 0.06, False),
    ("p=0.10",          0.10, 0.06, False),
    ("p=0.20 (paper)",  0.20, 0.06, False),
    ("homeostasis",     0.05, 0.06, True),
    ("p=0.20 + homeo",  0.20, 0.06, True),
    ("+ low beta",      0.20, 0.01, True),
]


def _built(p, beta, scaling, seed, orders):
    m = WordOrderLearner(
        num_nouns=4, num_verbs=2, mood_orders=orders,
        n=N, k=K, p=p, beta=beta, seed=seed,
        conjunctive_arc=True, synaptic_scaling=scaling)
    m.train(SENTENCES)
    return m


def _arc(m, mi, q):
    """ARC assembly for (state q, mood mi), on a deepcopy so nothing moves."""
    live = m.brain
    m.brain = copy.deepcopy(live)
    try:
        with m.brain.frozen():
            m._mood_now = mi
            m.brain.activate(MOOD, mi)
            m._activate_role(0, q, firings=3)
            m.brain.project({}, {HELPER[q]: [m._syn(q)], MOOD: [m._syn(q)]})
            m._form_arc(q)
            return read_assembly(m.brain, ARC)
    finally:
        m.brain = live


def separations(m):
    a = {q: _arc(m, 0, q) for q in CONSTITUENTS}
    ks = list(a)
    state = statistics.mean(assembly_overlap(a[x], a[y])
                            for i, x in enumerate(ks) for y in ks[i + 1:])
    mood = statistics.mean(assembly_overlap(a[q], _arc(m, 1, q)) for q in ks)
    return state, mood


def single_mood_floor(p, beta, scaling):
    ok = 0
    for name in sorted(ORDERS):
        m = _built(p, beta, scaling, 1, {0: ORDERS[name]})
        ok += "".join(m.generate(0)) == name
    return ok


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n  #93 -- can p, beta or homeostasis move the arc's 2:1 trade?")
    print(f"  n={N} k={K}, {SENTENCES} sentences, {len(SEEDS)} seeds, "
          f"NEURON IDS throughout")
    print(f"  a real conjunction needs BOTH separations below 0.5\n")
    print(f"  {'cell':>16} {'p':>5} {'beta':>5} {'homeo':>6} "
          f"{'across-STATE':>13} {'across-MOOD':>12} {'multi':>7} {'1-mood':>7}")

    rows = []
    for label, p, beta, scaling in CELLS:
        st, mo, ok, tot = [], [], 0, 0
        for a, b in PAIRS:
            for seed in SEEDS:
                m = _built(p, beta, scaling, seed, {0: ORDERS[a], 1: ORDERS[b]})
                got = ["".join(m.generate(i)) for i in (0, 1)]
                ok += (got[0] == a) + (got[1] == b)
                tot += 2
                if (a, b) == PAIRS[0]:
                    s, o = separations(m)
                    st.append(s); mo.append(o)
        floor = single_mood_floor(p, beta, scaling)
        s, o = statistics.mean(st), statistics.mean(mo)
        rows.append((label, p, beta, scaling, s, o, ok, tot, floor))
        print(f"  {label:>16} {p:>5.2f} {beta:>5.2f} {str(scaling):>6} "
              f"{s:>13.2f} {o:>12.2f} {ok:>4}/{tot:<2} {floor:>5}/4",
              flush=True)

    print("\n  READING\n")
    base = rows[0]
    p_cells = [r for r in rows if r[1] > 0.05 and not r[3]]
    homeo = [r for r in rows if r[3]]

    p1 = any(r[4] < base[4] for r in p_cells)
    print(f"    P1 raising p lowers across-STATE:        {str(p1):>5}   "
          f"{base[4]:.2f} -> {min(r[4] for r in p_cells):.2f}")
    p2 = any(r[4] < base[4] and r[5] <= base[5] + 0.05 for r in homeo)
    print(f"    P2 homeostasis moves the RATIO:          {str(p2):>5}   "
          f"best homeo cell "
          f"{min(homeo, key=lambda r: r[4] + r[5])[4]:.2f} / "
          f"{min(homeo, key=lambda r: r[4] + r[5])[5]:.2f}")
    winners = [r for r in rows if r[4] < 0.5 and r[5] < 0.5]
    print(f"    P3 some cell has BOTH below 0.5:         "
          f"{str(bool(winners)):>5}   "
          f"{[r[0] for r in winners] if winners else 'none'}")

    best = min(rows, key=lambda r: r[4] + r[5])
    print(f"\n    best sum: {best[0]} -> {best[4]:.2f} + {best[5]:.2f} = "
          f"{best[4] + best[5]:.2f}   (baseline "
          f"{base[4] + base[5]:.2f}), multi {best[6]}/{best[7]}, "
          f"floor {best[8]}/4")
    if not winners:
        print()
        print("    NO CELL IS A CONJUNCTION. Across every axis on which our")
        print("    setup violates the theorem -- density, plasticity, per-round")
        print("    homeostasis -- one shared k-WTA area still cannot hold")
        print("    (state, symbol). The remedy is then architectural, not")
        print("    parametric: an arc area per state, or the paper's mutual")
        print("    inhibition over the ROLE areas, which this repo has measured")
        print("    as never once firing ([[mutual-inhibition-prefers-untrained]]).")
    if any(r[8] < 4 for r in rows if r[4] + r[5] < base[4] + base[5]):
        print()
        print("    WARNING: a cell that improved the separations FAILED the")
        print("    single-mood floor. Improvement there is not a better")
        print("    conjunction, it is a model that stopped working -- the")
        print("    failure mode that made the #95 arc arms uninterpretable.")


if __name__ == "__main__":
    main()
