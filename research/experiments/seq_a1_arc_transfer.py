"""A1 follow-up 4: the arc's TRANSFER FUNCTION, ours vs the reference.

Established: the arc drifts first (crosses 0.9 at step 1.4 vs the state's 3.5)
and the state area is cleaning up after it, so the sequence fails because the
arc amplifies a small state error faster than the state area can correct it.
Both arcs are 0.000-overlap conjunctions, so sharpness alone does not explain
why OURS degrades and the reference's holds.

This measures the map directly. Perturb the state cue by swapping m of its k
neurons for UNUSED ones (never part of any state assembly, so no competing
state is injected), and read how far the arc assembly moves:

    input overlap  = (k - m) / k
    output overlap = overlap(arc assembly under perturbation, unperturbed)

PREDICTION, STATED BEFORE THE NUMBERS. If the arc is expansive -- our k-WTA is
documented to AMPLIFY input differences with beta as the gain,
[[kwta-amplifies-input-overlap]] -- output overlap falls faster than input
overlap, i.e. the slope d(output)/d(input) exceeds 1 and small perturbations
are magnified. A contractive or unit-slope map would tolerate the ~0.96 state
the sequence actually supplies at step 2.

Concretely: at m = 3 (input overlap 0.957, about what step 2 delivers) an
amplifying arc lands well below 0.9 while a faithful one stays near 0.95. If
ours amplifies and the reference does not, that is the whole discrepancy and
it is a property of the winner selection, not of the architecture.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import _snap, activate_assembly
from neural_assemblies.core.index_spaces import NeuronIds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_a1_fsm_parity import K, PRESENTATIONS, build

SWAPS = (0, 1, 2, 3, 5, 7, 10, 14)
SEEDS = (1, 2, 3)
PROBE_STATE, PROBE_SYMBOL = "1", "4"


def perturbed_ids(base_ids, spare_ids, m, rng):
    """Swap m of base for m unused neurons -- degrade WITHOUT adding a rival state."""
    if m == 0:
        return base_ids.copy()
    drop = rng.choice(len(base_ids), size=m, replace=False)
    add = rng.choice(len(spare_ids), size=m, replace=False)
    out = base_ids.copy()
    out[drop] = spare_ids[add]
    return out


def ours(seed):
    brain, fsm = build(seed, presentations=PRESENTATIONS)
    rng = np.random.default_rng(seed)
    n_state = brain.areas[fsm.state_area].n
    used = set()
    for st in fsm.states:
        used.update(int(i) for i in fsm.state_assembly(st).winners)
    spare = np.array(sorted(set(range(n_state)) - used), dtype=np.uint32)
    base = np.asarray(fsm.state_assembly(PROBE_STATE).winners, dtype=np.uint32)

    def arc_for(ids):
        with brain.probe():
            brain.inhibit_areas([fsm.arc_area, fsm.state_area])
            activate_assembly(brain, Assembly(fsm.state_area, NeuronIds(ids)))
            brain.areas[fsm.state_area].fix_assembly()
            brain.project({fsm._sym_stim[PROBE_SYMBOL]: [fsm.arc_area]},
                          {fsm.state_area: [fsm.arc_area]})
            out = _snap(brain, fsm.arc_area)
            brain.areas[fsm.state_area].unfix_assembly()
            return out

    ref = arc_for(base)
    return [float(overlap(arc_for(perturbed_ids(base, spare, m, rng)), ref))
            for m in SWAPS]


def reference(seed):
    from neural_assemblies.reference.nemo_numpy.fsm_network import (
        FSMNetwork, build_mod3_symbols_states, mod3_transition_list,
    )
    rng = np.random.default_rng(seed)
    fsm = FSMNetwork(1000, 500, 5000, K, 0.2, 0.1, rng)
    symbols, states = build_mod3_symbols_states(K, rng)
    for _ in range(PRESENTATIONS):
        for fr, sym, to in mod3_transition_list():
            fsm.train(symbols[sym], states[fr], states[to])

    spare = np.arange(len(states) * K, 500, dtype=int)
    base = states[int(PROBE_STATE)]
    prng = np.random.default_rng(seed)

    def arc_for(ids):
        fsm.inhibit()
        fsm.arc_area.forward([symbols[int(PROBE_SYMBOL)], ids], update=False)
        return fsm.arc_area.read()

    ref = arc_for(base)
    return [float(len(np.intersect1d(
        arc_for(perturbed_ids(base, spare, m, prng)), ref)) / K)
        for m in SWAPS]


def main():
    print("=== arc transfer function: state perturbation -> arc displacement ===")
    header = "  input overlap " + " ".join(f"{(K - m) / K:6.3f}" for m in SWAPS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    mine = np.array([ours(s) for s in SEEDS])
    refs = np.array([reference(s) for s in (42, 7, 13)])
    print("  ours (arc out) " + " ".join(f"{v:6.3f}" for v in mine.mean(axis=0)))
    print("  reference      " + " ".join(f"{v:6.3f}" for v in refs.mean(axis=0)))

    inp = np.array([(K - m) / K for m in SWAPS])
    for name, arr in (("ours", mine), ("reference", refs)):
        y = arr.mean(axis=0)
        # slope of output-loss against input-loss over the small-perturbation
        # end, which is the regime a sequence actually visits
        lo = slice(0, 4)
        slope = np.polyfit(1 - inp[lo], 1 - y[lo], 1)[0]
        print(f"\n  {name:<10s} amplification d(output loss)/d(input loss) "
              f"= {slope:.2f}" + ("  (>1 = amplifies)" if slope > 1 else ""))
        at3 = y[SWAPS.index(3)]
        print(f"             at m=3 (input {inp[SWAPS.index(3)]:.3f}, "
              f"roughly what step 2 delivers): arc overlap {at3:.3f}")

    from _results import write_result
    out = write_result("sequence", "seq_a1_arc_transfer_results.json",
                       {"swaps": list(SWAPS), "input_overlap": inp.tolist(),
                        "ours": mine.tolist(), "reference": refs.tolist()})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
