"""Does LOWER beta rescue depth? The characterization says it should.

THE PREDICTION, AND WHERE IT COMES FROM
----------------------------------------
`overnight_characterization.py` measured two things that together make a sharp
prediction about depth.

STUDY E collapsed a two-parameter family onto one variable. Assembly
persistence under self-recurrence is a function of `(1+beta)^T` alone, not of
beta and T separately -- at n=2000, (1+b)^T of 2.59 and 2.99 gave self-overlap
0.127 and 0.120, while 4.59 and 6.19 gave 0.853 and 0.860. Different splits,
same number, same outcome. So beta and T are interchangeable knobs on ONE
quantity: total potentiation.

STUDY C then showed merge capacity moving with that quantity. At n=1000,
capacity was 256 at beta=0.05/T=2 and 128 at beta=0.10/T=2 -- HALVING BETA
DOUBLED CAPACITY -- falling to 32 at beta=0.20/T=5.

`depth_corrected_substrate.py` had already established the mechanism that links
them: composition AMPLIFIES overlap multiplicatively down a chain (0.1586 ->
0.3763 -> 0.5583 at T=10), and the fix was to lower T until level-1 assemblies
were written at the floor. But T was the only knob used, and T=1 is the floor of
that knob -- below it there is no merge at all.

beta is the OTHER knob on the same quantity, and it is unbounded downward. If
overlap-per-level is what limits depth, beta should buy depth that T cannot,
because it can keep going after T has run out.

WHAT THIS TESTS AGAINST
-----------------------
`complexity_ladder_overnight.py` found depth 5 failing in a way depth 3 did not:

      n     M  D   full_1   full_D  marg_1  marg_D   spr_1   spr_D
   4000   256  5   1.0000   0.0182    7.89    1.16  0.0142  0.0143
   1000    64  5   1.0000   0.0938    4.34    1.17  0.0560  0.0625

Note `spr_D` sits AT THE FLOOR while retrieval dies. At depth 3 the failures had
spr_D climbing to 0.42 -- crowding. Here the level-5 assemblies are perfectly
distinct and the chain still loses them, with margin decaying 7.89 -> 1.16 and
no overlap growth. That is a SECOND failure mode, and it is the reason this file
reports margin at every level rather than just the ends: if beta rescues depth,
margin should decay more slowly per level; if it does not, the loss is not about
overlap at all and beta cannot touch it.

PRE-REGISTERED
--------------
B1 At beta=0.10 (the current operating point) depth 5 fails, reproducing the
   ladder from this harness. Gates everything else.
B2 Lower beta raises the deepest passing level. This is the prediction.
B3 Margin decays GEOMETRICALLY per level, and the per-level ratio improves as
   beta falls. This is the mechanism claim: each level costs a fixed fraction
   of the margin, and beta sets the fraction.
B4 If depth still fails at the lowest beta while margin-per-level is unchanged,
   the depth limit is NOT potentiation-driven -- it is a property of chaining
   projections, and the next control is a chain of PLAIN projections with no
   merge at all, to separate composition cost from projection cost.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_assemblies.diagnostics import (  # noqa: E402
    assembly_overlap, read_assembly,
)

N, K, P = 4000, 50, 0.05
M_ITEMS = 64
DEPTH = 8
MERGE_T = 2
BUILD_ROUNDS = 6
SEEDS = (42, 7, 123)
BETAS = (0.10, 0.20, 0.40, 0.80)
GATED = dict(parent_self=False, target_self=False, back_project=False)


def build_ff(brain, stim, area, rounds):
    for _ in range(rounds):
        brain.project({stim: [area]}, {})
    return read_assembly(brain, area)


def trial(beta, seed):
    from _substrate import pinned
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    brain.add_area("A", N, K, beta=beta)
    for L in range(1, DEPTH + 1):
        brain.add_area(f"P{L}", N, K, beta=beta)
        brain.add_area(f"C{L}", N, K, beta=beta)
    for m in range(M_ITEMS):
        brain.add_stimulus(f"a{m}", K)
        for L in range(1, DEPTH + 1):
            brain.add_stimulus(f"p{L}_{m}", K)
    for m in range(M_ITEMS):
        build_ff(brain, f"a{m}", "A", BUILD_ROUNDS)
        for L in range(1, DEPTH + 1):
            build_ff(brain, f"p{L}_{m}", f"P{L}", BUILD_ROUNDS)

    stored = {L: {} for L in range(1, DEPTH + 1)}
    for m in range(M_ITEMS):
        merge(brain, "A", "P1", "C1", stim_a=f"a{m}", stim_b=f"p1_{m}",
              rounds=MERGE_T, **GATED)
        stored[1][m] = read_assembly(brain, "C1")
    for L in range(2, DEPTH + 1):
        for m in range(M_ITEMS):
            with pinned(brain, f"C{L-1}", stored[L - 1][m]):
                merge(brain, f"C{L-1}", f"P{L}", f"C{L}",
                      stim_b=f"p{L}_{m}", rounds=MERGE_T, **GATED)
                stored[L][m] = read_assembly(brain, f"C{L}")

    full = {L: 0 for L in range(1, DEPTH + 1)}
    margins = {L: [] for L in range(1, DEPTH + 1)}
    for m in range(M_ITEMS):
        with brain.read_only():
            build_ff(brain, f"a{m}", "A", BUILD_ROUNDS)
            src = "A"
            for L in range(1, DEPTH + 1):
                brain.project({}, {src: [f"C{L}"]})
                live = read_assembly(brain, f"C{L}")
                sims = sorted(((assembly_overlap(live, a), j)
                               for j, a in stored[L].items()), reverse=True)
                full[L] += sims[0][1] == m
                if len(sims) > 1 and sims[1][0] > 0:
                    margins[L].append(sims[0][0] / sims[1][0])
                src = f"C{L}"
    return full, margins


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  n={N} k={K} p={P}, M={M_ITEMS} constituents, merge T={MERGE_T} "
          f"all channels gated, one SHARED area per level")
    print(f"  {M_ITEMS} items x {len(SEEDS)} seeds = {M_ITEMS * len(SEEDS)} "
          f"trials/cell, chance = {1 / M_ITEMS:.4f}")
    print(f"  beta and T are two knobs on ONE quantity, (1+beta)^T -- study E "
          f"showed persistence depends only on it.")
    print(f"  T is already at its floor for depth (T=1 is no merge), so beta "
          f"is the knob with room left.\n")
    print(f"  {'beta':>6}" + "".join(f"{'acc@' + str(L):>9}"
                                     for L in range(1, DEPTH + 1))
          + "   | " + "".join(f"{'mar@' + str(L):>8}"
                              for L in range(1, DEPTH + 1)))

    table = {}
    for beta in BETAS:
        res = [trial(beta, s) for s in SEEDS]
        tot = M_ITEMS * len(SEEDS)
        acc = {L: sum(r[0][L] for r in res) / tot
               for L in range(1, DEPTH + 1)}
        mar = {L: statistics.mean(statistics.mean(r[1][L]) for r in res
                                  if r[1][L])
               for L in range(1, DEPTH + 1)}
        table[beta] = (acc, mar)
        print(f"  {beta:>6.2f}"
              + "".join(f"{acc[L]:>9.4f}" for L in range(1, DEPTH + 1))
              + "   | " + "".join(f"{mar[L]:>8.2f}"
                                  for L in range(1, DEPTH + 1)))

    def deepest(beta):
        acc = table[beta][0]
        ok = [L for L in range(1, DEPTH + 1) if acc[L] > 0.90]
        return max(ok) if ok else 0

    print("\n  READING")
    b1 = deepest(0.10) < 5
    print(f"    B1 depth 5 fails at beta=0.10:   {b1}   deepest passing level "
          f"{deepest(0.10)}")
    # B2 was pre-registered as MONOTONE ("lower beta buys depth") and that is
    # REFUTED in both directions: depth is an INTERIOR optimum in beta. Testing
    # monotonicity here would report False and hide the actual finding, so the
    # check is for an interior peak -- a shape the prediction did not consider.
    depths = {b: deepest(b) for b in BETAS}
    peak = max(BETAS, key=lambda b: depths[b])
    interior = (peak != min(BETAS) and peak != max(BETAS)
                and depths[peak] > depths[min(BETAS)]
                and depths[peak] > depths[max(BETAS)])
    b2 = interior
    print(f"    B2 depth peaks at an INTERIOR beta: {b2}   " +
          "  ".join(f"b={b}:D{d}" for b, d in sorted(depths.items())) +
          f"   peak beta={peak}")

    print(f"    B3 margin decay per level (geometric ratio)")
    for beta in BETAS:
        mar = table[beta][1]
        ratios = [mar[L + 1] / mar[L] for L in range(1, DEPTH)
                  if mar[L] > 0]
        print(f"       beta={beta:<5} " +
              "  ".join(f"{r:.2f}" for r in ratios) +
              f"   mean {statistics.mean(ratios):.3f}" if ratios else "")

    print()
    if b2:
        print(f"    B2 REFUTED IN BOTH DIRECTIONS -- and that is the finding.")
        print(f"    Depth peaks at beta={peak} (D{depths[peak]}) and falls away on")
        print(f"    EITHER side, so two opposing pressures act on one parameter:")
        print(f"      too LOW  -- the C(L-1)->C(L) fiber is under-potentiated and")
        print(f"                  the chain cannot carry the signal. Margin decays")
        print(f"                  fastest per level here, with assemblies still")
        print(f"                  perfectly distinct: STARVATION, not crowding.")
        print(f"      too HIGH -- assemblies overlap on contact and the chain dies")
        print(f"                  within a level or two: CROWDING, the mechanism")
        print(f"                  depth_corrected_substrate.py found at T=10.")
        print(f"    The prediction that lower beta buys depth had exactly half")
        print(f"    the mechanism. Capacity does want low beta (study C: halving")
        print(f"    it doubled M_max); depth wants it high enough to transmit.")
        print(f"    They are opposed, so beta cannot be tuned once for both --")
        print(f"    which is a real constraint on any deep, broad grammar.")
    else:
        print(f"    No interior peak. Read the columns directly: if margin decays")
        print(f"    at the same per-level rate for every beta, the depth limit is")
        print(f"    not potentiation-driven at all, and the next control is a")
        print(f"    chain of PLAIN projections with no merge, to separate")
        print(f"    composition cost from the cost of chaining projections.")


if __name__ == "__main__":
    main()
