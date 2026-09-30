"""Are the PARENTS distinct before we ask whether composites are? (step 6)

`universality_composition.py` returned overlap EXACTLY 1.0000 for every pair in
every bin, with zero variance across seeds. That is not a weak gradient or a
partial collapse -- it is identity, and identity is far more likely to be a
harness defect than a fact about the substrate. Reporting "universality fails"
off that number would be the exact error class this project keeps catching:
a plausible number, no exception raised, wrong question answered.

There are two candidate stages, and they are separable.

  PARENTS    `learn_symbols` projects a0..a5 into SRC_A one after another with
             plasticity on. If the area has a strong recurrent attractor, the
             first symbol carves an assembly and every later symbol is pulled
             into it. Then all parents are the same assembly and composites are
             trivially identical -- nothing to do with merge or composition.

  COMPOSITES the parents are distinct, but the merge target COMP collapses
             across the 30 merges into it, which is the pathology task #31
             recorded for VP (94 constituents, one assembly).

This measures both, separately, and also checks the ONE-SHOT case: a single
merge into a fresh area, which cannot have collapsed because nothing was merged
before it. If one-shot composites are distinct and repeated ones are not, the
defect is accumulation in the target, not the operation.
"""

from __future__ import annotations

import itertools
import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

N, K, P, BETA = 1000, 50, 0.05, 0.10
M = 6


def pairwise(assemblies):
    from neural_assemblies.assembly_calculus.assembly import overlap

    vals = [overlap(a, b) for a, b in itertools.combinations(assemblies, 2)]
    return (statistics.mean(vals), min(vals), max(vals)) if vals else (0, 0, 0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus.ops import merge, project

    seed = 42
    print(f"\n  n={N} k={K} p={P} beta={BETA} seed={seed}  chance={K / N:.4f}")

    # -- stage 1: are the parents distinct? ---------------------------------
    brain = Brain(p=P, seed=seed)
    print(f"  norm_init={getattr(brain, 'norm_init', '?')}")
    for a in ("SRC_A", "SRC_B", "COMP"):
        brain.add_area(a, N, K, beta=BETA)
    for i in range(M):
        brain.add_stimulus(f"a{i}", K)
        brain.add_stimulus(f"b{i}", K)

    parents_a = [project(brain, f"a{i}", "SRC_A", rounds=12) for i in range(M)]
    parents_b = [project(brain, f"b{j}", "SRC_B", rounds=12) for j in range(M)]
    print(f"\n  parents in SRC_A: mean={pairwise(parents_a)[0]:.4f} "
          f"min={pairwise(parents_a)[1]:.4f} max={pairwise(parents_a)[2]:.4f}")
    print(f"  parents in SRC_B: mean={pairwise(parents_b)[0]:.4f} "
          f"min={pairwise(parents_b)[1]:.4f} max={pairwise(parents_b)[2]:.4f}")

    # Re-projecting a symbol must return the SAME assembly, or "shared parent"
    # is meaningless even if the parents are distinct from each other.
    from neural_assemblies.assembly_calculus.assembly import overlap

    again = project(brain, "a0", "SRC_A", rounds=12)
    print(f"  symbol stability (a0 re-projected): {overlap(parents_a[0], again):.4f}")

    # -- stage 2: composites into one shared, accumulating target -----------
    grid = [(i, j) for i in range(M) for j in range(M)]
    acc = [merge(brain, "SRC_A", "SRC_B", "COMP",
                 stim_a=f"a{i}", stim_b=f"b{j}", rounds=10) for i, j in grid]
    m, lo, hi = pairwise(acc)
    print(f"\n  composites, shared target ({len(grid)} merges into COMP):")
    print(f"    mean={m:.4f} min={lo:.4f} max={hi:.4f}")

    # -- stage 3: one-shot, each composite into its OWN fresh target --------
    # Nothing has been merged into these areas before, so they cannot have
    # accumulated an attractor. If these are distinct, the operation is fine
    # and the shared target is the problem.
    brain2 = Brain(p=P, seed=seed)
    for a in ("SRC_A", "SRC_B"):
        brain2.add_area(a, N, K, beta=BETA)
    for i in range(M):
        brain2.add_stimulus(f"a{i}", K)
        brain2.add_stimulus(f"b{i}", K)
    for i in range(M):
        project(brain2, f"a{i}", "SRC_A", rounds=12)
        project(brain2, f"b{i}", "SRC_B", rounds=12)

    subset = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 3)]
    one_shot = []
    for t, (i, j) in enumerate(subset):
        tgt = f"COMP_{t}"
        brain2.add_area(tgt, N, K, beta=BETA)
        one_shot.append(merge(brain2, "SRC_A", "SRC_B", tgt,
                              stim_a=f"a{i}", stim_b=f"b{j}", rounds=10))
    m2, lo2, hi2 = pairwise(one_shot)
    print(f"\n  composites, one fresh target each ({len(subset)} merges):")
    print(f"    mean={m2:.4f} min={lo2:.4f} max={hi2:.4f}")
    print("    NOTE: distinct targets have independent neuron indices, so a")
    print("    LOW number here is expected and is not evidence of structure --")
    print("    it only rules the operation IN if the shared-target case is 1.0.")

    print("\n  reading")
    pa = pairwise(parents_a)[0]
    if pa > 0.9:
        print("    PARENTS COLLAPSED. Every symbol is the same assembly, so the")
        print("    composition result was measuring nothing. Fix this first.")
    elif m > 0.9:
        print("    PARENTS FINE, TARGET COLLAPSED. Same pathology as task #31")
        print("    (94 VPs, one assembly) -- accumulation in the merge target.")
    else:
        print("    Neither collapsed; the flat bins came from somewhere else.")


if __name__ == "__main__":
    main()
