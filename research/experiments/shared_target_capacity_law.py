"""How many bindings can one shared target area hold before they stop being
retrievable -- and does the ceiling track M, or alpha = Mk/n?

WHY THIS AND NOT THE EARLIER SWEEP. `bound_pathway_capacity_sweep.py` swept the
same M and found the effect FLAT to M=32. It measured with `input_drive`, which
is now known to be structurally blind to which assembly a source was bound to
(`test_drive_and_binding_are_orthogonal.py`). A flat curve was the only thing it
could have produced. Same design, right instrument.

THE DESIGN ISOLATES TARGET LOAD, and it does so by construction rather than by
argument:

  * the SRC population is built ONCE and never changes, so source separation is
    held EXACTLY fixed while M varies -- source crowding cannot be the moving
    part;
  * bindings are written INCREMENTALLY, with retrieval measured at checkpoints,
    so every point on the curve is the same substrate at a different load rather
    than a different substrate;
  * stored targets are snapshotted AT BIND TIME and never refreshed, which is
    what `role_lexicons` actually holds. Re-snapshotting would measure the
    pathway against a target it just produced -- the fake-perfect shape.

READOUT DIFFICULTY IS HELD CONSTANT, which is the part the parser measurement
could not do. Retrieval is always scored among a random 6 of the M written so
far, so chance stays 1/6 at every point and the ONLY thing varying is how many
other bindings share the area. Raw rank-1-of-M would confound interference with
the arithmetic of having more candidates, and would fall even if the substrate
were perfect.

MARGIN IS REPORTED ALONGSIDE ACCURACY, because accuracy is a step function of a
continuous quantity: it reports "fine, fine, fine, broken" and hides the
approach. The merge-capacity work already learned this once.

TWO AREA SIZES, to separate M from alpha = Mk/n. If the curves at n=2000 and
n=4000 collapse when plotted against alpha, capacity is extensive and alpha is
the law. If they collapse against M, it is not.

CONTEXT. The parser's role areas sit at alpha ~ 0.36-0.46 with retrieval 0.375
and ret@6 0.729, while the storage regime's critical load is alpha* ~ 1.15. If
binding into a shared target saturates near 0.36, that 3x shortfall is the
number that says how much grammar this substrate can hold -- and it binds long
before vocabulary does, since the lexicon has no known ceiling.
"""
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                                     # noqa: E402

from _substrate import read, similarity                                # noqa: E402
from neural_assemblies.assembly_calculus.assembly import Assembly      # noqa: E402
from neural_assemblies.core.brain import Brain                         # noqa: E402
from neural_assemblies.assembly_calculus.ops import activate_assembly  # noqa: E402

SRC, DST = "SRC", "DST"
K, P, BETA = 40, 0.05, 0.10
BUILD_ROUNDS, BIND_ROUNDS = 6, 10
M_MAX = 96
CHECKPOINTS = [6, 12, 24, 48, 96]
CANDIDATES = 6            # readout difficulty, held constant
SUBSETS = 40
N_GRID = [2000, 4000]
SEEDS = [42, 7, 123]


def _retrieval_at_fixed_difficulty(brain, sources, targets, rng):
    """Rank-1 among a random `CANDIDATES` of what has been written so far."""
    words = list(targets)
    if len(words) < CANDIDATES:
        return float("nan"), float("nan")
    live = {}
    for w in words:
        with brain.probe():
            activate_assembly(brain, Assembly(SRC, sources[w]))
            brain.project({}, {SRC: [DST]})
            live[w] = read(brain, DST)
    hits = total = 0
    margins = []
    for _ in range(SUBSETS):
        subset = list(rng.choice(words, size=CANDIDATES, replace=False))
        for w in subset:
            sc = sorted(((similarity(live[w], targets[o]), o) for o in subset),
                        reverse=True)
            hits += int(sc[0][1] == w)
            total += 1
            if len(sc) > 1 and sc[1][0] > 0:
                margins.append(sc[0][0] / sc[1][0])
    return (hits / total if total else float("nan"),
            statistics.fmean(margins) if margins else float("nan"))


def _spread(targets, cap=30):
    items = list(targets.values())[:cap]
    pairs = list(combinations(items, 2))
    return statistics.fmean(similarity(x, y) for x, y in pairs) if pairs else float("nan")


def run(n, seed):
    brain = Brain(p=P, seed=seed, norm_init=True)
    brain.add_area(SRC, n, K, beta=BETA)
    brain.add_area(DST, n, K, beta=BETA)
    names = [f"a{i}" for i in range(M_MAX)]
    for nm in names:
        brain.add_stimulus(nm, K)

    # Built ONCE. Source separation is identical at every checkpoint.
    sources = {}
    for nm in names:
        for _ in range(BUILD_ROUNDS):
            brain.project({nm: [SRC]}, {})
        sources[nm] = read(brain, SRC)
    src_spread = _spread(sources)

    rng = np.random.default_rng(seed)
    targets, out = {}, {}
    for i, nm in enumerate(names, start=1):
        for _ in range(BIND_ROUNDS):
            brain.project({nm: [SRC]}, {SRC: [DST]})
        targets[nm] = read(brain, DST)          # snapshot AT BIND TIME
        if i in CHECKPOINTS:
            acc, margin = _retrieval_at_fixed_difficulty(
                brain, sources, targets, np.random.default_rng(seed))
            out[i] = (acc, margin, _spread(targets))
    return src_spread, out


def main():
    print(f"k={K} p={P} candidates={CANDIDATES} (chance {1/CANDIDATES:.3f}) "
          f"subsets={SUBSETS} seeds={SEEDS}")
    print("SRC population built ONCE at M_MAX and never changed, so source")
    print("separation is fixed and only TARGET load varies.")
    print()
    hdr = (f"{'n':>6} {'M':>5} {'alpha':>7} {'ret@6':>8} {'margin':>8} "
           f"{'tgt spread':>11} {'src spread':>11}")
    print(hdr)
    print("-" * len(hdr))

    curves = {}
    for n in N_GRID:
        per_seed = [run(n, s) for s in SEEDS]
        src_sp = statistics.fmean(x[0] for x in per_seed)
        for m in CHECKPOINTS:
            rows = [x[1][m] for x in per_seed if m in x[1]]
            if not rows:
                continue
            acc = statistics.fmean(r[0] for r in rows)
            mg = statistics.fmean(r[1] for r in rows)
            sp = statistics.fmean(r[2] for r in rows)
            curves[(n, m)] = (m * K / n, acc, mg)
            print(f"{n:>6} {m:>5} {m * K / n:>7.3f} {acc:>8.3f} {mg:>8.3f} "
                  f"{sp:>11.4f} {src_sp:>11.4f}")
        print()

    print("=" * 62)
    print("DOES THE CEILING TRACK M OR alpha? Same accuracy at same alpha means")
    print("capacity is EXTENSIVE and alpha is the law; same accuracy at same M")
    print("means it is not.")
    print(f"{'alpha':>7} " + "  ".join(f"n={n}" for n in N_GRID))
    print("-" * 40)
    alphas = sorted({round(v[0], 3) for v in curves.values()})
    for a in alphas:
        cells = []
        for n in N_GRID:
            hit = [v for (nn, _m), v in curves.items()
                   if nn == n and abs(v[0] - a) < 1e-6]
            cells.append(f"{hit[0][1]:.3f}" if hit else "  -  ")
        print(f"{a:>7.3f} " + "  ".join(f"{c:>6}" for c in cells))

    print()
    print("The parser sits at alpha 0.36-0.46 with ret@6 = 0.729/0.721. Read")
    print("this table at that alpha: if the toy substrate is much higher there,")
    print("the parser is losing something this design does not model.")


if __name__ == "__main__":
    main()
