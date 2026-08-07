"""Does the COLT-22 overlap law PREDICT the parser's collisions?

THE CHAIN THIS CLOSES. `colt22_overlap_preservation.py` measured, on the bare
substrate with the paper's own stimulus-class model, that assembly overlap is an
AMPLIFYING function of input overlap alpha:

    beta      alpha=0.10   alpha=0.25   alpha=0.50   alpha=0.75
    0.05        0.100        0.283        0.697        1.023
    0.10        0.170        0.497        0.897        1.023
    >=0.50      0.683        0.977        1.000        1.000

If that law is what drives the parser, then measuring alpha between each pair of
words' GROUNDING FEATURES should predict their measured assembly overlap. This
file does that -- it is a prediction test on the production system, not another
sweep.

WHY THIS IS THE RIGHT TEST. The earlier arc failed to reproduce the parser in a
two-area model and I concluded the behaviour was "not derivable from the minimal
model". That was wrong, and for a specific reason: every toy ran at alpha = 0 --
independent random stimuli, which is the ONE column of the table above where
nothing happens. The minimal model reproduces the parser fine once its inputs
share features. So this is also a check on that correction.

ALPHA FOR UNEQUAL SETS. The paper defines |S_A ∩ S_B| = alpha*k with both
classes of size exactly k. Grounding feature sets differ in size, so alpha is
taken as the Dice coefficient 2|A∩B| / (|A|+|B|), which reduces to the paper's
alpha when the sets are the same size. Reported rather than hidden because the
choice moves the x-axis.

WHAT WOULD FALSIFY THE PREDICTION. If assembly overlap is flat in alpha -- pairs
with disjoint grounding colliding as often as pairs with shared grounding --
then the substrate law is not what drives the parser and the collisions come
from somewhere else (the phon pathway, the training order, the curriculum).
That is a real possible outcome: only 1 of 4-5 duplicate clusters was already
identical at the input, so alpha is demonstrably NOT 1.0 for most of them.

A CONFOUND NAMED IN ADVANCE. Words are trained SEQUENTIALLY into a shared area,
so a late word meets a connectome already shaped by every earlier word. The
substrate sweep has the same structure (A built, then B), so the comparison is
like-for-like -- but it means "alpha predicts overlap" would be consistent with
either alpha or training order being the driver. Training rank is therefore
reported alongside, and if rank predicts as well as alpha does, the attribution
is not settled by this file.
"""
import os
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import numpy as np                                                     # noqa: E402

from _substrate import similarity                                      # noqa: E402
from neural_assemblies.assembly_calculus.emergent.core.areas import (   # noqa: E402
    NOUN_CORE,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    train_parser_to_depth,
)

DEPTH = "SENTENCES"
SEEDS = [11, 42]
BINS = [(0.0, 0.001), (0.001, 0.15), (0.15, 0.35), (0.35, 0.65),
        (0.65, 0.95), (0.95, 1.01)]

#: The substrate law, measured in colt22_overlap_preservation.py at n=10000,
#: k=100, p=0.05, r=0.9. The parser runs beta=0.10.
SUBSTRATE_LAW = {0.10: 0.170, 0.25: 0.497, 0.50: 0.897, 0.75: 1.023}


def _features(parser, word):
    """Flatten a GroundingContext into the set of features it fires."""
    ctx = getattr(parser, "word_grounding", {}).get(word)
    if ctx is None:
        return frozenset()
    feats = set()
    for field in ("visual", "motor", "properties", "spatial", "social",
                  "temporal", "emotional"):
        vals = getattr(ctx, field, None) or []
        feats.update(f"{field}:{v}" for v in vals)
    if not feats and isinstance(ctx, (list, tuple, set, frozenset)):
        feats.update(map(str, ctx))
    return frozenset(feats)


def analyse(parser, seed):
    core = parser.core_lexicons.get(NOUN_CORE, {})
    area = parser.brain.areas[NOUN_CORE]
    n, k = int(area.n), int(area.k)
    words = sorted(core)
    # Training order: dict insertion order is the order train_lexicon wrote them.
    rank = {w: i for i, w in enumerate(core)}
    asm = {w: np.asarray(core[w].winners, dtype=np.int64) for w in words}
    feats = {w: _features(parser, w) for w in words}
    empty = [w for w in words if not feats[w]]
    print(f"  NOUN_CORE n={n} k={k} M={len(words)}  "
          f"words with NO grounding features: {len(empty)}")

    pairs = []
    for a, b in combinations(words, 2):
        fa, fb = feats[a], feats[b]
        if not fa or not fb:
            continue
        alpha = 2 * len(fa & fb) / (len(fa) + len(fb))
        pairs.append((alpha, similarity(asm[a], asm[b]),
                      abs(rank[a] - rank[b]), a, b))
    if not pairs:
        print("  no usable pairs")
        return None

    print(f"  usable pairs: {len(pairs)}  (chance overlap k/n = {k / n:.4f})")
    print()
    hdr = (f"  {'alpha bin':>14} {'pairs':>7} {'mean alpha':>11} "
           f"{'mean overlap':>13} {'frac identical':>15} {'substrate law':>14}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lo, hi in BINS:
        sel = [p for p in pairs if lo <= p[0] < hi]
        if not sel:
            continue
        ma = statistics.fmean(p[0] for p in sel)
        mo = statistics.fmean(p[1] for p in sel)
        ident = sum(1 for p in sel if p[1] >= 0.999) / len(sel)
        near = min(SUBSTRATE_LAW, key=lambda x: abs(x - ma))
        pred = SUBSTRATE_LAW[near] if abs(near - ma) < 0.18 else float("nan")
        pred_s = f"{pred:.3f}@{near:.2f}" if pred == pred else "   --"
        print(f"  [{lo:.3f},{hi:.2f}) {len(sel):>7} {ma:>11.3f} "
              f"{mo:>13.4f} {ident:>15.1%} {pred_s:>14}")

    # Is it alpha, or is it training order? Spearman-free: compare the
    # correlation of overlap with alpha against its correlation with rank gap.
    ov = np.array([p[1] for p in pairs])
    al = np.array([p[0] for p in pairs])
    rk = np.array([float(p[2]) for p in pairs])

    def _r(x, y):
        xr = np.argsort(np.argsort(x)).astype(float)
        yr = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(xr, yr)[0, 1])

    print()
    print(f"  rank correlation  overlap vs ALPHA      {_r(al, ov):+.3f}")
    print(f"  rank correlation  overlap vs RANK GAP   {_r(rk, ov):+.3f}")
    print("  (if the second is comparable to the first, this file does not")
    print("   separate feature overlap from training order)")
    return pairs


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Does the substrate's overlap law predict the parser's collisions?")
    print("Substrate at beta=0.10:  alpha 0.10 -> 0.170,  0.25 -> 0.497,")
    print("                         alpha 0.50 -> 0.897,  0.75 -> 1.023")
    print()
    for seed in SEEDS:
        print(f"=== depth={DEPTH} seed={seed} ===")
        parser = train_parser_to_depth(DEPTH, seed=seed)
        analyse(parser, seed)
        print()
    print("=" * 72)
    print("Assembly overlap RISING with grounding alpha, and tracking the")
    print("substrate law, means the parser's collisions are the SAME mechanism")
    print("measured on the bare substrate -- not a parser-specific defect.")


if __name__ == "__main__":
    main()
