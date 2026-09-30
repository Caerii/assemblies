"""PHASE A: does beta 0.10 -> 0.05 buy the parser what the substrate law says?

ONE VARIABLE, deliberately. The substrate sweep
(`colt22_overlap_preservation.py`) measured that assembly overlap amplifies
input overlap alpha with beta as the gain:

    beta      alpha=0.10   alpha=0.25   alpha=0.50
    0.05        0.100        0.283        0.697
    0.10        0.170        0.497        0.897

and the parser runs beta = 0.10. The architecture fix (#125, PHON->LEX1/LEX2)
is the other candidate. Changing both at once would leave neither attributable,
so beta moves alone here and the architecture moves alone later.

PREDICTION, pre-registered. If the substrate law drives the parser then at
beta = 0.05:
  * mean assembly overlap in the alpha >= 0.35 bin FALLS (0.55 is the beta=0.10
    reading);
  * the exact-duplicate fraction FALLS (0.239-0.268 of words at beta=0.10);
  * role retrieval RISES or holds.

THE DEGENERATE ARM, and it is the whole reason retrieval is measured here.
Lowering beta can "improve distinctness" by failing to learn at all -- an area
that never consolidates has beautifully distinct, useless assemblies. This
project has the memory for it: plastic over-generalises, frozen
under-generalises, and distinctness is not information. So three competence
guards run alongside:

  * ret@6 on the role areas -- the downstream task, at matched difficulty;
  * M, the number of core and role lexicon entries actually written -- if the
    low-beta arm writes fewer, the comparison is not like-for-like;
  * the k/n floor printed next to every spread, via `_substrate.check_distinct`
    rather than a hand-rolled mean, so the partial-collapse detector runs.

A beta = 0.05 arm with better distinctness and WORSE retrieval is not a win; it
is the degenerate arm, and it is reported as one.

CACHE. `beta` only became cache-key material in cf6cb0f. Before that both arms
would have shared one pickled backbone and this study would have measured
exactly zero -- see that commit. Nothing here needs to disable the cache now,
but the arms MUST differ in the digest, which the regression tests pin.
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

from _substrate import check_distinct, similarity, mean_ci             # noqa: E402
from core_assembly_duplicates import _clusters, _retrieval_over_pool   # noqa: E402
from parser_grounding_alpha import _features                           # noqa: E402
from neural_assemblies.assembly_calculus.emergent.core.areas import (  # noqa: E402
    NOUN_CORE, ROLE_AGENT, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    train_parser_to_depth,
)

DEPTH = "SENTENCES"
SEEDS = [11, 42, 7]
BETAS = [0.10, 0.05]
ROLE_AREAS = (ROLE_PATIENT, ROLE_AGENT)
CANDIDATES = 6
HI_ALPHA = 0.35


def measure(parser, seed):
    core = parser.core_lexicons.get(NOUN_CORE, {})
    area = parser.brain.areas[NOUN_CORE]
    n, k = int(area.n), int(area.k)
    words = sorted(core)
    asm = {w: np.asarray(core[w].winners, dtype=np.int64) for w in words}

    spread, note = check_distinct(list(asm.values()), n, k, where="NOUN_CORE")
    groups = _clusters(core)
    dup_words = {w for ws in groups.values() if len(ws) > 1 for w in ws}

    feats = {w: _features(parser, w) for w in words}
    hi, lo = [], []
    for a, b in combinations(words, 2):
        fa, fb = feats[a], feats[b]
        if not fa or not fb:
            continue
        alpha = 2 * len(fa & fb) / (len(fa) + len(fb))
        (hi if alpha >= HI_ALPHA else lo).append(similarity(asm[a], asm[b]))

    out = {
        "M_core": len(words),
        "distinct": len(groups) / max(len(words), 1),
        "dup_frac": len(dup_words) / max(len(words), 1),
        "spread": spread,
        "floor": k / n,
        "note": note,
        "hi_overlap": statistics.fmean(hi) if hi else float("nan"),
        "lo_overlap": statistics.fmean(lo) if lo else float("nan"),
        "n_hi": len(hi),
    }
    for ra in ROLE_AREAS:
        rl = parser.role_lexicons.get(ra, {})
        pool = sorted(w for w in rl if w in core)
        out[f"M_{ra}"] = len(pool)
        res = _retrieval_over_pool(parser, ra, pool, np.random.default_rng(seed))
        out[f"ret_{ra}"] = (res[0] / res[1]) if res else float("nan")
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("PHASE A -- beta alone. Substrate law predicts overlap in the")
    print(f"alpha >= {HI_ALPHA} bin should FALL at beta=0.05, with retrieval held.")
    print(f"depth={DEPTH} seeds={SEEDS} betas={BETAS} candidates={CANDIDATES}")
    print()

    results = {b: [] for b in BETAS}
    for seed in SEEDS:
        for beta in BETAS:
            parser = train_parser_to_depth(DEPTH, seed=seed, beta=beta)
            m = measure(parser, seed)
            results[beta].append(m)
            print(f"  seed={seed} beta={beta:<5} M={m['M_core']:<4} "
                  f"distinct={m['distinct']:.3f} dup={m['dup_frac']:.3f} "
                  f"spread={m['spread']:.4f} (floor {m['floor']:.4f}) "
                  f"hi_ov={m['hi_overlap']:.4f} "
                  f"retP={m[f'ret_{ROLE_PATIENT}']:.3f} "
                  f"retA={m[f'ret_{ROLE_AGENT}']:.3f} {m['note']}")
    print()

    fields = [
        ("M_core", "core lexicon size", False),
        ("distinct", "distinct-assembly frac", True),
        ("dup_frac", "duplicated-word frac", False),
        ("spread", "mean spread", False),
        ("hi_overlap", f"overlap, alpha>={HI_ALPHA}", False),
        ("lo_overlap", "overlap, alpha<0.35", False),
        (f"ret_{ROLE_PATIENT}", "ret@6 ROLE_PATIENT", True),
        (f"ret_{ROLE_AGENT}", "ret@6 ROLE_AGENT", True),
    ]
    hdr = (f"{'metric':<26} {'beta=0.10':>18} {'beta=0.05':>18} "
           f"{'delta':>9} {'paired':>8}")
    print(hdr)
    print("-" * len(hdr))
    verdict = {}
    for key, label, higher_better in fields:
        a = [r[key] for r in results[0.10]]
        b = [r[key] for r in results[0.05]]
        ma, ha = mean_ci(a)
        mb, hb = mean_ci(b)
        # PAIRED, same seed: the arms share a seed, so the per-seed difference
        # removes the between-brain variance that swamps a 3-seed unpaired test.
        diffs = [y - x for x, y in zip(a, b)]
        md, hd = mean_ci(diffs)
        sign = "+" if md > 0 else ""
        excl = "yes" if (hd == hd and abs(md) > hd) else "no"
        verdict[key] = (md, excl, higher_better)
        print(f"{label:<26} {ma:>10.4f}+-{ha:<6.4f} {mb:>10.4f}+-{hb:<6.4f} "
              f"{sign}{md:>8.4f} {excl:>8}")

    print()
    print("VERDICT")
    d_hi = verdict["hi_overlap"][0]
    d_dup = verdict["dup_frac"][0]
    d_ret = statistics.fmean([verdict[f"ret_{ra}"][0] for ra in ROLE_AREAS])
    d_m = verdict["M_core"][0]
    if abs(d_m) > 0.5:
        print(f"  ** ARMS NOT LIKE-FOR-LIKE: core lexicon size moved by "
              f"{d_m:+.1f} words. Every other delta is confounded by how much "
              f"got written, not by beta. **")
    if d_hi < 0 and d_ret >= -0.02:
        print("  BETA HELPS. Overlap in the high-alpha bin fell and retrieval")
        print("  held, which is what the substrate law predicts. Adopt beta=0.05")
        print("  as the parser default BEFORE testing the architecture, so #125")
        print("  is measured against the better baseline.")
    elif d_hi < 0 and d_ret < -0.02:
        print("  DEGENERATE ARM. Distinctness improved and RETRIEVAL FELL")
        print(f"  ({d_ret:+.3f}) -- lower beta bought separation by learning")
        print("  less. This is the failure this file was built to catch; do not")
        print("  report the distinctness gain as a win.")
    else:
        print("  NO EFFECT IN THE PREDICTED DIRECTION. The substrate law does")
        print("  not transfer to the parser through beta alone, so the")
        print("  architecture (#125) is the remaining candidate rather than an")
        print("  additional one.")


if __name__ == "__main__":
    main()
