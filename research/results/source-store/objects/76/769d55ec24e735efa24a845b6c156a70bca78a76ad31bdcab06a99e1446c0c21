"""A/B the NumpySparseEngine stim-vector fast path on a full parser training.

Runs an identical TWO_WORD (or deeper) curriculum with
``ASSEMBLIES_STIM_FASTPATH`` off and on and requires the two engines to end up
bit-identical: winners and ever-fired counts per area, every stimulus and area
connectome array under ``np.array_equal``, and all pairwise assembly overlaps.
Also checks that the same seed twice gives the same result.

This is the slow, end-to-end counterpart of
``neural_assemblies/tests/test_stim_fastpath_equivalence.py``, which covers the
same invariants on a synthetic workload fast enough for the unit suite.

Usage::

    python scripts/check_stim_fastpath_equivalence.py
    python scripts/check_stim_fastpath_equivalence.py --depth SENTENCES --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ASSEMBLIES_BACKBONE_CACHE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_assemblies.core.numpy_engine import _sparse as sparse_mod  # noqa: E402
from neural_assemblies.tests.test_stim_fastpath_equivalence import (  # noqa: E402
    overlaps, snapshot,
)


def collect_errors(a, b):
    import numpy as np

    errs = []
    for section in ("areas", "stim_conns", "area_conns"):
        if set(a[section]) != set(b[section]):
            errs.append(f"{section}: key sets differ")
            continue
        for key in sorted(a[section], key=str):
            va, vb = a[section][key], b[section][key]
            if section == "areas":
                for field in ("w", "pool_ptr"):
                    if va[field] != vb[field]:
                        errs.append(f"areas[{key}].{field}: {va[field]} != {vb[field]}")
                if not np.array_equal(va["winners"], vb["winners"]):
                    errs.append(f"areas[{key}].winners differ")
                if va["map"] != vb["map"]:
                    errs.append(f"areas[{key}].compact_to_neuron_id differ")
            elif va.shape != vb.shape:
                errs.append(f"{section}[{key}]: shape {va.shape} != {vb.shape}")
            elif not np.array_equal(va, vb):
                errs.append(f"{section}[{key}]: {int((va != vb).sum())} cells differ")
    oa, ob = overlaps(a), overlaps(b)
    for key in sorted(oa, key=str):
        if oa[key] != ob[key]:
            errs.append(f"overlap{key}: {oa[key]} != {ob[key]}")
    return errs


def train(depth, seed, n, k):
    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        default_holdout_set, train_parser_to_depth,
    )
    t0 = time.perf_counter()
    parser = train_parser_to_depth(
        depth, n=n, k=k, seed=seed,
        holdout_words=default_holdout_set(), fast_training=True,
    )
    return parser.brain._engine, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="TWO_WORD")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("-n", type=int, default=3000)
    ap.add_argument("-k", type=int, default=30)
    args = ap.parse_args()

    train(args.depth, args.seeds[0], args.n, args.k)  # warm lazy imports
    ok = True

    for seed in args.seeds:
        sparse_mod.set_stim_fastpath(False)
        legacy, t_off = train(args.depth, seed, args.n, args.k)
        snap_off = snapshot(legacy)

        sparse_mod.set_stim_fastpath(True)
        fast, t_on = train(args.depth, seed, args.n, args.k)
        snap_on = snapshot(fast)

        errs = collect_errors(snap_off, snap_on)
        ok &= not errs
        print(f"[{'IDENTICAL' if not errs else 'DIVERGED'}] {args.depth} "
              f"seed={seed} n={args.n} k={args.k}  "
              f"off={t_off:.2f}s on={t_on:.2f}s  ({t_off / t_on:.2f}x)")
        for err in errs[:20]:
            print("     ", err)

    sparse_mod.set_stim_fastpath(True)
    again, _ = train(args.depth, args.seeds[0], args.n, args.k)
    errs = collect_errors(snapshot(again), snap_on if args.seeds[-1] == args.seeds[0]
                          else snapshot(train(args.depth, args.seeds[0], args.n, args.k)[0]))
    ok &= not errs
    print(f"[{'DETERMINISTIC' if not errs else 'NONDETERMINISTIC'}] "
          f"{args.depth} seed={args.seeds[0]} run twice")
    for err in errs[:10]:
        print("     ", err)

    print("\nRESULT:", "ALL IDENTICAL" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
