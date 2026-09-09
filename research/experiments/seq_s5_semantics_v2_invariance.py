"""V-S5: are the S5 findings invariant under drive semantics v2?

Implements the science-invariance bar of
`research/notes/substrate/PREREG_drive_semantics_v2.md`. The registered S5 study's
trained arms re-run with ASSEMBLIES_VIRTUAL_WEIGHTS=1 (same builds, seeds,
words); the bar is that the four REGISTERED VERDICTS are unchanged. Per-seed
exact flags may differ in isolated cells -- ulp ties are real -- and every
flip is printed.
"""
from __future__ import annotations

import json
import os
import sys

os.environ["ASSEMBLIES_VIRTUAL_WEIGHTS"] = "1"     # inherited by the pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from seq_s5_word_problem import (  # noqa: E402
    GROUP_NAMES, LENGTHS, SEEDS, run_tiered, worker,
)

REGISTERED = os.path.join(_HERE, "seq_s5_word_problem_results.json")


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    with open(REGISTERED, encoding="utf-8") as fh:
        reg = json.load(fh)

    print("=== V-S5: semantics v2 invariance ===")
    print("    trained arms, gate ON, judged against registered verdicts\n")
    r = run_tiered([(g, s, "trained") for g in GROUP_NAMES for s in seeds],
                   budget_gb=20.0)

    flips = 0
    table = {}
    for g in GROUP_NAMES:
        row = {}
        for L in LENGTHS:
            ok = sum(bool(r[(g, s, "trained")][str(L)]) for s in seeds)
            reg_ok = round(reg["curve"][g][str(L)][0] * len(seeds))
            row[str(L)] = ok
            if ok != reg_ok:
                flips += abs(ok - reg_ok)
                print(f"    CELL MOVED: {g} L={L}: {reg_ok}/10 -> {ok}/10")
        table[g] = row
        print(f"    {g:7s} " + "  ".join(
            f"L={L}:{row[str(L)]:2d}/10" for L in LENGTHS), flush=True)

    a5_100 = sum(bool(r[("A5", s, "trained")]["100"]) for s in seeds)
    s1 = a5_100 >= 9
    s2 = all(table[g]["500"] >= table[g]["10"] for g in GROUP_NAMES)
    solv = [table[g]["100"] for g in GROUP_NAMES if g in ("Z60", "A4xZ5")]
    hard = [table[g]["100"] for g in GROUP_NAMES if g in ("A5", "S5")]
    s3 = min(hard) >= min(solv) - 2          # the registered form's spirit:
    # non-solvable not systematically below solvable at L=100.
    print("\n=== BARS (registered verdicts: S1 PASS, S2 FAIL, S3 PASS) ===")
    v = {"S1": s1, "S2": s2, "S3": s3}
    reg_v = {"S1": True, "S2": False, "S3": True}
    invariant = all(v[k] == reg_v[k] for k in v)
    for k in v:
        mark = "UNCHANGED" if v[k] == reg_v[k] else "FLIPPED"
        print(f"  {k}: registered {reg_v[k]}, v2 {v[k]}  [{mark}]")
    print(f"\n  cells moved: {flips}")
    print(f"  V-S5 {'PASS -- findings invariant' if invariant else 'FAIL'}")

    out = {"seeds": seeds, "table": table, "verdicts": v,
           "cells_moved": flips, "invariant": invariant}
    path = os.path.join(_HERE, "seq_s5_semantics_v2_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
