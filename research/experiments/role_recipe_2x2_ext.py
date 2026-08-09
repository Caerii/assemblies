"""#52 sequential extension -- execution of the plan PRE-STATED in
role_recipe_2x2.py: "n=1e5 arms run seeds 42-46 first (runtime), extended
to 51 before ANY conclusion if the interaction is within its CI of zero."

The 5-seed n=1e5 interaction read -0.0056 +/- 0.0154 -- within its CI of
zero -- so this script runs seeds 47-51 at n=1e5 for both gains, merges
into role_recipe_2x2_results.json, and recomputes the analysis block over
the FULL seed sets (n=3000: 42-51 unchanged; n=1e5: 42-51).

No new predictions and no new bars: the registered decision rule in
role_recipe_2x2.py is evaluated on the merged analysis.

Run: python role_recipe_2x2_ext.py   (after role_recipe_2x2.py)
"""
from __future__ import annotations

import json

from role_recipe_2x2 import OUT_PATH, SEEDS_SMALL, cell, mci

SEEDS_LARGE_FULL = list(range(42, 52))
NEW_SEEDS = list(range(47, 52))


def main():
    with open(OUT_PATH, encoding="utf-8") as f:
        prev = json.load(f)
    cells = {k: dict(v) for k, v in prev["cells"].items()}
    ret_rows = dict(prev["ret_rows"])

    for gain in (1.0, 4.0):
        for seed in NEW_SEEDS:
            c = cell(seed, 100000, gain)
            key = f"100000-G{int(gain)}-{seed}"
            ret_rows[key] = c.pop("ret_rows")
            cells[key] = c
            print(f"n=100000 G{int(gain)} seed={seed} "
                  f"parse={c['parse_acc']:.3f} "
                  f"(A {c['parse_active']:.2f}/P {c['parse_passive']:.2f}) "
                  f"ret={round(c['ret_acc'], 3) if c['ret_acc'] is not None else None} "
                  f"(n={c['ret_n']})", flush=True)

    def arm(n, gain, key, seeds):
        return mci([cells[f"{n}-G{int(gain)}-{s}"][key] for s in seeds])

    analysis = {}
    for n, seeds in ((3000, SEEDS_SMALL), (100000, SEEDS_LARGE_FULL)):
        for gain in (1.0, 4.0):
            tag = f"n{n}_G{int(gain)}"
            analysis[f"{tag}_parse"] = arm(n, gain, "parse_acc", seeds)
            analysis[f"{tag}_ret"] = arm(n, gain, "ret_acc", seeds)
            analysis[f"{tag}_gap"] = arm(n, gain, "gap_mean", seeds)
        ds = []
        for s in seeds:
            a = cells[f"{n}-G4-{s}"]["ret_acc"]
            b = cells[f"{n}-G1-{s}"]["ret_acc"]
            if a is not None and b is not None:
                ds.append(a - b)
        analysis[f"interaction_ret_delta_n{n}"] = mci(ds)

    out = {"cells": cells, "ret_rows": ret_rows, "analysis": analysis}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(analysis, indent=2))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
