"""End-to-end verification of the #49 fix, on the cell that actually failed.

THE FAILURE. During a real ladder run the runtime self-check fired at n=4000
M=512 D=3: the serial reference produced 512 distinct assemblies at margin
6.34, while the spawned worker produced 1, margin 1.00, spread 0.9998. Bisection
localised it to `_resolve_module` loading a `__main__` worker's file under a
SYNTHETIC name, creating a second module object for the same source. Resolving
by the module's real name instead was verified to fix the mechanism -- but the
cell that collapsed was never re-run, so the fix was never verified END TO END.
This does that.

WHY THIS FILE IS A SCRIPT. The fault only occurs when the worker function lives
in `__main__`, which is true of every experiment here and false under pytest.
`cell` is therefore defined below, in this module, and this file must be RUN,
not imported. That is the same reason test_parallel_seeds_main exists.

WHY IT ALSO RUNS A NEGATIVE CONTROL. A regression test that cannot fail proves
nothing, and this project has already shipped one: test_parallel_seeds_main
passed with the bug deliberately reintroduced, because its n=400 M=6 cell is
far too small to show a failure that needs M=512. So this is run TWICE --

    SUBSTRATE_LEGACY_LOADER=0   the fix    -> must AGREE
    SUBSTRATE_LEGACY_LOADER=1   the bug    -> must DIVERGE

and only the pair is meaningful. An "agree" from the first arm alone would be
indistinguishable from a test with no power.

Run both arms:  python verify_49.py            (fix arm)
                SUBSTRATE_LEGACY_LOADER=1 python verify_49.py   (control arm)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("LADDER_BETA", "0.20")
os.environ["SUBSTRATE_VERIFY_PARALLEL"] = "1"   # the check under test

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402
from _substrate import parallel_seeds  # noqa: E402

N, M, D = 4000, 512, 3
SEEDS = (42, 7)


def cell(n, m_items, depth, seed):
    """Worker. Lives in __main__ ON PURPOSE -- that is the failing config."""
    return ladder.trial(n, m_items, depth, seed)


def brief(r):
    distinct, total, margins, spreads = r
    return (f"distinct={distinct} total={total} "
            f"marg={ {k: round(v, 2) for k, v in margins.items()} } "
            f"spr={ {k: round(v, 4) for k, v in spreads.items()} }")


if __name__ == "__main__":
    legacy = os.environ.get("SUBSTRATE_LEGACY_LOADER") == "1"
    arm = "CONTROL (legacy loader, bug present)" if legacy else "FIX"
    print(f"\n  verify #49 -- {arm}")
    print(f"  n={N} M={M} D={D} beta={ladder.BETA} seeds={SEEDS}")
    print(f"  worker module: {cell.__module__}\n")

    diverged, err = False, ""
    try:
        out = parallel_seeds(cell, SEEDS, N, M, D)
        print(f"    parallel[0]  {brief(out[0])}")
    except AssertionError as exc:
        diverged, err = True, str(exc)
        print(f"    DIVERGED: {str(exc)[:220]}")

    if legacy:
        ok = diverged
        print(f"\n  control arm expected DIVERGENCE: "
              f"{'yes -- the test has power' if ok else 'NO -- TEST IS BLIND'}")
    else:
        ok = not diverged
        print(f"\n  fix arm expected AGREEMENT: "
              f"{'yes -- parallel == serial' if ok else 'NO -- STILL BROKEN'}")

    sys.exit(0 if ok else 1)
