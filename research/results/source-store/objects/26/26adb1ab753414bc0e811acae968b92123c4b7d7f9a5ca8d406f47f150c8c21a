"""Is `trial` self-contained, or does it depend on process history? (task #49)

This is the hypothesis left standing after everything else was measured away.

`parallel_seeds` validates itself by running one seed serially IN THE PARENT
and comparing against the worker's result. That comparison is only meaningful
if `trial(...)` is a pure function of its arguments. Neither the ladder nor
`_substrate` reseeds numpy or `random` per trial, and this project has already
been bitten by a GLOBAL RNG LEAK making `Brain(seed=)` non-reproducible.

If trial is history-dependent then by the time the ladder reaches its tenth
cell the parent has run thousands of trials while each worker is comparatively
fresh -- so serial and parallel would diverge for a reason that has nothing to
do with multiprocessing, and the self-check would be blaming the wrong thing.

THE TEST. Run seed 42, then a different seed, then seed 42 again, all in one
process. A pure function returns the same thing both times.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("LADDER_BETA", "0.20")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402

N = int(os.environ.get("SC_N", "4000"))
M = int(os.environ.get("SC_M", "64"))
D = int(os.environ.get("SC_D", "3"))


def brief(r) -> str:
    distinct, total, margins, spreads = r
    return (f"distinct={distinct} "
            f"marg={ {k: round(v, 3) for k, v in margins.items()} } "
            f"spr={ {k: round(v, 5) for k, v in spreads.items()} }")


if __name__ == "__main__":
    print(f"\n  trial self-containment  (n={N} M={M} D={D} "
          f"beta={ladder.BETA})\n")

    first = ladder.trial(N, M, D, 42)
    print(f"    seed 42 (fresh process)   {brief(first)}")

    other = ladder.trial(N, M, D, 43)
    print(f"    seed 43 (intervening)     {brief(other)}")

    again = ladder.trial(N, M, D, 42)
    print(f"    seed 42 (after history)   {brief(again)}")

    pure = repr(first) == repr(again)
    print(f"\n    trial is a pure function of its arguments: {pure}")
    if not pure:
        print("    => the parallel/serial self-check is CONFOUNDED: it "
              "compares a\n       history-laden parent against a fresh "
              "worker, so it can report\n       divergence with no "
              "multiprocessing fault at all.")
    else:
        print("    => trial is reproducible in-process; process history is "
              "NOT the\n       explanation and the divergence is real.")
