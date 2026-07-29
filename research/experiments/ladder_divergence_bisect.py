"""Bisect the ladder's parallel/serial divergence (task #49).

Ruled out so far, each by direct measurement rather than argument:

  * PROCESS STATE -- engine class, module, norm_init, projection fidelity,
    numpy version and resulting winners are all identical parent vs child.
  * SCALE -- a spawned child building M=512 stimuli into one shared area gets
    512 distinct assemblies, exactly like the parent.
  * PYTHONHASHSEED ASYMMETRY -- `parallel_seeds` sets it in the parent after
    startup, so children inherit 0 while the parent keeps a random seed.
    Reproducing that asymmetry changes nothing.

What is left is the ladder's own `trial`, reached through `_resolve_module`.
This script runs the SAME failing cell twice: once with the ladder imported as
an ordinary module, once with its `__main__` path forced. If only the second
diverges, the loader is at fault; if both do, the loader is exonerated and the
cause is inside `trial`.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("LADDER_BETA", "0.20")
# The self-check is what we are studying; let it report rather than abort.
os.environ["SUBSTRATE_VERIFY_PARALLEL"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402
from _substrate import parallel_seeds  # noqa: E402

N, M, D = 4000, 512, 3
SEEDS = (42, 43)


def summarise(r) -> str:
    """The first element of a trial result is {level: distinct count}."""
    try:
        distinct, total, margins, spreads = r
        return (f"distinct={distinct} total={total} "
                f"marg={ {k: round(v, 2) for k, v in margins.items()} } "
                f"spr={ {k: round(v, 4) for k, v in spreads.items()} }")
    except Exception:  # noqa: BLE001
        return repr(r)[:160]


if __name__ == "__main__":
    print(f"\n  ladder divergence bisect  (n={N} M={M} D={D} "
          f"beta={ladder.BETA})\n")
    print(f"    ladder module     : {ladder.__name__}")
    print(f"    trial.__module__  : {ladder.trial.__module__}\n")

    serial = ladder.trial(N, M, D, SEEDS[0])
    print(f"    SERIAL            {summarise(serial)}")

    par = parallel_seeds(ladder.trial, SEEDS, N, M, D)
    print(f"    PARALLEL (module) {summarise(par[0])}")

    agree = repr(serial) == repr(par[0])
    print(f"\n    imported-as-module agrees with serial: {agree}")
    if agree:
        print("    => the loader is implicated: divergence needs __main__.")
    else:
        print("    => loader EXONERATED: parallel diverges even from an "
              "ordinary module, so the cause is inside trial/parallel_seeds.")
