"""What does the file-path-loaded __main__ copy actually contain? (task #49)

Established by bisection: with the ladder imported as an ORDINARY module,
`parallel_seeds` matches serial bit for bit at the cell that fails (n=4000
M=512 D=3). Force the `__main__` path and it collapses. So the fault is in
`_resolve_module`'s file-path load, not in `trial`, not in scale, not in
process state, and not in the PYTHONHASHSEED asymmetry.

`_resolve_module` re-EXECUTES the script's top level under a synthetic name.
Any module-level constant derived from the environment, from argv, or from
anything the parent computed at run time is therefore RECOMPUTED in the child
rather than inherited. If it recomputes differently, the worker runs the same
code with different constants -- which is exactly "plausible numbers, silently
wrong".

This script IS the experiment: it is run as `__main__`, so its own globals go
through the same loader the ladder's do.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Mirror the ladder's own module-level, environment-derived constants.
BETA = float(os.environ.get("LADDER_BETA", "0.10"))
GATED = dict(parent_self=False, target_self=False, back_project=False)
ARGV_TAIL = sys.argv[1:]

from _substrate import parallel_seeds  # noqa: E402


def report(tag, seed):
    """Runs in the worker. Reports the constants the worker actually holds."""
    return {
        "tag": tag,
        "seed": seed,
        "BETA": BETA,
        "GATED": dict(GATED),
        "ARGV_TAIL": list(ARGV_TAIL),
        "__name__": __name__,
        "__file__": os.path.basename(__file__),
        "env_LADDER_BETA": os.environ.get("LADDER_BETA"),
        "sys_path0": os.path.basename(sys.path[0]) if sys.path[0] else "",
    }


if __name__ == "__main__":
    print(f"\n  __main__ loader state probe   (parent BETA={BETA})\n")

    parent = report("x", 42)
    got = parallel_seeds(report, (42, 43), "x")
    child = got[0]

    width = max(len(k) for k in parent)
    ndiff = 0
    for k in parent:
        same = parent[k] == child[k]
        ndiff += not same
        print(f"  {'   ' if same else '<<<'} {k:<{width}}  "
              f"parent={parent[k]!r:<26} child={child[k]!r}")
    print(f"\n  {ndiff} field(s) differ between parent and file-path-loaded "
          f"child")
