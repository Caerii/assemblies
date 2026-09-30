"""parallel_seeds must be correct when called from a SCRIPT, not just a module.

WHY THIS FILE EXISTS. `parallel_seeds` was added with an equivalence check that
compared parallel against serial and reported IDENTICAL. It passed, and the
helper was still broken: the check imported the experiment as a normal module,
and the bug only appears when the worker function lives in `__main__` -- which
is the case for every experiment in this directory, because they are all run as
scripts. The worker resolved `fn.__module__ == "__main__"` with
`importlib.import_module`, which under spawn returns the CHILD's own main
module, so it ran something else entirely. It did not raise. It returned
plausible numbers: a ladder cell that gives accuracy 1.0000 and spread 0.0059
when run directly came back as 0.0020 and 0.9999 through the pool.

So the test verifies the configuration that is actually used. Run directly:

    python research/experiments/test_parallel_seeds_main.py

It is a script rather than a pytest module ON PURPOSE -- under pytest this file
would be imported, `__name__` would not be `"__main__"`, and the test would
once again exercise only the safe path. `run_checks()` is importable for a
pytest wrapper, but the meaningful assertion is the one below it.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from _substrate import (  # noqa: E402
    assert_distinct, parallel_seeds, read, spread,
)

N, K, P, BETA, M = 400, 20, 0.05, 0.10, 6


def build_cell(n, m_items, seed):
    """A small but real trial: distinct stimuli into one shared area."""
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    brain.add_area("L", n, K, beta=BETA)
    for i in range(m_items):
        brain.add_stimulus(f"w{i}", K)
    stored = []
    for i in range(m_items):
        for _ in range(4):
            brain.project({f"w{i}": ["L"]}, {})
        stored.append(read(brain, "L"))
    # Return something order- and value-sensitive enough that a worker running
    # different code cannot match it by luck.
    return [sorted(int(x) for x in a) for a in stored]


def run_checks() -> list:
    problems = []
    seeds = (11, 22, 33)

    serial = [build_cell(N, M, s) for s in seeds]
    par = parallel_seeds(build_cell, seeds, N, M)

    if serial != par:
        problems.append(
            "parallel_seeds does NOT match serial when the worker function "
            "lives in __main__ -- the workers are not running this module")

    # The invariant helper must fire on a genuinely collapsed set and stay
    # quiet on a healthy one, or it is decoration.
    try:
        assert_distinct([a for a in serial[0]], N, K, where="selftest")
    except AssertionError:
        problems.append("assert_distinct fired on a HEALTHY set (false alarm)")

    collapsed = [list(range(K))] * M
    try:
        assert_distinct(collapsed, N, K, where="selftest")
        problems.append("assert_distinct did NOT fire on a collapsed set")
    except AssertionError:
        pass
    return problems


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  parallel_seeds from __main__  (n={N} k={K} M={M})")
    found = run_checks()
    print(f"    serial == parallel:            {'FAIL' not in str(found)}")
    for p in found:
        print(f"    PROBLEM: {p}")
    if found:
        print(f"\n  {len(found)} PROBLEM(S)")
        sys.exit(1)
    print("\n  all checks pass: workers run this module, and the collapse "
          "invariant fires on collapse and only on collapse")
