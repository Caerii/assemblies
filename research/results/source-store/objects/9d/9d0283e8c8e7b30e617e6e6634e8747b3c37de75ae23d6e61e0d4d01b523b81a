"""Which green tests are actually coin flips?

WHY
---
`test_associate_creates_shared_response` had been green since it was written and
was not measuring what it claimed. It asserted one seed's overlap against a
3x-chance threshold; measured over twelve seeds, that threshold sat essentially
AT the mean and the test passed a little over half the time. It only went red
when an unrelated change re-rolled which seed landed low, and it took a sweep to
tell "regression" from "this was never established".

That is a CLASS, not an instance. Any test that fixes a seed and asserts a
threshold is reporting the seed unless the effect clears the threshold by a wide
margin. The tests most exposed are exactly the ones that encode scientific
claims -- conformance to the Assembly Calculus primitives -- because those are
the numbers a paper would cite.

WHAT THIS DOES
--------------
Re-runs a target test file once per seed by overriding the module-level SEED,
and reports the pass rate per test. Interpretation:

    12/12   the effect clears its threshold reliably. Fine.
    7/12    the assertion is near the effect's mean. The test reports the seed.
            EITHER the claim is weaker than the assertion implies (fix the
            claim) OR the assertion is badly stated (fix the test) -- but the
            single-seed version cannot tell you which, and that is the point.
    0/12    genuinely broken, and the one green seed was luck.

It does NOT decide which fix is right. It tells you where a decision is owed.

RESULT (2026-07-28), 29 tests x 12 seeds
-----------------------------------------
    0 of 29 seed-dependent -- every conformance test passed at every seed.

Believed only because the harness was checked against a known coin flip first.
`neural_assemblies/tests/test_zz_audit_control.py` contains one assertion that
is ~50/50 by construction and one that always holds; the audit scores them
10/12 and 12/12. Without that control, "0 of 29" is indistinguishable from a
harness where the seed override never reached the module -- which is exactly
the silent failure this file exists to find, one level up.

Note test_ac_conformance.py already sweeps its own 5 seeds internally, so it
was never as exposed as test_assembly_calculus.py, which fixed SEED = 42.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, Sequence

SEEDS = (42, 7, 123, 2024, 5, 99, 314, 1618, 271, 8, 55, 2718)

#: Files whose tests encode conformance claims about the primitives. These are
#: the ones whose numbers would end up in a paper.
TARGETS = (
    "neural_assemblies/tests/test_assembly_calculus.py",
    "neural_assemblies/tests/test_ac_conformance.py",
)


def run_one(path: str, seed: int) -> Dict[str, str]:
    """{test_name: 'passed'|'failed'} for one file at one seed.

    The seed arrives through ASSEMBLIES_AUDIT_SEED, which the target file reads
    as the default for its module-level SEED. Rewriting the file per seed was
    the alternative and is worse: the audit's result would then depend on the
    audit having cleaned up after itself.

    PYTHONHASHSEED is pinned so that per-process hash randomisation is not
    confounded with the seed being swept. Those were once the same bug here --
    hash()-derived RNG seeds made Brain(seed=) differ across processes -- and an
    audit that mixed them would blame the wrong one.
    """
    env = dict(os.environ, ASSEMBLIES_AUDIT_SEED=str(seed),
               PYTHONHASHSEED="0")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q", "--no-header",
         "-p", "no:cacheprovider", "-rA", "--tb=no"],
        capture_output=True, text=True, env=env,
    )
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR)\s+(\S+)", line.strip())
        if m:
            out[m.group(2)] = m.group(1).lower()
    return out


def audit(targets: Sequence[str] = TARGETS,
          seeds: Sequence[int] = SEEDS) -> Dict[str, Dict[int, str]]:
    results: Dict[str, Dict[int, str]] = defaultdict(dict)
    for path in targets:
        for seed in seeds:
            for name, status in run_one(path, seed).items():
                results[name][seed] = status
    return results


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seeds = SEEDS
    results = audit(seeds=seeds)

    rows = []
    for name, per_seed in results.items():
        passed = sum(1 for s in per_seed.values() if s == "passed")
        rows.append((passed / max(len(per_seed), 1), passed, len(per_seed), name))
    rows.sort()

    print(f"\n  {len(rows)} tests x {len(seeds)} seeds\n")
    print(f"  {'pass rate':>10}  test")
    flagged = 0
    for rate, passed, total, name in rows:
        if rate >= 1.0:
            continue
        flagged += 1
        print(f"  {passed:>4}/{total:<5}  {name}")
    if not flagged:
        print("       all   every test passed at every seed")
    print(f"\n  {flagged} of {len(rows)} tests are seed-dependent.")
    print("  A rate below 1.0 does not mean the test is wrong -- it means the")
    print("  single-seed version was reporting the seed, and someone owes a")
    print("  decision about whether the claim or the assertion is the problem.")


if __name__ == "__main__":
    main()
