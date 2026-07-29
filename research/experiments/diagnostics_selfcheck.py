"""Does `neural_assemblies.diagnostics` actually catch the four known failures?

A diagnostic module written FROM a set of bugs must be validated AGAINST them,
or it only encodes the belief that they were understood. Each case below
reconstructs a failure that really happened on 2026-07-28/29 and asserts that
the corresponding verdict fires -- and, just as importantly, that it does NOT
fire on the healthy control, since a check that always warns is noise.

    1. COLLAPSE     recurrent lexicon build at M=64, n=1000. Measured: rank-1
                    0.0312, pairwise overlap 0.2392 vs floor 0.0500.
    2. DEAD PROBE   reading an unmaterialised area under read_only(). Measured:
                    accuracy EXACTLY chance, margin EXACTLY 1.00.
    3. STABILITY    the same collapsed area re-cues at identity 0.7763 while
                    rank-1 is 0.0143 -- healthy-looking and useless.
    4. DRIVE SHARE  multi-mood word order: the conditioning input controls ~4%
                    of the drive, which is why five interventions failed.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from neural_assemblies.diagnostics import (  # noqa: E402
    area_health, drive_breakdown, format_report, read_assembly,
    recurrence_audit,
)

N, K, P, BETA, ROUNDS = 1000, 50, 0.05, 0.10, 6
AREA = "L"


def make(brain_seed, m_words, recurrent):
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=brain_seed)
    brain.add_area(AREA, N, K, beta=BETA)
    for m in range(m_words):
        brain.add_stimulus(f"w{m}", K)

    def drive(m):
        if recurrent:
            project(brain, f"w{m}", AREA, rounds=ROUNDS, recurrent=True)
        else:
            for _ in range(ROUNDS):
                brain.project({f"w{m}": [AREA]}, {})

    stored = {}
    for m in range(m_words):
        drive(m)
        stored[m] = read_assembly(brain, AREA)
    cues = {m: (lambda mm=m: drive(mm)) for m in range(m_words)}
    return brain, stored, cues


def has(health, label):
    return any(v.label == label and not v.ok for v in health.verdicts)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    passed = []

    print("\n  1. COLLAPSE  (recurrent build, M=64) vs CONTROL (feed-forward)")
    bad_brain, bad_stored, bad_cues = make(42, 64, recurrent=True)
    bad = area_health(bad_brain, AREA, bad_stored, bad_cues)
    good_brain, good_stored, good_cues = make(42, 64, recurrent=False)
    good = area_health(good_brain, AREA, good_stored, good_cues)
    print(f"     collapsed: spread {bad.spread:.4f}  acc {bad.accuracy:.4f}  "
          f"flagged={bad.collapsed}")
    print(f"     healthy:   spread {good.spread:.4f}  acc {good.accuracy:.4f}  "
          f"flagged={good.collapsed}")
    passed.append(("collapse detected", bad.collapsed))
    passed.append(("no false alarm on healthy", not good.collapsed))

    print("\n  3. STABILITY MISTAKEN FOR DISCRIMINABILITY (same collapsed area)")
    print(f"     identity {bad.identity:.4f} vs rank-1 {bad.accuracy:.4f}  "
          f"flagged={has(bad, 'stability != identity')}")
    passed.append(("stability trap detected",
                   has(bad, "stability != identity")))

    print("\n  2. DEAD PROBE  (read an unmaterialised area under read_only)")
    from neural_assemblies.core.brain import Brain
    b = Brain(p=P, seed=7)
    b.add_area("EMPTY", N, K, beta=BETA)
    b.add_area("SRC", N, K, beta=BETA)
    for m in range(8):
        b.add_stimulus(f"s{m}", K)
    src_stored = {}
    for m in range(8):
        for _ in range(ROUNDS):
            b.project({f"s{m}": ["SRC"]}, {})
        src_stored[m] = read_assembly(b, "SRC")
    dead_stored, dead_cues = {}, {}
    with b.read_only():
        for m in range(8):
            b.project({}, {"SRC": ["EMPTY"]})
            dead_stored[m] = read_assembly(b, "EMPTY")
        for m in range(8):
            dead_cues[m] = (lambda mm=m: b.project({}, {"SRC": ["EMPTY"]}))
        dead = area_health(b, "EMPTY", dead_stored, dead_cues)
    print(f"     acc {dead.accuracy:.4f} (chance {1/8:.4f})  "
          f"margin {dead.margin:.2f}  flagged={not dead.trustworthy}")
    passed.append(("dead probe detected", not dead.trustworthy))

    print("\n  4. DRIVE SHARE  (a weak source cannot decide the k-WTA)")
    d = drive_breakdown(bad_brain, AREA, [AREA], expect_controlling=AREA)
    print(format_report([d]))
    passed.append(("drive breakdown returns a share",
                   d.share(AREA) == d.share(AREA)))

    print("\n  5. RECURRENCE AUDIT")
    aud_bad = [v for v in recurrence_audit(bad_brain, [AREA]) if not v.ok]
    aud_good = [v for v in recurrence_audit(good_brain, [AREA]) if not v.ok]
    print(f"     recurrent build flagged: {bool(aud_bad)}"
          + (f"  {aud_bad[0].detail}" if aud_bad else ""))
    print(f"     feed-forward flagged:    {bool(aud_good)}")
    passed.append(("self-fiber flagged on recurrent arm", bool(aud_bad)))
    passed.append(("self-fiber quiet on feed-forward", not aud_good))

    print("\n  REPORT on the collapsed area")
    print(format_report([bad]))

    print("\n  SELF-CHECK")
    for name, ok in passed:
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {sum(1 for _, ok in passed if ok)}/{len(passed)} checks pass")
    if not all(ok for _, ok in passed):
        sys.exit(1)


if __name__ == "__main__":
    main()
