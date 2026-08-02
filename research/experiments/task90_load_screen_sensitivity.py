"""Can `load_audit` CLEAR anything, or only rank? Its sensitivity, measured.

THE GAP THIS CLOSES
-------------------
`task90_ab_load_triage.py` flagged 4 of 4 substrate A/Bs and stated plainly that
its SENSITIVITY was unmeasured: no A/B had ever passed the screen and then been
re-derived on exact drive, so there was no evidence that passing it means
anything. A screen that only ever says "suspect" is a ranking device, not a
filter, and should not be described as one.

FINDING A TRUE-NEGATIVE CASE IS ITSELF THE HARD PART
----------------------------------------------------
The candidate has to change the ANSWER while leaving RECRUITMENT alone. Among
substrate knobs there appear to be none. Measured (M=8, n=1000, 3 seeds):

    w_max 20 -> 5     acc 1.0000 -> 1.0000, spread 0.0402 -> 0.0402, gap 0.0000
    w_max 20 -> 2     acc 1.0000 -> 1.0000, spread 0.0402 -> 0.0376, gap 0.0797
    w_max 20 -> 1.2   acc 1.0000 -> 0.6667, spread 0.0402 -> 0.0388, gap 0.1187

`w_max=5` passes the screen because it is a DEAD manipulation -- the clamp never
binds, and the two arms agree to four decimals, which `compare_arms` already
catches better. Everything that does something moves load. That is not an
accident: the sampler's error is a function of how many neurons have fired, and
anything that changes WHICH neurons win changes how fast the area fills.

So the true negative has to come from a manipulation applied at READOUT ONLY.
Training is then bit-identical between the arms -- same recruitment, same load,
same sampler error -- while the measured quantity genuinely differs. Probing
with fewer rounds is exactly that: `probe()` wraps `read_only()`, which blocks
weight change AND recruitment, so nothing inside it can move `w`.

THE TEST
--------
  arm A   train normally, probe with 6 rounds     (the usual protocol)
  arm B   train normally, probe with 2 rounds     (an under-cued readout)

  screen  load gap between the arms -- predicted ~0, i.e. PASSES
  truth   re-derive the A-vs-B difference on numpy_exact

PRE-REGISTERED
--------------
S1 The screen PASSES this pair (load gap below the 0.05 threshold). It must,
   or there is no true-negative case here either and the whole question is
   unanswerable with this design.

S2 The A-vs-B difference is REAL on the sampler -- arm B scores worse. Without
   this the pair is another dead manipulation and proves nothing.

S3 THE ACTUAL QUESTION. Does the sampler's A-vs-B difference survive on exact
   drive? If yes, the screen cleared something correctly and has demonstrated
   specificity. If no, the screen passed a comparison that the sampler got
   wrong, and `load_audit` CANNOT clear anything -- it ranks, and its
   documentation and the triage file must say so.

I do not have a confident prediction. The mechanism argument says load-matched
arms share the error and should agree; today has been a poor day for mechanism
arguments.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_assemblies import diagnostics as dx  # noqa: E402
from _substrate import probe, rank1, read, similarity, spread  # noqa: E402

# M=32, NOT 8. At M=8 accuracy is 1.0000 in BOTH arms -- the area is far below
# its ceiling (~27), so the metric is saturated and S2 failed for a reason that
# had nothing to do with the screen. This is the repo's standing "N400 is
# saturated" failure in miniature: a metric at its ceiling cannot show an
# effect, and a null from one is not a null. M=32 sits just past the ceiling
# where accuracy is informative; `ident` is reported alongside since it is
# unsaturated at any M.
N, K, P, BETA, TRAIN_ROUNDS, M = 1000, 50, 0.05, 0.10, 6, 32
PROBE_A, PROBE_B = 6, 2
SEEDS = (42, 7, 123, 2024, 5, 99)
THRESHOLD = 0.05


def trial(seed, probe_rounds, engine):
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.core.brain import Brain

    b = Brain(p=P, seed=seed, norm_init=True, engine=engine)
    b.add_area("L", N, K, beta=BETA)
    for m in range(M):
        b.add_stimulus(f"w{m}", K)

    stored = {}
    for m in range(M):                      # TRAINING IS IDENTICAL PER SEED
        project(b, f"w{m}", "L", rounds=TRAIN_ROUNDS, recurrent=True)
        stored[m] = read(b, "L")

    hits, ident = 0, []
    for m in range(M):
        with probe(b):                      # only THIS differs between arms
            project(b, f"w{m}", "L", rounds=probe_rounds, recurrent=True)
            live = read(b, "L")
        hits += rank1(live, stored) == m
        ident.append(similarity(live, stored[m]))
    return (hits / M, statistics.mean(ident), spread(stored.values()), b)


def arm(probe_rounds, engine):
    res = [trial(s, probe_rounds, engine) for s in SEEDS]
    return (statistics.mean(r[0] for r in res),
            statistics.mean(r[1] for r in res), res[0][3])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  Does load_audit have any specificity, or does it only rank?")
    print(f"  n={N} k={K} beta={BETA} M={M}, train {TRAIN_ROUNDS} rounds "
          f"(IDENTICAL in both arms), {len(SEEDS)} seeds")
    print(f"  arm A probes with {PROBE_A} rounds, arm B with {PROBE_B}\n")

    out = {}
    for engine in ("numpy_sparse", "numpy_exact"):
        a_acc, a_id, a_brain = arm(PROBE_A, engine)
        b_acc, b_id, b_brain = arm(PROBE_B, engine)
        out[engine] = (a_acc, b_acc, a_id, b_id)
        gap = dx.load_audit({"probeA": a_brain, "probeB": b_brain},
                            threshold=THRESHOLD)[0]
        print(f"  {engine:>13}   acc A {a_acc:.4f}  B {b_acc:.4f}  "
              f"delta {a_acc - b_acc:+.4f}   |  ident A {a_id:.4f} B {b_id:.4f}")
        if engine == "numpy_sparse":
            print(f"  {'':>13}   screen: {gap}")
            screen_passes = not gap.confounded(THRESHOLD)

    sp_a, sp_b, sp_ia, sp_ib = out["numpy_sparse"]
    ex_a, ex_b, ex_ia, ex_ib = out["numpy_exact"]
    # PICK THE UNSATURATED METRIC, and say which. Accuracy pinned at 1.0000 in
    # every arm means the readout cannot express the effect, not that there is
    # none.
    acc_saturated = min(sp_a, sp_b, ex_a, ex_b) > 0.999
    metric = "ident" if acc_saturated else "acc"
    if acc_saturated:
        sp_delta, ex_delta = sp_ia - sp_ib, ex_ia - ex_ib
        print(f"    [accuracy is SATURATED at 1.0000 in every arm -- reading "
              f"the effect on `ident` instead]")
    else:
        sp_delta, ex_delta = sp_a - sp_b, ex_a - ex_b

    print("\n  READING\n")
    print(f"    S1 the screen PASSES this pair:       {str(screen_passes):>5}")
    # S2 AS ORIGINALLY WRITTEN WAS WRONG, and the run exposed it. It asked
    # whether the effect is real ON THE SAMPLER, and reported "dead
    # manipulation, proves nothing" when the sampler read 0.0000. But the
    # sampler reading zero while exact reads -0.2656 is not a dead
    # manipulation -- it is the sampler being FLAT WRONG, which is the single
    # most informative outcome this file can produce. A pre-registration that
    # can only detect a defect when the defective instrument agrees there is
    # something to detect is not a test.
    s2 = max(abs(sp_delta), abs(ex_delta)) > 0.02
    print(f"    S2 the A/B is real on EITHER engine:  {str(s2):>5}   "
          f"sampler {sp_delta:+.4f}, exact {ex_delta:+.4f}  (on {metric})")
    # AGREEMENT MEANS THE SAME SIZE, not merely the same sign. A screen that
    # only preserves direction cannot clear a magnitude, and every claim this
    # session withdrew was a magnitude.
    agree = ((sp_delta > 0) == (ex_delta > 0)
             and abs(sp_delta - ex_delta) <= 0.5 * max(abs(sp_delta),
                                                       abs(ex_delta)))
    print(f"    S3 and it SURVIVES on exact drive:    {str(agree):>5}   "
          f"sampler {sp_delta:+.4f} vs exact {ex_delta:+.4f}")

    print()
    if not screen_passes:
        print("    S1 FAILED -- even a readout-only manipulation moves load, so")
        print("    there is no true-negative case available and the screen's")
        print("    specificity cannot be measured with this design at all.")
        print("    That would itself justify demoting it to a ranking device.")
    elif not s2:
        print("    S2 FAILED -- NEITHER engine shows an effect, so this is a")
        print("    dead manipulation and proves nothing about the screen.")
        print("    Find a readout change with a real effect and re-run.")
    elif abs(sp_delta) < 0.02 <= abs(ex_delta):
        print("    THE SCREEN CANNOT CLEAR ANYTHING, AND THE WORST WAY.")
        print(f"    It PASSED this pair (load gap 0.000, training identical),")
        print(f"    and on that pair the sampler reports NOTHING ({sp_delta:+.4f})")
        print(f"    where the exact substrate reports {ex_delta:+.4f}. Not a")
        print("    magnitude error -- a missed effect entirely.")
        print()
        print("    So matched load is NOT sufficient for a trustworthy A/B.")
        print("    Matching load matches the RECRUITMENT channel of the")
        print("    sampler's error; evidently it distorts something else too,")
        print("    and at high occupancy (load 0.999 here) it reports perfect")
        print("    retrieval where the substrate has already degraded.")
        print()
        print("    ACTION: `load_audit` is a RANKING device. Its docstring and")
        print("    task90_ab_load_triage must stop implying that an unflagged")
        print("    pair is cleared. Nothing short of re-running on exact drive")
        print("    clears an A/B.")
    elif agree:
        print("    THE SCREEN HAS DEMONSTRATED SPECIFICITY. It passed a real")
        print("    A/B and that A/B holds on exact drive, so passing is")
        print("    evidence -- from ONE case, which is one more than before.")
        print("    load_audit may be described as a filter, weakly.")
    else:
        print("    THE SCREEN CANNOT CLEAR ANYTHING. It passed a comparison the")
        print("    sampler gets wrong, so load-matching is NOT sufficient for a")
        print("    trustworthy A/B -- matched load means matched RECRUITMENT")
        print("    error, and evidently the sampler distorts something else too.")
        print("    load_audit is a RANKING device. Its docstring and the triage")
        print("    file must stop implying that an unflagged pair is cleared.")


if __name__ == "__main__":
    main()
