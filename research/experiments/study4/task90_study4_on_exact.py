"""#90: study IV's H1 (0.7566 -> 0.1756) re-derived on exact drive.

WHY THIS FILE EXISTS AT ALL, AND THE FIRST FINDING
--------------------------------------------------
`research/notes/PREREG_gated_recurrence.md` records study IV's results. Its
HARNESS was never committed -- `ntp.py` and `ntp_ctx.py` lived only in a session
scratchpad, so the numbers in that note could not be reproduced from the
repository by anyone, including me. They are now in `research/experiments/
study4/` unchanged except for an `engine` parameter, which is the whole reason
this re-derivation is possible.

THE CLAIM UNDER TEST
--------------------
H1: freezing the CONTEXT self-fiber (beta_rec = 0) takes cross-prefix CONTEXT
overlap from **0.7566 to 0.1756**, a 4.3x distinctness gain. H2 then found that
this bought NO measurable prediction (C - B = +0.0106, CI covers zero), which is
the [[distinctness-is-not-information]] result.

Two reasons that 4.3x is suspect, both established today:

1. `graded_similarity_and_sampler_load.md` predicted the frozen arm's 0.1756 is
   inflated -- at comparable load the sampler reads ~6.5x chance where exact
   reads 1.8x -- so the true value is plausibly near chance, which would make
   arm C an even purer prefix hash and STRENGTHEN the H2 diagnosis.
2. Plastic-vs-frozen changes which neurons win, hence recruitment, hence load.
   That is exactly the shape of A/B whose magnitude did not survive for
   norm_init (8.0x -> 1.0x). And `task90_load_screen_sensitivity.py` showed
   the load screen cannot clear such a pair anyway.

WHAT IS MEASURED, AND WHAT IS NOT
---------------------------------
H1 only: the CONTEXT cross-prefix overlap for arms B (plastic) and C (frozen),
on both engines. The MRR arms (H2/H3) cost ~10x more and are left for a
follow-up -- named here rather than quietly skipped, because the H2 conclusion
depends on the MRR difference and this file does NOT re-derive it.

PRE-REGISTERED
--------------
D1 Both arms read LOWER on exact than on the sampler. Follows from the load
   measurement, and is the boring outcome.
D2 The 4.3x RATIO does not survive intact. This is the claim H1 actually
   makes, and the norm_init precedent says ratios across a load-moving
   manipulation are what breaks.
D3 The DIRECTION survives -- frozen is still more distinct than plastic. If
   even this fails, H1 is not a finding about the substrate at all and the
   whole of study IV needs redoing rather than rescaling.

If D3 holds and D2 fails, the honest summary is: freezing does separate the
prefixes, and "4.3x" was the instrument.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ntp_ctx  # noqa: E402

SEEDS = tuple(range(42, 52))
ENGINES = ("numpy_sparse", "numpy_exact")
BETA = 0.10
CHANCE = 100 / 5000.0        # k/n for the CONTEXT area in ntp.py -- printed,
                             # and corrected below from the real constants.


def overlap_arm(engine, rec_beta):
    vals = []
    for s in SEEDS:
        _mrr, ov = ntp_ctx.run(s, beta=BETA, ctx_probe=True,
                               rec_beta=rec_beta, engine=engine)
        vals.append(float(ov))
    return statistics.mean(vals), vals


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import ntp
    chance = ntp.K / ntp.N

    print(f"\n  #90 -- study IV H1, cross-prefix CONTEXT overlap, on exact drive")
    print(f"  n={ntp.N} k={ntp.K} beta={BETA}, {len(SEEDS)} seeds "
          f"(42..{SEEDS[-1]}), chance k/n = {chance:.4f}")
    print(f"  recorded on the sampler: plastic 0.7566, frozen 0.1756 "
          f"(4.3x distinctness)\n")

    out = {}
    print(f"  {'engine':>13} {'plastic (B)':>13} {'frozen (C)':>12} "
          f"{'ratio B/C':>11}")
    for e in ENGINES:
        b_mean, _ = overlap_arm(e, None)
        c_mean, _ = overlap_arm(e, 0.0)
        out[e] = (b_mean, c_mean)
        print(f"  {e:>13} {b_mean:>13.4f} {c_mean:>12.4f} "
              f"{(b_mean / c_mean if c_mean else float('nan')):>11.2f}x",
              flush=True)

    sb, sc = out["numpy_sparse"]
    eb, ec = out["numpy_exact"]
    print("\n  READING\n")

    d1 = eb < sb and ec < sc
    print(f"    D1 both arms lower on exact:          {str(d1):>5}   "
          f"B {sb:.4f}->{eb:.4f}, C {sc:.4f}->{ec:.4f}")
    r_sp = sb / sc if sc else float("nan")
    r_ex = eb / ec if ec else float("nan")
    d2 = abs(r_ex - r_sp) > 0.25 * r_sp
    print(f"    D2 the distinctness RATIO moves:      {str(d2):>5}   "
          f"{r_sp:.2f}x -> {r_ex:.2f}x")
    d3 = ec < eb
    print(f"    D3 frozen is still more distinct:     {str(d3):>5}   "
          f"{ec:.4f} vs {eb:.4f} on exact")
    print(f"\n    arm C against chance: sampler {sc / chance:.1f}x, "
          f"exact {ec / chance:.1f}x  (chance {chance:.4f})")

    print()
    if d3 and d2:
        print("    H1 HOLDS IN DIRECTION, NOT IN SIZE. Freezing the self-fiber")
        print("    does separate the prefixes on the exact substrate, and the")
        print(f"    '4.3x' is the instrument: it reads {r_ex:.2f}x here.")
        print("    Same shape as norm_init's 8.0x -> 1.0x. The H2 conclusion")
        print("    (distinctness bought no prediction) is UNAFFECTED by this")
        print("    file, because H2 is a difference of MRRs and is not")
        print("    re-derived here -- that is the follow-up.")
    elif d3:
        print("    H1 SURVIVES INTACT, ratio and direction. The load-moving")
        print("    objection does not bite here, which is a genuine negative")
        print("    result about the screen's reach and worth recording.")
    else:
        print("    D3 FAILED -- on exact drive frozen is NOT more distinct than")
        print("    plastic. H1 is then not a finding about the substrate, and")
        print("    study IV needs redoing rather than rescaling. Check first")
        print("    that the frozen arm's fiber is actually open and carrying")
        print("    drive on this engine (the original H0 gate) before")
        print("    believing it.")


if __name__ == "__main__":
    main()
