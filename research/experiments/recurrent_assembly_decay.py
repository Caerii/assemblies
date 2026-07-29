"""Why does a RECURRENTLY-trained assembly dissolve under self-projection?

WHAT PROVOKED THIS
------------------
`ops.project` was routing through `Brain.project_rounds`, whose fast path drops
target self-recurrence unless `recurrent_projection` is set (it defaults False).
So every default-built assembly was STIMULUS-ONLY -- not an assembly in the
defining sense (Dabagia et al. 2024: k neurons whose INTERNAL weights are
strengthened). Fixing `ops.project` to run its own documented protocol breaks
exactly eight tests, and they are not thresholds drifting. They are the
attractor tests, and the numbers are categorical:

    test_self_projection_converges_to_fixed_point
        consecutive-step overlap 0.53 -> 0.37 -> 0.14 -> 0.00 -> 0.01 -> 0.00
    test_training_strengthens_attractor
        1 round of training -> 1.000 stability;  10 rounds -> 0.020
    test_self_projection_preserves_trained_assembly     overlap 0.010
    test_autonomous_step_method                         overlap 0.000

An assembly trained WITH recurrence does not survive its own recurrence, and
MORE recurrent training makes it strictly worse. That inverts the defining
property, so something real is wrong -- either in the fix, in `norm_init`, or in
the substrate -- and which one it is cannot be settled by argument.

THE THREE CANDIDATES, and what separates them
----------------------------------------------
  H1 DEGREE BIAS / HUBS. `project_rounds` documents this exact failure for the
     recurrent path: with weights initialized at 1, drive is essentially the
     count of active afferents, so k-cap elects the random graph's high-in-
     degree hubs, and recurrence compounds it (measured there: winners' in-degree
     z-score +1.66 by round 15). `norm_init` was written to remove it. If H1 is
     live DESPITE norm_init, then norm_init is not doing its job once Hebbian
     potentiation has broken the normalization it applied once at init.
     SIGNATURE: winners' in-degree z-score climbs, and successive steps converge
     onto a COMMON set (the hubs) rather than churning.

  H2 RUNAWAY POTENTIATION. Recurrent training multiplies A->A weights inside the
     assembly. If they grow without bound, the drive distribution becomes
     dominated by a few synapses and the top-k flips between steps.
     SIGNATURE: churn -- consecutive overlap near zero AND no common attractor,
     with A->A weight range exploding relative to its init.

  H3 RECRUITMENT. Under lazy materialization a neuron gets a row only once it
     wins. If self-projection keeps recruiting NEW neurons, the assembly is
     rebuilt from fresh cells every step and cannot persist.
     SIGNATURE: `area.w` grows on every autonomous step.

H1 and H2 make OPPOSITE predictions about whether the churn has a fixed point,
which is why the trajectory is recorded against BOTH the original assembly and
the previous step. H3 is read straight off `w`.

CONTROLS, without which none of the above is interpretable
-----------------------------------------------------------
  build   stim-only (the old, silently-non-recurrent behaviour) vs recurrent
          (the protocol). The stim-only arm is what made these tests pass, so it
          is the reference the failures must be read against.
  norm    norm_init on vs off. `norm_init` is the stated prerequisite for safe
          self-recurrence, so if the decay is identical with it on and off, it
          is not protecting anything here and H1's premise is wrong.

Chance overlap is k/n and is printed, because "overlap 0.010" means nothing
without it.

RESULT (2026-07-28): H3 (RECRUITMENT), AND IT RESOLVES TO A QUANTITATIVE LAW
-----------------------------------------------------------------------------
At the failing tests' own regime, n=10000 k=100 p=0.05 beta=0.1, 3 seeds:

    build=stim-only  either norm   1.00 flat                  <- why they passed
    build=recurrent  norm=True     0.65->0.00, w 462->1051, z +1.22->+0.74
    build=recurrent  norm=False    0.99 flat,  w 330->338,  z +1.45 stable

Overlap with the PREVIOUS step decays too, so it is CHURN, not drift. `w` more
than doubles: H3. Winner in-degree z FALLS, refuting H1; A->A max/mean 2.3,
refuting H2.

Instrumenting the winner selection itself (intercepting `_select_winner_indices`
and splitting the drive array at `tgt.w`) shows the mechanism directly:

    norm=False  mat_top_k 25.7 28.1 30.9 34.1 37.6  RISING; cand_top_k 11.6 flat
    norm=True   mat_top_k .0279 .0261 .0243 .0231 .0208 FALLING; cand .0229 flat

Recruitment begins at step 0, BEFORE the means cross, because top-k draws from
the COMBINED array: the materialized distribution is wide (max .0446, top-k mean
.0279) while the candidate band is tight (.0229-.0256), so the assembly's
weakest third sinks below it. Under norm=False the ENTIRE assembly (25.7) is
above every candidate (max 13.0) and nothing is displaced.

WHY. Without normalization the winners are the high-in-degree neurons -- that is
why k-cap elected them -- so their raw drive is ~10 rather than the typical ~5,
and potentiation multiplies that. `norm_init` divides exactly that degree
advantage out, leaving potentiation alone: 1.1^10 = 2.59 on a typical draw of 5
gives ~13, against a candidate pool whose top order statistic over ~10^4 fresh
neurons is ~15. Dead heat.

THE LAW, and it is predictive rather than descriptive. Stability is governed by
`(1+beta)^T`, with beta and T NOT mattering separately:

    (1+beta)^T   2.6    6.2    6.7   13.8
    overlap     0.150  0.973  0.997  0.983
                       ^beta=.2,T=10  ^beta=.1,T=20 -- same value, same result

and the threshold RISES with n, as a larger candidate pool must. At
(1+beta)^T = 2.6: overlap 0.630 at n=2000, 0.150 at n=10000, 0.007 at n=50000;
at 8.1 all three are >= 0.99. That n-scaling is the prediction the extreme-value
account makes and the hub account does not, which is what promotes this from a
consistent story to a tested one.

CONSEQUENCE. beta=0.1 with 10 rounds gives 2.59 and is BELOW threshold. That was
the fixture in `test_assembly_calculus.py` and `test_autonomous_recurrence.py`,
and it is why eight attractor tests failed the moment `ops.project` stopped
silently dropping self-recurrence. Raised to 20 rounds (6.7), which is
independently correct: [PNAS20] sec 2 gives O(log n) and log2(10000) = 13.3, so
10 was already short of the paper's own bound.

ENGINE FIX ABLATED. `_norm_candidate_divisor` was dividing candidates by the
population mean `n*p` while `_norm_scale` divides materialized neurons by their
own in-degree. Replacing it with the same estimator, `d = drive + p*(n-active)`,
shifts the stability curve left at every point in and above the transition
(0.567->0.717, 0.863->0.920, 0.953->0.973, 0.987->0.997) and is noise below it.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

N, K, P, BETA = 1000, 100, 0.05, 0.10
SEEDS = (42, 7, 123)
TRAIN_ROUNDS = 10
STEPS = 8


def indegree_z(brain, area_name, winners):
    """z-score of the winners' recurrent in-degree against the whole area.

    H1's signature. Reads the A->A block directly: if k-cap is electing hubs,
    the winners' incoming weight sum sits far above the area mean.
    """
    engine = brain._engine_for(brain.areas[area_name])
    conn = getattr(engine, "_area_conns", {}).get(area_name, {}).get(area_name)
    w = getattr(conn, "weights", None)
    if w is None or getattr(w, "shape", (0, 0))[0] == 0:
        return float("nan")
    w = np.asarray(w.todense() if hasattr(w, "todense") else w)
    deg = w.sum(axis=0)
    cols = [int(j) for j in winners if int(j) < len(deg)]
    if not cols or deg.std() == 0:
        return float("nan")
    return float((deg[cols].mean() - deg.mean()) / deg.std())


def weight_range(brain, area_name):
    """max/mean of the A->A block. H2's signature if it explodes."""
    engine = brain._engine_for(brain.areas[area_name])
    conn = getattr(engine, "_area_conns", {}).get(area_name, {}).get(area_name)
    w = getattr(conn, "weights", None)
    if w is None or getattr(w, "shape", (0, 0))[0] == 0:
        return float("nan")
    w = np.asarray(w.todense() if hasattr(w, "todense") else w)
    nz = w[w > 0]
    return float(nz.max() / nz.mean()) if nz.size else float("nan")


def trial(build, norm, seed):
    from neural_assemblies.assembly_calculus.assembly import overlap
    from neural_assemblies.assembly_calculus.ops import _snap
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed, norm_init=norm)
    brain.add_area("A", N, K, beta=BETA)
    brain.add_stimulus("s", K)

    # Build. The two arms differ ONLY in whether the target's self-recurrence is
    # open during training -- which is precisely the `ops.project` change.
    brain.project({"s": ["A"]}, {})
    for _ in range(TRAIN_ROUNDS - 1):
        brain.project({"s": ["A"]}, {"A": ["A"]} if build == "recurrent" else {})
    trained = _snap(brain, "A")

    w0 = brain.areas["A"].w
    z0 = indegree_z(brain, "A", brain.areas["A"].winners)

    # Autonomous: stimulus removed, self-projection only. This is the operation
    # the failing tests assert an assembly must survive.
    vs_orig, vs_prev, ws = [], [], []
    prev = trained
    for _ in range(STEPS):
        brain.project({}, {"A": ["A"]})
        cur = _snap(brain, "A")
        vs_orig.append(overlap(cur, trained))
        vs_prev.append(overlap(cur, prev))
        ws.append(brain.areas["A"].w)
        prev = cur

    return dict(vs_orig=vs_orig, vs_prev=vs_prev, w0=w0, ws=ws,
                z0=z0, z1=indegree_z(brain, "A", brain.areas["A"].winners),
                wr=weight_range(brain, "A"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chance = K / N
    print(f"\n  n={N} k={K} p={P} beta={BETA}, train={TRAIN_ROUNDS} rounds, "
          f"{STEPS} autonomous steps, {len(SEEDS)} seeds")
    print(f"  chance overlap k/n = {chance:.3f}\n")

    for build in ("stim-only", "recurrent"):
        for norm in (True, False):
            res = [trial(build, norm, s) for s in SEEDS]
            orig = [statistics.mean(r["vs_orig"][i] for r in res)
                    for i in range(STEPS)]
            prev = [statistics.mean(r["vs_prev"][i] for r in res)
                    for i in range(STEPS)]
            w0 = statistics.mean(r["w0"] for r in res)
            wend = statistics.mean(r["ws"][-1] for r in res)
            z0 = statistics.mean(r["z0"] for r in res)
            z1 = statistics.mean(r["z1"] for r in res)
            wr = statistics.mean(r["wr"] for r in res)
            print(f"  build={build:<10} norm_init={str(norm):<5}")
            print(f"    vs ORIGINAL  " + " ".join(f"{v:.2f}" for v in orig))
            print(f"    vs PREVIOUS  " + " ".join(f"{v:.2f}" for v in prev))
            print(f"    w {w0:.0f} -> {wend:.0f}   in-degree z {z0:+.2f} -> "
                  f"{z1:+.2f}   A->A max/mean {wr:.1f}")

    print("\n  READING")
    print("   vs PREVIOUS stays HIGH while vs ORIGINAL falls -> the assembly")
    print("     DRIFTS to another fixed point. Recurrence works; the identity")
    print("     of the attractor is what moves.")
    print("   BOTH near chance -> CHURN, no fixed point at all (H2).")
    print("   in-degree z climbing with vs PREVIOUS high -> HUBS (H1); if that")
    print("     happens with norm_init=True, norm_init is not holding once")
    print("     Hebbian potentiation has broken the one-time normalization.")
    print("   w growing every step -> RECRUITMENT (H3): the assembly is rebuilt")
    print("     from fresh cells each step and cannot persist.")


if __name__ == "__main__":
    main()
