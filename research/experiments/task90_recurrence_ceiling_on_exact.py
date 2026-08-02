"""#90: does the norm_init recurrence ceiling survive an engine that does not sample?

WHAT IS BEING RE-DERIVED
------------------------
`core/brain.py` carries a table, quoted verbatim in a comment that is the stated
justification for keeping `recurrent_projection` off:

    M           2      4      8     16     32     64    128
    rec RAW  1.000  1.000  0.667  0.188  0.031  0.016  0.009
    rec norm 1.000  1.000  1.000  1.000  1.000  0.039  0.018
    ff  norm 1.000  1.000  1.000  1.000  1.000  1.000  1.000

and a conclusion drawn from it -- "the ceiling scales with n (M=32 / 64 / 256 at
n=1000 / 2000 / 4000), which is what identifies it as accumulated potentiation
rather than degree bias."

Every one of those numbers was measured on `numpy_sparse`, which invents a drive
for neurons that have never fired. That approximation was since measured to be
LOAD-DEPENDENT: it merges fully disjoint inputs at 18x chance when few neurons
have fired, and the error decays as the area fills
(`research/notes/graded_similarity_and_sampler_load.md`).

A ceiling that "scales with n" is exactly the shape a load-dependent instrument
would manufacture, because larger n at fixed k means lower load at every M. So
the scaling claim cannot be read off an instrument whose error moves with the
same variable. This file re-runs the identical protocol on `numpy_exact`, which
computes the drive rather than sampling it.

PRE-REGISTERED
--------------
H1 The two engines DISAGREE about `spread` under recurrence. Stated before
   running the sweep, on the strength of one seed at M=2 (sparse 0.00, exact
   0.86). If they agree, the sampler is not implicated here and the table stands.

H2 The disagreement is DIRECTIONAL: the sampler reports MORE distinctness than
   the substrate has. Mechanism, if so: each new item's k-WTA draws from a pool
   of never-fired neurons whose drive is invented, which is a recruitment
   channel the real substrate does not offer -- so the sampler hands every new
   item fresh territory and manufactures separation.

H3 `acc` (rank-1 identity) is much less affected than `spread`. If assemblies
   merge to 0.90 overlap and are STILL individually retrievable, then the
   ceiling as DEFINED (acc > 0.90) can survive a collapse that `spread` sees --
   distinctness and information are different quantities
   ([[distinctness-is-not-information]]).

H4 If H1-H3 hold, the n-scaling of the ceiling is not established by the data
   behind it. That is a statement about what the evidence supports, NOT a claim
   that the ceiling does not scale -- deciding that needs the n sweep, which is
   `task90_ceiling_n_scaling.py`.

WHAT WOULD MAKE THIS FILE WRONG
-------------------------------
* A DEAD PROBE: `acc` at chance while `ident` reads ~1.00 means every item
  re-cues to the same frozen assembly and nothing is being measured. Flagged
  per row rather than left for the reader to spot.
* `spread` at the floor with low `acc` is STARVATION, not distinctness; `spread`
  rising with `acc` holding is CROWDING. The two look alike in one column and
  are read here as a pair ([[spread-blind-to-partial-collapse]]).

PREREGISTRATION PROVENANCE -- H1/H2 WERE FORMED ON A BROKEN ENGINE
------------------------------------------------------------------
Recorded rather than quietly rewritten, because the hypotheses above are not
worth much if their origin is hidden.

The one-seed observation that motivated H2 (sparse spread 0.00, exact 0.86 at
M=2) was a DEFECT IN THE EXACT ENGINE, not a property of the substrate.
`_stim_norm` divided a stimulus fiber's base count by its own in-degree, which
is that count -- giving a drive of exactly 1.0 at every neuron, so k-WTA fell
through to the index tie-break and every stimulus elected the same k neurons.

The first full sweep "confirmed" H1 and H2 at max gap 0.8701 and would have
been reported as a large result. What refused was not a hypothesis test but the
CONTROL: feed-forward with no recurrence also read spread 0.89, and a control
that cannot collapse had collapsed. The `rec norm` and `rec RAW` rows were also
identical to four decimals, which is what a parameter that is not reaching the
engine looks like -- and it was not: `Brain` forwards `norm_init` only when
True, so this engine's `True` default silently ignored `norm_init=False`.

Both are fixed and both are now pinned by tests. The lesson is the cheap one:
the control was worth more than the contrast.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import MIN_TRIALS  # noqa: E402
from norm_init_recurrence_limit import (ARMS, BETA, K_, LABEL, N, P_,  # noqa: E402
                                        SEEDS, trial)

#: Capped below the original sweep's 128. The exact engine's cost grows with the
#: number of stored outer products, so M=128 x 6 seeds x 3 arms is hours; the
#: ceiling the comment claims (M=32) sits inside this range, so the range covers
#: the claim under test. Truncation is stated rather than silent.
M_SWEEP = (2, 4, 8, 16, 32, 64)
ENGINES = ("numpy_sparse", "numpy_exact")


def run(m_words, recurrent, norm_init, engine):
    res = [trial(m_words, recurrent, norm_init, s, engine) for s in SEEDS]
    tot = sum(x[1] for x in res)
    return (sum(x[0] for x in res) / tot,           # acc
            statistics.mean(x[2] for x in res),     # ident
            statistics.mean(x[3] for x in res),     # spread
            tot)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    floor = K_ / N

    print(f"\n  #90 -- the brain.py recurrence table, re-derived on exact drive")
    print(f"  n={N} k={K_} beta={BETA} p={P_}, one area, {len(SEEDS)} seeds")
    print(f"  chance overlap (floor) = {floor:.4f}\n")
    print(f"  {'M':>4} {'arm':>9} | {'acc':>7}{'ident':>8}{'spread':>8}"
          f" | {'acc':>7}{'ident':>8}{'spread':>8} |")
    print(f"  {'':>4} {'':>9} | {'--- numpy_sparse ---':^23}"
          f" | {'--- numpy_exact ----':^23} |")

    table = {}
    for m_words in M_SWEEP:
        for name, rec, norm in ARMS:
            row = {e: run(m_words, rec, norm, e) for e in ENGINES}
            for e, r in row.items():
                table[(m_words, name, norm, e)] = r
            flags = []
            for e, r in row.items():
                if r[0] <= 1.5 / m_words and r[1] > 0.95:
                    flags.append(f"DEAD PROBE on {e}")
                if r[3] < MIN_TRIALS:
                    flags.append(f"UNDER-POWERED on {e}")
            cells = "".join(
                f" | {row[e][0]:>7.4f}{row[e][1]:>8.4f}{row[e][2]:>8.4f}"
                for e in ENGINES)
            print(f"  {m_words:>4} {LABEL[(name, norm)]:>9}{cells} |"
                  + ("  <-- " + "; ".join(flags) if flags else ""))
        print()

    def ceiling(name, norm, engine):
        ok = [m for m in M_SWEEP
              if table[(m, name, norm, engine)][0] > 0.90]
        return max(ok) if ok else 0

    print("  READING\n")

    def signed_gap(name, norm, col=2):
        """exact - sparse. Positive = the SAMPLER reports the more distinct."""
        return [table[(m, name, norm, "numpy_exact")][col]
                - table[(m, name, norm, "numpy_sparse")][col] for m in M_SWEEP]

    # H1/H2 are read on BOTH recurrent arms, not just `rec norm`. Reading one
    # arm would have reported a small consistent bias and missed that the bias
    # REVERSES between them, which is the finding.
    for name, rec, norm in ARMS:
        g = signed_gap(name, norm)
        print(f"    spread, exact - sparse, {LABEL[(name, norm)]:>9}:  "
              + " ".join(f"{s:+.3f}" for s in g))
    print(f"    {'':>34}" + "".join(f"{('M=' + str(m)):>7}" for m in M_SWEEP))

    rec_norm, rec_raw = signed_gap("rec", True), signed_gap("rec", False)
    h1 = max(abs(s) for s in rec_norm + rec_raw) > 0.10
    h2 = all(s > 0 for s in rec_norm + rec_raw)
    reverses = (max(rec_norm) > 0) and (min(rec_raw) < 0)

    acc_gap = max(abs(s) for s in signed_gap("rec", True, col=0))
    h3 = acc_gap < max(abs(s) for s in rec_norm)

    print()
    print(f"    H1 engines disagree on spread:        {str(h1):>5}")
    print(f"    H2 sampler over-reports distinctness: {str(h2):>5}"
          + ("   (it does on rec norm and NOT on rec RAW -- the sign REVERSES)"
             if reverses else ""))
    print(f"    H3 acc moves less than spread:        {str(h3):>5}   "
          f"max acc gap {acc_gap:.4f} vs max spread gap "
          f"{max(abs(s) for s in rec_norm):.4f}")

    print()
    for name, rec, norm in ARMS:
        cs = {e: ceiling(name, norm, e) for e in ENGINES}
        print(f"    ceiling({LABEL[(name, norm)]:>9}, acc>0.90):  "
              + "   ".join(f"{e} M={cs[e]}" for e in ENGINES))

    # THE claim the brain.py comment actually rests on: that norm_init BUYS
    # capacity under recurrence. Measured as a ratio of ceilings, per engine.
    print()
    for e in ENGINES:
        c_norm, c_raw = ceiling("rec", True, e), ceiling("rec", False, e)
        gain = (c_norm / c_raw) if c_raw else float("inf")
        print(f"    norm_init's capacity gain under recurrence on {e:>12}: "
              f"M={c_raw} -> M={c_norm}  ({gain:.1f}x)")

    print()
    if reverses:
        print("    THE SAMPLER'S ERROR CHANGES SIGN BETWEEN THE TWO ARMS.")
        print("    With norm_init on it reports slightly MORE distinctness than")
        print("    the substrate has; with norm_init off it reports dramatically")
        print("    LESS. So it is not a bias that a constant correction removes.")
        print()
        print("    THAT BREAKS THE STANDING REASSURANCE. The graded-similarity")
        print("    note argued that absolute overlaps were suspect but PAIRED")
        print("    comparisons survived, since both arms share the engine and the")
        print("    load. This pairing is norm_init on vs off -- and it does not")
        print("    survive: the sampler's own error is what differs between the")
        print("    arms, because norm_init changes which neurons win, hence")
        print("    recruitment, hence load, hence the size of the sampler's")
        print("    invention. A/B is only safe when the manipulation does not")
        print("    move load. norm_init moves it.")
    elif h1:
        print("    The engines disagree, but the sign does not reverse. The")
        print("    sampler is implicated as a consistent bias here, which is the")
        print("    correctable case -- state the direction and carry on.")
    else:
        print("    The engines AGREE. The sampler is not implicated in this")
        print("    protocol and the brain.py table stands as measured.")


if __name__ == "__main__":
    main()
