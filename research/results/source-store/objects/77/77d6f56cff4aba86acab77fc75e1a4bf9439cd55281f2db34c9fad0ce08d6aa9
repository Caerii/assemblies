"""How many bits survive each level of composition? (task #46)

The depth results so far are stated in accuracy and margin, which are properties
of a particular read-out. This restates them in bits, which is not.

TWO STANDARD TOOLS, USED FOR WHAT THEY ACTUALLY GIVE.

  DATA PROCESSING INEQUALITY. A composition chain is a Markov chain
  X -> A_1 -> ... -> A_D, where X is the item identity and A_L the assembly at
  level L: level L+1 is computed from level L and nothing else. Therefore

      I(X; A_1) >= I(X; A_2) >= ... >= I(X; A_D)

  and this is a THEOREM, not a measurement. Information about the item cannot
  be recovered further down a chain, only lost. Any scheme promising to restore
  discriminability at depth by post-processing the deep layer alone is ruled
  out a priori.

  FANO'S INEQUALITY turns the measured retrieval accuracy into a numerical
  floor under the surviving information. With M equiprobable items and error
  probability P_e,

      H(X | A_L) <= h(P_e) + P_e * log2(M - 1)
      I(X; A_L)  >= log2 M - h(P_e) - P_e * log2(M - 1)

  so accuracy alone bounds the bits retained from below.

WHAT THIS IS AND IS NOT. Fano gives a LOWER bound, so a falling bound does not
by itself prove information is falling -- but the DPI already guarantees the
true quantity is non-increasing, so the two together give a monotone quantity
with a measured floor. What the bound cannot do is prove information is still
present when it reads zero: a bound of zero bits means "not demonstrated",
not "none".

WHY BOTHER. Bits are comparable across system sizes, depths and vocabularies in
a way that accuracy is not: retrieval of 0.9 among 32 items and among 512 items
are very different achievements, and the bit count says so. It also converts
"how deep can we go" into a budget -- if each level costs a roughly constant
number of bits, the maximum depth follows by division.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSVS = [x for x in os.environ.get("INFO_CSVS", "sweep.csv").split(",") if x]


def h2(x):
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -(x * math.log2(x) + (1 - x) * math.log2(1 - x))


def fano_bits(acc, m_items):
    """Lower bound on I(X; A) in bits, from accuracy over M equiprobable items."""
    if m_items <= 1:
        return 0.0
    pe = max(0.0, min(1.0, 1.0 - acc))
    bound = math.log2(m_items) - h2(pe) - pe * math.log2(max(m_items - 1, 1))
    return max(0.0, bound)


def load(paths):
    rows = []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(HERE, p)
        if not os.path.exists(full):
            continue
        with open(full, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


if __name__ == "__main__":
    rows = load(CSVS)
    if not rows:
        raise SystemExit(f"no data in {CSVS}")

    # One curve per (cut, n, M, p, gain, depth): bits against level.
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["cut"], int(r["n"]), int(r["M"]), float(r["p"]),
               float(r["gain"]), int(r["depth"]))
        by[key][int(r["level"])].append(float(r["acc"]))

    print("\n  BITS RETAINED PER LEVEL   (Fano lower bound on I(X; A_L))")
    print("  DPI guarantees the true value is non-increasing in level.\n")

    shown = 0
    for key in sorted(by):
        cut, n, M, p, gain, depth = key
        lv = sorted(by[key])
        bits = [fano_bits(statistics.mean(by[key][L]), M) for L in lv]
        if max(bits) <= 0.0:
            continue          # nothing survived even at level 1; not informative
        cap = math.log2(M)

        # NOT a constant rate. Measured, the bound holds essentially FLAT at
        # the full vocabulary capacity for several levels and then falls off a
        # cliff -- e.g. at n=2000, M=96: 6.58, 5.17, 0.04, 0, 0. Reporting an
        # average "bits lost per level" would describe a smooth decay that does
        # not happen, and would imply a budget that divides, which it does not.
        #
        # The honest summary is the USABLE DEPTH: the deepest level still
        # carrying most of the vocabulary. Composition behaves as though it
        # works, then stops.
        usable = 0
        for L, b in zip(lv, bits):
            if b >= 0.5 * cap:
                usable = L
            else:
                break

        print(f"  {cut:<13} n={n:<5} M={M:<4} p={p:<5g} g={gain:<5.2f} "
              f"D={depth}  cap={cap:.1f}b")
        print(f"      bits: " + "  ".join(f"L{L}:{b:5.2f}"
                                          for L, b in zip(lv, bits)))
        print(f"      usable depth {usable} (levels holding >= half of "
              f"{cap:.1f} bits)")
        shown += 1
        if shown >= 12:
            print("\n  (truncated; set INFO_CSVS or filter for more)")
            break

    # A bound that is zero everywhere would make the whole exercise vacuous, so
    # say plainly how often it bound anything at all.
    total = len(by)
    live = sum(1 for k in by
               if max(fano_bits(statistics.mean(by[k][L]), k[2])
                      for L in by[k]) > 0)
    print(f"\n  {live}/{total} configurations retain a demonstrable "
          f"(non-zero) bound at some level.")
