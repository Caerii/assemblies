"""Where on the complexity ladder does this substrate stop representing?

WHY THIS LADDER AND NOT THE CHOMSKY ONE
----------------------------------------
`universality_composition.py` established that composition is a property of the
substrate rather than of language. The natural next question is what CLASS of
structure the substrate can hold, and the honest way to ask it is to climb from
the bottom until something breaks.

The Chomsky hierarchy is the famous ladder and it is the wrong tool here: its
bottom rung (regular) is enormous, and a system that fails on regular languages
has failed somewhere much earlier without saying where. The SUBREGULAR hierarchy
refines exactly that region, and each of its classes has a canonical minimal
language and a crisp statement of the one capability it adds:

    rung  class            language                        needs
    0     SL_1 (CONTROL)   last symbol is `a`              recency only
    1     SL_2             no `ab` substring               a local window
    2     SP_2             no `a` ... later `b`            unbounded "seen an a"
    3     LTT              at least two `a`s               counting to a threshold
    4     Star-free        every `a` later followed by `b` FO[<], no modular count
    5     Regular \\ SF     even number of `a`s (parity)     MODULAR counting
    6     Context-free     Dyck-1, balanced brackets       an unbounded counter
    7     Context-sensitive a^n b^n c^n                    two coupled counters

Rung 0 is a POSITIVE CONTROL, not a result. It is decodable from the most
recent input alone, so any system whose state retains anything at all must pass
it. If rung 0 fails, the harness is broken and every rung above it is
uninterpretable -- the same role `test_zz_audit_control.py` plays for the
lucky-seed audit, and the reason that audit's "0 of 29" was believable.

WHAT IS ACTUALLY MEASURED
-------------------------
Not "does the parser classify", which would measure the parser. Symbols are
projected one at a time into SEQ, which drives a recurrent STATE area; after the
string, STATE's assembly is snapshotted. A nearest-centroid readout built from
TRAINING finals then classifies held-out strings.

So this is a REPRESENTATIONAL PROBE: does the recurrent assembly state encode
the distinction, linearly recoverable? That is the right form of the question
for a substrate. It is deliberately generous -- the readout is handed to the
system rather than learned by it -- which means a FAILURE is strong evidence
(the information is not there at all) and a success is weaker (the information
is there and might still not be usable downstream).

Run at the substrate. Not through EmergentParser, for the reason
`grounding_lesion.py` measured: categories there come from three hand-authored
routes, so anything routed through it would be testing the annotation.

THE TWO CONTROLS THAT DECIDE WHETHER A STALL MEANS ANYTHING
------------------------------------------------------------
A stall at rung X admits two readings and they need separating:

  CAPACITY  the brain is too small to hold the states this rung needs. Then
            accuracy rises with n. Swept over n to test.

  CLASS     the mechanism cannot represent this kind of structure at all. Then
            accuracy is flat at chance no matter how large n gets.

and LENGTH GENERALIZATION separates finite-state from genuinely unbounded
memory: train on short strings, test on longer ones. A finite-state solution
transfers in length for regular languages and must fail for context-free ones,
because the counter it did not build cannot be extrapolated. A system that
scores well in-distribution and collapses out-of-length has memorised, not
represented.

Chance is 0.500 by construction -- every rung's test set is balanced -- and is
printed on every row so "above chance" is never taken on trust.

PRE-REGISTERED PREDICTIONS
--------------------------
C0 rung 0 passes well above chance. Otherwise stop; nothing else is readable.

C1 rungs 1-4 pass. These need only bounded memory of bounded-length windows,
   thresholds, or a monotone "have I seen it" flag, and an assembly with
   recurrence is a large finite-state machine, which is more than enough.

C2 rung 5 (PARITY) is the first real risk. It needs exact modular counting: the
   state must flip on every `a` and never drift. Assemblies under k-WTA are
   attractors, and attractors resist flipping. Predicted FAIL or marginal, and
   predicted to stay failed as n grows -- because parity needs 2 states, not
   many, so if it were a capacity problem more neurons would be irrelevant.
   Parity failing while rung 4 passes would be the sharpest result available
   here: it is the exact boundary between star-free and regular.

C3 rungs 6-7 fail, and fail WORSE out-of-length than in-distribution. Both need
   an unbounded counter; a bounded assembly state cannot have one. In-length
   success with out-of-length collapse is the signature of memorisation and is
   the outcome to expect, not clean failure at both.

C4 The stall is a CLASS boundary, not capacity: whichever rung breaks stays
   broken across the n sweep. If instead accuracy climbs with n, the honest
   headline is "undersized", not "cannot represent", and this file's framing is
   wrong.

Recorded before the first run.
"""

from __future__ import annotations

import os
import random
import statistics
import sys
from typing import Callable, Dict, List, Sequence, Tuple

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

K = 50
P = 0.05
BETA = 0.10
SEEDS = (42, 7)
N_SWEEP = (500, 1000, 2000)

SEQ, STATE = "SEQ", "STATE"
START = "start"

#: Rounds of recurrence per symbol. Kept at 1 for the reason
#: `universality_composition.py` had to learn the hard way: recurrence into a
#: shared area merges everything that passes through it. At high round counts
#: every final state collapses onto one assembly and every rung reads at chance
#: -- which would look exactly like "the substrate cannot represent anything"
#: and would be an artifact. `state_spread` below is the guard.
ROUNDS_PER_SYMBOL = 1

TRAIN_PER_CLASS = 60
TEST_PER_CLASS = 30
SHORT = (4, 10)
LONG = (14, 22)


# --------------------------------------------------------------------------
# The ladder. Each entry: (name, class, alphabet, membership predicate)
# --------------------------------------------------------------------------

def _dyck_ok(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += 1 if ch == "a" else -1
        if depth < 0:
            return False
    return depth == 0


def _abc_ok(s: str) -> bool:
    i = 0
    while i < len(s) and s[i] == "a":
        i += 1
    na = i
    while i < len(s) and s[i] == "b":
        i += 1
    nb = i - na
    while i < len(s) and s[i] == "c":
        i += 1
    nc = i - na - nb
    return i == len(s) and na == nb == nc and na > 0


LADDER: List[Tuple[int, str, str, str, Callable[[str], bool]]] = [
    (0, "SL_1 (CONTROL)", "last symbol is a", "ab",
     lambda s: bool(s) and s[-1] == "a"),
    (1, "SL_2", "no `ab` substring", "ab",
     lambda s: "ab" not in s),
    (2, "SP_2", "no a ... later b", "ab",
     lambda s: not any(s[i] == "a" and "b" in s[i + 1:] for i in range(len(s)))),
    (3, "LTT", "at least two a", "ab",
     lambda s: s.count("a") >= 2),
    (4, "Star-free", "every a later followed by b", "ab",
     lambda s: all("b" in s[i + 1:] for i, ch in enumerate(s) if ch == "a")),
    (5, "Regular \\ SF", "PARITY: even count of a", "ab",
     lambda s: s.count("a") % 2 == 0),
    (6, "Context-free", "Dyck-1 balanced", "ab", _dyck_ok),
    (7, "Context-sensitive", "a^n b^n c^n", "abc", _abc_ok),
]


def sample_strings(rung, lo: int, hi: int, per_class: int, rng: random.Random):
    """Balanced positive/negative sample, so chance is exactly 0.500.

    Rejection sampling over uniform random strings finds no positives at all for
    the deep rungs -- a random string is essentially never balanced Dyck and
    never a^n b^n c^n -- so positives for those are CONSTRUCTED and negatives
    are perturbations of them. Negatives being near-misses rather than random
    junk makes those rungs HARDER, not easier, which is the right direction for
    a claim about failure.
    """
    _, _, _, alpha, ok = rung
    pos: List[str] = []
    neg: List[str] = []

    for _ in range(per_class * 400):
        if len(pos) >= per_class and len(neg) >= per_class:
            break
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(lo, hi)))
        bucket = pos if ok(s) else neg
        if len(bucket) < per_class:
            bucket.append(s)

    # Constructive fill for rungs where positives are vanishingly rare.
    guard = 0
    while len(pos) < per_class and guard < per_class * 200:
        guard += 1
        if rung[0] == 6:
            n = rng.randint(max(1, lo // 2), max(1, hi // 2))
            s = "a" * n + "b" * n
            if rng.random() < 0.5:  # nesting, not just one flat block
                m = rng.randint(1, max(1, n - 1))
                s = "a" * m + "b" * m + "a" * (n - m) + "b" * (n - m)
        elif rung[0] == 7:
            n = rng.randint(max(1, lo // 3), max(1, hi // 3))
            s = "a" * n + "b" * n + "c" * n
        else:
            break
        if ok(s) and lo <= len(s) <= hi:
            pos.append(s)

    while len(neg) < per_class and pos:
        base = list(rng.choice(pos))
        i = rng.randrange(len(base))
        base[i] = rng.choice([c for c in rung[3] if c != base[i]])
        s = "".join(base)
        if not ok(s) and lo <= len(s) <= hi:
            neg.append(s)

    m = min(len(pos), len(neg), per_class)
    return [(s, True) for s in pos[:m]] + [(s, False) for s in neg[:m]]


# --------------------------------------------------------------------------

def build(n: int, seed: int, alphabet: str):
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    brain.add_area(SEQ, n, K, beta=BETA)
    brain.add_area(STATE, n, K, beta=BETA)
    brain.add_stimulus(START, K)
    for ch in alphabet:
        brain.add_stimulus(f"sym_{ch}", K)
    return brain


def encode(brain, s: str) -> np.ndarray:
    """Run the string through SEQ -> STATE with recurrence; return final STATE.

    STATE is reset by driving a dedicated START stimulus rather than by poking
    `winners` directly: a start state the dynamics actually settle into is a
    real initial condition, whereas an emptied winner array is a state the
    system can never otherwise be in.
    """
    brain.project({START: [STATE]}, {})
    for ch in s:
        brain.project({f"sym_{ch}": [SEQ]}, {SEQ: [STATE], STATE: [STATE]})
        for _ in range(ROUNDS_PER_SYMBOL - 1):
            brain.project({}, {SEQ: [STATE], STATE: [STATE]})
    return np.array(brain.areas[STATE].winners, dtype=np.int64)


def centroid(finals: Sequence[np.ndarray], k: int) -> set:
    """The k most frequently firing neurons across a class's final states."""
    counts: Dict[int, int] = {}
    for f in finals:
        for x in f:
            counts[int(x)] = counts.get(int(x), 0) + 1
    return set(sorted(counts, key=lambda x: -counts[x])[:k])


def state_spread(finals: Sequence[np.ndarray]) -> float:
    """Mean pairwise overlap of final states -- the collapse guard.

    Near 1.0 means every string ended in the same assembly, so the readout has
    nothing to read and EVERY rung would score at chance. That is an artifact of
    too much recurrence, not a fact about the complexity class, and it is the
    failure mode this whole file would otherwise silently report as a result.
    """
    if len(finals) < 2:
        return float("nan")
    vals = []
    for i in range(min(len(finals), 30)):
        for j in range(i + 1, min(len(finals), 30)):
            a, b = set(int(x) for x in finals[i]), set(int(x) for x in finals[j])
            vals.append(len(a & b) / max(1, min(len(a), len(b))))
    return statistics.mean(vals) if vals else float("nan")


def run_rung(rung, n: int, seed: int) -> Dict[str, float]:
    rng = random.Random(seed * 1000 + rung[0])
    brain = build(n, seed, rung[3])

    train = sample_strings(rung, *SHORT, TRAIN_PER_CLASS, rng)
    test = sample_strings(rung, *SHORT, TEST_PER_CLASS, rng)
    long_test = sample_strings(rung, *LONG, TEST_PER_CLASS, rng)
    if not train or not test:
        return {}

    # Training pass: plasticity ON. This is the only place the substrate learns.
    finals = {True: [], False: []}
    for s, label in train:
        finals[label].append(encode(brain, s))

    cen = {lab: centroid(fs, K) for lab, fs in finals.items() if fs}
    if len(cen) < 2:
        return {}

    def score(items) -> float:
        hits = 0
        # read_only: probing must not train. Every test string would otherwise
        # move the weights that classify the next one.
        with brain.read_only():
            for s, label in items:
                f = set(int(x) for x in encode(brain, s))
                pred = len(f & cen[True]) > len(f & cen[False])
                hits += int(pred == label)
        return hits / len(items)

    return {
        "short": score(test),
        "long": score(long_test),
        "spread": state_spread(finals[True] + finals[False]),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    n = int(os.environ.get("LADDER_N", "1000"))
    print(f"\n  substrate only: n={n} k={K} p={P} beta={BETA} "
          f"rounds/symbol={ROUNDS_PER_SYMBOL}, seeds {list(SEEDS)}")
    print(f"  train len {SHORT}, test len {SHORT}, length-gen test {LONG}")
    print(f"  chance = 0.500 by construction (balanced)\n")
    print(f"  {'rung':<4} {'class':<19} {'language':<32} "
          f"{'in-len':>7} {'out-len':>8} {'spread':>7}")

    for rung in LADDER:
        short, long_, spread = [], [], []
        for seed in SEEDS:
            r = run_rung(rung, n, seed)
            if r:
                short.append(r["short"])
                long_.append(r["long"])
                spread.append(r["spread"])
        if not short:
            print(f"  {rung[0]:<4} {rung[1]:<19} {rung[2]:<32} "
                  f"{'(no sample)':>7}")
            continue
        ms, ml = statistics.mean(short), statistics.mean(long_)
        flag = ""
        if statistics.mean(spread) > 0.9:
            flag = "  <- COLLAPSED, not a result"
        print(f"  {rung[0]:<4} {rung[1]:<19} {rung[2]:<32} "
              f"{ms:>7.3f} {ml:>8.3f} {statistics.mean(spread):>7.3f}{flag}")


if __name__ == "__main__":
    main()
