"""Theorem 4 (COLT 2022): does assembly overlap PRESERVE stimulus overlap?

WHY THIS IS THE EXPERIMENT. Measuring the parser found that words with
DIFFERENT grounding land on IDENTICAL core assemblies -- 24% of NOUN_CORE are
exact duplicates, and the semantic clusters ({death fear life love ...},
{face foot head mouth nose}) are stable across seeds. I proposed fixing that by
reserving a private fraction of k per word.

That proposal was a hand-rolled restatement of a theorem. Dabagia,
Papadimitriou & Vempala (COLT 2022), Theorem 4 (Multiple Assemblies), with two
stimulus classes satisfying |S_A ∩ S_B| = alpha*k:

    "the overlap in the core sets A* and B* will preserve the overlap of the
     stimulus classes, so that |A* ∩ B*| <= alpha*k"

Overlap preservation is PROVED, not designed. So the parser is not missing a
feature -- it is violating a guarantee, and the question becomes whether the
guarantee holds in OUR implementation at OUR parameters.

THE ARITHMETIC THAT MAKES THIS NON-OBVIOUS. Theorem 1 requires a plasticity
LOWER bound,

    beta >= beta0 = (1/r^2) * [ sqrt(2-r^2)*sqrt(2 ln(n/k)) + sqrt(6) ]
                             / [ sqrt(kp) + sqrt(2 ln(n/k)) ]

and bounds the support by |A*| <= k / (1 - exp(-(beta/beta0)^2)).

Evaluated below, that bound is VACUOUS at every beta anyone actually runs --
including the acquisition paper's own beta = 0.06. So this is emphatically NOT
"we violate the hypothesis, therefore that is the bug". The correct reading is
that the theorems are asymptotic sufficient conditions with unoptimised
constants, and they do not certify any practical regime. Whether overlap
preservation HOLDS far below beta0 is an empirical question, and it is exactly
the question that decides whether vocabulary scales.

THE MODEL IS THE PAPER'S, NOT OURS, and that is the point. A stimulus class A is
a distribution: a set S_A of k sensory neurons, and to draw a stimulus each
i in S_A fires with probability r while each i outside fires with probability
qk/n. The assembly is formed from a STREAM of such samples, and the "core set"
is what survives the sampling.

Our lexicon fires ONE FIXED deterministic pattern per word. There is no
sampling, so there is no core to extract -- which may be the whole divergence.
This file implements the paper's model so that question can be asked at all.

VALIDATION BEFORE INTERPRETATION. Theorem 3 (Recall) predicts that a fresh
sample from class A produces a cap overlapping A* by at least 1 - e^{-kpr}. That
is a checkable prediction of the same setup, and it runs FIRST: if recall fails,
the setup is wrong and the Theorem 4 numbers mean nothing. This project has
shipped conclusions from setups that never formed an assembly, so the
validation is not optional decoration.

WHAT A DEGENERATE ARM SCORES, named in advance. If the area collapses so that
A* = B* regardless of the inputs, overlap reads 1.0 at EVERY alpha -- which is
the violation this file is looking for, so it cannot be distinguished from the
finding by the headline number alone. It is separated by two other columns:
a collapsed run also has |A*|/k far above 1 and recall that does NOT degrade
with alpha. Both are printed.

READING:
  * overlap tracks alpha (slope ~1, and <= alpha)
        -> Theorem 4 holds in our implementation, well below beta0. The
           substrate preserves input structure, and the parser's collisions come
           from the LEXICON handing it inputs that are already too close --
           which is an architecture question (PHON->LEX1/LEX2), not a
           substrate one.
  * overlap saturates at 1 above some alpha
        -> there is a critical input overlap past which the substrate quantizes.
           That threshold IS the design rule for the featural code, and it is a
           number we can build against.
  * overlap high at every alpha, |A*|/k large
        -> the substrate does not preserve overlap at our beta. Then beta0 is
           not merely conservative and the plasticity regime is implicated.
"""
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                                     # noqa: E402

from neural_assemblies.core.brain import Brain                         # noqa: E402

SENSE, B = "SENSE", "B"
N, K, P = 10000, 100, 0.05
R, Q = 0.9, 0.0            # paper's stimulus-class parameters
ROUNDS = 10                # O(log k); log2(100) ~ 6.6
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75]
#: Spans BOTH sides of beta0 = 1.349. Testing only below it would let "higher
#: beta is worse" be reported without ever entering the regime where the
#: theorem's hypothesis actually holds -- which is the one claim this sweep
#: exists to check.
BETAS = [0.05, 0.10, 0.50, 1.00, 1.50, 2.00, 4.00]
SEEDS = [42, 7, 123]


def beta0(n=N, k=K, p=P, r=R):
    L = math.sqrt(2 * math.log(n / k))
    return (1 / r ** 2) * ((math.sqrt(2 - r * r) * L + math.sqrt(6))
                           / (math.sqrt(k * p) + L))


def support_bound(k, beta, b0):
    x = 1 - math.exp(-(beta / b0) ** 2)
    return k / x if x > 0 else float("inf")


def _sample(core, n, k, rng, r=R, q=Q):
    """Draw a stimulus from the class: i in S_A w.p. r, else w.p. qk/n."""
    on = core[rng.random(core.size) < r]
    if q > 0:
        outside = rng.random(n) < (q * k / n)
        outside[core] = False
        on = np.union1d(on, np.flatnonzero(outside))
    return on.astype(np.int64)


def _fire(brain, idx):
    """Present `idx` in the sensory area as this step's firing set."""
    area = brain.areas[SENSE]
    area.winners = np.asarray(idx, dtype=np.int64)
    area.fix_assembly()


def _build(brain, core, rng, rounds=ROUNDS):
    """Present a stream of samples; return (union of caps, final cap)."""
    union = set()
    cap = np.empty(0, dtype=np.int64)
    for _ in range(rounds):
        _fire(brain, _sample(core, N, K, rng))
        brain.project({}, {SENSE: [B]})
        cap = np.asarray(brain.areas[B].winners, dtype=np.int64)
        union.update(int(x) for x in cap)
    return union, cap


def _recall(brain, core, rng, star):
    """Theorem 3: a fresh sample should land back on the stored assembly."""
    with brain.read_only():
        _fire(brain, _sample(core, N, K, rng))
        brain.project({}, {SENSE: [B]})
        cap = set(int(x) for x in np.asarray(brain.areas[B].winners))
    return len(cap & star) / max(len(cap), 1)


def trial(alpha, beta, seed):
    rng = np.random.default_rng(seed)
    brain = Brain(p=P, seed=seed, engine="numpy_exact", norm_init=True)
    brain.add_area(SENSE, N, K, beta=beta)
    brain.add_area(B, N, K, beta=beta)

    # Two classes with EXACTLY alpha*k shared sensory neurons.
    shared = int(round(alpha * K))
    pool = rng.permutation(N)
    core_a = np.sort(pool[:K])
    core_b = np.sort(np.concatenate([core_a[:shared], pool[K:2 * K - shared]]))
    assert core_b.size == K
    assert np.intersect1d(core_a, core_b).size == shared

    a_star, cap_a = _build(brain, core_a, rng)
    rec_a = _recall(brain, core_a, rng, a_star)
    b_star, cap_b = _build(brain, core_b, rng)
    rec_b = _recall(brain, core_b, rng, b_star)

    inter = len(a_star & b_star)
    # TWO READINGS OF "core set", because the paper's phrase is ambiguous and
    # the choice could manufacture the result. A* is defined as the UNION of
    # caps, which is what `inter` uses; but "the overlap in the core sets" may
    # mean the settled assembly rather than the union over transients. The
    # final caps are the narrower reading and are reported alongside.
    cap_inter = len(set(int(x) for x in cap_a) & set(int(x) for x in cap_b))
    # CHANCE FLOOR. The bound alpha*k demands ZERO overlap at alpha=0, but two
    # independent sets of size |A*| and |B*| in n neurons already share
    # |A*||B*|/n by chance. Scoring alpha=0 against a literal 0 would report a
    # violation that is pure baseline, so the floor is subtracted explicitly.
    chance = (len(a_star) * len(b_star) / N) / K
    return {
        "overlap": inter / K,           # the theorem's |A* ∩ B*| / k
        "cap_overlap": cap_inter / K,
        "chance": chance,
        "alpha": alpha,
        "support_a": len(a_star) / K,
        "support_b": len(b_star) / K,
        "recall": (rec_a + rec_b) / 2,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    b0 = beta0()
    print("COLT 2022 Theorem 4 -- does assembly overlap preserve stimulus "
          "overlap?")
    print(f"n={N} k={K} p={P} r={R} q={Q} rounds={ROUNDS} seeds={SEEDS} "
          f"engine=numpy_exact")
    print()
    print("THE THEOREM'S OWN REGIME, evaluated. beta0 is a LOWER bound on")
    print("plasticity; the support bound is k / (1 - exp(-(beta/beta0)^2)).")
    print(f"  beta0(n={N}, k={K}, p={P}, r={R}) = {b0:.3f}")
    for nm, n_, k_, p_, bt in (("parser NOUN_CORE", 3000, 30, 0.05, 0.10),
                               ("our toy", 2000, 40, 0.05, 0.10),
                               ("acquisition paper", 100000, 100, 0.05, 0.06),
                               ("this experiment", N, K, P, 0.10)):
        bb = beta0(n_, k_, p_, R)
        sb = support_bound(k_, bt, bb)
        print(f"  {nm:<19} beta={bt:<5} beta0={bb:6.3f}  "
              f"beta/beta0={bt / bb:5.3f}  support bound |A*| <= {sb / k_:8.1f}k")
    print("  -> VACUOUS everywhere, including the paper's own parameters. The")
    print("     theorems are asymptotic sufficient conditions, so beta0 is NOT")
    print("     evidence that our beta is wrong. Whether overlap preservation")
    print("     survives this far below beta0 is the empirical question.")
    print()

    print(f"Theorem 3 (Recall) predicts overlap >= 1 - e^-kpr = "
          f"{1 - math.exp(-K * P * R):.3f}")
    print()
    hdr = (f"{'beta':>6} {'alpha':>7} {'overlap':>9} {'cap ov':>8} "
           f"{'chance':>8} {'bound':>7} {'amp':>6} {'holds':>6} "
           f"{'|A*|/k':>8} {'recall':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for beta in BETAS:
        for alpha in ALPHAS:
            res = [trial(alpha, beta, s) for s in SEEDS]
            ov = statistics.fmean(r["overlap"] for r in res)
            cv = statistics.fmean(r["cap_overlap"] for r in res)
            ch = statistics.fmean(r["chance"] for r in res)
            sa = statistics.fmean(r["support_a"] for r in res)
            sb = statistics.fmean(r["support_b"] for r in res)
            rc = statistics.fmean(r["recall"] for r in res)
            # Bound compared against the CHANCE-CORRECTED observation, so a
            # baseline-level reading at alpha=0 is not scored as a violation.
            # One neuron of slack (1/k). Without it, an alpha=0 cell sitting
            # EXACTLY at the chance floor scores as a violation on rounding
            # noise, and the headline count reports baseline as a finding.
            ok = "yes" if (ov - ch) <= alpha + 1.0 / K else "NO"
            amp = (ov - ch) / alpha if alpha > 0 else float("nan")
            rows.append((beta, alpha, ov, sa, sb, rc, ok, cv, ch, amp))
            print(f"{beta:>6.2f} {alpha:>7.2f} {ov:>9.4f} {cv:>8.4f} "
                  f"{ch:>8.4f} {alpha:>7.2f} {amp:>6.2f} {ok:>6} "
                  f"{sa:>8.2f} {rc:>8.3f}")
        print()

    print("=" * len(hdr))
    bad_recall = [r for r in rows if r[5] < 1 - math.exp(-K * P * R) - 0.15]
    if bad_recall:
        print(f"** RECALL FAILED on {len(bad_recall)} of {len(rows)} cells "
              f"(Theorem 3 predicts >= {1 - math.exp(-K * P * R):.3f}). The "
              f"setup did not form assemblies there, so their overlap numbers "
              f"are NOT evidence about Theorem 4. **")
    viol = [r for r in rows if r[6] == "NO"]
    print(f"Theorem 4 bound |A*interB*| <= alpha*k held in "
          f"{len(rows) - len(viol)}/{len(rows)} cells.")
    if viol:
        print("Violations (beta, alpha, overlap):")
        for r in viol:
            print(f"   beta={r[0]:.2f} alpha={r[1]:.2f} overlap={r[2]:.4f}")
    print()
    print("A run where overlap reads ~1.0 at EVERY alpha with |A*|/k large and")
    print("recall flat is a COLLAPSED area, not a refutation of the theorem.")


if __name__ == "__main__":
    main()
