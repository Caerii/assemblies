"""PNAS 2020 tier-A parity: our numbers against the PAPER's numbers.

Every other pnas2020_* golden in this directory was recorded by running our own
code, which pins today's output against last month's but cannot detect a shared
misreading of the paper. The targets here are quoted from
`research/literature/papers/papadimitriou2020_pnas.pdf` and exist independently
of this repo.

Claims implemented (quotes are verbatim from the PDF):

  A1  regime   "we typically use n = 10^7, k = 10^4, p = 0.001, and beta = 0.1"
               (Fig. 1); the body gives "n = 10^6-7, p = 10^-3, k = 10^2-3,
               and beta = 0.1", and separately assumes "the assembly size k is
               about less than the square root of n".

  A2  project  "One would expect y2 to overlap substantially with y1 -- this
               overlap can be calculated in our mathematical model to be
               roughly 50% for a broad range of parameters."
               y1 is the first k-WTA in B driven by x alone; y2 is the next,
               driven by x AND the recurrent input from y1.  "A broad range of
               parameters" is what makes this testable below paper scale.

  A3  associate "The results of ref. 15 suggest an overlap between associated
               assemblies in the MTL of about 8 to 10% of the size of an
               assembly."  Fig. 2E protocol: two stable assemblies are built in
               A and B, then "the assemblies in A and B are projected
               simultaneously into C", and the overlap of their C-projections
               is measured.

  A4  complete Fig. 2 F2: "An assembly is created in an area A by repeated
               firing of its parent; the number of times the parent fires is
               depicted in the horizontal axis. Next, 40% of neurons of the
               assembly are selected at random and fire for a number of steps
               ... With more reinforcement of the original assembly, the subset
               recovers nearly all of the original assembly."
               Note 40%, not the 50% our pnas2020_pattern_complete golden uses,
               and note the claim is a TREND in reinforcement, not one number.

The protocols are written directly against `Brain.project` rather than through
`assembly_calculus.ops`, because the wrappers are part of what is under test --
routing a parity check through them would hide a wrapper-level protocol error,
which is exactly the defect class that produced the re-recorded goldens of
2026-07-30.

NORMALIZATION IS AN OPEN DIVERGENCE.  The paper assumes ongoing homeostasis:
"synaptic weights are renormalized, at a slower time scale, so that the sum of
presynaptic weights at each neuron stays relatively stable".  The authors'
reference implementation instead normalizes ONCE at reset (this repo's
`norm_init`).  Both arms are run and reported; neither is assumed correct.

SCALING DOWN: PRESERVE k*p, NOT p.  The paper runs at n = 10^6-7, which we
cannot.  The quantity that governs the dynamics is the expected afferent count
k*p -- how many active presynaptic partners a candidate neuron actually has.
At the paper's regime that is 10^4 * 10^-3 = 10.  Copying p = 0.001 to
n = 2x10^4 with k = 100 gives k*p = 0.1, i.e. almost no neuron in the target
receives ANY input, k-WTA falls through to its index tie-break, and every
measurement below becomes an artifact of that tie-break: the first run of this
harness returned overlap(y1,y2) = 0.0000 exactly, and two "separate" assemblies
that already overlapped 0.82 before association.  `--afferent` sets p from
k and the target afferent count so this cannot happen silently.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Tuple

import numpy as np

from neural_assemblies.core.brain import Brain

# Verbatim targets. Do not "adjust" these to match output.
PAPER = {
    "A2_project_y1_y2_overlap": (0.50, "roughly 50%"),
    "A3_associate_overlap": ((0.08, 0.10), "about 8 to 10%"),
    "A4_complete_cue_fraction": (0.40, "40% of neurons"),
}


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _mk(n: int, k: int, p: float, beta: float, seed: int,
        norm_init: bool) -> Brain:
    _seed_all(seed)
    return Brain(p=p, seed=seed, save_winners=True, norm_init=norm_init,
                 recurrent_projection=True, engine="numpy_sparse")


def _ov(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return len(np.intersect1d(a, b)) / max(1, len(a))


# ------------------------------------------------------------------ A2

def claim_a2(n: int, k: int, p: float, beta: float, seed: int,
             norm_init: bool) -> float:
    """overlap(y1, y2) where y2 adds recurrent input from y1. Paper: ~0.50."""
    b = _mk(n, k, p, beta, seed, norm_init)
    b.add_stimulus("x", k)
    b.add_area("B", n, k, beta)

    b.project({"x": ["B"]}, {})                      # y1: afferent only
    y1 = np.asarray(b.areas["B"].winners).copy()
    b.project({"x": ["B"]}, {"B": ["B"]})            # y2: afferent + recurrent
    y2 = np.asarray(b.areas["B"].winners).copy()
    return _ov(y1, y2)


# ------------------------------------------------------------------ A3

def claim_a3(n: int, k: int, p: float, beta: float, seed: int,
             norm_init: bool, build_rounds: int, assoc_rounds: int) -> Dict[str, float]:
    """Fig. 2E. Two stable assemblies co-projected into C. Paper: 0.08-0.10."""
    b = _mk(n, k, p, beta, seed, norm_init)
    b.add_stimulus("sx", k)
    b.add_stimulus("sy", k)
    for area in ("A", "B", "C"):
        b.add_area(area, n, k, beta)

    for stim, area in (("sx", "A"), ("sy", "B")):     # stable parents
        for _ in range(build_rounds):
            b.project({stim: [area]}, {area: [area]})

    def project_into_c(stim: str, src: str) -> np.ndarray:
        for _ in range(build_rounds):
            b.project({stim: [src]}, {src: ["C"], "C": ["C"]})
        return np.asarray(b.areas["C"].winners).copy()

    xc_before = project_into_c("sx", "A")
    yc_before = project_into_c("sy", "B")
    before = _ov(xc_before, yc_before)

    for _ in range(assoc_rounds):                     # simultaneous firing
        b.project({"sx": ["A"], "sy": ["B"]},
                  {"A": ["C"], "B": ["C"], "C": ["C"]})

    xc_after = project_into_c("sx", "A")
    yc_after = project_into_c("sy", "B")
    after = _ov(xc_after, yc_after)
    return {"before": before, "after": after, "chance": k / n}


# ------------------------------------------------------------------ A4

def claim_a4(n: int, k: int, p: float, beta: float, seed: int,
             norm_init: bool, reinforcements: List[int],
             cue_fraction: float, complete_rounds: int) -> List[Tuple[int, float]]:
    """Fig. 2 F2: recovery from a 40% cue, as a function of reinforcement."""
    out = []
    for t in reinforcements:
        b = _mk(n, k, p, beta, seed, norm_init)
        b.add_stimulus("x", k)
        b.add_area("A", n, k, beta)
        for _ in range(t):
            b.project({"x": ["A"]}, {"A": ["A"]})
        full = np.asarray(b.areas["A"].winners).copy()

        rng = np.random.default_rng(seed)
        cue = rng.choice(full, size=max(1, int(round(cue_fraction * len(full)))),
                         replace=False)
        b.areas["A"].winners = cue
        for _ in range(complete_rounds):
            b.project({}, {"A": ["A"]})
        out.append((t, _ov(np.asarray(b.areas["A"].winners), full)))
    return out


# ----------------------------------------------------------------- main

def run(n: int, k: int, p: float, beta: float, seeds: List[int],
        norm_init: bool) -> Dict:
    import statistics as st
    tag = "norm_init=True" if norm_init else "norm_init=False"
    print(f"\n{'='*66}\n{tag}   n={n} k={k} p={p} beta={beta}  "
          f"(k<sqrt(n)? {k < n ** 0.5})\n{'='*66}")

    a2 = [claim_a2(n, k, p, beta, s, norm_init) for s in seeds]
    t2 = PAPER["A2_project_y1_y2_overlap"][0]
    print(f"\nA2 project overlap(y1,y2)   ours {st.mean(a2):.4f} "
          f"+/- {st.stdev(a2) if len(a2) > 1 else 0:.4f}    paper ~{t2:.2f}")

    a3 = [claim_a3(n, k, p, beta, s, norm_init, 10, 10) for s in seeds]
    lo, hi = PAPER["A3_associate_overlap"][0]
    print(f"A3 associate overlap        ours {st.mean(r['after'] for r in a3):.4f}"
          f"   (before {st.mean(r['before'] for r in a3):.4f}, "
          f"chance {a3[0]['chance']:.4f})    paper {lo:.2f}-{hi:.2f}")

    a4 = claim_a4(n, k, p, beta, seeds[0], norm_init, [1, 2, 5, 10, 20], 0.40, 10)
    print(f"A4 complete from 40% cue    (paper: rises toward ~1.0 with reinforcement)")
    for t, ov in a4:
        print(f"      reinforcement {t:>3}  ->  overlap {ov:.4f}")

    return {"a2": a2, "a3": a3, "a4": a4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--p", type=float, default=None,
                    help="explicit p; overrides --afferent")
    ap.add_argument("--afferent", type=float, default=10.0,
                    help="target expected afferent count k*p (paper: 10^4*10^-3 = 10)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    p = a.p if a.p is not None else min(0.5, a.afferent / a.k)
    print(f"regime: n={a.n} k={a.k} p={p:.4g} beta={a.beta}   "
          f"expected afferent k*p = {a.k * p:.2f} (paper 10)")

    results = {}
    for norm in (False, True):
        results[f"norm_init={norm}"] = run(a.n, a.k, p, a.beta, a.seeds, norm)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"parameters": vars(a), "paper_targets":
                       {k: v[1] for k, v in PAPER.items()},
                       "results": str(results)}, f, indent=2)


if __name__ == "__main__":
    main()
