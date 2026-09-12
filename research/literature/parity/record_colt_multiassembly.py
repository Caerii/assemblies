#!/usr/bin/env python3
"""Record [COLT22] Theorems 1/3/4 golden -- creation, recall, MULTIPLE assemblies.

Companion to ``record_colt_halfspace.py`` (Theorem 6). This one covers the
theorems about forming assemblies for stimulus CLASSES and about what happens
when two classes share sensory neurons.

WHAT THIS GOLDEN ASSERTS AND WHAT IT MERELY PINS -- the distinction is the point:

  * Theorem 3 (Recall) is ASSERTED against the paper's own bound
    ``1 - e^{-kpr}``. It holds with wide margin, so the threshold tests the
    effect rather than the seed set (cf. paper-parity-process).

  * Theorem 4 (Multiple Assemblies) is PINNED AS A DIVERGENCE. The theorem says
    ``|A* ∩ B*| <= alpha*k``; we measure amplification instead. Recording it as
    a golden means the divergence cannot drift silently, and a future change
    that FIXES it will fail this file loudly rather than pass unnoticed.

  * Theorem 1's support bound is RECORDED, NOT ASSERTED, because it is vacuous
    at every beta anyone runs -- 182k here, and 506k at the acquisition paper's
    own beta=0.06. Asserting a bound that nothing can violate is the "weak test"
    failure mode this repo has already catalogued.
"""

from __future__ import annotations

import json
from pathlib import Path

from research.json_documents import write_new_document

from neural_assemblies.programs.colt_multiassembly_numpy import (
    run_colt_multiassembly,
    support_bound,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = (ROOT / "research" / "literature" / "parity" / "golden"
       / "colt2022_multiassembly.json")


def main() -> None:
    result = run_colt_multiassembly()
    k = int(result.parameters["k"])
    golden = {
        "protocol": "colt2022_multiassembly",
        "cite_tag": "COLT22",
        "claim": (
            "Theorem 3 (Recall): a fresh sample from class A yields a cap "
            "overlapping A* by at least 1 - e^{-kpr}. "
            "Theorem 4 (Multiple Assemblies): with |S_A n S_B| = alpha*k, "
            "|A* n B*| <= alpha*k."
        ),
        "source": "COLT 2022 paper text; protocol implemented in "
                  "neural_assemblies/programs/colt_multiassembly_numpy.py",
        "recorded": "2026-08-07",
        "parameters": dict(result.parameters),
        "metrics": {
            "recall": round(result.recall, 4),
            "recall_floor": round(result.recall_floor, 4),
            "alphas": list(result.alphas),
            "overlaps": [round(x, 4) for x in result.overlaps],
            "chance": [round(x, 4) for x in result.chance],
            "supports": [round(x, 4) for x in result.supports],
            "beta0": round(result.beta0, 4),
            "support_bound_in_k": round(
                support_bound(k, float(result.parameters["beta"]),
                              result.beta0) / k, 1),
        },
        "thresholds": {
            # ASSERTED: the theorem's own recall bound.
            "recall_min": round(result.recall_floor, 4),
        },
        "divergences": {
            "theorem_4_overlap_preservation": {
                "status": "DOES NOT HOLD",
                "expected": "|A* n B*| / k <= alpha",
                "measured": {str(a): round(o, 4)
                             for a, o in zip(result.alphas, result.overlaps)},
                "note": (
                    "Overlap is AMPLIFIED, with beta as the gain. Raising beta "
                    "to and past beta0 (1.5/2.0/4.0 vs beta0=1.349) does NOT "
                    "restore preservation -- those arms are identical to "
                    "beta=0.5. Every practical beta is far below beta0, "
                    "including the papers' own simulations, so the theorem's "
                    "hypothesis is not in force and this is a divergence from "
                    "the guarantee rather than a refutation of the theorem."
                ),
            },
        },
        "notes": [
            "Engine PINNED to numpy_exact: the numpy_sparse candidate sampler "
            "invents drive for neurons that have not fired, so an overlap "
            "measurement there is partly a property of the sampler.",
            "alpha=0 is scored against the CHANCE floor |A*||B*|/n, not "
            "against the theorem's literal 0, which no finite substrate can "
            "deliver.",
            "Stimulus classes are DISTRIBUTIONS (r=0.9, q=0.0), not the fixed "
            "deterministic patterns our lexicon uses -- without sampling there "
            "is no core set for these theorems to describe.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_new_document(OUT, golden)
    print(f"wrote {OUT}")
    print(f"  recall {result.recall:.4f} (>= {result.recall_floor:.4f})")
    for a, o, c in zip(result.alphas, result.overlaps, result.chance):
        # CHANCE-CORRECTED. The bound alpha*k demands exactly 0 at alpha=0,
        # which no finite substrate delivers -- two independent supports share
        # |A*||B*|/n by default. Flagging that baseline as "AMPLIFIED" would
        # report the floor as a finding.
        flag = "OK" if (o - c) <= a + 1.0 / k else "AMPLIFIED"
        print(f"  alpha={a:.2f} -> overlap {o:.4f} (chance {c:.4f})  [{flag}]")


if __name__ == "__main__":
    main()
