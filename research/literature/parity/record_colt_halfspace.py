#!/usr/bin/env python3
"""Record [COLT22] Theorem 6 (Learning Linear Thresholds) golden.

Source protocol: ``.reference/mdabagia-learning-with-assemblies/Halfspace.ipynb``
(the paper's own notebook), ported in
``neural_assemblies/programs/colt_halfspace_numpy.py``.

Unlike ``reference_pnas_golden.json``, this golden is NOT self-recorded from
this package's ops -- it comes from a transcription of the authors' code, so it
is an independent target for the engine to be measured against (cf. task #54).
"""

from __future__ import annotations

import json
from pathlib import Path

from research.json_documents import write_new_document

from neural_assemblies.programs.colt_halfspace_numpy import run_colt_halfspace

ROOT = Path(__file__).resolve().parents[3]
OUT = (ROOT / "research" / "literature" / "parity" / "golden"
       / "colt2022_halfspace.json")


def main() -> None:
    result = run_colt_halfspace()
    k = result.parameters["cap_size"]
    golden = {
        "protocol": "colt2022_halfspace",
        "cite_tag": "COLT22",
        "claim": (
            "Theorem 6: presenting Omega(log k) samples from D+ forms an "
            "assembly A* such that a fresh D+ sample's cap overlaps at least "
            "3k/4 of A*, and a D- sample's at most k/4."
        ),
        "source": "learning-with-assemblies/Halfspace.ipynb numpy port",
        "recorded": "2026-07-30",
        "parameters": result.parameters,
        "metrics": {
            "pos_overlap": round(result.pos_overlap, 4),
            "neg_overlap": round(result.neg_overlap, 4),
            "per_seed_pos": [round(x, 4) for x in result.per_seed_pos],
            "per_seed_neg": [round(x, 4) for x in result.per_seed_neg],
            "n_on_pos": result.n_on_pos,
            "n_on_neg": result.n_on_neg,
        },
        "thresholds": {
            # The theorem's own bounds. Both hold with wide margin here
            # (measured ~99 and ~10 against 75 and 25), so they are assertable
            # rather than seed-set-dependent -- see paper-parity-process.
            "pos_overlap_min": 0.75 * k,
            "neg_overlap_max": 0.25 * k,
        },
        "notes": [
            "beta=1.0, not the 0.1 used elsewhere in this repo.",
            "Weights are column-normalized at init AND AGAIN after training, "
            "before evaluation -- the theorem assumes homeostasis BETWEEN "
            "training and evaluation, which one-time norm_init does not do.",
            "Evaluation is recurrent: k_cap(out @ W + x @ A).",
            "Classes are a halfspace with margin and OVERLAPPING support, not "
            "two disjoint supports.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_new_document(OUT, golden)
    print(f"wrote {OUT}")
    print(f"  D+ overlap {result.pos_overlap:.2f} (>= {0.75 * k:.0f})")
    print(f"  D- overlap {result.neg_overlap:.2f} (<= {0.25 * k:.0f})")
    print(f"  separates: {result.separates}")


if __name__ == "__main__":
    main()
