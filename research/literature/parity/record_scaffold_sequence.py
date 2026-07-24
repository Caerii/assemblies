#!/usr/bin/env python3
"""Record scaffold vs simple sequence recall golden (dabagia.org/nemo/sequences/)."""

from __future__ import annotations

import json
from pathlib import Path

from neural_assemblies.assembly_calculus.scaffold import compare_scaffold_vs_simple

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "literature" / "parity" / "golden" / "nemo2025_scaffold.json"

SEED = 42
SEQ_LEN = 20
N = 1000
K = 30
P = 0.4
BETA = 0.1
N_PRESENTATIONS = 10


def main() -> None:
    result = compare_scaffold_vs_simple(
        seq_len=SEQ_LEN,
        n=N,
        k=K,
        density=P,
        n_presentations=N_PRESENTATIONS,
        seed=SEED,
    )

    golden = {
        "protocol": "nemo2025_scaffold",
        "source": "neural_assemblies compare_scaffold_vs_simple (mdabagia/nemo ScaffoldNetwork)",
        "demo_url": "https://dabagia.org/nemo/sequences/",
        "recorded": "2026-06-23",
        "parameters": {
            "seed": SEED,
            "seq_len": SEQ_LEN,
            "n": N,
            "k": K,
            "p": P,
            "beta": BETA,
            "n_presentations": N_PRESENTATIONS,
        },
        "metrics": {
            "simple_last_recall": round(result.simple_last, 4),
            "scaffold_last_recall": round(result.scaffold_last, 4),
            "scaffold_beats_simple": result.scaffold_beats_simple,
            "simple_history": [round(x, 4) for x in result.simple_history],
            "scaffold_history": [round(x, 4) for x in result.scaffold_history],
        },
        "thresholds": {
            "simple_last_recall_min": 0.0,
            "scaffold_last_recall_min": 0.0,
            "metrics_match_tolerance": 0.02,
        },
        "notes": "Site figure uses seq_len=20; nemo-demo.ipynb uses 25.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(golden["metrics"], indent=2))


if __name__ == "__main__":
    main()
