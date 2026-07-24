#!/usr/bin/env python3
"""Record PNAS 2020 param-regime reconciliation golden (CI vs paper canonical)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "literature" / "parity" / "golden" / "pnas2020_scaling.json"


def _record_regime(name: str, n: int, k: int, p: float, beta: float, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.ops import overlap, project, separate
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse")
    brain.add_stimulus("s1", k)
    brain.add_stimulus("s2", k)
    brain.add_area("A", n, k, beta)
    a1 = project(brain, "s1", "A", rounds=10)
    a1b = project(brain, "s1", "A", rounds=10)
    _, _, sep_ov = separate(brain, "s1", "s2", "A", rounds=10)
    return {
        "n": n,
        "k": k,
        "p": p,
        "beta": beta,
        "seed": seed,
        "chance_overlap": round(k / n, 6),
        "project_persistence": round(overlap(a1, a1b), 4),
        "separate_overlap": round(sep_ov, 4),
    }


def main() -> None:
    golden = {
        "protocol": "pnas2020_scaling",
        "source": "neural_assemblies package — dual regime reconciliation",
        "recorded": "2026-06-23",
        "regimes": {
            "ci_parity": _record_regime("ci_parity", n=5000, k=80, p=0.05, beta=0.1, seed=42),
            "paper_canonical": _record_regime(
                "paper_canonical", n=10000, k=100, p=0.01, beta=0.05, seed=42,
            ),
            "test_assembly_calculus": _record_regime(
                "test_assembly_calculus", n=10000, k=100, p=0.05, beta=0.1, seed=42,
            ),
        },
        "notes": {
            "ci_parity": "Pinned in test_literature_parity + reference_pnas_golden.json",
            "paper_canonical": "PNAS sparse regime n=10^4, k=100, p=0.01, beta=0.05",
            "test_assembly_calculus": "Package unit test defaults",
        },
        "config_ref": "parity/configs/pnas2020.yaml",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(golden["regimes"], indent=2))


if __name__ == "__main__":
    main()
