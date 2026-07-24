#!/usr/bin/env python3
"""Record DIRECT causal binding golden metrics (Kopadi & Kalles 2026)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "literature" / "parity" / "golden" / "direct2026_pearl.json"


def main() -> None:
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.programs.direct import (
        direct_bind,
        measure_directional_asymmetry,
        validate_direct_do_calculus,
    )

    seed, n, k, p, beta, rounds = 42, 5000, 80, 0.05, 0.1, 8
    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse")
    brain.add_stimulus("cause_s", k)
    brain.add_stimulus("effect_s", k)
    brain.add_area("CAUSE", n, k, beta)
    brain.add_area("EFFECT", n, k, beta)
    brain.add_area("BIND", n, k, beta)
    direct_bind(
        brain, "CAUSE", "EFFECT", "BIND",
        cause_stim="cause_s", effect_stim="effect_s", rounds=rounds,
    )
    fwd, rev = measure_directional_asymmetry(brain, "CAUSE", "EFFECT", "BIND")
    _, _, do_fwd = validate_direct_do_calculus(brain, "CAUSE", "EFFECT", "BIND")

    golden = {
        "protocol": "direct2026_pearl",
        "source": "neural_assemblies.programs.direct (seed=42)",
        "recorded": "2026-06-23",
        "paper_figures": {
            "fig9": "Ground-truth Alzheimer DAG vs learned structure",
            "fig10": "Interventional ATE sign/magnitude match",
            "fig11": "Counterfactual direction consistency",
        },
        "parameters": {
            "seed": seed,
            "n": n,
            "k": k,
            "p": p,
            "beta": beta,
            "rounds": rounds,
            "engine": "numpy_sparse",
        },
        "metrics": {
            "forward_overlap": round(fwd, 4),
            "reverse_overlap": round(rev, 4),
            "do_effect_forward_overlap": round(do_fwd, 4),
            "forward_reverse_ratio": round(fwd / max(rev, 1e-9), 4),
        },
        "thresholds": {
            "forward_overlap_min": 0.1,
            "reverse_overlap_min": 0.0,
            "do_effect_forward_min_fraction_of_forward": 0.45,
        },
        "paper_claims_note": (
            "Full Alzheimer DAG recovery requires tabular SCM pipeline; "
            "this golden pins toy DIRECT binding asymmetry + do-calculus readout."
        ),
        "config_ref": "parity/configs/direct2026.yaml",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(golden["metrics"], indent=2))


if __name__ == "__main__":
    main()
