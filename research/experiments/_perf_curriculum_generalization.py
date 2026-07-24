#!/usr/bin/env python3
"""Curriculum depth generalization sweep — holdout metrics by training depth."""

from __future__ import annotations

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
    DEFAULT_DEPTH_CHECKPOINTS,
    format_curriculum_sweep_table,
    run_curriculum_depth_sweep,
)


def _parse_depths() -> list[str]:
    raw = os.environ.get("SWEEP_DEPTHS", "").strip()
    if not raw:
        return list(DEFAULT_DEPTH_CHECKPOINTS)
    return [d.strip() for d in raw.split(",") if d.strip()]


def main() -> None:
    depths = _parse_depths()
    seed = int(os.environ.get("SWEEP_SEED", "42"))
    qa_subset = int(os.environ.get("SWEEP_QA_SUBSET", "12"))

    print("=== CURRICULUM DEPTH GENERALIZATION SWEEP ===")
    print(f"depths: {depths}")
    print(f"seed={seed} qa_subset={qa_subset} fast_training=1")
    print()

    sweep = run_curriculum_depth_sweep(
        depths=depths,
        seed=seed,
        max_bridge_probes=20,
        qa_subset=qa_subset,
        fast_training=True,
    )
    print(format_curriculum_sweep_table(sweep))
    print()
    if os.environ.get("DECOMPOSE", "1").strip() in ("1", "true", "yes"):
        from neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference import (
            format_holdout_decomposition,
        )

        print("=== HOLDOUT DECOMPOSITION BY DEPTH ===")
        for depth, entry in sweep["depths"].items():  # type: ignore[union-attr]
            decomp = entry["metrics"].get("holdout_decomposition")
            if decomp:
                print(f"\n--- {depth} ---")
                print(format_holdout_decomposition(decomp))
    print()
    print("Phase hints:")
    for depth, entry in sweep["depths"].items():  # type: ignore[union-attr]
        print(f"  {depth:20} {entry['phases_hint']}")


if __name__ == "__main__":
    main()
