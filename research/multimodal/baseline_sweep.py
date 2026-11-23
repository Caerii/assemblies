"""Seed-paired R1k structural generalization baseline sensitivity sweep.

Runs only *unprivileged* train-only baselines; no native Brain execution.
Dataset seed remains private from each learner. All metrics are per-seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from .zero_shot_gate import (
        Config, evaluate, ShiftRoleLearner, HebbianRoleMap,
        ExactPairMemory, CopyBaseline,
    )
except ImportError:
    from zero_shot_gate import (
        Config, evaluate, ShiftRoleLearner, HebbianRoleMap,
        ExactPairMemory, CopyBaseline,
    )


BASELINES = (
    ("shared_shift", ShiftRoleLearner),
    ("hebbian_pairs", HebbianRoleMap),
    ("exact_pair_memory", ExactPairMemory),
    ("identity_copy", CopyBaseline),
)


@dataclass(frozen=True)
class Sweep:
    geometry: tuple[tuple[int, int], ...] = ((128, 8), (256, 8), (256, 16), (512, 16))
    seeds: tuple[int, ...] = tuple(range(12))
    model_seed: int = 1729
    contents: int = 12

    def validate(self):
        if not self.geometry or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("nonempty geometry and unique seeds required")
        if type(self.model_seed) is not int or self.model_seed < 0:
            raise ValueError("invalid public model seed")
        if any(type(s) is not int or s < 0 or s == self.model_seed for s in self.seeds):
            raise ValueError("private data seeds must be nonnegative and distinct from model seed")
        if len(set(self.geometry)) != len(self.geometry):
            raise ValueError("duplicate geometry")
        for n, k in self.geometry:
            Config(n=n, k=k, contents=self.contents, seed=self.seeds[0],
                   model_seed=self.model_seed, transform_family="cyclic_shift").validate()
        return self


def run(spec: Sweep) -> dict:
    spec.validate()
    rows = []
    for n, k in spec.geometry:
        for seed in spec.seeds:
            cfg = Config(n=n, k=k, contents=spec.contents, seed=seed,
                         model_seed=spec.model_seed, transform_family="cyclic_shift")
            for name, factory in BASELINES:
                result = evaluate(cfg, factory)
                metrics = result["summary"]
                rows.append({
                    "n": n, "k": k, "dataset_seed": seed, "model": name,
                    "train_exact": metrics["train"]["exact"],
                    "heldout_exact": metrics["heldout"]["exact"],
                    "heldout_overlap": metrics["heldout"]["overlap"],
                    "heldout_strict_rank1": metrics["heldout"]["strict_rank1"],
                    "targets_sha256": result["dataset"]["targets_sha256"],
                })
    summary = []
    for n, k in spec.geometry:
        for name, _ in BASELINES:
            selected = [row for row in rows if (row["n"], row["k"], row["model"]) == (n, k, name)]
            summary.append({
                "n": n, "k": k, "model": name, "seeds": len(selected),
                **{field: {"mean": float(np.mean([r[field] for r in selected])),
                           "min": float(np.min([r[field] for r in selected])),
                           "max": float(np.max([r[field] for r in selected]))}
                   for field in ("train_exact", "heldout_exact", "heldout_overlap",
                                 "heldout_strict_rank1")},
            })
    serialized_rows = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "synthetic_train_only_baselines_not_native_brain",
        "spec": asdict(spec), "rows_sha256": hashlib.sha256(serialized_rows).hexdigest(),
        "summary": summary, "rows": rows,
        "limitations": [
            "SharedShiftLearner knows the cyclic-shift hypothesis family.",
            "All data are synthetic; this cannot demonstrate neural assembly compositionality.",
            "The seed-level range is descriptive, not a population confidence interval.",
            "The baseline storage/computation budgets are not matched.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = run(Sweep(seeds=tuple(a.seeds)))
    data = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if a.output:
        a.output.write_text(data)
    else:
        print(data)


if __name__ == "__main__":
    main()
