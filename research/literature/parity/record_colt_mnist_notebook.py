#!/usr/bin/env python3
"""Record COLT 2022 MNIST numpy notebook protocol golden."""

from __future__ import annotations

import json
from pathlib import Path

from neural_assemblies.programs.colt_mnist_numpy import run_colt_mnist_numpy

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "literature" / "parity" / "golden" / "colt2022_mnist_notebook.json"


def main() -> None:
    result = run_colt_mnist_numpy(seed=42, n_examples=500)
    golden = {
        "protocol": "colt2022_mnist_notebook",
        "source": "learning-with-assemblies/MNIST.ipynb numpy port",
        "recorded": "2026-06-23",
        "parameters": result.parameters,
        "metrics": {
            "mean_accuracy": round(result.mean_accuracy, 4),
            "per_class_accuracy": [round(float(x), 4) for x in result.per_class_accuracy],
            "data_source": result.data_source,
        },
        "thresholds": {
            "mean_accuracy_min": 0.08,
            "metrics_match_tolerance": 0.02,
        },
        "notes": (
            "Full notebook uses n_examples=5000; golden uses 500 for CI speed. "
            "Place mnist_train.csv + mnist_test.csv in data/mnist/ for real MNIST."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(golden["metrics"], indent=2))


if __name__ == "__main__":
    main()
