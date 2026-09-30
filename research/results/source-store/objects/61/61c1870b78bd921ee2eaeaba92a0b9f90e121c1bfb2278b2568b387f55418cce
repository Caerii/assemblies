#!/usr/bin/env python3
"""Record COLT 2022 ten-class separable classification golden."""

from __future__ import annotations

import json
from pathlib import Path

from research.json_documents import write_new_document

from neural_assemblies.assembly_calculus.ops import project
from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.learn import classify, learn_separable_classes

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "literature" / "parity" / "golden" / "colt2022_mnist.json"

SEED = 42
N = 5000
K = 80
P = 0.05
BETA = 0.1
ROUNDS = 8
N_CLASSES = 10


def main() -> None:
    brain = Brain(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    brain.add_area("CLASS", N, K, BETA)
    class_stimuli = {str(i): f"s{i}" for i in range(N_CLASSES)}
    for stim in class_stimuli.values():
        brain.add_stimulus(stim, K)

    lexicon, min_pairwise = learn_separable_classes(brain, class_stimuli, "CLASS", rounds=ROUNDS)
    correct = 0
    for label, stim in class_stimuli.items():
        query = project(brain, stim, "CLASS", rounds=3)
        pred = classify(lexicon, query, threshold=0.3)
        if pred == label:
            correct += 1
    accuracy = correct / N_CLASSES

    golden = {
        "protocol": "colt2022_mnist",
        "source": "10-class learn_separable_classes (MNIST hierarchical port pending)",
        "recorded": "2026-06-23",
        "parameters": {
            "seed": SEED,
            "n": N,
            "k": K,
            "p": P,
            "beta": BETA,
            "rounds": ROUNDS,
            "n_classes": N_CLASSES,
        },
        "metrics": {
            "train_classify_accuracy": round(accuracy, 4),
            "min_pairwise_overlap": round(min_pairwise, 6),
            "chance_overlap": round(K / N, 6),
        },
        "tolerance": 0.02,
        "thresholds": {
            "train_classify_accuracy_min": 1.0,
            "min_pairwise_overlap_max": 0.5,
        },
        "notes": "Full MNIST hierarchical learner: legacy/root_modules/image_learner.py — TODO port.",
        "legacy_target": {
            "module": "legacy/root_modules/image_learner.py",
            "areas": ["LOW_LEVEL", "MID_LEVEL", "HIGH_LEVEL", "CLASS_AREA"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_new_document(OUT, golden)
    print(json.dumps(golden["metrics"], indent=2))


if __name__ == "__main__":
    main()
