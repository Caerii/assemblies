#!/usr/bin/env python3
"""Record NEMO 2025 developmental curriculum golden metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "literature" / "parity" / "golden" / "nemo2025_curriculum.json"

# Fast pinned curriculum path (matches literature parity scale)
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.pop("EMERGENT_DEV_CURRICULUM", None)


def main() -> None:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.evaluation import EvaluationSuite
    from neural_assemblies.assembly_calculus.emergent.curriculum import CurriculumTrainer

    from neural_assemblies.assembly_calculus.emergent.core.grounding import VOCABULARY

    modality_to_pos = {
        "visual": "NOUN",
        "motor": "VERB",
        "properties": "ADJ",
        "spatial": "PREP",
        "social": "PRON",
        "temporal": "ADV",
        "none": "DET",
    }
    test_vocab = {
        word: modality_to_pos[ctx.dominant_modality]
        for word, ctx in VOCABULARY.items()
    }

    parser = EmergentParser(n=5000, k=80, p=0.05, beta=0.1, seed=42, rounds=10)
    trainer = CurriculumTrainer(parser)
    stages_run = []
    stage_metrics = {}

    for stage in (
        "BABBLE",
        "FIRST_WORDS",
        "VOCABULARY_SPURT",
        "TWO_WORD",
        "SENTENCES",
    ):
        result = trainer.train_stage(stage)
        stages_run.append(stage)
        stage_metrics[stage] = {
            "classification_accuracy": round(result.classification_accuracy, 4),
            "beta": result.beta,
            "sentences_trained": result.sentences_trained,
            "phases_run": list(result.phases_run),
        }

    suite = EvaluationSuite(parser)
    pos = suite.evaluate_classification(test_vocab)
    wo = suite.evaluate_word_order(target="SVO")
    role_probes = suite.evaluate_roles([
        {"words": ["the", "dog", "runs"], "expected_roles": {"dog": "AGENT"}},
        {"words": ["the", "cat", "chases", "the", "bird"], "expected_roles": {"cat": "AGENT", "bird": "PATIENT"}},
    ])

    golden = {
        "protocol": "nemo2025_curriculum",
        "source": "EmergentParser CurriculumTrainer fast path",
        "recorded": "2026-06-23",
        "parameters": {
            "n": 5000,
            "k": 80,
            "p": 0.05,
            "beta": 0.1,
            "seed": 42,
            "rounds": 10,
            "EMERGENT_FAST_TRAINING": "1",
            "stages": stages_run,
        },
        "metrics": {
            "stage_metrics": stage_metrics,
            "pos_accuracy": round(pos["accuracy"], 4),
            "word_order_correct": wo.get("correct", wo.get("is_correct")),
            "role_probes": role_probes,
        },
        "thresholds": {
            "pos_accuracy_min": 0.80,
            "word_order_correct": True,
        },
        "config_ref": "parity/configs/nemo2025.yaml",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(golden["metrics"], indent=2))


if __name__ == "__main__":
    main()
