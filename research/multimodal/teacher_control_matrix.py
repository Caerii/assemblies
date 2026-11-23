"""Paired supervised-learning controls for the independent-target R1f benchmark.

This module is backend agnostic: models must implement train(i,role,cue,target)
and predict(i,role,cue). The same dataset and test queries are used across
correct, shuffled, and no-plasticity conditions. Native Brain results require
running the native model in a complete repository checkout.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict
from typing import Callable

import numpy as np

try:
    from .zero_shot_gate import Config, Dataset, _validate_output, _rank
except ImportError:
    from zero_shot_gate import Config, Dataset, _validate_output, _rank

CONDITIONS = ("correct_teacher", "shuffled_teacher", "no_plasticity")


def _teacher_labels(dataset: Dataset, condition: str) -> dict:
    """Deterministic derangement of TRAIN labels only, within each role."""
    if condition not in CONDITIONS:
        raise ValueError(f"invalid condition: {condition}")
    labels = {pair: dataset.target(*pair) for pair in dataset.train}
    if condition != "shuffled_teacher":
        return labels
    # Rotation is a derangement of labels (but codes can still collide).
    groups = defaultdict(list)
    for item, role in dataset.train:
        groups[role].append(item)
    for role, items in groups.items():
        if len(items) < 2:
            raise ValueError("need at least two trained examples per role")
        prior = [labels[i, role].copy() for i in items]
        for index, item in enumerate(items):
            labels[item, role] = prior[(index + 1) % len(items)]
    return labels


def _fingerprint(labels: dict) -> str:
    parts = [f"{i}:{role}:".encode() + target.tobytes()
             for (i, role), target in sorted(labels.items())]
    return hashlib.sha256(b"".join(parts)).hexdigest()


def run_conditions(cfg: Config, model_factory: Callable):
    """Factory(cfg, condition) constructs an independent seed-matched model.

    Evaluation never calls train with a held-out label. A repeated prediction
    detects non-repeatable outputs but cannot prove synaptic non-mutation.
    """
    cfg.validate()
    dataset = Dataset(cfg)
    output = []
    for condition in CONDITIONS:
        model = model_factory(cfg, condition)
        labels = _teacher_labels(dataset, condition)
        for pair in dataset.train:
            i, role = pair
            model.train(i, role, dataset.cues[i].copy(), labels[pair].copy())
        if hasattr(model, "train_pairs") and set(model.train_pairs) != set(dataset.train):
            raise AssertionError("training pair accounting mismatch")
        observations = []
        for partition, pairs in (("train", dataset.train), ("heldout", dataset.test)):
            for i, role in pairs:
                query = dataset.cues[i].copy()
                pred1 = _validate_output(model.predict(i, role, query.copy()), cfg.n, cfg.k)
                pred2 = _validate_output(model.predict(i, role, query.copy()), cfg.n, cfg.k)
                stable = (pred1 is None and pred2 is None) or (
                    pred1 is not None and pred2 is not None and np.array_equal(pred1, pred2))
                target = dataset.target(i, role)
                negatives = [dataset.target(j, role) for j in range(cfg.contents) if j != i]
                metrics = _rank(pred1, target, negatives)
                observations.append({"partition": partition, "item": i, "role": role,
                                     "repeatable": int(stable), **metrics})
        summary = {}
        for partition in ("train", "heldout"):
            chosen = [r for r in observations if r["partition"] == partition]
            summary[partition] = {metric: float(np.mean([r[metric] for r in chosen]))
                                  for metric in ("exact", "overlap", "strict_rank1", "repeatable")}
        output.append({"condition": condition, "train_labels_sha256": _fingerprint(labels),
                       "summary": summary, "observations": observations,
                       "training_calls": getattr(model, "training_calls", None),
                       "clamp_diagnostics": getattr(model, "clamp_diagnostics", None)})
    return {"status": "controlled_teacher_comparison_not_native_validation",
            "config": asdict(cfg), "dataset": dataset.manifest(),
            "conditions": output,
            "limitations": [
                "Repeated predictions test output consistency, not weight immutability.",
                "Shuffled training labels are permuted within a role and use no held-out targets.",
                "Teacher/no-plasticity controls must initialize the same connectome for causal comparison.",
                "A native numerical claim requires a complete repository checkout and run records.",
            ]}
