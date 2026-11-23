"""Exact structural zero-shot calibration, not a learned neural model.

A fixed invertible role permutation factors a lexical code into a role-coded
representation without writing any role/content pairing. This is an oracle
upper-bound control, not evidence of neural compositionality.

Run: uv run python -m research.multimodal.zero_shot_baselines
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class Config:
    n: int = 1024
    k: int = 20
    contents: int = 16
    seed: int = 42
    roles: tuple[str, str] = ("SUBJ", "OBJ")

    def validate(self):
        if not 0 < self.k < self.n or self.contents < 4 or self.contents % 2:
            raise ValueError("invalid geometry or balanced split")
        if len(set(self.roles)) != 2:
            raise ValueError("roles must be distinct")
        return self


def split(cfg: Config):
    cfg.validate()
    train, heldout = [], []
    for i in range(cfg.contents):
        for j, role in enumerate(cfg.roles):
            (train if (i + j) % 2 == 0 else heldout).append((i, role))
    return train, heldout


def experiment(cfg: Config):
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    # Lexical codes allocated independently of syntactic role assignments.
    lexical = np.stack([
        np.sort(rng.choice(cfg.n, cfg.k, replace=False))
        for _ in range(cfg.contents)
    ])
    # Fixed role operators; no per-combination optimization.
    role_maps = {role: rng.permutation(cfg.n) for role in cfg.roles}
    inverse = {role: np.argsort(perm) for role, perm in role_maps.items()}

    def encode(i, role):
        return np.sort(role_maps[role][lexical[i]])

    def decode(code, role):
        restored = np.sort(inverse[role][code])
        hits = [
            len(np.intersect1d(restored, x, assume_unique=True))
            for x in lexical
        ]
        ordered = sorted(range(cfg.contents), key=lambda j: (-hits[j], j))
        return ordered[0], hits[ordered[0]] - hits[ordered[1]]

    train, heldout = split(cfg)
    memorized = {(i, role): i for i, role in train}
    results = []
    for part, pairs in (("train", train), ("heldout", heldout)):
        for i, role in pairs:
            guess, margin = decode(encode(i, role), role)
            results.append({
                "partition": part, "content": i, "role": role,
                "factorized_correct": int(guess == i and margin > 0),
                "margin": int(margin),
                "pair_lookup_correct": int(memorized.get((i, role)) == i),
            })

    def accuracy(part, field):
        values = [r[field] for r in results if r["partition"] == part]
        return float(np.mean(values))

    return {
        "status": "oracle_structural_calibration_not_assembly_learning",
        "config": asdict(cfg), "train": train, "heldout": heldout,
        "code_sha256": hashlib.sha256(
            lexical.astype("<i8").tobytes()
        ).hexdigest(),
        "summary": {
            "factorized_train": accuracy("train", "factorized_correct"),
            "factorized_heldout": accuracy("heldout", "factorized_correct"),
            "lookup_train": accuracy("train", "pair_lookup_correct"),
            "lookup_heldout": accuracy("heldout", "pair_lookup_correct"),
        },
        "results": results,
        "limitations": [
            "Fixed invertible role maps are an engineered oracle, not learned assembly calculus.",
            "All lexical codes are already known; no unseen lexical entities.",
            "No visual grounding, semantic reasoning, plasticity or neural inference.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(experiment(Config()), indent=2))
