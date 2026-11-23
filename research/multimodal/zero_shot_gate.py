"""Leakage-resistant zero-shot role transformation benchmark.

Synthetic independent targets; NOT native assembly-calculus evidence.
Adapter: train(item, role, cue, target), predict(item, role, cue).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class Config:
    n: int = 512
    k: int = 16
    contents: int = 12
    seed: int = 42
    roles: tuple[str, str] = ("SUBJ", "OBJ")
    transform_family: str = "random_permutation"

    def validate(self):
        if not isinstance(self.n, int) or not isinstance(self.k, int) or not 0 < self.k < self.n:
            raise ValueError("invalid population and winner count")
        if not isinstance(self.contents, int) or self.contents < 4 or self.contents % 2:
            raise ValueError("contents must be even and at least 4")
        if len(self.roles) != 2 or len(set(self.roles)) != 2:
            raise ValueError("two distinct roles required")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.transform_family not in ("random_permutation", "cyclic_shift"):
            raise ValueError("unknown transform family")
        return self


class Dataset:
    def __init__(self, cfg: Config):
        self.cfg = cfg.validate()
        rng = np.random.default_rng(cfg.seed)
        self.cues = tuple(np.sort(rng.choice(cfg.n, cfg.k, replace=False).astype(np.uint32))
                          for _ in range(cfg.contents))
        if cfg.transform_family == "cyclic_shift":
            self.maps = {role: (np.arange(cfg.n) + int(rng.integers(1, cfg.n))) % cfg.n
                         for role in cfg.roles}
        else:
            self.maps = {role: rng.permutation(cfg.n) for role in cfg.roles}
        self.train = tuple((i, role) for i in range(cfg.contents)
                           for j, role in enumerate(cfg.roles) if (i + j) % 2 == 0)
        self.test = tuple((i, role) for i in range(cfg.contents)
                          for j, role in enumerate(cfg.roles) if (i + j) % 2 == 1)
        self._targets = {(i, role): np.sort(self.maps[role][self.cues[i]].astype(np.uint32))
                         for i in range(cfg.contents) for role in cfg.roles}

    def target(self, item, role):
        return self._targets[item, role].copy()

    def manifest(self):
        src = b"".join(x.tobytes() for x in self.cues)
        target = b"".join(self._targets[i, role].tobytes()
                          for i in range(self.cfg.contents) for role in self.cfg.roles)
        return {"config": asdict(self.cfg), "cues_sha256": hashlib.sha256(src).hexdigest(),
                "targets_sha256": hashlib.sha256(target).hexdigest(),
                "train": self.train, "heldout": self.test}


def _validate_output(value, n, k):
    if value is None:
        return None
    a = np.asarray(value)
    if (a.ndim != 1 or len(a) != k or not np.issubdtype(a.dtype, np.integer)
            or (a < 0).any() or (a >= n).any() or len(np.unique(a)) != k):
        raise ValueError("model returned invalid assembly; cannot score it")
    return np.sort(a.astype(np.uint32))


def _rank(got, target, distractors):
    if got is None:
        return {"exact": 0, "overlap": 0., "strict_rank1": 0, "margin": -1}
    hits = int(np.intersect1d(got, target, assume_unique=True).size)
    best_other = max((np.intersect1d(got, x, assume_unique=True).size
                      for x in distractors), default=-1)
    margin = hits - best_other
    return {"exact": int(np.array_equal(got, target)), "overlap": hits / len(target),
            "strict_rank1": int(margin > 0), "margin": int(margin)}


def evaluate(cfg: Config, model_factory: Callable, *, expose_labels=False):
    """Never train on held-out pair. Only the engineered oracle gets labels."""
    d = Dataset(cfg)
    model = model_factory(d if expose_labels else cfg)
    for i, role in d.train:
        model.train(i, role, d.cues[i].copy(), d.target(i, role))
    records = []
    for part, pairs in (("train", d.train), ("heldout", d.test)):
        for i, role in pairs:
            cue = d.cues[i].copy()
            expected = d.target(i, role)
            predicted = _validate_output(model.predict(i, role, cue), cfg.n, cfg.k)
            alternatives = [d.target(j, role) for j in range(cfg.contents) if j != i]
            metrics = _rank(predicted, expected, alternatives)
            records.append({"partition": part, "item": i, "role": role, **metrics})
    return {"status": "synthetic_independent_targets_not_native_brain",
            "model": type(model).__name__, "dataset": d.manifest(),
            "summary": {part: {metric: float(np.mean([r[metric] for r in records
                                                   if r["partition"] == part]))
                               for metric in ("exact", "overlap", "strict_rank1")}
                        for part in ("train", "heldout")},
            "records": records,
            "caveats": ["Synthetic teacher role permutations, not learned language syntax",
                        "Not memory-budget matched", "No native Brain/FPGA/Rust results"]}


class ExactPairMemory:
    def __init__(self, _):
        self.memory = {}

    def train(self, item, role, cue, target):
        self.memory[item, role] = target.copy()

    def predict(self, item, role, cue):
        return self.memory.get((item, role))


class CopyBaseline:
    def __init__(self, _):
        pass

    def train(self, item, role, cue, target):
        pass

    def predict(self, item, role, cue):
        return cue.copy()


class KnownPermutationOracle:
    """Upper bound has access to full ground-truth maps (not learned)."""
    def __init__(self, dataset: Dataset):
        self.maps = dataset.maps

    def train(self, item, role, cue, target):
        pass

    def predict(self, item, role, cue):
        return np.sort(self.maps[role][cue].astype(np.uint32))



class HebbianRoleMap:
    """Sparse local coactivation baseline, with no held-out writes."""
    def __init__(self, cfg: Config):
        self.n = cfg.n
        self.k = cfg.k
        self.weights = {role: np.zeros((cfg.n, cfg.n), dtype=np.float32)
                        for role in cfg.roles}

    def train(self, item, role, cue, target):
        self.weights[role][np.ix_(cue, target)] += np.float32(1.)

    def predict(self, item, role, cue):
        drive = np.sum(self.weights[role][cue], axis=0)
        order = np.lexsort((np.arange(self.n), -drive))
        return np.sort(order[:self.k].astype(np.uint32))

def demo():
    cfg = Config()
    return {"oracle": evaluate(cfg, KnownPermutationOracle, expose_labels=True),
            "pair_memory": evaluate(cfg, ExactPairMemory),
            "copy": evaluate(cfg, CopyBaseline),
            "hebbian": evaluate(cfg, HebbianRoleMap)}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))


class ShiftRoleLearner:
    """Learn a shared cyclic role shift from train-only exemplar pairs."""
    def __init__(self, cfg: Config):
        self.n = cfg.n
        self.k = cfg.k
        self.scores = {role: np.zeros(cfg.n, dtype=np.int64) for role in cfg.roles}
        self.count = {role: 0 for role in cfg.roles}

    def train(self, item, role, cue, target):
        # For each candidate shift count role-conditioned target overlaps.
        target_bits = np.zeros(self.n, dtype=np.int64)
        target_bits[target] = 1
        for shift in range(self.n):
            self.scores[role][shift] += int(np.sum(target_bits[(cue + shift) % self.n]))
        self.count[role] += 1

    def predict(self, item, role, cue):
        if not self.count[role]:
            return None
        shift = int(np.argmax(self.scores[role]))
        return np.sort(((cue + shift) % self.n).astype(np.uint32))
