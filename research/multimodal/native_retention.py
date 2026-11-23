"""Native Brain stimulus-retention experiment (R1b).

Run from repository root:
    uv run python research/multimodal/native_retention.py --seeds 0 1 2

This tests identity retention of learned assemblies, NOT retrieval of a
specified arbitrary target: the first learned winner set is the label.
Refraction's masked readout is intentionally NOT claimed or implemented.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import platform
import sys

import numpy as np


@dataclass(frozen=True)
class NativeConfig:
    n: int = 512
    k: int = 20
    p: float = 0.1
    beta: float = 0.1
    write_rounds: int = 8
    read_rounds: int = 8
    loads: tuple[int, ...] = (4, 8, 16)
    seeds: tuple[int, ...] = (0, 1, 2)
    arms: tuple[str, ...] = ("recurrent", "feedforward", "no_plasticity")

    def validate(self):
        if not 0 < self.k <= self.n or not 0 < self.p <= 1:
            raise ValueError("invalid population or connectivity")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must be in [0,1)")
        if self.write_rounds < 1 or self.read_rounds < 1:
            raise ValueError("rounds must be positive")
        if not self.loads or any(x < 1 for x in self.loads):
            raise ValueError("loads must be positive")
        if tuple(sorted(set(self.loads))) != self.loads:
            raise ValueError("loads must be strictly increasing")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        supported = {"recurrent", "feedforward", "no_plasticity"}
        if not self.arms or len(set(self.arms)) != len(self.arms) or set(self.arms) - supported:
            raise ValueError("unsupported or repeated arm")
        return self


def canonical(winners, k: int):
    values = np.asarray(winners, dtype=np.uint32)
    if len(values) != k or len(np.unique(values)) != k:
        raise AssertionError("invalid winner cardinality")
    return np.sort(values)


def overlap(a, b) -> float:
    intersection = np.intersect1d(a, b, assume_unique=True).size
    return float(intersection) / len(a)


def _native():
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus import project
    return Brain, project


def _evaluate(brain, project, name: str, original: np.ndarray,
              cfg: NativeConfig, seed: int, i: int, recurrent: bool):
    # Both read-only and frozen are required: one restores state/recruitment,
    # while the other prevents synaptic mutation. Keep the seed pinned.
    with ExitStack() as stack:
        stack.enter_context(brain.read_only(seed=seed * 100_000 + i))
        stack.enter_context(brain.frozen())
        brain.clear_activity(["memory"])
        retrieved = project(brain, name, "memory", rounds=cfg.read_rounds,
                            recurrent=recurrent)
        observed = canonical(retrieved.winners, cfg.k)
    return overlap(original, observed), bool(np.array_equal(original, observed))


def trial(seed: int, arm: str, cfg: NativeConfig):
    Brain, project = _native()
    beta = 0.0 if arm == "no_plasticity" else cfg.beta
    recurrent = arm != "feedforward"
    brain = Brain(engine="numpy_exact", seed=seed, p=cfg.p, norm_init=False)
    brain.add_area("memory", n=cfg.n, k=cfg.k, beta=beta)
    for i in range(max(cfg.loads)):
        brain.add_stimulus(f"item_{i:04d}", cfg.k)
    original = []
    records = []
    for i in range(max(cfg.loads)):
        name = f"item_{i:04d}"
        brain.clear_activity(["memory"])
        written = project(brain, name, "memory", rounds=cfg.write_rounds,
                          recurrent=recurrent)
        original.append(canonical(written.winners, cfg.k))
        load = i + 1
        if load not in cfg.loads:
            continue
        similarities = []
        exact = []
        for j in range(load):
            sim, ok = _evaluate(brain, project, f"item_{j:04d}", original[j],
                                cfg, seed, j, recurrent)
            similarities.append(sim)
            exact.append(ok)
        records.append({
            "seed": seed, "arm": arm, "load": load,
            "mean_overlap": float(np.mean(similarities)),
            "exact_recovery": float(np.mean(exact)),
            "early_overlap": float(np.mean(similarities[:max(1, load // 4)])),
            "early_exact": float(np.mean(exact[:max(1, load // 4)])),
            "stored_assembly_digest": hashlib.sha256(
                b"".join(a.tobytes() for a in original)).hexdigest(),
        })
    return records


def run(cfg: NativeConfig):
    cfg.validate()
    records = [row for seed in cfg.seeds for arm in cfg.arms
               for row in trial(seed, arm, cfg)]
    summary = []
    for arm in cfg.arms:
        for load in cfg.loads:
            subset = [r for r in records if r["arm"] == arm and r["load"] == load]
            summary.append({
                "arm": arm, "load": load, "seeds": len(subset),
                **{metric: {"mean": float(np.mean([r[metric] for r in subset])),
                            "min": float(np.min([r[metric] for r in subset])),
                            "max": float(np.max([r[metric] for r in subset]))}
                   for metric in ("mean_overlap", "exact_recovery",
                                  "early_overlap", "early_exact")},
            })
    return {
        "status": "native_brain_identity_retention_not_binding",
        "config": asdict(cfg), "python": sys.version.split()[0],
        "numpy": np.__version__, "platform": platform.platform(),
        "readout_mode": "read_only_frozen_net_unmasked",
        "records": records, "summary": summary,
        "limitations": [
            "Measures recall of the original learned assembly, not specified target binding.",
            "Read-only inhibits recruitment; this is a deliberate observation policy.",
            "Refraction masked readout is NOT implemented in this experiment.",
            "No native Rust or GPU acceleration is benchmarked.",
            "Stimulus order and registration are fixed per seed and arm.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--p", type=float, default=.1)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--write-rounds", type=int, default=8)
    parser.add_argument("--read-rounds", type=int, default=8)
    parser.add_argument("--loads", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+",
                        default=["recurrent", "feedforward", "no_plasticity"])
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(NativeConfig(
        n=args.n, k=args.k, p=args.p, beta=args.beta,
        write_rounds=args.write_rounds, read_rounds=args.read_rounds,
        loads=tuple(args.loads), seeds=tuple(args.seeds), arms=tuple(args.arms)))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
