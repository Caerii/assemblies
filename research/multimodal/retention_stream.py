"""Deterministic continual-association diagnostic on one explicit fiber.

This is not native Brain parity, language acquisition, or visual grounding.
Run: uv run python research/multimodal/retention_stream.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

try:
    from .multiplicative_reference import MultiplicativeFiber
except ImportError:
    from multiplicative_reference import MultiplicativeFiber


@dataclass(frozen=True)
class Config:
    n: int = 256
    k: int = 8
    p: float = 0.1
    beta: float = 0.15
    w_max: float = 20.0
    presentations: int = 1
    loads: tuple[int, ...] = (16, 32, 64, 128)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def validate(self):
        if not 0 < self.k <= self.n:
            raise ValueError("require 0 < k <= n")
        if not 0 <= self.p <= 1 or not 0 <= self.beta or self.w_max <= 0:
            raise ValueError("invalid p/beta/w_max")
        if not isinstance(self.presentations, int) or self.presentations < 1:
            raise ValueError("presentations must be a positive integer")
        if not self.loads or any(x <= 0 for x in self.loads):
            raise ValueError("loads must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        return self


def exact(predicted, expected):
    return bool(np.array_equal(predicted, expected))


def jaccard(predicted, expected):
    common = len(np.intersect1d(predicted, expected, assume_unique=True))
    return common / (len(predicted) + len(expected) - common)


def trial(seed: int, config: Config):
    rng = np.random.default_rng(seed)
    max_load = max(config.loads)
    synapses = (rng.random((config.n, config.n)) < config.p).astype(np.float32)
    fiber = MultiplicativeFiber(synapses, k=config.k, beta=config.beta, w_max=config.w_max)
    sources = np.array(
        [rng.choice(config.n, config.k, replace=False) for _ in range(max_load)],
        dtype=np.uint32,
    )
    targets = np.array(
        [rng.choice(config.n, config.k, replace=False) for _ in range(max_load)],
        dtype=np.uint32,
    )
    stream_digest = hashlib.sha256(sources.tobytes() + targets.tobytes()).hexdigest()
    records = []
    for step in range(max_load):
        fiber.cofire(sources[step], targets[step], rounds=config.presentations)
        load = step + 1
        if load not in config.loads:
            continue
        assembly_exact, assembly_jaccard, episodic_exact, active_coverage = [], [], [], []
        for i in range(load):
            got = fiber.predict(sources[i])
            accessible = np.any(synapses[np.sort(sources[i])][:, targets[i]] > 0, axis=0)
            active_coverage.append(float(accessible.mean()))
            assembly_exact.append(exact(got, np.sort(targets[i])))
            assembly_jaccard.append(jaccard(got, targets[i]))
            matches = [
                j for j in range(load)
                if np.array_equal(np.sort(sources[j]), np.sort(sources[i]))
            ]
            episodic_exact.append(exact(np.sort(targets[matches[-1]]), np.sort(targets[i])))
        early = max(1, load // 4)
        records.append({
            "seed": seed, "load": load, "stream_sha256": stream_digest,
            "assembly_exact": float(np.mean(assembly_exact)),
            "assembly_jaccard": float(np.mean(assembly_jaccard)),
            "assembly_early_exact": float(np.mean(assembly_exact[:early])),
            "target_input_coverage": float(np.mean(active_coverage)),
            "episodic_exact": float(np.mean(episodic_exact)),
            "nonzero_synapses": int(np.count_nonzero(fiber.weights)),
            "synapse_bytes": int(fiber.weights.nbytes),
            "episodic_min_bytes": int(load * config.k * 2 * np.dtype(np.uint32).itemsize),
        })
    return records


def run(config: Config):
    config.validate()
    observations = [row for seed in config.seeds for row in trial(seed, config)]
    metrics = ("assembly_exact", "assembly_jaccard", "assembly_early_exact",
               "episodic_exact", "target_input_coverage")
    summary = []
    for load in config.loads:
        rows = [row for row in observations if row["load"] == load]
        summary.append({"load": load, "seeds": len(rows), **{
            key: {
                "mean": float(np.mean([r[key] for r in rows])),
                "min": float(np.min([r[key] for r in rows])),
                "max": float(np.max([r[key] for r in rows])),
            } for key in metrics
        }})
    return {
        "status": "diagnostic_not_native_brain",
        "config": asdict(config), "python": sys.version.split()[0],
        "numpy": np.__version__, "platform": platform.platform(),
        "observations": observations, "summary": summary,
        "limitations": [
            "The episodic dictionary baseline is not memory-budget matched.",
            "The model is an explicit one-fiber toy, not the native Brain engine.",
            "Independent synthetic codes do not demonstrate composition or grounding.",
            "Input coverage is not a strict mathematical recall upper bound.",
            "No held-out model selection or baseline tuning was performed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--p", type=float, default=.1)
    parser.add_argument("--beta", type=float, default=.15)
    parser.add_argument("--presentations", type=int, default=1)
    parser.add_argument("--loads", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(Config(n=args.n, k=args.k, p=args.p, beta=args.beta,
                        presentations=args.presentations,
                        loads=tuple(args.loads), seeds=tuple(args.seeds)))
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
