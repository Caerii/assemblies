"""R1c: paired masked/net half-cue recall on one trained native Brain.

Run from repo root:
    uv run python research/multimodal/refraction_paired.py --seeds 42 43 44

One brain is trained ONCE for each (seed, arm), and the very same stored
assemblies and probe indices are read in both modes. No numerical results
are claimed until this file has run on the full repository installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PairedConfig:
    n: int = 2000
    k: int = 60
    p: float = 0.5
    beta: float = 0.1
    rounds: int = 8
    stimulus_parts: int = 10
    refraction_factor: float = 0.5
    loads: tuple[int, ...] = (8, 16, 32)
    recall_sample: int = 16
    seeds: tuple[int, ...] = (42, 43, 44)

    def validate(self):
        if not 0 < self.k <= self.n or not 0 < self.p <= 1:
            raise ValueError("invalid n/k/p")
        if not 0 < self.beta < 1 or not 0 <= self.refraction_factor <= 1:
            raise ValueError("invalid beta/refraction_factor")
        if (self.stimulus_parts < 1 or self.k % self.stimulus_parts
                or self.rounds < 1 or self.recall_sample < 1):
            raise ValueError("invalid stimulus parts, rounds or recall size")
        if (not self.loads or self.loads != tuple(sorted(set(self.loads)))
                or self.loads[0] < 2 or not self.seeds
                or len(self.seeds) != len(set(self.seeds))):
            raise ValueError("invalid checkpoints or seeds")
        return self


def _native():
    from neural_assemblies.core.brain import Brain
    return Brain


def _snapshot(eng) -> np.ndarray:
    return np.sort(np.asarray(eng.get_winners("memory"), dtype=np.int64).copy())


def _recall_one(brain, eng, stored: list[np.ndarray], item: int, masked: bool,
                rounds: int, seed: int) -> tuple[bool, float]:
    # The native engine here is fully materialized, so winner indices are
    # actual neuron IDs. This assumption is NOT valid for lazy sparse areas.
    cue = np.asarray(stored[item][:len(stored[item]) // 2], dtype=np.int64)
    try:
        brain.set_masked_readout("memory", masked)
        with ExitStack() as scope:
            scope.enter_context(brain.read_only(seed=seed))
            scope.enter_context(brain.frozen())
            eng.set_winners("memory", cue)
            for _ in range(rounds):
                eng.project_into("memory", [], ["memory"], plasticity_enabled=False)
            got = set(map(int, eng.get_winners("memory")))
        overlaps = [len(got.intersection(map(int, x))) for x in stored]
        own_overlap = overlaps[item] / len(stored[item])
        return bool(int(np.argmax(overlaps)) == item), float(own_overlap)
    finally:
        # Readout mode is a backend flag, not a transient winner state.
        brain.set_masked_readout("memory", False)


def trial(seed: int, refracted: bool, cfg: PairedConfig) -> list[dict]:
    Brain = _native()
    brain = Brain(p=cfg.p, seed=seed, engine="numpy_sparse",
                  norm_init=True, recurrent_projection=True,
                  synaptic_scaling=False)
    brain.add_area("memory", cfg.n, cfg.k, cfg.beta)
    brain.materialize_area("memory", storage="dense")
    if refracted:
        brain.set_refracted("memory", True, cfg.beta * cfg.refraction_factor)
    eng = brain._engine_for(brain.areas["memory"])
    stored: list[np.ndarray] = []
    output = []
    rng = np.random.default_rng(seed)
    for i in range(max(cfg.loads)):
        parts = [f"cue_{i}_{j}" for j in range(cfg.stimulus_parts)]
        for name in parts:
            brain.add_stimulus(name, cfg.k // cfg.stimulus_parts)
        brain.clear_activity(["memory"])
        for _ in range(cfg.rounds):
            eng.project_into("memory", parts, ["memory"], plasticity_enabled=True)
        now = _snapshot(eng)
        if len(now) != cfg.k or len(np.unique(now)) != cfg.k:
            raise AssertionError("invalid native winner snapshot")
        stored.append(now)
        load = i + 1
        if load not in cfg.loads:
            continue
        selected = np.sort(rng.choice(load, min(load, cfg.recall_sample),
                                      replace=False)).tolist()
        metrics = {}
        for masked in (False, True):
            outcome = [_recall_one(brain, eng, stored, j, masked,
                                   cfg.rounds, seed * 100000 + j) for j in selected]
            metrics["masked" if masked else "net"] = {
                "rank1": float(np.mean([x[0] for x in outcome])),
                "self_overlap": float(np.mean([x[1] for x in outcome])),
            }
        # Same selected items, same trained weights, same readout seed.
        output.append({
            "seed": seed, "refracted": refracted, "load": load,
            "selected": selected,
            "stored_sha256": hashlib.sha256(
                b"".join(x.tobytes() for x in stored)).hexdigest(),
            "net": metrics["net"], "masked": metrics["masked"],
            "paired_delta_rank1": metrics["masked"]["rank1"] - metrics["net"]["rank1"],
        })
    return output


def run(cfg: PairedConfig):
    cfg.validate()
    rows = [r for seed in cfg.seeds for refracted in (False, True)
            for r in trial(seed, refracted, cfg)]
    return {
        "status": "native_paired_refraction_probe",
        "config": asdict(cfg), "rows": rows,
        "summary": [{
            "refracted": arm, "load": m,
            "mean_masked_minus_net_rank1": float(np.mean([
                r["paired_delta_rank1"] for r in rows
                if r["refracted"] == arm and r["load"] == m])),
            "n_seeds": len(cfg.seeds),
        } for arm in (False, True) for m in cfg.loads],
        "limitations": [
            "This is paired cue recovery, not cross-modal binding.",
            "Uses explicit full materialization and internal engine projection.",
            "Rank-1 ties go to the earliest stored assembly.",
            "A read-only probe uses a different recruitment policy from normal inference.",
            "Small load grids do not establish memory-capacity ceilings.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--loads", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--recall-sample", type=int, default=16)
    ap.add_argument("--output")
    args = ap.parse_args()
    result = run(PairedConfig(seeds=tuple(args.seeds), n=args.n, k=args.k,
                              loads=tuple(args.loads),
                              recall_sample=args.recall_sample))
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
