"""R1i: paired native teacher/no-plasticity/shuffled-label experiment.

Run in full checkout:
  uv run python -m research.multimodal.native_teacher_controls \
    --family cyclic_shift --seed 42 --output /tmp/r1i.json

No result is claimed until native Brain actually executes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.multimodal.zero_shot_gate import Config
from research.multimodal.teacher_control_matrix import run_conditions
from research.multimodal.native_teacher_drive import NativeTeacherDriven


def factory(cfg: Config, condition: str, *, gain: float, presentations: int):
    if condition not in ("correct_teacher", "shuffled_teacher", "no_plasticity"):
        raise ValueError("unknown condition")
    beta = 0.0 if condition == "no_plasticity" else 0.1
    return NativeTeacherDriven(cfg, teacher_gain=gain,
                               presentations=presentations, beta=beta)


def run(cfg: Config, *, gain=100., presentations=3):
    result = run_conditions(
        cfg, lambda config, condition: factory(
            config, condition, gain=gain, presentations=presentations))
    for condition in result["conditions"]:
        # Repeated equal predictions cannot prove synaptic immutability.
        # Require full explicit-engine fingerprints and per-write diagnostics.
        if not condition.get("weight_audit_available", False):
            raise AssertionError("native weight fingerprint audit is unavailable")
        clamp = condition.get("clamp_diagnostics")
        if not clamp or len(clamp) != len(result["dataset"]["train"]):
            raise AssertionError("missing per-training-step synapse measurements")
        if condition["condition"] == "no_plasticity" and any(
                row["active_edge_changed"] != 0 for row in clamp):
            raise AssertionError("beta=0 control changed synaptic weights")
    result["status"] = "native_r1i_teacher_controls_unvalidated"
    result["hyperparameters"] = {"teacher_gain": gain, "presentations": presentations,
                                 "plastic_beta": 0.1, "control_beta": 0.0}
    result["limitations"].extend([
        "Fixed teacher gain and repeated presentations are exploratory parameters.",
        "Read repeatability is weaker than a direct proof of weight immutability.",
        "No structural learning claim follows from success on trained pairs.",
    ])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("cyclic_shift", "random_permutation"),
                    default="cyclic_shift")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--contents", type=int, default=12)
    ap.add_argument("--gain", type=float, default=100.)
    ap.add_argument("--presentations", type=int, default=3)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(Config(seed=args.seed, n=args.n, k=args.k,
                        contents=args.contents, transform_family=args.family),
                 gain=args.gain, presentations=args.presentations)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
