"""R1h native teacher-guided projection using explicit target drive.

The only supervision is each observed cue/target pair. The hidden role
transform is NEVER provided. Training drives the target area's winners
with a finite additive signal; inference runs with no teacher drive.
This is a supervised native baseline, not a guarantee of learned rules.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from research.multimodal.zero_shot_gate import Config, evaluate


class NativeTeacherDriven:
    def __init__(self, cfg: Config, teacher_gain: float = 100.0,
                 presentations: int = 3, beta: float = 0.1):
        from neural_assemblies.core.brain import Brain
        if not np.isfinite(teacher_gain) or teacher_gain <= 0:
            raise ValueError("teacher_gain must be positive and finite")
        if type(presentations) is not int or presentations < 1:
            raise ValueError("presentations must be a positive integer")
        if not np.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("beta must be finite and in [0, 1)")
        self.cfg = cfg.validate()
        self.teacher_gain = float(teacher_gain)
        self.presentations = presentations
        self.beta = float(beta)
        self.brain = Brain(engine="numpy_exact", seed=cfg.seed, p=.1,
                           norm_init=False)
        # External drive requires the native dense explicit engine.
        self.brain.add_area("LEX", n=cfg.n, k=cfg.k, beta=self.beta, explicit=True)
        for role in cfg.roles:
            self.brain.add_area(role, n=cfg.n, k=cfg.k, beta=self.beta,
                                explicit=True)
        self.train_pairs = set()
        self.training_calls = 0

    def train(self, item, role, cue, target):
        from neural_assemblies.assembly_calculus.ops import activate_assembly
        from neural_assemblies.assembly_calculus.assembly import Assembly
        from neural_assemblies.core.index_spaces import NeuronIds
        cue = np.asarray(cue, dtype=np.uint32)
        target = np.asarray(target, dtype=np.uint32)
        if (len(cue) != self.cfg.k or len(target) != self.cfg.k
                or len(np.unique(cue)) != self.cfg.k
                or len(np.unique(target)) != self.cfg.k
                or np.any(cue >= self.cfg.n) or np.any(target >= self.cfg.n)):
            raise ValueError("invalid training assembly")
        drive = np.zeros(self.cfg.n, dtype=np.float32)
        drive[target] = self.teacher_gain
        for _ in range(self.presentations):
            activate_assembly(self.brain, Assembly("LEX", NeuronIds(cue)))
            self.brain.clear_activity([role])
            self.brain.project({}, {"LEX": [role]},
                               external_drive={role: drive})
        self.train_pairs.add((item, role))
        self.training_calls += 1

    def predict(self, item, role, cue):
        from neural_assemblies.assembly_calculus.ops import activate_assembly
        from neural_assemblies.assembly_calculus.assembly import Assembly
        from neural_assemblies.core.index_spaces import NeuronIds
        with self.brain.read_only(seed=self.cfg.seed * 100000 + item):
            with self.brain.frozen():
                activate_assembly(self.brain, Assembly(
                    "LEX", NeuronIds(np.asarray(cue, dtype=np.uint32))))
                self.brain.clear_activity([role])
                # No target or teacher drive in inference.
                self.brain.project({}, {"LEX": [role]})
                return np.sort(np.asarray(
                    Assembly.from_area(self.brain, role).neuron_ids,
                    dtype=np.uint32))


def run(cfg: Config):
    result = evaluate(cfg, NativeTeacherDriven)
    result["status"] = "native_teacher_driven_projection_pending_validation"
    result["caveats"].extend([
        "Teacher targets appear only during observed training pairs.",
        "Teacher drive is an experimental clamp, not a native role-transform learning law.",
        "Full native engine execution and parameter sweeps are still required.",
        "Exact target recall may fail despite teacher-driven training.",
    ])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", choices=["cyclic_shift", "random_permutation"],
                        default="cyclic_shift")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--contents", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(Config(seed=args.seed, n=args.n, k=args.k,
                                contents=args.contents,
                                transform_family=args.family)), indent=2))


if __name__ == "__main__":
    main()
