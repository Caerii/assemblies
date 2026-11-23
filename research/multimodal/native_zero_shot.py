"""Native Brain projection baseline for the R1f independent-target gate.

This deliberately tests UNSUPERVISED native source-to-role projection.
It does not pretend to train the teacher's role transform: targets are
never supplied to Brain. It is a lower-bound native baseline, NOT a
supervised adapter or evidence of learned compositionality.

Run from checkout:
 uv run python -m research.multimodal.native_zero_shot --seed 42
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.multimodal.zero_shot_gate import Config, evaluate


class NativeProjectionBaseline:
    def __init__(self, cfg: Config):
        from neural_assemblies.core.brain import Brain
        self.cfg = cfg
        self.brain = Brain(engine="numpy_exact", seed=cfg.seed, p=0.1, norm_init=False)
        self.brain.add_area("LEX", n=cfg.n, k=cfg.k, beta=0.1)
        self.brain.materialize_area("LEX", storage="dense")
        for role in cfg.roles:
            self.brain.add_area(role, n=cfg.n, k=cfg.k, beta=0.1)
            self.brain.materialize_area(role, storage="dense")
        self.writes = 0

    def _activate(self, cue):
        from neural_assemblies.core.index_spaces import NeuronIds
        from neural_assemblies.assembly_calculus.assembly import Assembly
        from neural_assemblies.assembly_calculus.ops import activate_assembly
        activate_assembly(self.brain, Assembly("LEX", NeuronIds(cue)))

    def train(self, item, role, cue, target):
        # Intentionally discard target: no native teacher-forced role
        # transformation is established. This is unsupervised projection.
        self._activate(cue)
        self.brain.clear_activity([role])
        self.brain.project({}, {"LEX": [role]})
        self.writes += 1

    def predict(self, item, role, cue):
        from neural_assemblies.assembly_calculus.assembly import Assembly
        with self.brain.read_only(seed=self.cfg.seed * 100000 + item):
            with self.brain.frozen():
                self._activate(cue)
                self.brain.clear_activity([role])
                self.brain.project({}, {"LEX": [role]})
                return np.sort(np.asarray(
                    Assembly.from_area(self.brain, role).neuron_ids,
                    dtype=np.uint32))


def run(cfg: Config):
    cfg.validate()
    outcome = evaluate(cfg, NativeProjectionBaseline)
    outcome["status"] = "native_unsupervised_projection_lower_bound"
    outcome["caveats"].extend([
        "Native baseline discards teacher targets during train; NOT supervised role learning.",
        "The native LEX area must support activation of arbitrary stable IDs.",
        "No guarantee the backend's projection materializes a full k winners.",
        "No held-out target or teacher permutation is passed to native Brain.",
    ])
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family", choices=("cyclic_shift", "random_permutation"),
                    default="cyclic_shift")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--contents", type=int, default=12)
    args = ap.parse_args()
    print(json.dumps(run(Config(seed=args.seed, transform_family=args.family,
                                n=args.n, k=args.k, contents=args.contents)),
                     indent=2))


if __name__ == "__main__":
    main()
