"""R1o optional NATIVE Brain composition adapter, never receives private rule.

Only if the local native package is installed:
    python -m research.multimodal.native_affine_chain --data-seed 23

This is a teacher-guided association baseline built on R1h and sequential
read-only projection. It is not assumed to learn affine algebra and no native
result is reported until run in a complete checkout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .noncommutative_composition import Protocol, run_one


class NativeAffineChain:
    def __init__(self, public, gain=100., presentations=3, beta=.1):
        # The R1h adapter accepts a separate zero-shot Config. Only the
        # PUBLIC model seed is used for brain initialization, never the
        # private data-generation seed or hidden operator parameters.
        from .zero_shot_gate import Config
        from .native_teacher_drive import NativeTeacherDriven
        baseline_cfg = Config(n=public.n, k=public.k, contents=4,
                              seed=public.seed, model_seed=public.seed + 1,
                              roles=public.roles, transform_family="cyclic_shift")
        self.inner = NativeTeacherDriven(baseline_cfg, teacher_gain=gain,
                                         presentations=presentations, beta=beta)
        self.n, self.k = public.n, public.k
        self.writes = 0

    def train(self, role, cue, target):
        # Exactly one atomic pair is taught here; no composite is written.
        self.inner.train(self.writes, role, cue, target)
        self.writes += 1

    def predict(self, roles, cue):
        state = np.asarray(cue, dtype=np.uint32).copy()
        for role in roles:
            # Each native prediction is isolated by the R1h read-only/frozen
            # scope. The composed trajectory has no training signal.
            state = self.inner.predict(0, role, state)
            if state is None:
                return None
        return state

    def fingerprint(self):
        return self.inner.state_fingerprint()

    def bytes_state(self):
        # 3 explicit areas, all 9 n*n float32 directed connectomes.
        # This is a model-layout estimate; actual RSS needs measurement.
        return 9 * self.n * self.n * np.dtype(np.float32).itemsize


def run_native(protocol: Protocol):
    result = run_one(protocol, factories=(("native_teacher_chain", NativeAffineChain),))
    result["status"] = "native_supervised_affine_composition_smoke"
    result["limitations"].extend([
        "Native R1h teacher drive is not a learned affine operator hypothesis.",
        "Every composed probe runs without teacher labels and without updating synapses.",
        "This run must execute in a full native Brain checkout before numerical claims.",
    ])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-seed", type=int, default=23)
    p.add_argument("--model-seed", type=int, default=8123)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--atomic-examples", type=int, default=8)
    p.add_argument("--test-cues", type=int, default=8)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    spec = Protocol(n=a.n, k=a.k, atomic_examples=a.atomic_examples,
                    test_cues=a.test_cues, data_seed=a.data_seed,
                    model_seed=a.model_seed)
    raw = json.dumps(run_native(spec), indent=2, sort_keys=True) + "\n"
    if a.output:
        a.output.write_text(raw)
    else:
        print(raw)

if __name__ == "__main__": main()
