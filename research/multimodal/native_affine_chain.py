"""R1o optional NATIVE Brain composition adapter with per-step read invariants.

Native Brain's actual numerical behavior requires a full checkout. This
module verifies a stricter contract at every read step and does not claim
that teacher-forced associations learn affine algebra.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .noncommutative_composition import Protocol, run_one, _valid_output


class NativeAffineChain:
    def __init__(self, public, gain=100., presentations=3, beta=.1):
        # Only the public learner seed reaches the neural model. The private
        # data seed and hidden operator parameters are never passed to it.
        from .zero_shot_gate import Config
        from .native_teacher_drive import NativeTeacherDriven

        baseline_cfg = Config(n=public.n, k=public.k, contents=4,
                              seed=public.seed, model_seed=public.seed + 1,
                              roles=public.roles, transform_family="cyclic_shift")
        self.inner = NativeTeacherDriven(baseline_cfg, teacher_gain=gain,
                                         presentations=presentations, beta=beta)
        self.n, self.k, self.roles = public.n, public.k, tuple(public.roles)
        self.writes = 0

    def train(self, role, cue, target):
        if role not in self.roles:
            raise ValueError(f"unknown atomic role: {role!r}")
        src = _valid_output(cue, self.n, self.k)
        dst = _valid_output(target, self.n, self.k)
        if src is None or dst is None:
            raise ValueError("cannot train on an absent assembly")
        self.inner.train(self.writes, role, src.copy(), dst.copy())
        self.writes += 1

    def predict(self, roles, cue):
        # Validate *before* invoking native code. Passing an invalid cue into
        # a neural simulator can mutate state before the scorer detects it.
        state = _valid_output(cue, self.n, self.k)
        if state is None:
            return None
        roles = tuple(roles)
        if not roles or any(role not in self.roles for role in roles):
            raise ValueError("nonempty known role sequence required")
        before = self.fingerprint()
        for step, role in enumerate(roles):
            # An adversarial intermediate step could alter the graph and a
            # later step could reverse that mutation, leaving the FINAL digest
            # unchanged. Verify on EVERY step, not only at the end.
            output = self.inner.predict(0, role, state.copy())
            after = self.fingerprint()
            if after != before:
                raise AssertionError(
                    f"native composition read mutated synapses at step {step} "
                    f"({role}); {before} -> {after}"
                )
            state = _valid_output(output, self.n, self.k)
            if state is None:
                return None
        return state.copy()

    def fingerprint(self):
        # NativeTeacherDriven hashes all explicit area-to-area matrices, not
        # just the trained LEX->role fibers. No partial audit is acceptable.
        return self.inner.state_fingerprint()

    def bytes_state(self):
        # Three explicit areas; all 9 directed dense n*n float32 matrices.
        # Not measured RSS; excludes runtime buffers, Python objects, etc.
        return 9 * self.n * self.n * np.dtype(np.float32).itemsize


def run_native(protocol: Protocol):
    result = run_one(protocol, factories=(("native_teacher_chain", NativeAffineChain),))
    result["status"] = "native_supervised_affine_composition_smoke"
    result["limitations"].extend([
        "Native R1h teacher drive is not a learned affine operator hypothesis.",
        "Each atomic composition read must preserve the full explicit synaptic graph.",
        "No composed examples are taught to the model.",
        "Native Brain results require execution in a full repository checkout.",
    ])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, default=23)
    parser.add_argument("--model-seed", type=int, default=8123)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--atomic-examples", type=int, default=8)
    parser.add_argument("--test-cues", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = Protocol(n=args.n, k=args.k,
                    atomic_examples=args.atomic_examples,
                    test_cues=args.test_cues,
                    data_seed=args.data_seed, model_seed=args.model_seed)
    raw = json.dumps(run_native(spec), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw)


if __name__ == "__main__":
    main()
