"""GATE-3 of DESIGN_sequence_port.md: A1's horizon on the hashed organ at width.

The registered curve -- first divergence index over 2000 random digits of
the mod-3 machine -- on 20 brains at p = 0.3 and p = 0.4, against the five
numpy seeds in `seq_a1_horizon_results.json`. Same machine (N_ARC 5000,
N_STATE 500, K 70, beta 0.1, 15 presentations, norm_init off, the Brain's
clip), same digit strings per seed (`random.Random(seed * 7919)`), the
numpy seeds 1..5 among the hashed 1..20 so a paired reading is possible.

    BAR (GATE-3). At both p, every numpy first-error index (NEVER = censored
    at 2000) lies inside the hashed [5th, 95th] percentile.
    If it fails: the sampler is the first suspect (the numpy seeds ran on
    the sampled engine; the hashed organ equals the materialized one).

    python -m research.runner a1-horizon --tag horizon-01 [--smoke]
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import inspect
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_assemblies.core.brain import Brain                            # noqa: E402
from neural_assemblies import describe_hashed_arc_fsm                    # noqa: E402
from neural_assemblies.programs.mod3_fsm import (                         # noqa: E402
    ALL_STATES, ALL_SYMBOLS, mod3_transition_table)
from seq_a1_fsm_parity import BETA, K, N_ARC, N_STATE, PRESENTATIONS      # noqa: E402
from _results import results_path  # noqa: E402
from research.runner import (  # noqa: E402
    experiment_parser, run_experiment, validate_registered_seeds,
    validate_seed_identities,
)

REGISTERED_SEEDS = tuple(range(1, 21))

LENGTH = 2000
P_VALUES = (0.3, 0.4)
CHECKPOINTS = (10, 50, 100, 500, 1000, 2000)
STRENGTH = 0.1
_HERE = os.path.dirname(os.path.abspath(__file__))



@dataclass(frozen=True)
class HorizonProtocol:
    length: int
    p_values: tuple[float, ...]
    n_arc: int
    n_state: int
    k: int
    beta: float
    presentations: int
    strength: float
    norm_init: bool
    w_max: float
    checkpoints: tuple[int, ...]
    device: str

    def __post_init__(self):
        for name in ("length", "n_arc", "n_state", "k", "presentations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.k > self.n_arc or self.n_state < len(ALL_STATES) * self.k:
            raise ValueError("area sizes must contain the winner set and assigned state blocks")
        for name in ("beta", "strength", "w_max"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.w_max < 1 or type(self.norm_init) is not bool:
            raise ValueError("w_max must be at least one and norm_init must be boolean")
        for name in ("p_values", "checkpoints"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (not self.p_values or any(type(p) not in (int, float) or not 0 < p <= 1 for p in self.p_values)
                or len(set(self.p_values)) != len(self.p_values)):
            raise ValueError("p_values must be unique probabilities in (0,1]")
        if (not self.checkpoints or any(type(c) is not int or c < 1 for c in self.checkpoints)
                or tuple(sorted(set(self.checkpoints))) != self.checkpoints):
            raise ValueError("checkpoints must be increasing positive integers")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be an explicit nonempty name")

    @classmethod
    def from_parameters(cls, parameters):
        return cls(**{field.name: parameters[field.name] for field in fields(cls)})


def digit_strings(seeds, length):
    from neural_assemblies.core._torch_ops import torch_ops
    out = torch_ops.zeros(len(seeds), length, dtype=torch_ops.int64)
    for b, seed in enumerate(seeds):
        rng = random.Random(seed * 7919)
        out[b] = torch_ops.tensor([rng.randrange(10) for _ in range(length)])
    return out


def run_width(seeds, p, protocol, organ_semantics=None):
    from neural_assemblies.core._torch_ops import torch_ops
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
    length = protocol.length
    fsm = HashedArcFSM(seeds, ALL_STATES, ALL_SYMBOLS, mod3_transition_table(),
                       n_arc=protocol.n_arc, n_state=protocol.n_state, k=protocol.k, p=p, beta=protocol.beta,
                       refracted_strength=protocol.strength, w_max=protocol.w_max, norm_init=protocol.norm_init,
                       max_potentiations=protocol.presentations * 4 + 8, prefix="_mod3",
                       device=protocol.device, organ_semantics=organ_semantics)
    t0 = time.perf_counter()
    fsm.train(protocol.presentations)
    fsm.check()
    digits = digit_strings(seeds, length).to(protocol.device)
    # ground truth: running residue per brain
    truth = torch_ops.cumsum(digits, dim=1) % 3
    fsm.arc.inhibit()
    fsm.cue_state("0")
    got = torch_ops.zeros_like(digits)
    exact = torch_ops.zeros(len(seeds), dtype=torch_ops.int64, device=protocol.device)
    for t in range(length):
        got[:, t] = fsm.step(digits[:, t])
        # exact: every winner inside the true block
        blk = fsm.state.winners // protocol.k
        exact += (blk == truth[:, t].view(-1, 1)).all(dim=1).to(torch_ops.int64)
    correct = (got == truth).cpu().numpy()
    exact = (exact.cpu().numpy() / length)
    rows = []
    for b, seed in enumerate(seeds):
        wrong = np.flatnonzero(~correct[b])
        fe = int(wrong[0]) + 1 if len(wrong) else None
        rows.append({"seed": int(seed), "p": p, "length": length, "first_error": fe,
                     "accuracy": float(correct[b].mean()),
                     "exact_fraction": float(exact[b]),
                     "prefix_correct": {c: bool(correct[b, :c].all())
                                        for c in protocol.checkpoints if c <= length}})
    print(f"    p={p}: {len(seeds)} brains, {length} steps  "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    return rows


def gate3(hashed_rows, numpy_rows, p, length=LENGTH):
    """Numpy seeds inside the hashed [5, 95] percentile; NEVER is length + 1."""
    def val(r):
        return float(length + 1 if r["first_error"] is None else r["first_error"])
    h = np.array([val(r) for r in hashed_rows if r["p"] == p])
    lo, hi = np.percentile(h, 5), np.percentile(h, 95)
    verdicts = []
    for r in numpy_rows:
        if r["p"] != p:
            continue
        v = val(r)
        verdicts.append((r["seed"], r["first_error"], bool(lo <= v <= hi)))
    return lo, hi, verdicts


def experiment(record):
    if record.get("mode", "study") == "study":
        validate_seed_identities(record["seeds"], REGISTERED_SEEDS)
    """Run the registered horizon measurement from resolved runner inputs."""
    smoke = record['mode'] == 'smoke'
    # Specification: neural_assemblies/ir/VERIFICATION.md#contract-horizon-execution
    protocol = HorizonProtocol.from_parameters(record["parameters"])
    length = protocol.length
    seeds = record['seeds']
    if smoke:
        print("*** SMOKE: API only. THESE NUMBERS ARE VOID. ***")
    with open(results_path("sequence", "seq_a1_horizon_results.json")) as fh:
        numpy_rows = json.load(fh)
    if set(protocol.p_values) - {row["p"] for row in numpy_rows}:
        raise ValueError("historical comparison has no reference for a requested probability")
    print(f"=== GATE-3: the horizon at width ({len(seeds)} brains, {length} digits) ===")
    rows, out = [], {"brains": seeds, "length": length, "rows": []}
    for p in protocol.p_values:
        got = run_width(
            seeds, p, protocol,
            record["execution_semantics"]["profiles"]["default"],
        )
        rows += got
        never = sum(r["first_error"] is None for r in got)
        fes = sorted(r["first_error"] for r in got if r["first_error"] is not None)
        print(f"      first errors: {fes}  NEVER: {never}/{len(got)}")
        for c in protocol.checkpoints:
            if c <= length:
                n_ok = sum(r["prefix_correct"].get(c, False) for r in got)
                print(f"         first {c:>5} steps perfect: {n_ok}/{len(got)}")
        lo, hi, verdicts = gate3(got, numpy_rows, p, length)
        print(f"      hashed [5th, 95th] = [{lo:.0f}, {hi:.0f}]")
        for seed, fe, ok in verdicts:
            print(f"        numpy seed {seed}: first error {fe if fe else 'NEVER':>6}  "
                  f"{'inside' if ok else 'OUTSIDE'}")
        out[f"gate3_p{p}"] = {"lo": lo, "hi": hi,
                              "verdicts": [(s, fe, ok) for s, fe, ok in verdicts]}
    out["rows"] = rows
    allok = all(ok for p in protocol.p_values for _, _, ok in out[f"gate3_p{p}"]["verdicts"])
    print(f"\n  GATE-3 {'PASS' if allok else 'FAIL'}"
          + ("" if not smoke else "  (SMOKE: VOID)"))
    out['verdict'] = 'VOID' if smoke else ('PASS' if allok else 'FAIL')
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "A1 horizon study", engines=('hashed_arc_fsm',),
                           default_seeds=REGISTERED_SEEDS)
    args = ap.parse_args(argv)
    validate_registered_seeds(ap, args, REGISTERED_SEEDS)
    w_max = inspect.signature(Brain).parameters['w_max'].default
    path = run_experiment(
        script=__file__, protocol='sequence.a1-horizon', protocol_version='2',
        registration='research/notes/sequence/DESIGN_sequence_port.md',
        engine=args.engine, seeds=args.seeds, tag=args.tag, smoke=args.smoke,
        minimum_study_seeds=20,
        organ_semantics=describe_hashed_arc_fsm(
            w_max=w_max, norm_init=False, refracted_strength=STRENGTH,
        ),
        input_artifacts=('research/results/sequence/seq_a1_horizon_results.json',),
        parameters=dict(length=50 if args.smoke else LENGTH, p_values=P_VALUES,
                        n_arc=N_ARC, n_state=N_STATE, k=K, beta=BETA,
                        presentations=PRESENTATIONS, strength=STRENGTH,
                        norm_init=False, checkpoints=CHECKPOINTS, device="cuda",
                        w_max=w_max,
                        comparison='historical sampled numpy; not valid sequence evidence'),
        measure=experiment,
    )
    print(f'  wrote {path}')


if __name__ == "__main__":
    main()
