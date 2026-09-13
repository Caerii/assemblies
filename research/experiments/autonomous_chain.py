"""Autonomous chain recall on the refracted arc. [[ARC-CONJUNCT-EXPOSURE]]

Registration: research/notes/sequence/PREREG_autonomous_chain.md.

The reference builds sequences from a feedforward RefractedArea arc driven by
`(symbol, state)`. Drive it with a SINGLE CONSTANT symbol and the chain becomes
autonomous: the tick carries no information, so every advance comes from the
state through the arc. That is the recall `ordered_recall` fails at, built on
the construction that works.

It is also the sharpest available test of the anti-swamping claim: the tick
appears in every transition while each state appears in one, so it is the most
over-exposed conjunct the construction admits. With refraction off, the arcs of
distinct states should collapse onto the tick and the chain should not move.

Run:  python -m research.runner autonomous-chain --tag UNIQUE
      (smoke: --smoke --seeds 1 2 3; VOID)
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from neural_assemblies import describe_hashed_arc_fsm
from neural_assemblies.diagnostics import ensemble_from_values
from research.runner import experiment_parser, run_experiment, validate_registered_seeds

PROTOCOL = "sequence.autonomous-chain"
VERSION = "1"
REGISTRATION = "research/notes/sequence/PREREG_autonomous_chain.md"
#: Seeds 42-45 were used by the exploratory probe that informed the bars, so
#: the study runs on a block no probe has touched.
REGISTERED_SEEDS = tuple(range(62, 82))

N_ARC, K, P, BETA, W_MAX = 10000, 100, 0.2, 0.10, 20.0
STRENGTH = BETA                 # the reference runs the refracted arc at s = plasticity
MAX_POTENTIATIONS = 96          # sized to the episode; the default overflows float32
PRESENTATIONS = 20
TICK = "tick"
#: name -> (length, presentations, strength ratio of beta, arc density)
ARM_SPECS = {
    "L32": (32, PRESENTATIONS, 1.0, P),
    "L128": (128, PRESENTATIONS, 1.0, P),
    "L32-no-refraction": (32, PRESENTATIONS, 0.0, P),
    "L32-pres5": (32, 5, 1.0, P),
    "L32-pres10": (32, 10, 1.0, P),
    "L32-p0.05": (32, PRESENTATIONS, 1.0, 0.05),
    "L32-p0.02": (32, PRESENTATIONS, 1.0, 0.02),
}
ARMS = tuple(ARM_SPECS)
SMOKE_ARMS = ("L32", "L32-no-refraction")


def profile_for(ratio):
    """The organ profile of one arm. `ratio` is a fraction of beta."""
    return describe_hashed_arc_fsm(w_max=W_MAX, norm_init=False,
                                   refracted_strength=ratio * BETA,
                                   tie_jitter=0.0, zero_or_size=False)


def chain_table(length):
    """A simple path q0 -> q1 -> ... -> qL on one constant symbol."""
    states = [f"q{i}" for i in range(length + 1)]
    return states, [(states[i], TICK, states[i + 1]) for i in range(length)]


def consecutive_correct(visited, length):
    """Correct visits from the start; visit t is correct when it reads q_{t+1}."""
    got = 0
    for t in range(length):
        if visited[t] == t + 1:
            got += 1
        else:
            break
    return got


def arc_collapse(fsm, states, nbrain):
    """Mean pairwise overlap between the ARC assemblies of distinct states.

    The anti-swamping measurement: if the constant tick swamps the state, every
    state drives the SAME arc and this rises towards 1. Chance is k / n_arc.
    """
    from neural_assemblies.core._torch_ops import torch_ops

    arcs = []
    for name in states[:-1]:                 # every state that has an outgoing arc
        fsm.arc.inhibit()
        fsm.cue_state(name)
        fsm.step(TICK)
        arcs.append(torch_ops.sort(fsm.arc.winners, dim=1).values.clone())
    stack = torch_ops.stack(arcs)            # [S, B, k]
    S, _, k = stack.shape
    total = torch_ops.zeros(nbrain, dtype=torch_ops.float32, device=stack.device)
    pairs = 0
    for i in range(S):
        for j in range(i + 1, S):
            a, b = stack[i], stack[j]
            idx = torch_ops.searchsorted(a.contiguous(), b.contiguous()).clamp_(max=k - 1)
            total += (torch_ops.gather(a, 1, idx) == b).sum(1).float() / k
            pairs += 1
    return (total / max(pairs, 1)).cpu().numpy().tolist()


def run_arm(seeds, length, presentations, ratio, density, *, device, organ_semantics):
    from neural_assemblies.core._torch_ops import torch_ops
    from neural_assemblies.core.semantics import OrganSemantics
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM

    if OrganSemantics.normalize(organ_semantics).mismatch(profile_for(ratio)):
        raise ValueError("recorded organ profile disagrees with the arm's construction")
    states, table = chain_table(length)
    fsm = HashedArcFSM(seeds, states, [TICK], table, n_arc=N_ARC, k=K, p=density,
                       beta=BETA, refracted_strength=ratio * BETA, w_max=W_MAX,
                       max_potentiations=MAX_POTENTIATIONS, device=device)
    fsm.train(presentations)
    fsm.check()
    syms = torch_ops.zeros(len(seeds), length, dtype=torch_ops.int64, device=device)
    visited = fsm.run(syms, states[0]).cpu().numpy()
    rows = [{"seed": int(s), "correct": consecutive_correct(visited[b], length),
             "visited": visited[b].tolist()}
            for b, s in enumerate(seeds)]
    collapse = arc_collapse(fsm, states, len(seeds))
    for row, value in zip(rows, collapse):
        row["arc_overlap"] = float(value)
    del fsm
    torch_ops.cuda.empty_cache()
    return rows


def _ens(values, label, seeds):
    e = ensemble_from_values([float(v) for v in values], label, keys=list(seeds))
    return {**asdict(e), "low": e.low, "high": e.high}


def experiment(record):
    p = record["parameters"]
    seeds = list(record["seeds"])
    smoke = record["mode"] == "smoke"
    chance = K / N_ARC
    arms = {}
    for name in p["arms"]:
        length, presentations, ratio, density = ARM_SPECS[name]
        if smoke:
            length = 8
        rows = run_arm(seeds, length, presentations, ratio, density,
                       device=p["device"],
                       organ_semantics=record["execution_semantics"]["profiles"][name])
        correct = [r["correct"] for r in rows]
        arms[name] = {
            "length": length, "presentations": presentations,
            "strength_ratio": ratio, "density": density, "rows": rows,
            "correct": _ens(correct, f"{name}:correct", seeds),
            "arc_overlap": _ens([r["arc_overlap"] for r in rows], f"{name}:arc", seeds),
            "exact_brains": sum(c == length for c in correct),
        }
        # the ensemble already carries the seed statistic with its interval;
        # printing a second hand-rolled mean would be a different claim
        overlap_mean = arms[name]["arc_overlap"]["mean"]
        print(f"  {name:<20s} L={length:<4d} correct {min(correct)}..{max(correct)}"
              f"  exact {arms[name]['exact_brains']}/{len(seeds)}"
              f"  arc overlap {overlap_mean:.4f}"
              f" ({overlap_mean / chance:.1f}x chance)", flush=True)

    bars, comparisons = {}, {}
    if not smoke:
        def arm(name):
            return arms[name]

        def all_exact(name):
            return arm(name)["exact_brains"] == len(seeds)

        def worst(name):
            return max(r["correct"] for r in arm(name)["rows"])
        comparisons = {
            "chance_arc_overlap": chance,
            "arc_overlap_by_arm": {k: v["arc_overlap"]["mean"] for k, v in arms.items()},
            "exact_brains_by_arm": {k: v["exact_brains"] for k, v in arms.items()},
        }
        bars = {
            "CL-1 it exceeds the reported band: every brain exact at L=32 and at L=128":
                all_exact("L32") and all_exact("L128"),
            "CL-2 refraction is necessary and total: 0 correct on every brain with strength 0":
                all(r["correct"] == 0 for r in arm("L32-no-refraction")["rows"]),
            "CL-3 the score is reachable and not free: <= 5 correct on every brain at 5 and 10 presentations":
                worst("L32-pres5") <= 5 and worst("L32-pres10") <= 5,
            "CL-4 the arc needs density: <= 5 correct on every brain at p = 0.05 and p = 0.02":
                worst("L32-p0.05") <= 5 and worst("L32-p0.02") <= 5,
            "CL-5 the mechanism is anti-swamping: arcs collapse without refraction (>= 0.5) and sit near chance with it (<= 3x)":
                all(r["arc_overlap"] >= 0.5 for r in arm("L32-no-refraction")["rows"])
                and all(r["arc_overlap"] <= 3 * chance for r in arm("L32")["rows"]),
        }
        bars = {name: bool(ok) for name, ok in bars.items()}
        for name, ok in bars.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    verdict = "VOID" if smoke else ("PASS" if all(bars.values()) else "FAIL")
    return {"verdict": verdict, "bars": bars, "arms": arms, "comparisons": comparisons,
            "scope": "autonomous chain recall on a teacher-forced path over one "
                     "constant symbol, hashed refracted arc, one operating point"}


def main(argv=None):
    parser = experiment_parser(
        "Autonomous chain recall on the refracted arc, driven by one constant symbol",
        engines=("hashed_arc_fsm",), default_seeds=REGISTERED_SEEDS,
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, REGISTERED_SEEDS)
    names = list(SMOKE_ARMS if args.smoke else ARMS)
    parameters = {"n_arc": N_ARC, "k": K, "p": P, "beta": BETA, "w_max": W_MAX,
                  "strength_ratio": 1.0, "max_potentiations": MAX_POTENTIATIONS,
                  "presentations": PRESENTATIONS, "symbol": TICK, "arms": names,
                  "arm_specs": {k: list(v) for k, v in ARM_SPECS.items() if k in names},
                  "device": args.device}
    profiles = {name: profile_for(ARM_SPECS[name][2]) for name in names}
    path = run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATION, engine=args.engine, seeds=args.seeds,
        tag=args.tag, smoke=args.smoke, minimum_study_seeds=20,
        parameters=parameters, organ_semantics=profiles, measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
