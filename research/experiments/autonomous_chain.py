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
VERSION = "3"          # v2: Amendment 1 limit grid; v3: Amendment 2 collidable states
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

#: Amendment 1. n_state is held FIXED across every limit cell: HashedArcFSM
#: otherwise sizes it from the chain length, so varying L would redraw the
#: connectome. The engine caps it at 65536 (topk_select packs a 16-bit index),
#: which bounds the chain at 65536/k - 1 = 639 here.
N_STATE_FIXED = 51200      # exactly L_max * k; larger wastes the memory budget
LIMIT_P = 0.3          # kp = 30 clears 3 ln n in every limit cell
LIMIT_ARCS = (1000, 2000, 3000, 4000)
LIMIT_LENGTHS = (160, 256, 384, 512)


def limit_arms():
    """name -> (length, presentations, strength ratio, density, n_arc)."""
    out = {f"n{n}-L{L}": (L, PRESENTATIONS, 1.0, LIMIT_P, n)
           for n in LIMIT_ARCS for L in LIMIT_LENGTHS}
    # the mechanism-disabled null carried over, at the smallest cell
    out[f"n{LIMIT_ARCS[0]}-L{LIMIT_LENGTHS[0]}-no-refraction"] = (
        LIMIT_LENGTHS[0], PRESENTATIONS, 0.0, LIMIT_P, LIMIT_ARCS[0])
    return out


LIMIT_SPECS = limit_arms()

#: Amendment 2. Every earlier result assigns each state a DISJOINT block, so
#: state collision is impossible by construction. These arms replace the code
#: with random k-subsets and shrink the state area until collision is forced.
#: L * k = 16000 here, so the last two arms cannot be disjoint.
STATE_L, STATE_ARC = 160, 3000
STATE_AREAS = (64000, 32000, 16000, 8000, 4000)


def state_arms():
    """name -> (n_state, code) where code is 'blocks' or 'random'."""
    out = {f"blocks-n{STATE_AREAS[0]}": (STATE_AREAS[0], "blocks")}
    out.update({f"random-n{n}": (n, "random") for n in STATE_AREAS})
    return out


STATE_SPECS = state_arms()


def state_code(seeds, n_state, kind, n_states, k, device):
    """[n_states, k] compact indices: disjoint blocks, or random k-subsets.

    Random subsets overlap at about chance k / n_state when the area is roomy
    and are FORCED to overlap once n_states * k exceeds it.
    """
    from neural_assemblies.core._torch_ops import torch_ops
    if kind == "blocks":
        if n_states * k > n_state:
            raise ValueError("disjoint blocks do not fit in the state area")
        return torch_ops.arange(n_states * k, device=device,
                                dtype=torch_ops.int64).view(n_states, k)
    gen = torch_ops.Generator(device="cpu")
    gen.manual_seed(int(seeds[0]))
    rows = [torch_ops.randperm(n_state, generator=gen)[:k] for _ in range(n_states)]
    return torch_ops.stack(rows).to(device).to(torch_ops.int64)


def decode_by_overlap(winners, code):
    """[B] index of the code row the winners overlap most.

    The block readout decodes by integer division and assumes contiguous
    disjoint blocks, so a collidable code needs this instead. The disjoint arm
    is decoded the same way, so the two are compared on ONE readout.
    """
    # [S, k] code against [B, k] winners -> [B, S] overlap counts
    hit = (winners.unsqueeze(1).unsqueeze(-1) == code.unsqueeze(0).unsqueeze(2))
    return hit.any(-1).sum(-1).argmax(1)


def code_overlap(code, k):
    """Mean pairwise overlap between distinct code rows, as a fraction of k."""
    from neural_assemblies.core._torch_ops import torch_ops
    srt = torch_ops.sort(code, dim=1).values
    S = srt.shape[0]
    total, pairs = 0.0, 0
    for i in range(S):
        a = srt[i].unsqueeze(0).expand(S - i - 1, k) if i + 1 < S else None
        if a is None:
            continue
        b = srt[i + 1:]
        idx = torch_ops.searchsorted(a.contiguous(), b.contiguous()).clamp_(max=k - 1)
        total += float((torch_ops.gather(a, 1, idx) == b).sum()) / k
        pairs += S - i - 1
    return total / max(pairs, 1)


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


#: sampled pairs for the collapse measurement; the full set is quadratic in L
PAIR_SAMPLE = 512


def arc_collapse(fsm, states, nbrain):
    """Mean pairwise overlap between the ARC assemblies of distinct states,
    over a fixed sample of pairs.

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
    # SAMPLE pairs rather than enumerate them: the full set is quadratic in the
    # chain length and at S = 512 that is 130k tensor comparisons, which took
    # 13 minutes on one cell before this was fixed. The capacity protocol
    # samples for the same reason.
    want = min(PAIR_SAMPLE, S * (S - 1) // 2)
    gen = torch_ops.Generator(device="cpu")
    gen.manual_seed(0)
    ia = torch_ops.randint(0, S, (want,), generator=gen)
    ib = torch_ops.randint(0, S, (want,), generator=gen)
    keep = ia != ib
    ia, ib = ia[keep].to(stack.device), ib[keep].to(stack.device)
    if ia.numel() == 0:
        return [0.0] * nbrain
    a, b = stack[ia], stack[ib]              # [P, B, k], sorted along k
    idx = torch_ops.searchsorted(a.contiguous(), b.contiguous()).clamp_(max=k - 1)
    hit = (torch_ops.gather(a, 2, idx) == b).sum(2).float() / k     # [P, B]
    return hit.mean(0).cpu().numpy().tolist()


def run_arm(seeds, length, presentations, ratio, density, *, device,
            organ_semantics, n_arc=N_ARC, n_state=None):
    from neural_assemblies.core._torch_ops import torch_ops
    from neural_assemblies.core.semantics import OrganSemantics
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM

    if OrganSemantics.normalize(organ_semantics).mismatch(profile_for(ratio)):
        raise ValueError("recorded organ profile disagrees with the arm's construction")
    states, table = chain_table(length)
    fsm = HashedArcFSM(seeds, states, [TICK], table, n_arc=n_arc, k=K, p=density,
                       n_state=n_state, beta=BETA, refracted_strength=ratio * BETA,
                       w_max=W_MAX, max_potentiations=MAX_POTENTIATIONS,
                       device=device)
    fsm.train(presentations)
    fsm.check()
    syms = torch_ops.zeros(len(seeds), length, dtype=torch_ops.int64, device=device)
    visited = fsm.run(syms, states[0]).cpu().numpy()
    rows = []
    for b, s in enumerate(seeds):
        seq = visited[b].tolist()
        rows.append({
            "seed": int(s), "correct": consecutive_correct(seq, length),
            # total correct distinguishes a chain DEATH from a dropped step
            "total_correct": sum(1 for t_, v in enumerate(seq) if v == t_ + 1),
            "visited": seq,
        })
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
    limit_mode = bool(p.get("limit_mode"))
    specs = LIMIT_SPECS if limit_mode else ARM_SPECS
    for name in p["arms"]:
        spec = specs[name]
        length, presentations, ratio, density = spec[:4]
        n_arc = spec[4] if len(spec) > 4 else N_ARC
        if smoke:
            length = 8
        rows = run_arm(seeds, length, presentations, ratio, density,
                       device=p["device"], n_arc=n_arc,
                       n_state=(N_STATE_FIXED if limit_mode else None),
                       organ_semantics=record["execution_semantics"]["profiles"][name])
        correct = [r["correct"] for r in rows]
        arms[name] = {
            "length": length, "presentations": presentations,
            "strength_ratio": ratio, "density": density, "n_arc": n_arc,
            "n_state": (N_STATE_FIXED if limit_mode else None), "rows": rows,
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
    if not smoke and limit_mode:
        def exact_length(n_arc):
            """Largest tested length at which EVERY brain recalls every visit.
            A grid quantity, never interpolated; the top of the grid is censored."""
            best = 0
            for L in LIMIT_LENGTHS:
                a = arms.get(f"n{n_arc}-L{L}")
                if a and a["exact_brains"] == len(seeds):
                    best = L
            return best
        exact = {n: exact_length(n) for n in LIMIT_ARCS}
        deaths = [(name, a) for name, a in arms.items()
                  if "no-refraction" not in name and a["exact_brains"] < len(seeds)]
        ratio_3x = (exact[LIMIT_ARCS[-1]] / exact[LIMIT_ARCS[0]]
                    if exact[LIMIT_ARCS[0]] else None)
        comparisons = {
            "exact_length_by_arc": exact,
            "censored": {n: exact[n] == LIMIT_LENGTHS[-1] for n in LIMIT_ARCS},
            "n_state_identical": len({a["n_state"] for a in arms.values()}) == 1,
            "exact_length_ratio_6000_over_2000": ratio_3x,
        }
        ordered = [exact[n] for n in LIMIT_ARCS]
        bars = {
            "AL-1 it breaks, and the break is a chain DEATH not a dropped step":
                bool(deaths) and all(
                    r["total_correct"] == r["correct"]
                    for _, a in deaths for r in a["rows"] if r["correct"] < a["length"]),
            "AL-2 exact_length is non-decreasing in n_arc":
                all(x <= y for x, y in zip(ordered, ordered[1:])),
            "AL-3 superlinear but not square: the 4x arc ratio lands strictly between 4 and 16":
                ratio_3x is not None and 4.0 < ratio_3x < 16.0,
            "AL-4 the state area cannot be the cause: n_state identical in every cell":
                comparisons["n_state_identical"],
            "AL-5 refraction still carries it: 0 correct with strength 0":
                all(r["correct"] == 0
                    for r in arms[f"n{LIMIT_ARCS[0]}-L{LIMIT_LENGTHS[0]}-no-refraction"]["rows"]),
        }
        bars = {name: bool(ok) for name, ok in bars.items()}
        for name, ok in bars.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"  exact_length by arc: {exact}; ratio {ratio_3x}")
    elif not smoke:
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
    parser.add_argument("--limit", action="store_true",
                        help="run Amendment 1's fixed-grid chain-length cells")
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, REGISTERED_SEEDS)
    specs = LIMIT_SPECS if args.limit else ARM_SPECS
    names = list(SMOKE_ARMS if args.smoke else specs)
    parameters = {"n_arc": N_ARC, "k": K, "p": P, "beta": BETA, "w_max": W_MAX,
                  "strength_ratio": 1.0, "max_potentiations": MAX_POTENTIATIONS,
                  "presentations": PRESENTATIONS, "symbol": TICK, "arms": names,
                  "arm_specs": {k: list(v) for k, v in specs.items() if k in names},
                  "limit_mode": bool(args.limit),
                  "n_state_fixed": (N_STATE_FIXED if args.limit else None),
                  "device": args.device}
    profiles = {name: profile_for(specs[name][2]) for name in names}
    path = run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATION, engine=args.engine, seeds=args.seeds,
        tag=args.tag, smoke=args.smoke, minimum_study_seeds=20,
        parameters=parameters, organ_semantics=profiles, measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
