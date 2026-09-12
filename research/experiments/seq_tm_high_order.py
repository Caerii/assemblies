"""PREREG_temporal_memory.md, cells B: high-order sequence prediction on the
hashed transducer as a temporal memory (state = previous arc, predicted
neurons win), the literature's own benchmark (Bouhadjar et al. 2022).

    set I    A D B E  and  F D B C          (shared middle D B; order 2)
    set II   six five-element sequences with shared elements
    set III  two twelve-element sequences identical in the middle ten
             (order 10)

Each set is presented PRESENTATIONS times to 20 brains; after each
presentation every sequence is replayed frozen and the next word
predicted at every position. TM-4: set I's continuation after D B on
>= 18/20 brains at g = 4. TM-5: set III's order-10 disambiguation on
>= 15/20 at g = 4. TM-6: the presentation at which set I is first
predicted perfectly, reported. The induced state (g = 0, projected) is
the control on every set.

    python research/experiments/seq_tm_high_order.py [--brains 20] [--smoke]
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from neural_assemblies import describe_hashed_transducer
from neural_assemblies.diagnostics import ensemble_from_values
from research.runner import ExperimentOutput, experiment_parser, run_experiment

N, N_ARC, K, P, ORGAN_P, BETA = 4000, 4000, 100, 0.05, 0.2, 0.10
PRESENTATIONS, ROUNDS, GROUND_ROUNDS = 40, 3, 5
SEEDS = list(range(42, 62))

SETS = {
    "I": [list("ADBE"), list("FDBC")],
    "II": [list("ADBEG"), list("FDBCH"), list("IJBEK"), list("LDMCN"), list("OJBCP"), list("QDMEK")],
    "III": [list("A") + list("DBEGHIJKLM") + list("N"),
            list("F") + list("DBEGHIJKLM") + list("O")],
}
VOCAB = sorted({w for seqs in SETS.values() for s in seqs for w in s})


def ambiguous_positions(seqs):
    """(sequence index, position) pairs where the prefix so far is shared
    with another sequence and the next word differs: the order test."""
    out = []
    for i, s in enumerate(seqs):
        for t in range(1, len(s) - 1):
            prefix = tuple(s[1:t + 1])           # after the first, distinguishing word
            for j, o in enumerate(seqs):
                if j != i and len(o) > t + 1 and tuple(o[1:t + 1]) == prefix and o[t + 1] != s[t + 1]:
                    out.append((i, t))
                    break
    return out


def run_set(name, seqs, seeds, gain, mode, presentations, organ_semantics):
    # Import the GPU implementation only after the shared parser and runner
    # have validated the invocation and reserved its immutable tag.
    from neural_assemblies.core.torch_engine._hashed_transducer import HashedTransducer
    t0 = time.perf_counter()
    tr = HashedTransducer(seeds, VOCAB, n=N, n_arc=N_ARC, k=K, p=P, beta=BETA,
                          organ_p=ORGAN_P, w_max=20.0, norm_init=True,
                          max_potentiations=presentations * 4 + 8,
                          state_mode=mode, predict_gain=gain,
                          n_state=(N_ARC if mode == "copy" else None),
                          organ_semantics=organ_semantics)
    tr.ground(rounds=GROUND_ROUNDS)
    B = len(seeds)
    amb = ambiguous_positions(seqs)
    rng = random.Random(0)
    first_perfect: list[int | None] = [None] * B
    per_pres = []
    n_amb = 0
    for pres in range(presentations):
        for s in seqs:
            tr.train_sentence(s, rounds=ROUNDS)
        # replay frozen: rank-1 at every position, and at the ambiguous ones
        hits_all = np.zeros(B); n_all = 0
        hits_amb = np.zeros(B); n_amb = 0
        for i, s in enumerate(seqs):
            tr.reset()
            for t, (a, nxt) in enumerate(zip(s, s[1:])):
                tr.tick(a, rounds=ROUNDS, freeze=True)
                ranked = tr.rank(tr.emit(), rng)
                ok = np.array([r[0] == nxt for r in ranked], dtype=float)
                hits_all += ok; n_all += 1
                if (i, t) in amb:
                    hits_amb += ok; n_amb += 1
        acc_all = hits_all / n_all
        acc_amb = hits_amb / max(n_amb, 1)
        per_pres.append({"presentation": pres + 1, "acc_all": acc_all.tolist(),
                         "acc_ambiguous": acc_amb.tolist()})
        for b in range(B):
            if first_perfect[b] is None and acc_all[b] >= 1.0:
                first_perfect[b] = pres + 1
    last = per_pres[-1]
    e_all = ensemble_from_values(last['acc_all'], "rank-1 all")
    e_amb = ensemble_from_values(last['acc_ambiguous'], "rank-1 ambiguous")
    print(f"    set {name:3s} mode {mode:7s} g {gain:<3}: rank-1 all {e_all.mean:.3f} +/- {e_all.ci:.3f}, "
          f"ambiguous {e_amb.mean:.3f} +/- {e_amb.ci:.3f} on {n_amb} positions; brains perfect "
          f"{sum(a >= 1.0 for a in last['acc_all'])}/{B}; first perfect at "
          f"{sorted(x for x in first_perfect if x)}  [{time.perf_counter() - t0:.0f}s]", flush=True)
    del tr
    import torch
    torch.cuda.empty_cache()
    return {"set": name, "mode": mode, "gain": gain, "ambiguous_positions": amb,
            "per_presentation": per_pres, "first_perfect": first_perfect,
            "final_acc_ambiguous": last["acc_ambiguous"], "final_acc_all": last["acc_all"]}


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    presentations = parameters["presentations"]
    rows = []
    arms = [("induced", 0.0), ("copy", 0.0), ("copy", 4.0)]
    for name, seqs in SETS.items():
        for mode, g in arms:
            key = f"{mode}-g{g:g}"
            rows.append(run_set(
                name, seqs, seeds, g, mode, presentations,
                record["execution_semantics"]["profiles"][key],
            ))
    def brains_ok(name, mode, g, bar):
        r = next(x for x in rows if x["set"] == name and x["mode"] == mode and x["gain"] == g)
        return sum(a >= 1.0 for a in r["final_acc_ambiguous"])
    tm4 = brains_ok("I", "copy", 4.0, 18)
    tm5 = brains_ok("III", "copy", 4.0, 15)
    r1 = next(x for x in rows if x["set"] == "I" and x["mode"] == "copy" and x["gain"] == 4.0)
    return ExperimentOutput({
        "verdict": "VOID" if record["mode"] == "smoke" else (
            "PASS" if tm4 >= 18 and tm5 >= 15 else "FAIL"
        ),
        "tm4": tm4, "tm5": tm5, "rows": rows,
        "scope": "high-order temporal transducer prediction",
    })


def main(argv=None):
    parser = experiment_parser(
        __doc__ or "High-order temporal memory study",
        engines=("hashed_transducer",),
        default_seeds=tuple(SEEDS),
    )
    parser.add_argument(
        "--presentations", type=int, default=PRESENTATIONS,
        help="Amendment 1: 20 sits inside the clip window",
    )
    parser.add_argument("--brains", type=int, default=len(SEEDS),
                        help="prefix length of the registered seed set")
    args = parser.parse_args(argv)
    if args.brains < 3 or args.brains > len(SEEDS):
        parser.error(f"--brains must be between 3 and {len(SEEDS)}")
    if args.presentations < 1:
        parser.error("--presentations must be positive")
    seeds = SEEDS[:3] if args.smoke else SEEDS[:args.brains]
    parameters = {
        "n": N, "n_arc": N_ARC, "k": K, "p": P, "organ_p": ORGAN_P,
        "beta": BETA, "presentations": 3 if args.smoke else args.presentations,
        "rounds": ROUNDS, "ground_rounds": GROUND_ROUNDS, "w_max": 20.0,
        "norm_init": True, "max_potentiations": (3 if args.smoke else args.presentations) * 4 + 8,
    }
    profiles = {
        "induced-g0": describe_hashed_transducer(
            w_max=20.0, norm_init=True, refracted_strength=0.1,
            state_mode="induced", predict_gain=0.0,
        ).to_dict(),
        "copy-g0": describe_hashed_transducer(
            w_max=20.0, norm_init=True, refracted_strength=0.1,
            state_mode="copy", predict_gain=0.0,
        ).to_dict(),
        "copy-g4": describe_hashed_transducer(
            w_max=20.0, norm_init=True, refracted_strength=0.1,
            state_mode="copy", predict_gain=4.0,
        ).to_dict(),
    }
    path = run_experiment(
        script=Path(__file__), protocol="sequence.temporal-memory-high-order",
        protocol_version="2", registration="research/notes/sequence/PREREG_temporal_memory.md",
        engine=args.engine, seeds=seeds, tag=args.tag, smoke=args.smoke,
        minimum_study_seeds=20, parameters=parameters,
        organ_semantics=profiles, measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
