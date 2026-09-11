"""PREREG_refraction_memory.md, Amendment 3: the anti-merging claim gated on
the NUMPY engine, whose k-WTA has no selector defect.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-model-semantics

Mirrors `seq_capacity_scaling.py`'s protocol on `numpy_sparse`, one brain at
a time: area n = 2000, k = 60, p = 0.5, beta = 0.1, w_max = 20, norm_init,
no scaling, MATERIALIZED; each item trained by its own stimulus alongside
recurrence for T = 8 rounds from an inhibited area. Readout is the
harness's HALF-CUE recall: the first k/2 neurons of the stored assembly set
as winners, T frozen recurrent rounds under `probe()`, the refraction bias
zeroed for the MASKED readout and restored after, ranked by overlap against
every stored assembly. Distinctness by exact duplicates. Bars N1-N3.

    python -m research.runner refraction-memory-numpy --tag UNIQUE
      [--seeds 42 43 44 45 46] [--arm ref|ctl|both]
"""
from __future__ import annotations

import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_assemblies.core.brain import Brain                              # noqa: E402
from neural_assemblies.core.semantics import describe_brain_model            # noqa: E402
from research.runner import experiment_parser, run_experiment               # noqa: E402
from _substrate import ceiling_from_curve                                   # noqa: E402

N, K, P, T, BETA, W_MAX, STRENGTH = 2000, 60, 0.5, 8, 0.10, 20.0, 0.05
#: THE STIMULUS MODEL. The engine's stimulus into a materialized area is
#: zero-or-size ([[add-stimulus-zero-or-size]]): one stimulus of size 60 at
#: p = 0.5 hands half the area weight 60 and the rest nothing, every item is
#: the same tie, and the control collapses to ONE assembly (measured:
#: distinct 0.018 at M = 512); at p = 0.05 an item is its ~100 connected
#: neurons tied at 60, and under refraction it ROTATES through them each
#: round, spreading its potentiation thin (masked half-cue recall 0.5-0.75
#: at M = 4-8 against the harness's 1.000). The harness's stimulus is a
#: Binomial(60, 0.5) count per neuron -- GRADED. The mirror sums STIM_PARTS
#: stimuli of size K / STIM_PARTS per item: drive = (K / parts) x
#: Binomial(parts, p), the same mean and a graded, item-specific ranking.
STIM_PARTS = 10
MS = (8, 16, 32, 64, 128, 256, 512)
RECALL_SAMPLE, PAIR_SAMPLE, THRESHOLD = 32, 200, 0.90
AREA = "A"
_HERE = os.path.dirname(os.path.abspath(__file__))


def build(seed, refracted):
    random.seed(seed)
    np.random.seed(seed)
    b = Brain(p=P, seed=seed, engine="numpy_sparse", w_max=W_MAX, norm_init=True,
              recurrent_projection=True, synaptic_scaling=False)
    b.add_area(AREA, N, K, BETA)
    eng = b._engine_for(b.areas[AREA])
    eng.materialize_area(AREA, storage="dense")
    if refracted:
        eng.set_refracted(AREA, True, STRENGTH)
    return b, eng


def half_cue_rank1(b, eng, stored, rng, masked):
    """Fraction of sampled items whose own assembly ranks first after a
    half-cue recall, frozen."""
    M = len(stored)
    samp = rng.choice(M, min(RECALL_SAMPLE, M), replace=False)
    # the engine is driven DIRECTLY (as the parity tests do): the Brain-level
    # call reads its own cached winners, not the half cue set on the engine.
    # The masked read is the engine's own mode (`Brain.set_masked_readout`),
    # honoured on reads only.
    b.set_masked_readout(AREA, bool(masked))
    hits = 0
    for a in samp:
        eng.set_winners(AREA, np.asarray(stored[a][: K // 2], dtype=np.int64))
        for _ in range(T):
            eng.project_into(AREA, [], [AREA], plasticity_enabled=False)
        rec = set(int(x) for x in eng.get_winners(AREA))
        ov = [len(rec & s) for s in stored_sets(stored)]
        if int(np.argmax(ov)) == int(a):
            hits += 1
    b.set_masked_readout(AREA, False)
    return hits / len(samp)


_SETS = {}


def stored_sets(stored):
    key = id(stored), len(stored)
    if key not in _SETS or len(_SETS[key]) != len(stored):
        _SETS[key] = [set(int(x) for x in s) for s in stored]
    return _SETS[key]


def measure(b, eng, stored, rng, masked):
    sets_ = stored_sets(stored)
    M = len(stored)
    distinct = len({frozenset(s) for s in sets_}) / M
    pw = []
    if M > 1:
        for _ in range(min(PAIR_SAMPLE, M * (M - 1) // 2)):
            i, j = rng.choice(M, 2, replace=False)
            pw.append(len(sets_[i] & sets_[j]) / K)
    pw_x = (float(np.mean(pw)) / (K / N)) if pw else 0.0
    fill = float(len(set().union(*sets_))) / N
    return dict(rank1=half_cue_rank1(b, eng, stored, rng, masked),
                pairwise_x=pw_x, distinct=distinct, fill=fill)


def run_brain(seed, refracted, masked):
    b, eng = build(seed, refracted)
    rng = np.random.default_rng(seed)
    stored, out = [], {}
    t0 = time.perf_counter()
    for a in range(max(MS)):
        parts = [f"s{a}_{j}" for j in range(STIM_PARTS)]
        for nm in parts:
            b.add_stimulus(nm, K // STIM_PARTS)
        b.inhibit_areas([AREA])
        for _ in range(T):
            eng.project_into(AREA, parts, [AREA], plasticity_enabled=True)
        stored.append(np.asarray(eng.get_winners(AREA), dtype=np.int64).copy())
        M = a + 1
        if M in MS:
            _SETS.clear()
            out[M] = measure(b, eng, stored, rng, masked)
            print(f"      seed {seed} {'REF' if refracted else 'CTL'} M={M:4d} "
                  f"rank1 {out[M]['rank1']:.3f} pw/chance {out[M]['pairwise_x']:5.2f} "
                  f"distinct {out[M]['distinct']:.3f} fill {out[M]['fill']:.3f}  "
                  f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    return out


def main(argv=None):
    ap = experiment_parser(
        __doc__.splitlines()[0], engines=("numpy_sparse",),
        default_seeds=(42, 43, 44, 45, 46),
    )
    ap.add_argument("--arm", choices=("ref", "ctl", "both"), default="both")
    args = ap.parse_args(argv)
    global MS
    if args.smoke:
        MS = (4, 8)
        print("SMOKE: API only; numbers VOID")
    def measure(record):
        results = {}
        for arm in (("ref", "ctl") if args.arm == "both" else (args.arm,)):
            seeds = record["seeds"]
            print(f"=== numpy {arm.upper()}  n={N} k={K} p={P} stimulus {STIM_PARTS}x{K // STIM_PARTS} T={T} beta={BETA} "
                  f"{'strength ' + str(STRENGTH) + ' masked' if arm == 'ref' else ''}")
            per = {s: run_brain(s, arm == "ref", masked=True) for s in seeds}
            ceilings = []
            for s in seeds:
                curve = {M: per[s][M]["rank1"] for M in MS}
                ceilings.append(str(ceiling_from_curve(list(curve.items()), THRESHOLD)))
            results[arm] = {
                "per_seed": {str(s): {str(M): v for M, v in per[s].items()} for s in seeds},
                "ceilings": ceilings,
            }
            print(f"    ceilings: {ceilings}")
        return results

    return run_experiment(
        script=__file__,
        protocol="memory.refraction-memory-numpy",
        protocol_version="1",
        registration="research/notes/memory/PREREG_refraction_memory.md",
        engine=args.engine,
        seeds=args.seeds,
        tag=args.tag,
        parameters={
            "n": N, "k": K, "p": P, "rounds": T, "beta": BETA,
            "weight_ceiling": W_MAX, "strength": STRENGTH,
            "stimulus_parts": STIM_PARTS, "recall_sample": RECALL_SAMPLE,
            "pair_sample": PAIR_SAMPLE, "threshold": THRESHOLD,
            "arm": args.arm,
        },
        measure=measure,
        smoke=args.smoke,
        model_semantics=describe_brain_model(
            args.engine, p=P, seed=0, w_max=W_MAX, norm_init=True,
            recurrent_projection=True, synaptic_scaling=False,
        ),
    )


if __name__ == "__main__":
    main()
