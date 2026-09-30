"""How often the k-WTA bar is tied, and what one last bit does to it. [[KWTA-TIE-FRAGILE]]

Registration: research/notes/substrate/PREREG_kwta_tie_fragility.md.

The drives are computed from the explicit engine's connectome weights the
way its first projection computes them (a float32 sum over the stimulus
rows), so no readout projection runs and nothing learns
(`observation_policy` none). Per seed the run retains a tie census on the
raw 0/1 drive, on the in-degree-normalized drive (exact rational
arithmetic), and on a jittered drive that has no ties (the null); a
constructed one-ulp perturbation of a tied outsider; and the number of
winners that differ between three arithmetic paths for the normalized
drive under the engine's lowest-id tie rule.

Run:  python -m research.runner kwta-tie-fragility --tag UNIQUE
      (smoke: --smoke --seeds 1 2 3; VOID)
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from neural_assemblies import Brain, describe_brain_model
from research.runner import (
    experiment_parser, run_experiment, validate_registered_seeds,
    validate_seed_identities,
)

PROTOCOL = "substrate.kwta-tie-fragility"
VERSION = "1"
REGISTRATION = "research/notes/substrate/PREREG_kwta_tie_fragility.md"
REGISTERED_SEEDS = tuple(range(42, 62))
N, K, P, STIM = 1000, 50, 0.1, 50
JITTER = 1e-3


def _lowest_id_topk(drive, k):
    """The engine's tie rule: highest drive first, lowest neuron id among ties."""
    d = np.asarray(drive, dtype=np.float64)
    order = np.lexsort((np.arange(len(d)), -d))
    return order[:k]


def _bar_ties(drive, k):
    d = np.asarray(drive, dtype=np.float64)
    bar = np.sort(d)[::-1][k - 1]
    return int((d == bar).sum())


def _rational_ties(count, indeg, k):
    exact = [Fraction(int(c), int(d)) for c, d in zip(count, indeg)]
    order = sorted(range(len(exact)), key=lambda i: (-exact[i], i))
    bar = exact[order[k - 1]]
    return sum(1 for x in exact if x == bar), np.array(order[:k])


def _one_ulp_flip(drive, k, *, tied_only):
    """Raise one candidate by one float32 ulp and report whether the winner set changes.

    With `tied_only`, the candidate is the highest-index neuron tied at the
    bar but outside the winner set (the constructed last-bit decision);
    otherwise it is the candidate at rank k + 1, which is what the same
    construction can do when no tie exists.
    """
    d = np.asarray(drive, dtype=np.float32).copy()
    winners = _lowest_id_topk(d, k)
    inside = np.zeros(len(d), dtype=bool)
    inside[winners] = True
    bar = np.float32(np.sort(d)[::-1][k - 1])
    if tied_only:
        candidates = np.flatnonzero((d == bar) & ~inside)
        if candidates.size == 0:
            return 0, -1
        target = int(candidates.max())
    else:
        target = int(_lowest_id_topk(d, k + 1)[k])
    perturbed = d.copy()
    perturbed[target] = np.nextafter(perturbed[target], np.float32(np.inf))
    moved = _lowest_id_topk(perturbed, k)
    return int(set(moved.tolist()) != set(winners.tolist())), target


def measure_seed(seed, *, engine, model_semantics):
    brain = Brain(engine=engine, p=P, seed=seed, norm_init=False,
                  model_semantics=model_semantics)
    brain.add_area("A", N, K, beta=0.1)
    brain.add_stimulus("s", STIM)
    # Register the fiber without projecting: the engine draws the stimulus
    # connectome at first use, so ask it to materialize the fiber and read the
    # 0/1 weights before any learning.
    weights = np.asarray(brain._engine._stim_conns["s"]["A"].weights)
    if weights.ndim != 2 or weights.shape != (STIM, N):
        raise RuntimeError(f"expected a ({STIM}, {N}) stimulus matrix, got {weights.shape}")
    if not np.array_equal(np.unique(weights), np.array([0.0, 1.0])):
        raise RuntimeError("stimulus weights must be untrained 0/1 values")
    raw = weights.astype(np.float32).sum(axis=0)              # the engine's own accumulation
    count = weights.astype(np.int64).sum(axis=0)
    # In-degree normalization divides by the column's full in-degree from the
    # whole source population; with one stimulus of STIM rows that is the
    # column sum, so raw/indeg is 1 wherever the count is nonzero. Draw a
    # second, larger 0/1 population as the normalizing source population so
    # the ratio is informative, as it is for an area fiber.
    rng = np.random.default_rng(seed + 10_000)
    population = (rng.random((10 * STIM, N)) < P)
    population[:STIM] = weights.astype(bool)                  # the presented rows are members
    indeg = np.maximum(population.sum(axis=0), 1).astype(np.int64)
    ties_norm, winners_exact = _rational_ties(count, indeg, K)
    d64 = count.astype(np.float64) / indeg.astype(np.float64)
    d32 = count.astype(np.float32) / indeg.astype(np.float32)
    r32 = count.astype(np.float32) * (np.float32(1.0) / indeg.astype(np.float32))
    w64, w32, wr = (set(_lowest_id_topk(v, K).tolist()) for v in (d64, d32, r32))
    jittered = raw.astype(np.float64) + JITTER * np.arange(N)
    flip_raw, target_raw = _one_ulp_flip(raw, K, tied_only=True)
    flip_jit, target_jit = _one_ulp_flip(jittered, K, tied_only=False)
    return {
        "seed": seed,
        "ties_raw": _bar_ties(raw, K),
        "ties_normalized": int(ties_norm),
        "ties_jittered": _bar_ties(jittered, K),
        "flip_one_ulp_raw": flip_raw, "flip_target_raw": target_raw,
        "flip_one_ulp_jittered": flip_jit, "flip_target_jittered": target_jit,
        "split_f32_div_vs_recip": len(w32 ^ wr),
        "split_f32_vs_f64": len(w32 ^ w64),
        "split_f64_vs_exact": len(w64 ^ set(winners_exact.tolist())),
    }


def experiment(record):
    if record.get("mode", "study") == "study":
        validate_seed_identities(record["seeds"], REGISTERED_SEEDS)
    p = record["parameters"]
    if (p["n"], p["k"], p["p"], p["stimulus_rows"]) != (N, K, P, STIM):
        raise ValueError("this protocol version does not implement changed sizes")
    required_model = record["execution_semantics"]["profiles"]["default"]
    rows = [measure_seed(s, engine=record["engine"], model_semantics=required_model)
            for s in record["seeds"]]
    for r in rows:
        print(f"  seed {r['seed']}: ties raw {r['ties_raw']:3d}  normalized {r['ties_normalized']:2d}  "
              f"jittered {r['ties_jittered']}  one-ulp flip raw {r['flip_one_ulp_raw']} / jittered "
              f"{r['flip_one_ulp_jittered']}  splits f32div/recip {r['split_f32_div_vs_recip']} "
              f"f32/f64 {r['split_f32_vs_f64']}", flush=True)
    n = len(rows)
    columns = {key: [r[key] for r in rows] for key in rows[0] if key != "seed"}
    smoke = record["mode"] == "smoke"
    bars = {} if smoke else {
        "KT-1 ties_raw >= 5 on every seed": all(v >= 5 for v in columns["ties_raw"]),
        "KT-2 one-ulp raise of a tied outsider flips on every seed, never on the jittered drive":
            all(v == 1 for v in columns["flip_one_ulp_raw"])
            and all(v == 0 for v in columns["flip_one_ulp_jittered"]),
        "KT-3 ties_normalized >= 2 on at least half the seeds":
            sum(v >= 2 for v in columns["ties_normalized"]) * 2 >= n,
    }
    for name, ok in bars.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    verdict = "VOID" if smoke else ("PASS" if all(bars.values()) else "FAIL")
    return {"verdict": verdict, "bars": bars, "rows": rows,
            "seeds": list(record["seeds"]), "columns": columns,
            "reported_not_judged": {
                "split_f32_div_vs_recip": columns["split_f32_div_vs_recip"],
                "split_f32_vs_f64": columns["split_f32_vs_f64"],
                "split_f64_vs_exact": columns["split_f64_vs_exact"],
            },
            "scope": "tie census on the explicit CPU engine's first-projection drive; "
                     "the GPU selector's split is not reproduced here by construction"}


def main(argv=None):
    parser = experiment_parser(
        "k-WTA tie census with a one-ulp perturbation and a tie-free null",
        engines=("numpy_explicit",), default_seeds=REGISTERED_SEEDS,
    )
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, REGISTERED_SEEDS)
    parameters = {"n": N, "k": K, "p": P, "stimulus_rows": STIM, "jitter": JITTER,
                  "normalizing_population_rows": 10 * STIM}
    path = run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATION, engine=args.engine, seeds=args.seeds,
        tag=args.tag, smoke=args.smoke, parameters=parameters,
        observation_policy="none",  # drives are read from the weights; no readout projection
        model_semantics=describe_brain_model("numpy_explicit", p=P, norm_init=False),
        measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
