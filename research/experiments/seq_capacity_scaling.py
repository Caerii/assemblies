"""How does the M-ceiling scale with n?  Registered in PREREG_capacity_scaling.md.

The ceiling study closed with "What is NOT established: how any of this scales
with n", blocked on compute. This runs the sweep on `batched_project_hashed`,
which historically trained 16 independent brains at once with a generated connectome and is
verified against `numpy_sparse` on all four substrate arms.

PROTOCOL. As registered: the area is inhibited between assemblies, and each
assembly is trained by firing its own STIMULUS every round alongside recurrence
-- `project({s: [AREA]}, {AREA: [AREA]})`. An earlier version substituted a
fixed initial winner set for the stimulus and that removed the ANCHOR, not just
the fiber: rank1 was 0.19 at M=4 (Amendment 1). The stimulus fiber is now
hash-generated and priced as the engine prices it.

ONE REMAINING DIFFERENCE from `PREREG_substrate_ceiling.md`: the stimulus fiber
here does not itself learn a per-cell connectome -- it stores pre-summed input,
which is what the engine stores, but its base is generated rather than drawn in
RNG order. Absolute ceilings are therefore not bit-comparable to that study's
M* = 41 / 104 at n=2000; the SCALING with n is the deliverable.

    python -m research.runner capacity-scaling --tag NAME --registration PATH [--smoke]

`--smoke` checks the API only. Its numbers are VOID. Protocol version 2
records full (arm,n,k,seed) coordinates through the shared runner. Historical
flat JSON and its automatic slope verdict are not produced by this version.

Protocol version 3 can execute the Hebbian control and refracted treatment in
one run with `--compare-refraction`. Their complete configurations and organ
semantics are separate, while seeds and measurement sampling remain paired.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-capacity-comparison
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.core.numpy_engine import _seeding                # noqa: E402
from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from neural_assemblies import describe_assembly_memory                  # noqa: E402
from research.experiments._substrate import ceiling_from_curve          # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

DEV = "cuda"
K = 60
P = 0.50
T = 8
BETA = 0.10
W_MAX = 20.0
HALF_BAR = 0.50
DISTINCT_GATE = 3.0
NS = (1000, 2000, 4000, 8000)
MS = (8, 16, 32, 64, 128, 256)
RECALL_SAMPLE = 32
PAIR_SAMPLE = 200
ARMS = {"B": dict(norm_init=True, synaptic_scaling=False),
        "G": dict(norm_init=True, synaptic_scaling=True)}


@dataclass(frozen=True)
class CapacityProtocol:
    checkpoints: tuple[int, ...] = MS
    p: float = P
    beta: float = BETA
    rounds: int = T
    stim_size: int | None = None
    refracted: bool = False
    readout: str = "net"
    converge: bool = False
    refracted_factor: float = 1.0
    w_max: float = W_MAX
    recall_sample: int = RECALL_SAMPLE
    pair_sample: int = PAIR_SAMPLE

    def __post_init__(self):
        if (not self.checkpoints or any(type(m) is not int or m < 2 for m in self.checkpoints)
                or tuple(sorted(set(self.checkpoints))) != self.checkpoints):
            raise ValueError("checkpoints must be strictly increasing integers >= 2")
        if not 0 < self.p <= 1 or not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("p must be in (0, 1] and beta finite and nonnegative")
        if (type(self.rounds) is not int or self.rounds < 1
                or (self.stim_size is not None and
                    (type(self.stim_size) is not int or self.stim_size < 1))):
            raise ValueError("rounds and stimulus size must be positive")
        if (not math.isfinite(self.w_max) or self.w_max < 1
                or any(type(v) is not int or v < 1 for v in (self.recall_sample, self.pair_sample))):
            raise ValueError("weight clip and measurement sample counts must be positive")
        if not math.isfinite(self.refracted_factor) or self.refracted_factor < 0:
            raise ValueError("refraction factor must be finite and nonnegative")
        if self.readout not in {"net", "masked"}:
            raise ValueError("unknown readout")


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def seeds_for(seeds):
    return [to_i32(_seeding.fnv1a_pair_seed(seed, "A", "A")) for seed in seeds]


def _fill(mem):
    """rows/n -- the fraction of the area that has EVER fired (CAP3's
    censoring guard reads this)."""
    return mem.fill.cpu().numpy()


def run_cell(n, k, protocol, seeds, rng, *, arm_settings, device,
             organ_semantics=None):
    """Train up to `m_max` assemblies, checkpointing at every M in MS.

    The protocol is `AssemblyMemory`: INHIBITED between assemblies, each
    written by T rounds of its own stimulus alongside recurrence
    (`brain.inhibit_areas([AREA])`, then `project({s: [AREA]}, {AREA:
    [AREA]})` x T); the class is the harness's numbers, bit-identical to
    the wrapper sequence this ran on before it (tested)."""
    import torch
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory

    sd = seeds_for(seeds)
    m_max = max(protocol.checkpoints)
    mem = AssemblyMemory(sd, n, k, protocol.p, beta=protocol.beta,
                         w_max=protocol.w_max, rounds=protocol.rounds,
                         strength=(protocol.refracted_factor if protocol.refracted else 0.0),
                         gate=protocol.converge, max_items=m_max, device=device,
                         organ_semantics=organ_semantics, **arm_settings)
    stored, used = [], []
    out = {}
    for a in range(m_max):
        ss = [to_i32(_seeding.fnv1a_pair_seed(seed, f"s{a}", "A")) for seed in seeds]
        stored.append(mem.store(ss, stim_size=(protocol.stim_size or k)))
        if protocol.converge:
            used.append(mem.rounds_used.clone())
        M = a + 1
        if M in protocol.checkpoints:
            out[M] = measure(n, mem, stored, len(seeds), rng, protocol)
            if protocol.converge:
                # Amendment 5, G3: rounds spent per item since the last
                # checkpoint, and the fraction that converged before T_max
                u = torch.stack(used).float()                    # [items, B]
                out[M]["rounds_used"] = u.mean(0).tolist()
                out[M]["converged"] = (u < protocol.rounds).float().mean(0).tolist()
                used.clear()
    return out


def _set_hash(X):
    """One int64 per winner set, order-canonical. `X` is [M, B, K] SORTED."""
    import torch

    pos = torch.arange(X.shape[2], device=X.device,
                       dtype=torch.int64).view(1, 1, -1)
    z = (X * 0x9E3779B97F4A7C15) ^ (pos * 0xBF58476D1CE4E5B9)
    z = z ^ (z >> 31)
    z = z * 0x94D049BB133111EB
    z = z ^ (z >> 29)
    return z.sum(dim=2)


def _overlaps(Ks, ia, ib):
    """|A ∩ B| / K for sampled pairs, without leaving the GPU.

    `Ks` is [M, B, K] sorted along K, so `searchsorted` finds each element of
    one set in the other and a gather confirms equality. Winners are distinct
    within a set, so no de-duplication is needed.
    """
    import torch

    A, Bv = Ks[ia], Ks[ib]                          # [P, B, K]
    K = Ks.shape[2]
    idx = torch.searchsorted(A.contiguous(), Bv.contiguous()).clamp_(max=K - 1)
    hit = torch.gather(A, 2, idx) == Bv
    return hit.sum(2).float() / K                   # [P, B]


def measure(n, mem, stored, nbrain, rng, protocol):
    """All four metrics on the GPU.

    The earlier version did `M x B` `.tolist()` calls for distinctness, a
    Python `set()` intersection per sampled pair, and an M-iteration gather
    loop inside EVERY recall. That made the study measurement-bound rather than
    GPU-bound: the kernels cost ~0.09 ms per brain-round and the study was
    paying ~0.65.
    """
    import torch

    M = len(stored)
    St = torch.stack(stored).long()                 # [M, B, K]
    K = St.shape[2]
    device = St.device
    Ks = torch.sort(St, dim=2).values                # canonical order

    # -- distinctness by a 64-bit set hash (possible collisions).
    # Hash each set to an int64, sort along M, count consecutive differences.
    h = _set_hash(Ks)                                # [M, B]
    sh, _ = torch.sort(h, dim=0)
    fresh = torch.ones_like(sh, dtype=torch.bool)
    fresh[1:] = sh[1:] != sh[:-1]
    dist = (fresh.sum(0).double() / M).cpu().numpy()

    # -- pairwise overlap on a sample of pairs
    pw = np.zeros(nbrain)
    if M > 1:
        npair = min(protocol.pair_sample, M * (M - 1) // 2)
        ia = rng.integers(0, M, npair)
        ib = rng.integers(0, M, npair)
        keep = ia != ib
        if keep.any():
            ia = torch.from_numpy(ia[keep]).to(device)
            ib = torch.from_numpy(ib[keep]).to(device)
            pw = _overlaps(Ks, ia, ib).mean(0).double().cpu().numpy()
    pw_x = pw / (K / n)

    # -- half-cue rank-1, frozen (the probe equivalent)
    samp = rng.choice(M, min(protocol.recall_sample, M), replace=False)
    off = (torch.arange(nbrain, device=device, dtype=torch.int64)
           * n).view(1, nbrain, 1)
    flat = (St + off).reshape(-1)                    # [M*B*K], built once
    hits = torch.zeros(nbrain, dtype=torch.int64, device=device)
    for a in samp:
        rec = mem.recall(St[a][:, : K // 2],
                         masked=(protocol.refracted and protocol.readout == "masked"))
        mask = torch.zeros(nbrain * n, dtype=torch.bool, device=device)
        mask[(rec + off[0]).reshape(-1)] = True
        # ONE gather for all M stored assemblies, instead of M gathers.
        ov = mask[flat].view(M, nbrain, K).sum(2)    # [M, B]
        hits += (ov.argmax(dim=0) == int(a)).long()
    rank1 = (hits.double() / len(samp)).cpu().numpy()
    return dict(rank1=rank1.tolist(), pairwise_x=pw_x.tolist(),
                distinct=dist.tolist(), fill=_fill(mem).tolist())


def _fill_at(cells, m_star):
    """rows/n interpolated at M*, in log2(M)."""
    pts = sorted((M, ensemble_from_values(cells[M]["fill"]).mean) for M in cells)
    if not pts or m_star is None or m_star <= 0:
        return None
    if m_star <= pts[0][0]:
        return pts[0][1]
    if m_star >= pts[-1][0]:
        return pts[-1][1]
    for (m0, f0), (m1, f1) in zip(pts, pts[1:]):
        if m0 <= m_star <= m1:
            t = ((math.log2(m_star) - math.log2(m0))
                 / max(math.log2(m1) - math.log2(m0), 1e-12))
            return f0 + t * (f1 - f0)
    return pts[-1][1]


def gated(cell, seeds, *, distinct_gate, distinct_low_bar):
    r = ensemble_from_values(cell["rank1"], keys=seeds)
    x = ensemble_from_values(cell["pairwise_x"], keys=seeds)
    d = ensemble_from_values(cell["distinct"], keys=seeds)
    ok = (x.high <= distinct_gate and d.low >= distinct_low_bar)
    return (r.mean if ok else 0.0), r, x, d


def _resolved_conditions(record):
    """Decode version-2 single or version-3 paired protocol configurations."""
    parameters = record["parameters"]
    if "conditions" not in parameters:
        values = dict(parameters["configuration"])
        values["checkpoints"] = tuple(values["checkpoints"])
        return {"default": CapacityProtocol(**values)}, False
    if ("configuration" in parameters or not isinstance(parameters["conditions"], dict)
            or set(parameters["conditions"]) != {"control", "refracted"}):
        raise ValueError("paired conditions must be exactly control and refracted")
    conditions = {}
    for name, raw in parameters["conditions"].items():
        if (not isinstance(name, str) or not name
                or not isinstance(raw, dict)):
            raise ValueError("condition names and configurations must be explicit")
        try:
            values = dict(raw)
            values["checkpoints"] = tuple(values["checkpoints"])
            conditions[name] = CapacityProtocol(**values)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid {name} condition configuration") from exc
    control, treatment = conditions["control"], conditions["refracted"]
    if (control.refracted or control.refracted_factor != 0.0
            or treatment.refracted is not True or treatment.refracted_factor <= 0
            or control.readout != "masked" or treatment.readout != "masked"
            or control.converge or treatment.converge):
        raise ValueError("paired conditions require an ungated masked refraction contrast")
    ignored = {"refracted", "refracted_factor"}
    if ({key: value for key, value in asdict(control).items() if key not in ignored}
            != {key: value for key, value in asdict(treatment).items()
                if key not in ignored}):
        raise ValueError("paired conditions may differ only in refraction")
    return conditions, True


def _capacity_cell(n, k, protocol, seeds, rng, *, arm, settings, device,
                   organ_semantics, distinct_gate, distinct_low_bar, half_bar):
    cells = run_cell(n, k, protocol, seeds, rng, arm_settings=settings,
                     device=device, organ_semantics=organ_semantics)
    curve = [(m, gated(cell, seeds, distinct_gate=distinct_gate,
                       distinct_low_bar=distinct_low_bar)[0])
             for m, cell in sorted(cells.items())]
    ceiling = ceiling_from_curve(curve, threshold=half_bar)
    fill = _fill_at(cells, ceiling.m_star)
    return {
        "arm": arm, "n": n, "k": k, "seeds": seeds,
        "checkpoints": cells,
        "ensembles": {m: {name: asdict(ensemble_from_values(values, keys=seeds,
                                                              label=name))
                          for name, values in cell.items()}
                      for m, cell in cells.items()},
        "ceiling": {"m_star": ceiling.m_star, "supported": bool(ceiling.supported),
                    "grid_lo": ceiling.lo, "grid_hi": ceiling.hi,
                    "grid_censored": bool(ceiling.censored),
                    "interior_points": ceiling.n_interior,
                    "fill_at_ceiling": fill,
                    "fill_censored": fill is not None and fill >= 0.95,
                    "alpha": ceiling.m_star * k / n if ceiling.m_star else None},
    }


def experiment(record):
    """Evaluate explicitly identified cells; completion leaves adoption UNJUDGED."""
    parameters = record["parameters"]
    # Specification: neural_assemblies/ir/VERIFICATION.md#contract-capacity-execution
    for name in ("distinct_gate", "distinct_low_bar", "half_bar"):
        value = parameters[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if not 0 < parameters["half_bar"] <= 1 or parameters["distinct_low_bar"] > 1:
        raise ValueError("half_bar must be in (0,1] and distinct_low_bar at most one")
    measurement_seed = parameters["measurement_seed"]
    if type(measurement_seed) is not int or not 0 <= measurement_seed < 2**64:
        raise ValueError("measurement_seed must be an unsigned 64-bit integer")
    settings = parameters["arm_settings"]
    if set(settings) != set(parameters["arms"]):
        raise ValueError("arm_settings must describe exactly the requested arms")
    for cfg in settings.values():
        if (set(cfg) != {"norm_init", "synaptic_scaling"}
                or any(type(value) is not bool for value in cfg.values())):
            raise ValueError("each arm must explicitly specify boolean normalization and scaling")
    device = parameters["device"]
    if not isinstance(device, str) or not device:
        raise ValueError("device must be an explicit nonempty name")
    conditions, paired = _resolved_conditions(record)
    seeds = record["seeds"]
    results = {}
    for condition, protocol in conditions.items():
        condition_cells = {}
        for arm in parameters["arms"]:
            for n, k in parameters["nk"]:
                profile = arm if not paired else f"{condition}.{arm}"
                # Restart measurement sampling for every cell and condition, so
                # treatment/control consume identical pair samples.
                cell = _capacity_cell(
                    n, k, protocol, seeds,
                    np.random.default_rng(measurement_seed),
                    arm=arm, settings=settings[arm], device=device,
                    organ_semantics=record["execution_semantics"]["profiles"][profile],
                    distinct_gate=parameters["distinct_gate"],
                    distinct_low_bar=parameters["distinct_low_bar"],
                    half_bar=parameters["half_bar"],
                )
                condition_cells[f"{arm}/{n}/{k}"] = cell
                print(f"{condition} {arm} n={n} k={k}: {cell['ceiling']['m_star']}")
        results[condition] = {"cells": condition_cells}
    common = {"verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED",
              "fit_status": "not evaluated; fit requires a separately specified estimand and uncertainty protocol"}
    if paired:
        return {"conditions": results, **common}
    return {"cells": list(results["default"]["cells"].values()), **common}


def main(argv=None):
    ap = experiment_parser(__doc__, engines=("hashed_assembly_memory",),
                           default_seeds=tuple(range(42, 62)))
    ap.add_argument("--registration", required=True,
                    help="repository path to the registration/amendment for this exact protocol")
    grid = ap.add_mutually_exclusive_group()
    grid.add_argument("--ns", help="comma-separated n values at k=60 (or --ksqrt)")
    grid.add_argument("--nk", help="explicit n:k pairs, including multiple k at the same n")
    ap.add_argument("--ksqrt", action="store_true")
    ap.add_argument("--ms", help="comma-separated, increasing M checkpoints")
    ap.add_argument("--arms", default="B,G")
    ap.add_argument("--p", type=float, default=P)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--rounds", type=int, default=T)
    ap.add_argument("--stim-size", type=int)
    ap.add_argument("--refracted", action="store_true")
    ap.add_argument("--refracted-factor", type=float, default=1.0)
    ap.add_argument("--readout", choices=("net", "masked"), default="net")
    ap.add_argument("--converge", action="store_true")
    ap.add_argument("--compare-refraction", action="store_true",
                    help="paired Hebbian control and refracted masked-readout conditions")
    ap.add_argument("--device", default=DEV)
    ap.add_argument("--distinct-gate", type=float, default=DISTINCT_GATE)
    ap.add_argument("--distinct-low-bar", type=float, default=0.9)
    args = ap.parse_args(argv)
    try:
        if args.nk and args.ksqrt:
            raise ValueError("--nk already supplies k; do not combine it with --ksqrt")
        ns = tuple(int(v) for v in args.ns.split(",")) if args.ns else ((1000, 2000) if args.smoke else NS)
        nk = ([tuple(int(v) for v in pair.split(":")) for pair in args.nk.split(",")]
              if args.nk else [(n, round(math.sqrt(n)) if args.ksqrt else K) for n in ns])
        if (not nk or any(len(pair) != 2 or not 2 <= pair[1] <= pair[0] for pair in nk)
                or len(set(nk)) != len(nk)):
            raise ValueError("grid needs unique (n,k) pairs with 2 <= k <= n")
        arms = args.arms.split(",")
        if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
            raise ValueError("arms must be a unique subset of B,G")
        checkpoints = tuple(int(v) for v in args.ms.split(",")) if args.ms else ((4, 8) if args.smoke else MS)
        if args.compare_refraction and (args.refracted or args.converge
                                        or args.readout != "net"):
            raise ValueError("--compare-refraction owns refraction, gate, and masked readout")
        config = CapacityProtocol(checkpoints=checkpoints, p=args.p, beta=args.beta,
                                  rounds=args.rounds, stim_size=args.stim_size,
                                  refracted=args.refracted, readout=args.readout,
                                  converge=args.converge, refracted_factor=args.refracted_factor)
    except ValueError as exc:
        ap.error(str(exc))
    conditions = ({
        "control": asdict(CapacityProtocol(
            checkpoints=checkpoints, p=args.p, beta=args.beta, rounds=args.rounds,
            stim_size=args.stim_size, refracted=False, readout="masked",
            converge=False, refracted_factor=0.0)),
        "refracted": asdict(CapacityProtocol(
            checkpoints=checkpoints, p=args.p, beta=args.beta, rounds=args.rounds,
            stim_size=args.stim_size, refracted=True, readout="masked",
            converge=False, refracted_factor=args.refracted_factor)),
    } if args.compare_refraction else None)
    profiles = {}
    for condition, values in ((conditions or {"default": asdict(config)}).items()):
        for arm in arms:
            profiles[arm if conditions is None else f"{condition}.{arm}"] = (
                describe_assembly_memory(
                    w_max=values["w_max"], beta=values["beta"],
                    strength=(values["refracted_factor"] if values["refracted"] else 0.0),
                    gate=values["converge"], **ARMS[arm],
                )
            )
    path = run_experiment(
        script=__file__, protocol="memory.capacity-scaling",
        protocol_version="3" if conditions else "2",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment,
        organ_semantics=profiles,
        parameters={**({"conditions": conditions} if conditions else
                       {"configuration": asdict(config)}), "nk": nk, "arms": arms,
                    "arm_settings": {arm: ARMS[arm] for arm in arms},
                    "measurement_seed": 1234, "half_bar": HALF_BAR,
                    "distinct_gate": args.distinct_gate, "distinct_low_bar": args.distinct_low_bar,
                    "device": args.device, "distinctness": "64-bit set hash; collisions possible"},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
