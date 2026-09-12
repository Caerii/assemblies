"""Can an explicit recruitment mechanism defeat Hebbian rich-get-richer?

Follow-up to `research/experiments/capacity/` (READ ITS `REPORT.md` FIRST).
That study established, on a SHARED-substrate lexicon model, that recruitment
into an area stops far short of the `n/k` tiling limit, that the cause is
Hebbian rich-get-richer rather than neuron exhaustion, and that the capacity
scaling exponent `d log V* / d log n` collapses from 0.84 (beta=0) to 0.29
(beta=0.05) to 0.14 (beta=0.2).

Its closing suggestion was that two mechanisms already present in the codebase
but never engaged might restore the beta=0 exponent while keeping beta>0
learning:

  1. `refracted` suppression -- `Area(refracted=True, refracted_strength=s)`.
     The sparse engine keeps a per-neuron `_cumulative_bias`, incremented by
     `s` every time that neuron fires, and SUBTRACTS it from the input vector
     before winner selection. Crucially the bias vector is indexed by COMPACT
     index, so it only ever penalises neurons that have already materialised;
     never-fired candidates are appended afterwards and are untouched. That is
     exactly an explicit recruitment pressure.

  2. `synaptic_scaling` -- `Brain(synaptic_scaling=True)`, homeostatic
     renormalisation of area->area fibers. The repo documents this as NOT YET
     CORRECT (`_sparse.py::_normalize_area_columns`): a per-fiber setpoint
     cancels net potentiation and broke attractor stability and pattern
     completion in earlier testing. Tested anyway.

  3. Also tested: LRI (`refractory_period` / `inhibition_strength`), a
     short-window decaying penalty on recently-fired neurons -- the same idea
     as (1) but with a finite memory instead of a cumulative one.

METHOD is deliberately identical to `capacity/lexicon_capacity.py`, whose
helpers are imported rather than re-implemented: one shared PHON->LEX
connectome for the whole vocabulary (per-word private stimulus matrices
reverse the direction of interference and hide forgetting -- finding 6 of the
prior report), fixed-k TopK, k=30, p=0.05, 6 training rounds, 400-word
vocabulary with a planted 8-category block structure, 5 seeds per cell.

EXTRA INSTRUMENTATION beyond the prior harness:
  * `probe_nobias`: at the LAST checkpoint, retrieval is repeated on a clone
    whose refracted bias has been zeroed. The accumulated bias is itself a
    distortion at read-out time (old assemblies carry the most suppression),
    so the honest report needs retrieval both with the mechanism left engaged
    and with it disengaged for the probe.
  * `exhausted_at` is already recorded by the prior harness; with strong
    refracted suppression it becomes the DOMINANT outcome, which is the point:
    the mechanism converts the soft plasticity horizon into the hard
    combinatorial floor.

Usage:
    python -m research.experiments.recruitment.recruitment_mechanisms
    python -m research.experiments.recruitment.recruitment_mechanisms --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.json_documents import write_checkpoint_document

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from neural_assemblies.core.brain import Brain  # noqa: E402
from research.experiments.capacity.lexicon_capacity import (  # noqa: E402
    _drive,
    _probe_checkpoint,
    build_phon_patterns,
)


# ======================================================================
# Configuration
# ======================================================================

@dataclass(frozen=True)
class Config:
    n: int = 3000
    k: int = 30
    p: float = 0.05
    beta: float = 0.05
    n_phon: int = 600
    k_phon: int = 30
    k_cat: int = 10
    n_categories: int = 8
    train_rounds: int = 6
    probe_rounds: int = 4
    vocab_max: int = 400
    policy: str = "topk"
    mode: str = "ff"
    seed: int = 0
    # --- mechanisms under test ---
    refracted_strength: float = 0.0   # 0 disables `refracted`
    lri_period: int = 0               # 0 disables LRI
    lri_strength: float = 0.0
    synaptic_scaling: bool = False

    def make_policy(self):
        return None  # fixed top-k throughout; policy interactions are out of
        # scope here (see capacity/REPORT.md Table 5: E%-WTA does not form
        # assemblies at all in this drive regime).

    @property
    def arm(self) -> str:
        if self.synaptic_scaling:
            return "scaling"
        if self.lri_period > 0:
            return "lri"
        if self.refracted_strength > 0:
            return "refracted"
        return "baseline"


DEFAULT_CHECKPOINTS: Tuple[int, ...] = (25, 50, 100, 200, 300, 400)


def make_brain(cfg: Config) -> Brain:
    b = Brain(p=cfg.p, seed=cfg.seed, engine="numpy_sparse",
              synaptic_scaling=cfg.synaptic_scaling)
    b.add_explicit_area("PHON", cfg.n_phon, cfg.k_phon)
    b.add_area("LEX", cfg.n, cfg.k, beta=cfg.beta,
               winner_policy=cfg.make_policy(),
               refracted=(cfg.refracted_strength > 0),
               refracted_strength=cfg.refracted_strength,
               refractory_period=cfg.lri_period,
               inhibition_strength=cfg.lri_strength)
    return b


def _bias_stats(b: Brain, cfg: Config) -> Dict[str, float]:
    st = b._engine._areas["LEX"]
    bias = getattr(st, "_cumulative_bias", None)
    if bias is None or len(bias) == 0:
        return {"bias_mean": 0.0, "bias_max": 0.0, "bias_p95": 0.0}
    arr = np.asarray(bias, dtype=float)
    return {"bias_mean": float(arr.mean()), "bias_max": float(arr.max()),
            "bias_p95": float(np.percentile(arr, 95))}


def _probe_nobias(b: Brain, cfg: Config, pats, cats, stored, V: int) -> Dict:
    """Repeat the checkpoint probe with the refracted bias zeroed.

    The bias is a read-out distortion as well as a recruitment pressure: an
    assembly learned early carries more accumulated suppression than one
    learned late, so leaving it engaged during retrieval confounds the
    forgetting gradient with the mechanism itself.
    """
    probe = b.clone()
    probe._explicit_engine = b._explicit_engine
    probe.disable_plasticity = True
    try:
        probe.clear_refracted_bias("LEX")
        probe.clear_refractory("LEX")
        probe.set_lri("LEX", 0, 0.0)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return _probe_checkpoint(probe, cfg, pats, cats, stored, V)


# ======================================================================
# One run
# ======================================================================

def run_one(cfg: Config, checkpoints: Sequence[int]) -> Dict:
    rng = np.random.default_rng(cfg.seed * 7919 + 13)
    pats, cats = build_phon_patterns(cfg, rng)
    b = make_brain(cfg)
    lex = b.areas["LEX"]

    stored: List[np.ndarray] = []
    per_word: List[Dict] = []
    cps = [c for c in checkpoints if c <= cfg.vocab_max]
    cp_results: List[Dict] = []
    t0 = time.perf_counter()

    exhausted_at: Optional[int] = None
    for i in range(cfg.vocab_max):
        w_before = int(lex.w)
        try:
            assembly = _drive(b, pats[i], cfg.train_rounds, cfg.mode)
        except RuntimeError as exc:
            if "too small to sample" not in str(exc):
                raise
            exhausted_at = i
            break
        recruited = int(np.sum(assembly >= w_before))
        w_after = int(lex.w)
        stored.append(assembly)
        per_word.append({
            "index": i,
            "size": int(len(assembly)),
            "recruited": recruited,
            "recruit_frac": recruited / max(1, len(assembly)),
            "w_before": w_before,
            "w_after": w_after,
            "w_over_n": w_after / cfg.n,
            "recruit_frac_null": 1.0 - w_before / cfg.n,
            "recruit_excess": (recruited / max(1, len(assembly)))
                              - (1.0 - w_before / cfg.n),
        })
        if (i + 1) in cps:
            cp_results.append(_probe_checkpoint(b, cfg, pats, cats, stored, i + 1))

    if exhausted_at is not None and (not cp_results
                                     or cp_results[-1]["V"] < len(stored)):
        cp_results.append(_probe_checkpoint(b, cfg, pats, cats, stored,
                                            len(stored)))

    nobias = None
    if cfg.refracted_strength > 0 and stored:
        V = cp_results[-1]["V"] if cp_results else len(stored)
        nobias = _probe_nobias(b, cfg, pats, cats, stored, V)

    return {
        "config": dict(cfg.__dict__),
        "arm": cfg.arm,
        "per_word": per_word,
        "checkpoints": cp_results,
        "probe_nobias": nobias,
        "bias_stats": _bias_stats(b, cfg),
        "train_seconds": time.perf_counter() - t0,
        "final_w": int(lex.w),
        "final_w_over_n": int(lex.w) / cfg.n,
        "exhausted_at": exhausted_at,
        "vocab_learned": len(stored),
    }


# ======================================================================
# Sweep
# ======================================================================

def build_cells(quick: bool) -> List[Dict]:
    ns = (1000, 3000) if quick else (1000, 3000, 10000)
    cells: List[Dict] = []

    # -- Arm 0/1: refracted strength sweep (includes rs=0 baseline) ---------
    rss = (0.0, 0.02, 0.15) if quick else (0.0, 0.005, 0.02, 0.05, 0.15, 0.5)
    for n in ns:
        for rs in rss:
            cells.append({"n": n, "refracted_strength": rs})

    # -- Arm 2: synaptic scaling, feedforward AND recurrent -----------------
    # ff is the matched comparison to the baseline above; rec is included
    # because the mechanism's documented failure mode is about attractors,
    # and because in ff mode the only area->area fiber is PHON->LEX.
    for n in ns:
        cells.append({"n": n, "synaptic_scaling": True})
        cells.append({"n": n, "mode": "rec"})
        cells.append({"n": n, "mode": "rec", "synaptic_scaling": True})

    # -- Arm 3: LRI (finite-memory refractory) ------------------------------
    for n in ns:
        cells.append({"n": n, "lri_period": 20, "lri_strength": 0.5})
    if not quick:
        cells.append({"n": 3000, "lri_period": 5, "lri_strength": 0.5})
        cells.append({"n": 3000, "lri_period": 20, "lri_strength": 2.0})

    return cells


def build_cells_beta(rs: float) -> List[Dict]:
    """Second pass: does refracted suppression restore the exponent at other
    plasticity rates? Run only after pass 1 picks the strength."""
    out = []
    for n in (1000, 3000, 10000):
        for beta in (0.01, 0.2):
            out.append({"n": n, "beta": beta, "refracted_strength": 0.0})
            out.append({"n": n, "beta": beta, "refracted_strength": rs})
    return out


def sweep(cells: Sequence[Dict], seeds: Sequence[int], base: Config,
          checkpoints: Sequence[int], out_path: Optional[Path] = None
          ) -> List[Dict]:
    out = []
    for spec in cells:
        for s in seeds:
            cfg = Config(**{**base.__dict__, **spec, "seed": s})
            t = time.perf_counter()
            res = run_one(cfg, checkpoints)
            out.append(res)
            ex = res.get("exhausted_at")
            print(f"  n={cfg.n:>6} arm={cfg.arm:<9} rs={cfg.refracted_strength:<6}"
                  f" lri={cfg.lri_period}/{cfg.lri_strength} mode={cfg.mode:<4}"
                  f" beta={cfg.beta:<5} ss={int(cfg.synaptic_scaling)} seed={s}"
                  f" w/n={res['final_w_over_n']:.3f} V={res['vocab_learned']}"
                  f"{f' EXHAUSTED@{ex}' if ex is not None else ''}"
                  f" ({time.perf_counter()-t:.1f}s)", flush=True)
        # Checkpoint after every cell: this machine is heavily contended and a
        # killed sweep must not lose hours of completed runs.
        if out_path is not None:
            write_checkpoint_document(out_path, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--beta-pass", type=float, default=None,
                    help="run the beta x refracted-strength second pass at "
                         "this refracted_strength")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", default=None,
                    help="i/N -- run every Nth cell starting at i. The host "
                         "is CPU-contended; several single-threaded shards "
                         "get a larger total scheduler share than one process.")
    args = ap.parse_args()

    if args.beta_pass is not None:
        cells = build_cells_beta(args.beta_pass)
        seeds = (0, 1, 2, 3, 4)
        base = Config(vocab_max=400)
        cps = DEFAULT_CHECKPOINTS
        default_out = "results_beta_pass.json"
    elif args.quick:
        cells = build_cells(True)
        seeds = (0, 1)
        base = Config(vocab_max=100)
        cps = (25, 50, 100)
        default_out = "results_quick.json"
    else:
        cells = build_cells(False)
        seeds = (0, 1, 2, 3, 4)
        base = Config(vocab_max=400)
        cps = DEFAULT_CHECKPOINTS
        default_out = "results_recruitment.json"

    if args.shard:
        i, N = (int(x) for x in args.shard.split("/"))
        cells = cells[i::N]
        default_out = default_out.replace(".json", f"_shard{i}of{N}.json")

    out = Path(args.out) if args.out else (Path(__file__).parent / default_out)
    print(f"recruitment sweep: {len(cells)} cells x {len(seeds)} seeds, "
          f"vocab_max={base.vocab_max} -> {out.name}", flush=True)
    t0 = time.perf_counter()
    results = sweep(cells, seeds, base, cps, out_path=out)
    print(f"done in {time.perf_counter()-t0:.1f}s")

    write_checkpoint_document(out, results)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
