"""
Re-measure assembly capacity V*(n) and its scaling exponent on the CORRECTED
substrate (norm_init ON) vs the collapsed substrate (norm_init OFF).

Background: the original capacity study (REPORT.md) ran on a substrate where
k-cap systematically selects the random graph's high-in-degree HUBS, because
incoming weights were never normalized. The fix -- one-time per-fiber incoming
weight normalization (`norm_init`, now the Brain default) -- removes that hub
bias. This script re-measures the headline capacity numbers with norm_init
toggled OFF vs ON, holding everything else identical to lexicon_capacity.py.

We REUSE lexicon_capacity.run_one / Config / recruit_horizon unchanged and only
monkeypatch make_brain so it forwards a norm_init flag to Brain().

Usage:
    python -m research.experiments.capacity.norm_init_capacity --smoke
    python -m research.experiments.capacity.norm_init_capacity --task1
    python -m research.experiments.capacity.norm_init_capacity --beta
    python -m research.experiments.capacity.norm_init_capacity --all
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("ASSEMBLIES_BACKBONE_CACHE", "0")

from research.json_documents import write_checkpoint_document

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from neural_assemblies.core.brain import Brain  # noqa: E402
from research.experiments.capacity import lexicon_capacity as lc  # noqa: E402
from research.experiments.capacity.analyze import recruit_horizon  # noqa: E402

HERE = Path(__file__).parent

# module-global toggle read by the patched make_brain
_NORM = {"v": True}


def make_brain_norm(cfg: lc.Config) -> Brain:
    """Identical to lexicon_capacity.make_brain but forwards norm_init."""
    b = Brain(p=cfg.p, seed=cfg.seed, engine="numpy_sparse",
              norm_init=_NORM["v"])
    b.add_explicit_area("PHON", cfg.n_phon, cfg.k_phon)
    b.add_area("LEX", cfg.n, cfg.k, beta=cfg.beta,
               winner_policy=cfg.make_policy())
    return b


lc.make_brain = make_brain_norm  # patch the name run_one resolves at call time


def run_cell(n, beta, mode, norm, seed, vocab_max, checkpoints):
    _NORM["v"] = norm
    cfg = lc.Config(n=n, beta=beta, mode=mode, policy="topk", seed=seed,
                    vocab_max=vocab_max)
    t = time.perf_counter()
    res = lc.run_one(cfg, checkpoints)
    res["norm_init"] = norm
    # CORRECTED V*: identification-based capacity (see lc.identification_capacity).
    idcap = lc.identification_capacity(res["checkpoints"], thresh=0.5)
    res["vstar"] = idcap["vstar"]
    res["vstar_censored"] = idcap["censored"]
    # keep the OLD churn-based metric under its own name for transparency /
    # before-after comparison -- it is NOT capacity on the norm_init substrate.
    res["vstar_recruit"] = recruit_horizon(res["per_word"])
    res["wall_seconds"] = time.perf_counter() - t
    return res


def sweep(cells, seeds, vocab_max, checkpoints, out_path, verbose=True):
    results = []
    t0 = time.perf_counter()
    for (n, beta, mode, norm) in cells:
        for s in seeds:
            res = run_cell(n, beta, mode, norm, s, vocab_max, checkpoints)
            results.append(res)
            if verbose:
                ex = res.get("exhausted_at")
                print(f"  n={n:>6} beta={beta:<5} mode={mode:<3} "
                      f"norm={int(norm)} seed={s} "
                      f"V*={res['vstar']:>3} Vlearn={res['vocab_learned']:>3} "
                      f"w/n={res['per_word'][-1]['w_over_n']:.3f} "
                      f"{f'EXH@{ex}' if ex is not None else ''} "
                      f"({res['wall_seconds']:.1f}s)", flush=True)
            # incremental save so a crash never loses everything
            write_checkpoint_document(Path(out_path), results)
    print(f"sweep done in {time.perf_counter()-t0:.1f}s -> {out_path}",
          flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task1", action="store_true")
    ap.add_argument("--beta", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ns", default=None, help="comma list overrides n set")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--vocab", type=int, default=400)
    ap.add_argument("--mode", default="rec")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        cells = [(1000, 0.05, "rec", False), (1000, 0.05, "rec", True),
                 (1000, 0.05, "ff", False), (1000, 0.05, "ff", True)]
        seeds = (0, 1)
        vocab = 100
        cps = (25, 50, 100)
        out = args.out or (HERE / "results_norm_smoke.json")
        sweep(cells, seeds, vocab, cps, out)
        return 0

    ns = [int(x) for x in args.ns.split(",")] if args.ns else [1000, 3000, 10000]
    seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else (0, 1, 2, 3)
    cps = lc.DEFAULT_CHECKPOINTS
    vocab = args.vocab

    cells = []
    if args.task1 or args.all:
        # headline: beta=0.05, OFF vs ON, requested mode, all n
        for n in ns:
            for norm in (False, True):
                cells.append((n, 0.05, args.mode, norm))
    if args.beta or args.all:
        # beta control with norm ON (+ OFF for direct contrast), all n
        for n in ns:
            for beta in (0.0, 0.01, 0.05, 0.2):
                for norm in (True, False):
                    cells.append((n, beta, args.mode, norm))
    # dedupe preserving order
    seen = set(); uniq = []
    for c in cells:
        if c not in seen:
            seen.add(c); uniq.append(c)
    cells = uniq

    tag = "all" if args.all else ("task1" if args.task1 else "beta")
    out = args.out or (HERE / f"results_norm_{tag}_{args.mode}.json")
    print(f"norm_init capacity sweep [{tag}, mode={args.mode}]: "
          f"{len(cells)} cells x {len(seeds)} seeds, vocab={vocab}", flush=True)
    sweep(cells, seeds, vocab, cps, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
