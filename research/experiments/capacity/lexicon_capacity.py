"""
Assembly recruitment vs reuse: where is the capacity limit of an area?

MOTIVATION
----------
In a TWO_WORD EmergentParser training run at n=3000, k=30, the lexical core
areas end up ~60% materialized (NOUN_CORE w=1808/3000, DET_CORE 1739,
VERB_CORE 1701). With k=30 that is ~60 assembly-sized slots for a training
vocabulary of the same order. The hypothesis under test: these areas are near
a tiling limit, so each new word RECRUITS fresh (never-fired) neurons rather
than REUSING materialized ones, and past some vocabulary size assemblies must
overlap -- the regime where interference and forgetting appear.

DESIGN
------
A controlled lexicon-in-one-area model, deliberately simpler than the full
44-area parser so that vocabulary size is the only thing that varies:

    PHON (explicit, n_phon neurons)  --->  LEX (sparse, n neurons, k winners)
                                            ^__|  (recurrent)

Every word is a fixed k-subset of PHON. All words share ONE PHON->LEX
connectome and ONE LEX->LEX recurrent connectome, so a later word's plasticity
writes into the same synapses an earlier word used. This is the point: giving
each word a private stimulus matrix (the `PHON_{word}` idiom used elsewhere in
this repo) makes retrieval trivially perfect and hides forgetting entirely.

Words are organised into categories. Words in the same category share a block
of `k_cat` PHON neurons; the rest is idiosyncratic. This lets us ask whether
category structure survives saturation.

INSTRUMENTATION
---------------
The numpy_sparse engine materialises neurons lazily: a target area's compact
winner index is < `area.w` iff that neuron has fired before. So for a word
learned when the area had `w_before` ever-fired neurons,

    recruited = |{j in assembly : j >= w_before}|      (first-time winners)
    reused    = |assembly| - recruited

That is an exact accounting, not an estimate.

MEASUREMENTS PER CHECKPOINT V (probed on a *clone*, so probing never
contaminates the recruitment curve):
  * recruit fraction of the V-th word, and cumulative w/n
  * retrieval fidelity: re-present each learned word's PHON pattern with
    plasticity off, overlap of the retrieved assembly with the one stored at
    learning time  (null: chance overlap k/n)
  * identification accuracy: does the retrieved assembly match the RIGHT
    stored assembly, over all V candidates  (null: 1/V)
  * pairwise overlap distribution among stored assemblies (null: k/n)
  * category separation: within-category minus between-category overlap

COMPETITION POLICIES
--------------------
  topk    -- fixed-k winner-take-all (classic Assembly Calculus)
  epsilon -- E%-WTA, Hoff et al. 2026 rule: h_j >= (1-eps) * h_max
  sigma   -- E%-WTA scale-invariant variant: h_j >= h_max - sigma_c * std(h)

Usage:
    python -m research.experiments.capacity.lexicon_capacity            # full
    python -m research.experiments.capacity.lexicon_capacity --quick    # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from neural_assemblies.core.brain import Brain  # noqa: E402
from neural_assemblies.compute.winner_policies import (  # noqa: E402
    EPercentPolicy,
    TopKPolicy,
)


# ======================================================================
# Configuration
# ======================================================================

@dataclass(frozen=True)
class Config:
    n: int = 3000                # LEX area size
    k: int = 30                  # assembly size (fixed-k policies)
    p: float = 0.05              # connection probability
    beta: float = 0.05           # Hebbian plasticity rate
    n_phon: int = 600            # PHON (input) area size
    k_phon: int = 30             # PHON neurons active per word
    k_cat: int = 10              # of those, how many are the category block
    n_categories: int = 8
    train_rounds: int = 6        # projection rounds per word during learning
    probe_rounds: int = 4        # projection rounds per word during retrieval
    vocab_max: int = 400
    policy: str = "topk"         # topk | epsilon | sigma
    mode: str = "ff"             # ff | rec   (see module docstring)
    seed: int = 0

    def make_policy(self):
        if self.policy == "topk":
            return None  # engine default is fixed top-k
        if self.policy == "epsilon":
            return EPercentPolicy.from_gamma(window="epsilon", min_winners=1)
        if self.policy == "sigma":
            return EPercentPolicy.from_gamma(window="sigma", sigma_c=1.7,
                                             min_winners=1)
        raise ValueError(f"unknown policy {self.policy!r}")


DEFAULT_CHECKPOINTS: Tuple[int, ...] = (25, 50, 100, 200, 300, 400)


# ======================================================================
# Set helpers
# ======================================================================

def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """|A n B| / min(|A|,|B|).  Comparable across variable assembly sizes."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    sa, sb = set(a.tolist()), set(b.tolist())
    m = min(len(sa), len(sb))
    return len(sa & sb) / m if m else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Set cosine |A n B| / sqrt(|A||B|) -- the identification score."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / float(np.sqrt(len(sa) * len(sb)))


# ======================================================================
# Vocabulary: shared PHON substrate with category structure
# ======================================================================

def build_phon_patterns(cfg: Config, rng: np.random.Generator
                        ) -> Tuple[List[np.ndarray], List[int]]:
    """Each word: k_cat neurons from its category block + idiosyncratic rest.

    Category blocks are disjoint slices of PHON, so within-category words
    share exactly k_cat input neurons by construction and between-category
    words share only what the idiosyncratic draw gives them.
    """
    block = cfg.n_phon // (cfg.n_categories + 1)
    cat_blocks = [np.arange(c * block, (c + 1) * block) for c in range(cfg.n_categories)]
    free_pool = np.arange(cfg.n_categories * block, cfg.n_phon)

    pats: List[np.ndarray] = []
    cats: List[int] = []
    for i in range(cfg.vocab_max):
        c = i % cfg.n_categories
        core = rng.choice(cat_blocks[c], cfg.k_cat, replace=False)
        rest = rng.choice(free_pool, cfg.k_phon - cfg.k_cat, replace=False)
        pats.append(np.sort(np.concatenate([core, rest])).astype(np.uint32))
        cats.append(c)
    return pats, cats


# ======================================================================
# Brain construction / drive
# ======================================================================

def make_brain(cfg: Config) -> Brain:
    b = Brain(p=cfg.p, seed=cfg.seed, engine="numpy_sparse")
    b.add_explicit_area("PHON", cfg.n_phon, cfg.k_phon)
    b.add_area("LEX", cfg.n, cfg.k, beta=cfg.beta,
               winner_policy=cfg.make_policy())
    return b


def _drive(b: Brain, pat: np.ndarray, rounds: int, mode: str) -> np.ndarray:
    """Inhibit LEX, then drive it from PHON for `rounds` projection rounds.

    mode="ff"  : PHON -> LEX only.
    mode="rec" : PHON -> LEX plus the plastic LEX -> LEX recurrent fiber from
                 round 1 onward (the classic Assembly Calculus project loop).
    """
    b.inhibit_areas(["LEX"])
    for r in range(rounds):
        if mode == "ff" or r == 0:
            proj = {"PHON": ["LEX"]}
        else:
            proj = {"PHON": ["LEX"], "LEX": ["LEX"]}
        b.project(external_inputs={"PHON": pat}, projections=proj)
    return np.array(b.areas["LEX"].winners, dtype=np.int64)


# ======================================================================
# One run: train a vocabulary sequentially, probing at checkpoints
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
        w_before = int(lex.w)                 # read BEFORE inhibit: the
        # Area.winners setter clobbers .w on inhibit, the engine restores it
        # on the next projection, so w_before must be sampled here.
        try:
            assembly = _drive(b, pats[i], cfg.train_rounds, cfg.mode)
        except RuntimeError as exc:
            # HARD CAPACITY LIMIT: fewer than k neurons in the area have never
            # fired, so the sparse engine cannot materialise another assembly.
            # This is a measurement, not a bug -- record where it happened,
            # probe what was learned, and stop.
            if "too small to sample" not in str(exc):
                raise
            exhausted_at = i
            break
        w_after = int(lex.w)
        recruited = int(np.sum(assembly >= w_before))
        stored.append(assembly)
        per_word.append({
            "index": i,
            "size": int(len(assembly)),
            "recruited": recruited,
            "reused": int(len(assembly)) - recruited,
            "recruit_frac": recruited / max(1, len(assembly)),
            "w_before": w_before,
            "w_after": w_after,
            "w_over_n": w_after / cfg.n,
            "materialized_delta": w_after - w_before,
            # NULL MODEL (random tiling / coupon collector): if winner
            # selection were unbiased by learning history, each winner would
            # be a uniform draw from the n neurons, so the chance it has never
            # fired is exactly 1 - w_before/n. Any shortfall is the
            # rich-get-richer effect of Hebbian potentiation on incumbents.
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

    return {
        "config": cfg.__dict__,
        "per_word": per_word,
        "checkpoints": cp_results,
        "train_seconds": time.perf_counter() - t0,
        "final_w": int(lex.w),
        "final_w_over_n": int(lex.w) / cfg.n,
        "exhausted_at": exhausted_at,
        "vocab_learned": len(stored),
    }


def _probe_checkpoint(b: Brain, cfg: Config, pats, cats, stored, V: int) -> Dict:
    """Probe retrieval on a CLONE so probing never grows w in the live brain."""
    probe = b.clone()
    # Brain.clone() drops _explicit_engine, and the lazy re-creation path in
    # _engine_for() does not re-register the area it is being asked about, so
    # a cloned brain cannot drive its own explicit areas. Share the parent's
    # explicit engine instead: PHON is never a projection TARGET here, so it
    # carries no plastic state, and its winners are overwritten on every
    # _drive() call. (Read-only workaround; core/ is off-limits.)
    probe._explicit_engine = b._explicit_engine
    probe.disable_plasticity = True

    retrieved = []
    probe_failures = 0
    for i in range(V):
        try:
            retrieved.append(_drive(probe, pats[i], cfg.probe_rounds, cfg.mode))
        except RuntimeError as exc:
            if "too small to sample" not in str(exc):
                raise
            probe_failures += 1
            retrieved.append(np.array([], dtype=np.int64))

    self_ov, ident, sizes = [], [], []
    for i in range(V):
        r = retrieved[i]
        if len(r) == 0:
            continue
        sizes.append(len(r))
        self_ov.append(_overlap(r, stored[i]))
        scores = np.array([_cosine(r, stored[j]) for j in range(V)])
        ident.append(1.0 if int(np.argmax(scores)) == i else 0.0)

    # Pairwise overlap among stored assemblies (subsample for large V)
    rng = np.random.default_rng(1234 + V)
    max_pairs = 4000
    all_pairs = [(i, j) for i in range(V) for j in range(i + 1, V)]
    if len(all_pairs) > max_pairs:
        idx = rng.choice(len(all_pairs), max_pairs, replace=False)
        pairs = [all_pairs[t] for t in idx]
    else:
        pairs = all_pairs
    pw, within, between = [], [], []
    for i, j in pairs:
        o = _overlap(stored[i], stored[j])
        pw.append(o)
        (within if cats[i] == cats[j] else between).append(o)

    pw_a = np.array(pw) if pw else np.array([0.0])
    nan = float("nan")
    dec = max(1, len(self_ov) // 10)

    def _m(xs, sl=slice(None)) -> float:
        seg = xs[sl]
        return float(np.mean(seg)) if len(seg) else nan

    return {
        "V": V,
        "probe_failures": probe_failures,
        "probes_ok": len(self_ov),
        "w": int(b.areas["LEX"].w),
        "w_over_n": int(b.areas["LEX"].w) / cfg.n,
        "retrieval_overlap_mean": _m(self_ov),
        "retrieval_overlap_std": float(np.std(self_ov)) if self_ov else nan,
        "retrieval_overlap_first_decile": _m(self_ov, slice(0, dec)),
        "retrieval_overlap_last_decile": _m(self_ov, slice(-dec, None)),
        "identification_acc": _m(ident),
        "identification_acc_first_decile": _m(ident, slice(0, dec)),
        "identification_acc_last_decile": _m(ident, slice(-dec, None)),
        "pairwise_overlap_mean": float(np.mean(pw_a)),
        "pairwise_overlap_p95": float(np.percentile(pw_a, 95)),
        "pairwise_overlap_max": float(np.max(pw_a)),
        "chance_overlap": cfg.k / cfg.n,
        "within_cat_overlap": float(np.mean(within)) if within else float("nan"),
        "between_cat_overlap": float(np.mean(between)) if between else float("nan"),
        "category_separation": (float(np.mean(within)) - float(np.mean(between)))
                               if (within and between) else float("nan"),
        "probe_size_mean": float(np.mean(sizes)) if sizes else nan,
        "stored_size_mean": float(np.mean([len(s) for s in stored[:V]])),
    }


def identification_capacity(checkpoints: Sequence[Dict],
                            thresh: float = 0.5) -> Dict:
    """Identification-based capacity V*: the largest checkpoint V at which
    mean identification accuracy is still >= `thresh`.

    WHY THIS AND NOT recruit_horizon: the original V* was the "recruitment
    horizon" -- the vocabulary size past which per-word recruit fraction
    (churn between an assembly and its predecessors) falls below 0.05. On the
    ORIGINAL collapsed substrate churn ceasing coincided with capacity
    exhaustion, so recruit_horizon was a serviceable proxy. On the norm_init
    substrate the two DECOUPLE: an assembly stabilises (churn -> 0) within a
    handful of words while identification stays 0.85-1.0 out to V=400, so
    recruit_horizon collapses to ~11 and badly under-reports capacity. Capacity
    is an identification property (can the area still tell its stored words
    apart?), so V* must be measured off identification_acc, not churn.

    Returns {"vstar": int, "censored": bool}. `censored` is True when the last
    checkpoint still clears the threshold (capacity is a right-censored lower
    bound at checkpoint resolution). A run whose FIRST checkpoint already fails
    yields vstar=0. NaN identification (exhausted probes) fails the threshold.
    """
    vstar = 0
    last_V = 0
    for c in checkpoints:
        V = c["V"]
        last_V = max(last_V, V)
        acc = c.get("identification_acc")
        if acc is not None and not (isinstance(acc, float) and np.isnan(acc)) \
                and acc >= thresh:
            vstar = V
    return {"vstar": vstar, "censored": vstar == last_V and last_V > 0}


# ======================================================================
# Sweep driver
# ======================================================================

def sweep(cells: Sequence[Tuple[int, str, str, float]], seeds: Sequence[int],
          base: Config, checkpoints: Sequence[int], verbose: bool = True
          ) -> List[Dict]:
    """`cells` are (n, policy, mode, beta); every cell runs on every seed."""
    out = []
    for n, pol, mode, beta in cells:
        for s in seeds:
            cfg = Config(**{**base.__dict__, "n": n, "policy": pol,
                            "mode": mode, "beta": beta, "seed": s})
            t = time.perf_counter()
            res = run_one(cfg, checkpoints)
            out.append(res)
            if verbose:
                ex = res.get("exhausted_at")
                print(f"  n={n:>6} pol={pol:<8} mode={mode:<4} beta={beta:<5} "
                      f"seed={s} w/n={res['final_w_over_n']:.3f} "
                      f"V={res['vocab_learned']}"
                      f"{f' EXHAUSTED@{ex}' if ex is not None else ''} "
                      f"({time.perf_counter()-t:.1f}s)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        ns = (1000, 3000)
        seeds = (0, 1)
        base = Config(vocab_max=100)
        cps = (25, 50, 100)
        cells = [(n, p, "ff", 0.05) for n in ns for p in ("topk", "sigma")]
        cells += [(n, "topk", "rec", 0.05) for n in ns]
        cells += [(3000, "topk", "ff", 0.0)]
    else:
        ns = (1000, 3000, 10000)
        policies = ("topk", "epsilon", "sigma")
        seeds = (0, 1, 2, 3, 4)
        base = Config(vocab_max=400)
        cps = DEFAULT_CHECKPOINTS
        # Main factorial: n x policy, feedforward shared substrate.
        cells = [(n, p, "ff", 0.05) for n in ns for p in policies]
        # Control 1: the plastic recurrent fiber engaged (classic AC loop).
        cells += [(n, "topk", "rec", 0.05) for n in ns]
        # Control 2: plasticity strength.  beta=0 is the random-tiling null --
        # no Hebbian bias toward incumbents, so recruitment should track the
        # coupon-collector curve 1 - w/n exactly.
        cells += [(n, "topk", "ff", b) for n in ns for b in (0.0, 0.01, 0.20)]

    print(f"lexicon capacity sweep: {len(cells)} cells x {len(seeds)} seeds, "
          f"vocab_max={base.vocab_max}", flush=True)
    t0 = time.perf_counter()
    results = sweep(cells, seeds, base, cps)
    print(f"done in {time.perf_counter()-t0:.1f}s")

    out = Path(args.out) if args.out else (
        Path(__file__).parent / ("results_quick.json" if args.quick
                                 else "results_lexicon_capacity.json"))
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
