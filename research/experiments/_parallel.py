"""Parallel (arm x seed) cell runner + fingerprinted baseline cache.

The wall-clock profile of every A/B here is arms x seeds SEQUENTIAL
trainings, each using roughly one core -- E1 spent ~40 minutes on 10 cells a
process pool finishes in ~2 rounds. Safe to parallelize because training is
BIT-IDENTICAL across processes since the one-seeding-path fixes (#65/#80);
each cell reseeds np/random itself, and cells share nothing.

Also here: `cached_parser`, because #129's DEFAULT arm, E1's OFF arm and
E2's OFF arm are THE SAME parser at the same seeds, retrained once per
experiment. The cache key includes `training_code_fingerprint()` (the same
traced-source fingerprint the backbone cache uses), so a source change
invalidates it -- the fingerprint-gap lesson: warm runs do not train, so the
key must cover everything that matters DURING training, which is exactly
what that fingerprint was rebuilt to trace (#106).

Windows spawn semantics: `worker` must be a TOP-LEVEL function of the
experiment module, its module must guard `if __name__ == "__main__"`, and
results must pickle.
"""
from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, Optional, Tuple

CACHE_DIR = os.path.join(os.path.dirname(__file__), "_parser_cache")


def run_cells(
    worker: Callable[..., dict],
    cells: Iterable[Tuple],
    max_workers: Optional[int] = None,
) -> Dict[Tuple, dict]:
    """Run `worker(*cell)` for every cell in a process pool.

    Returns {cell: result}. A worker exception propagates -- an experiment
    with a dead cell must fail loudly, not average over the survivors
    (the NaN-filter lesson: dropping the failed half is not hygiene).
    """
    cells = list(cells)
    if max_workers is None:
        max_workers = min(len(cells), max(1, (os.cpu_count() or 4) - 2))
    # Pin BLAS to one thread per worker BEFORE the pool spawns (children
    # inherit env; the thread count is read at numpy import). Without this,
    # N workers x M BLAS threads oversubscribes the box severalfold and the
    # cells fight each other. Each cell is ~single-core work anyway.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    results: Dict[Tuple, dict] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, *cell): cell for cell in cells}
        for fut in as_completed(futures):
            cell = futures[fut]
            results[cell] = fut.result()
            print(f"[done {cell}]", flush=True)
    return results


def forked_parser(base_tag: str, seed: int,
                  build_and_train_pre: Callable[[], object],
                  arm_setup: Callable[[object], None],
                  train_final: Callable[[object], object]):
    """Fork per-arm training from a shared pre-stage checkpoint.

    THE OBSERVATION THIS EXPLOITS: an experiment's arms usually differ only
    in mechanisms that act during the FINAL curriculum stage (feature
    phases run only in SENTENCES; scoped scaling touches only areas that
    receive plasticity there), so the earlier stages are BIT-IDENTICAL
    across arms -- and were being retrained once per arm. Train them once
    per seed, checkpoint (parser + global RNG states, so the continuation
    consumes the same draws full training would), then per arm: load a
    fresh copy, apply the arm's flags, train the final stage only.

    Equivalence is NOT assumed: `research/experiments/fork_equivalence.py`
    pins full-train == fork-train exactly before any experiment relies on
    this. `arm_setup` must set flags that work POST-CONSTRUCTION (engine
    attribute for scoped scaling, parser attribute for novelty gain).
    """
    import random as _random

    import numpy as _np

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(
        CACHE_DIR, _cache_key(f"pre-{base_tag}", seed) + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            parser, np_state, py_state = pickle.load(f)
    else:
        parser = build_and_train_pre()
        np_state = _np.random.get_state()
        py_state = _random.getstate()
        with open(path, "wb") as f:
            pickle.dump((parser, np_state, py_state), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    _np.random.set_state(np_state)
    _random.setstate(py_state)
    arm_setup(parser)
    return train_final(parser)


def _cache_key(tag: str, seed: int) -> str:
    from neural_assemblies.assembly_calculus.emergent.evaluation.sweep \
        import training_code_fingerprint

    return f"{tag}-{training_code_fingerprint()[:16]}-s{seed}"


def cached_parser(tag: str, seed: int, build_and_train: Callable[[], object]):
    """Return a trained parser, from cache when code+config are unchanged.

    `tag` must encode EVERY config choice the closure makes (n, k, stages,
    vocabulary, mechanism arms); the code side is covered by the traced
    source fingerprint. Anything not in the key that affects training is a
    silent staleness bug -- when in doubt, put it in the tag.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _cache_key(tag, seed) + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    parser = build_and_train()
    with open(path, "wb") as f:
        pickle.dump(parser, f, protocol=pickle.HIGHEST_PROTOCOL)
    return parser
