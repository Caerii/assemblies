"""
Deep performance profiling of Assembly Calculus operations.

Profiles each operation end-to-end, per-step, and by internal phase.
Compares sparse engine (n=10000) vs explicit engine (n=1000).
"""

import cProfile
import io
import pstats
import time
import sys
import os
import copy
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.brain import Brain
from src.assembly_calculus import (
    project, reciprocal_project, associate, merge,
    pattern_complete, separate,
)
from src.assembly_calculus.ops import _snap


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
ROUNDS = 10

SPARSE_CFG = dict(n=10_000, k=100, p=0.05, beta=0.1, engine="numpy_sparse")
EXPLICIT_CFG = dict(n=1_000, k=100, p=0.05, beta=0.1, engine="numpy_explicit")


def _brain(cfg):
    return Brain(p=cfg["p"], save_winners=True, seed=SEED, engine=cfg["engine"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that records elapsed wall-clock time."""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.t0


def fmt_ms(seconds):
    return f"{seconds * 1000:8.2f} ms"


def banner(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# 1. End-to-end operation timings
# ---------------------------------------------------------------------------

def bench_operations(cfg, label):
    banner(f"Operation Timings — {label} (n={cfg['n']})")
    n, k, beta = cfg["n"], cfg["k"], cfg["beta"]

    results = {}

    # --- project ---
    b = _brain(cfg)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    with Timer() as t:
        project(b, "stim", "A", rounds=ROUNDS)
    results["project"] = t.elapsed
    print(f"  project          {fmt_ms(t.elapsed)}")

    # --- reciprocal_project ---
    b = _brain(cfg)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    project(b, "stim", "A", rounds=ROUNDS)
    b.areas["A"].fix_assembly()
    with Timer() as t:
        reciprocal_project(b, "A", "B", rounds=ROUNDS)
    results["reciprocal_project"] = t.elapsed
    print(f"  reciprocal_proj  {fmt_ms(t.elapsed)}")

    # --- associate ---
    b = _brain(cfg)
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    project(b, "stimA", "A", rounds=ROUNDS)
    project(b, "stimB", "B", rounds=ROUNDS)
    with Timer() as t:
        associate(b, "A", "B", "C", stim_a="stimA", stim_b="stimB", rounds=ROUNDS)
    results["associate"] = t.elapsed
    print(f"  associate        {fmt_ms(t.elapsed)}")

    # --- merge ---
    b = _brain(cfg)
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    project(b, "stimA", "A", rounds=ROUNDS)
    project(b, "stimB", "B", rounds=ROUNDS)
    with Timer() as t:
        merge(b, "A", "B", "C", stim_a="stimA", stim_b="stimB", rounds=ROUNDS)
    results["merge"] = t.elapsed
    print(f"  merge            {fmt_ms(t.elapsed)}")

    # --- pattern_complete ---
    b = _brain(cfg)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    project(b, "stim", "A", rounds=ROUNDS)
    with Timer() as t:
        pattern_complete(b, "A", fraction=0.5, rounds=5, seed=42)
    results["pattern_complete"] = t.elapsed
    print(f"  pattern_complete {fmt_ms(t.elapsed)}")

    # --- separate ---
    b = _brain(cfg)
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    with Timer() as t:
        separate(b, "stimA", "stimB", "A", rounds=ROUNDS)
    results["separate"] = t.elapsed
    print(f"  separate         {fmt_ms(t.elapsed)}")

    return results


# ---------------------------------------------------------------------------
# 2. Per-step timing within project()
# ---------------------------------------------------------------------------

def bench_per_step(cfg, label):
    banner(f"Per-Step Breakdown — {label} (n={cfg['n']})")
    n, k, beta = cfg["n"], cfg["k"], cfg["beta"]

    b = _brain(cfg)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)

    step_times = []

    # Step 1: stim -> A (no recurrence)
    with Timer() as t:
        b.project({"stim": ["A"]}, {})
    step_times.append(("step 1 (stim->A only)", t.elapsed))

    # Steps 2-ROUNDS: stim -> A + A -> A
    for i in range(2, ROUNDS + 1):
        with Timer() as t:
            b.project({"stim": ["A"]}, {"A": ["A"]})
        step_times.append((f"step {i} (stim->A + A->A)", t.elapsed))

    total = sum(e for _, e in step_times)
    for label_s, elapsed in step_times:
        pct = elapsed / total * 100 if total > 0 else 0
        print(f"  {label_s:30s} {fmt_ms(elapsed)} ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s} {fmt_ms(total)}")


# ---------------------------------------------------------------------------
# 3. Per-step for associate (3 phases)
# ---------------------------------------------------------------------------

def bench_associate_phases(cfg, label):
    banner(f"Associate Phase Breakdown — {label} (n={cfg['n']})")
    n, k, beta = cfg["n"], cfg["k"], cfg["beta"]

    b = _brain(cfg)
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    project(b, "stimA", "A", rounds=ROUNDS)
    project(b, "stimB", "B", rounds=ROUNDS)

    # Phase 1: A -> C
    with Timer() as t:
        b.project({"stimA": ["A"]}, {"A": ["A", "C"]})
        for _ in range(ROUNDS - 1):
            b.project({"stimA": ["A"]}, {"A": ["A", "C"], "C": ["C"]})
    print(f"  Phase 1 (A->C, {ROUNDS} steps)     {fmt_ms(t.elapsed)}")

    # Phase 2: B -> C
    with Timer() as t:
        b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
        for _ in range(ROUNDS - 1):
            b.project({"stimB": ["B"]}, {"B": ["B", "C"], "C": ["C"]})
    print(f"  Phase 2 (B->C, {ROUNDS} steps)     {fmt_ms(t.elapsed)}")

    # Phase 3: interleave
    with Timer() as t:
        for _ in range(ROUNDS):
            b.project(
                {"stimA": ["A"], "stimB": ["B"]},
                {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]},
            )
    print(f"  Phase 3 (interleave, {ROUNDS} steps) {fmt_ms(t.elapsed)}")


# ---------------------------------------------------------------------------
# 4. cProfile of a single project() call
# ---------------------------------------------------------------------------

def profile_project(cfg, label):
    banner(f"cProfile — single project() — {label} (n={cfg['n']})")
    n, k, beta = cfg["n"], cfg["k"], cfg["beta"]

    b = _brain(cfg)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)

    pr = cProfile.Profile()
    pr.enable()
    project(b, "stim", "A", rounds=ROUNDS)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


# ---------------------------------------------------------------------------
# 5. Internal primitive timings (sparse engine only)
# ---------------------------------------------------------------------------

def bench_primitives_sparse():
    banner("Sparse Engine Internal Primitives (n=10000)")
    n, k, p = 10_000, 100, 0.05

    from src.compute.sparse_simulation import SparseSimulationEngine
    from src.compute.winner_selection import WinnerSelector

    rng = np.random.default_rng(SEED)
    sparse_sim = SparseSimulationEngine(rng)
    winner_sel = WinnerSelector(rng)

    # sample_new_winner_inputs
    input_sizes = [k]  # one source of size k
    times = []
    for _ in range(50):
        with Timer() as t:
            sparse_sim.sample_new_winner_inputs(input_sizes, n, 0, k, p)
        times.append(t.elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  sample_new_winner_inputs (1 source)  {fmt_ms(avg)} ± {fmt_ms(std)}")

    # sample_new_winner_inputs with 2 sources
    input_sizes_2 = [k, k]
    times = []
    for _ in range(50):
        with Timer() as t:
            sparse_sim.sample_new_winner_inputs(input_sizes_2, n, 0, k, p)
        times.append(t.elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  sample_new_winner_inputs (2 sources) {fmt_ms(avg)} ± {fmt_ms(std)}")

    # sample_new_winner_inputs with w > 0 (existing assembly)
    times = []
    for _ in range(50):
        with Timer() as t:
            sparse_sim.sample_new_winner_inputs([k], n, k * 5, k, p)
        times.append(t.elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  sample_new_winner_inputs (w=500)     {fmt_ms(avg)} ± {fmt_ms(std)}")

    # heapq_select_top_k on typical array
    arr = rng.random(k + k)  # w previous + k potential
    times = []
    for _ in range(1000):
        with Timer() as t:
            winner_sel.heapq_select_top_k(arr, k)
        times.append(t.elapsed)
    avg = np.mean(times)
    print(f"  heapq_select_top_k (n={len(arr)}, k={k})  {fmt_ms(avg)}")

    # heapq_select_top_k on larger array
    arr_big = rng.random(5000)
    times = []
    for _ in range(1000):
        with Timer() as t:
            winner_sel.heapq_select_top_k(arr_big, k)
        times.append(t.elapsed)
    avg = np.mean(times)
    print(f"  heapq_select_top_k (n=5000, k={k})  {fmt_ms(avg)}")

    # compute_input_splits
    first_inputs = [int(x) for x in rng.integers(3, 8, size=50)]
    times = []
    for _ in range(500):
        with Timer() as t:
            sparse_sim.compute_input_splits([k], first_inputs)
        times.append(t.elapsed)
    avg = np.mean(times)
    print(f"  compute_input_splits (50 winners)    {fmt_ms(avg)}")

    # Matrix operations: vstack/hstack for expansion
    mat = (rng.random((200, 200)) < p).astype(np.float32)
    times = []
    for _ in range(500):
        new_rows = (rng.random((50, 200)) < p).astype(np.float32)
        with Timer() as t:
            np.vstack([mat, new_rows])
        times.append(t.elapsed)
    avg = np.mean(times)
    print(f"  np.vstack (200x200 + 50x200)         {fmt_ms(avg)}")

    times = []
    for _ in range(500):
        new_cols = (rng.random((200, 50)) < p).astype(np.float32)
        with Timer() as t:
            np.hstack([mat, new_cols])
        times.append(t.elapsed)
    avg = np.mean(times)
    print(f"  np.hstack (200x200 + 200x50)         {fmt_ms(avg)}")

    # Plasticity: ix_ based update
    mat = rng.random((300, 300)).astype(np.float32)
    rows = np.arange(100)
    cols = np.arange(100)
    times = []
    for _ in range(1000):
        with Timer() as t:
            ix = np.ix_(rows, cols)
            mat[ix] *= 1.1
        times.append(t.elapsed)
    avg = np.mean(times)
    print(f"  plasticity ix_ update (100x100)      {fmt_ms(avg)}")

    # scipy.stats.binom.ppf (cold)
    from scipy.stats import binom
    times = []
    for _ in range(50):
        with Timer() as t:
            binom.ppf(0.99, k, p)
        times.append(t.elapsed)
    avg = np.mean(times)
    print(f"  binom.ppf (single call)              {fmt_ms(avg)}")

    # scipy.stats.truncnorm.rvs
    from scipy.stats import truncnorm
    mu = k * p
    import math
    std_val = math.sqrt(k * p * (1 - p))
    a = (3 - mu) / std_val
    times = []
    for _ in range(50):
        with Timer() as t:
            truncnorm.rvs(a, np.inf, scale=std_val, size=k, random_state=rng)
        times.append(t.elapsed)
    avg = np.mean(times)
    print(f"  truncnorm.rvs (size={k})              {fmt_ms(avg)}")


# ---------------------------------------------------------------------------
# 6. Scaling: how does project() time scale with n?
# ---------------------------------------------------------------------------

def bench_scaling():
    banner("Scaling: project() time vs n (sparse engine)")
    sizes = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
    k = 100

    for n in sizes:
        b = Brain(p=0.05, save_winners=True, seed=SEED, engine="numpy_sparse")
        b.add_stimulus("stim", k)
        b.add_area("A", n, k, 0.1)

        times = []
        for trial in range(3):
            b2 = Brain(p=0.05, save_winners=True, seed=SEED + trial, engine="numpy_sparse")
            b2.add_stimulus("stim", k)
            b2.add_area("A", n, k, 0.1)
            with Timer() as t:
                project(b2, "stim", "A", rounds=ROUNDS)
            times.append(t.elapsed)

        avg = np.mean(times)
        std = np.std(times)
        print(f"  n={n:>6d}  {fmt_ms(avg)} ± {fmt_ms(std)}")


# ---------------------------------------------------------------------------
# 7. deepcopy cost (used in associate/merge tests)
# ---------------------------------------------------------------------------

def bench_deepcopy():
    banner("deepcopy cost (relevant for associate/merge tests)")
    n, k, beta = 10_000, 100, 0.1

    b = Brain(p=0.05, save_winners=True, seed=SEED, engine="numpy_sparse")
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    project(b, "stimA", "A", rounds=ROUNDS)
    project(b, "stimB", "B", rounds=ROUNDS)
    associate(b, "A", "B", "C", stim_a="stimA", stim_b="stimB", rounds=ROUNDS)

    times = []
    for _ in range(10):
        with Timer() as t:
            _ = copy.deepcopy(b)
        times.append(t.elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  deepcopy(brain, 3 areas, after assoc) {fmt_ms(avg)} ± {fmt_ms(std)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Assembly Calculus — Deep Performance Profiling")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")

    # End-to-end
    sparse_results = bench_operations(SPARSE_CFG, "Sparse")
    explicit_results = bench_operations(EXPLICIT_CFG, "Explicit")

    # Comparison table
    banner("Sparse vs Explicit Comparison")
    print(f"  {'Operation':<22s} {'Sparse':>10s} {'Explicit':>10s} {'Ratio':>8s}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8}")
    for op in sparse_results:
        s = sparse_results[op]
        e = explicit_results[op]
        ratio = s / e if e > 0 else float("inf")
        print(f"  {op:<22s} {fmt_ms(s)} {fmt_ms(e)} {ratio:7.2f}x")

    # Per-step
    bench_per_step(SPARSE_CFG, "Sparse")
    bench_per_step(EXPLICIT_CFG, "Explicit")

    # Associate phases
    bench_associate_phases(SPARSE_CFG, "Sparse")

    # cProfile
    profile_project(SPARSE_CFG, "Sparse")
    profile_project(EXPLICIT_CFG, "Explicit")

    # Primitives
    bench_primitives_sparse()

    # Scaling
    bench_scaling()

    # deepcopy
    bench_deepcopy()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
