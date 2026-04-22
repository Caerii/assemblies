#!/usr/bin/env python3
"""Performance probes for optional C++ and CUDA implementations."""

from __future__ import annotations

import importlib
import importlib.util
import time

import numpy as np
import pytest


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def test_cuda_availability() -> None:
    """Skip when the CUDA extension is absent; fail only on broken imports."""
    if not _module_available("cuda_brain"):
        pytest.skip("cuda_brain extension is not installed")
    importlib.import_module("cuda_brain")


def test_cpp_availability() -> None:
    """Skip when the C++ extension is absent; fail only on broken imports."""
    if not _module_available("neural_assemblies.core.brain_cpp"):
        pytest.skip("BrainCPP extension is not installed")
    importlib.import_module("neural_assemblies.core.brain_cpp")


def run_cpp_benchmark(n, k, p, beta, overlap_iter):
    """Run the C++ benchmark helper."""
    from neural_assemblies.simulation.association_simulator_cpp import association_sim_cpp

    print(f"Running C++ benchmark: n={n}, k={k}")
    start_time = time.time()

    try:
        _, winners = association_sim_cpp(n, k, p, beta, overlap_iter, verbose=0)
        cpp_time = time.time() - start_time
        print(f"C++ completed in {cpp_time:.4f} seconds")
        print(f"Generated {len(winners)} winner sets")
        return cpp_time, len(winners), True
    except Exception as exc:
        cpp_time = time.time() - start_time
        print(f"C++ failed: {exc}")
        return cpp_time, 0, False


def run_cuda_benchmark(n, k, p, beta, overlap_iter):
    """Run the CUDA benchmark helper."""
    start_time = time.time()

    try:
        from cuda_brain_wrapper import BrainCUDA

        print(f"Running CUDA benchmark: n={n}, k={k}")

        brain = BrainCUDA(p=p, beta=beta, max_weight=10000.0, seed=7777)
        brain.add_area("A", n, k, beta=beta)
        brain.add_area("B", n, k, beta=beta)
        brain.add_area("C", n, k, beta=beta)

        brain.add_stimulus("stimA", k)
        brain.add_stimulus("stimB", k)

        brain.add_fiber("stimA", "A")
        brain.add_fiber("stimB", "B")
        brain.add_fiber("A", "A")
        brain.add_fiber("B", "B")
        brain.add_fiber("C", "C")
        brain.add_fiber("A", "C")
        brain.add_fiber("B", "C")

        print("  Phase 1: Stabilizing A/B")
        for _ in range(9):
            brain.project({"stimA": ["A"], "stimB": ["B"]}, {"A": ["A"], "B": ["B"]})

        print("  Phase 2: A->C")
        for _ in range(10):
            brain.project({"stimA": ["A"]}, {"A": ["A", "C"]})

        print("  Phase 3: B->C")
        for _ in range(10):
            brain.project({"stimB": ["B"]}, {"B": ["B", "C"]})

        print("  Phase 4: A,B->C overlap")
        for _ in range(overlap_iter):
            brain.project({"stimA": ["A"], "stimB": ["B"]}, {"A": ["A"], "B": ["B"], "C": ["C"]})

        print("  Phase 5: Final B-only")
        for _ in range(10):
            brain.project({"stimB": ["B"]}, {"B": ["B", "C"]})

        cuda_time = time.time() - start_time
        print(f"CUDA completed in {cuda_time:.4f} seconds")
        return cuda_time, 1, True

    except Exception as exc:
        cuda_time = time.time() - start_time
        print(f"CUDA failed: {exc}")
        return cuda_time, 0, False


def run_performance_comparison():
    """Run a manual comparison across available implementations."""
    print("CUDA vs C++ performance comparison")
    print("=" * 60)

    cuda_available = _module_available("cuda_brain")
    cpp_available = _module_available("neural_assemblies.core.brain_cpp")

    if not cpp_available:
        print("Cannot run comparison without the C++ implementation.")
        return

    test_cases = [
        {"n": 1000, "k": 50, "name": "Small"},
        {"n": 5000, "k": 100, "name": "Medium"},
        {"n": 10000, "k": 200, "name": "Large"},
        {"n": 50000, "k": 500, "name": "Very Large"},
    ]

    p = 0.05
    beta = 0.1
    overlap_iter = 2

    results = []

    for test_case in test_cases:
        n = test_case["n"]
        k = test_case["k"]
        name = test_case["name"]

        print(f"\n{name} test (n={n:,}, k={k})")
        print("-" * 40)

        cpp_time, _, cpp_success = run_cpp_benchmark(n, k, p, beta, overlap_iter)

        if cuda_available:
            cuda_time, _, cuda_success = run_cuda_benchmark(n, k, p, beta, overlap_iter)
        else:
            cuda_time, cuda_success = 0.0, False

        results.append(
            {
                "name": name,
                "n": n,
                "cpp_time": cpp_time,
                "cpp_success": cpp_success,
                "cuda_time": cuda_time,
                "cuda_success": cuda_success,
                "speedup": cpp_time / cuda_time if cuda_success and cuda_time > 0 else 0.0,
            }
        )

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Test':<12} {'C++ Time':<12} {'CUDA Time':<12} {'Speedup':<10} {'Status'}")
    print("-" * 60)

    for result in results:
        if result["cpp_success"] and result["cuda_success"]:
            status = "Both"
        elif result["cpp_success"]:
            status = "C++ Only"
        elif result["cuda_success"]:
            status = "CUDA Only"
        else:
            status = "Failed"

        speedup_str = f"{result['speedup']:.2f}x" if result["speedup"] > 0 else "N/A"
        print(
            f"{result['name']:<12} {result['cpp_time']:<12.4f} {result['cuda_time']:<12.4f} "
            f"{speedup_str:<10} {status}"
        )

    successful_tests = [r for r in results if r["cpp_success"] and r["cuda_success"]]
    if successful_tests:
        avg_speedup = np.mean([r["speedup"] for r in successful_tests])
        max_speedup = max([r["speedup"] for r in successful_tests])
        min_speedup = min([r["speedup"] for r in successful_tests])

        print("\nOverall statistics:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Maximum speedup: {max_speedup:.2f}x")
        print(f"  Minimum speedup: {min_speedup:.2f}x")
        print(f"  Successful tests: {len(successful_tests)}/{len(results)}")


if __name__ == "__main__":
    run_performance_comparison()
