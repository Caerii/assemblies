#!/usr/bin/env python3
"""
Test script for C++ performance comparison.
"""

import sys
sys.path.append('.')
from src.simulation.association_simulator_cpp import benchmark_comparison
import time

def main():
    print('Performance Benchmark: C++ vs Python')
    print('=' * 50)

    # Run benchmark with smaller parameters for quick comparison
    results = benchmark_comparison(2000, 50, 0.05, 0.1, 2)

    print('\nBenchmark Results:')
    print(f'  C++ Available: {results.get("cpp_available", "Unknown")}')
    print(f'  C++ Success: {results.get("cpp_success", "Unknown")}')
    if results.get('cpp_success'):
        print(f'  C++ Time: {results["cpp_time"]:.4f} seconds')
        print(f'  C++ Winners: {results["cpp_winners"]}')

    print(f'  Python Success: {results.get("python_success", "Unknown")}')
    if results.get('python_success'):
        print(f'  Python Time: {results["python_time"]:.4f} seconds')
        print(f'  Python Winners: {results["python_winners"]}')

    if results.get('speedup'):
        print(f'\n🚀 SPEEDUP: {results["speedup"]:.2f}x faster with C++!')
        print(f'   Time saved: {results["python_time"] - results["cpp_time"]:.4f} seconds')
        print(f'   Efficiency gain: {(results["speedup"] - 1) * 100:.1f}%')

if __name__ == "__main__":
    main()
