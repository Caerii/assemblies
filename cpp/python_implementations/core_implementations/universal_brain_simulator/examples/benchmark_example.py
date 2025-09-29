#!/usr/bin/env python3
"""
Benchmark Example - Universal Brain Simulator
============================================

This example shows how to run benchmarks and compare different
configurations to find the optimal setup for your hardware.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from universal_brain_simulator.client import BrainSimulator, quick_benchmark, compare_configurations


def main():
    print("⚡ BENCHMARK EXAMPLE")
    print("=" * 50)
    
    # Single benchmark - Large Scale
    print("\n📊 Single Benchmark (Large Scale)")
    sim = BrainSimulator(
        neurons=50000000,     # 50 million neurons
        active_percentage=0.01,
        areas=5,
        use_optimized_cuda=True
    )
    
    benchmark_results = sim.benchmark(warmup_steps=5, measure_steps=10)
    
    # Quick benchmark using convenience function - Billion Scale
    print("\n📊 Quick Benchmark (Billion Scale)")
    quick_results = quick_benchmark(neurons=1000000000, optimized=True)  # 1 billion neurons
    
    # Compare multiple configurations
    print("\n📊 Configuration Comparison")
    configs = [
        {
            'neurons': 100000000,    # 100 million neurons
            'active_percentage': 0.01,
            'areas': 5,
            'use_optimized_cuda': True,
            'use_gpu': True
        },
        {
            'neurons': 100000000,    # 100 million neurons
            'active_percentage': 0.01,
            'areas': 5,
            'use_optimized_cuda': False,
            'use_gpu': True
        },
        {
            'neurons': 10000000,     # 10 million neurons (CPU can't handle 100M)
            'active_percentage': 0.01,
            'areas': 5,
            'use_optimized_cuda': False,
            'use_gpu': False
        }
    ]
    
    comparison_results = compare_configurations(configs, steps=20)
    
    # Find best configuration
    best_result = max(comparison_results, 
                     key=lambda x: x['result']['summary']['steps_per_second'])
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Steps/sec: {best_result['result']['summary']['steps_per_second']:.1f}")
    print(f"   Neurons/sec: {best_result['result']['summary']['neurons_per_second']:,.0f}")
    print(f"   Config: {best_result['config']}")


if __name__ == "__main__":
    main()
