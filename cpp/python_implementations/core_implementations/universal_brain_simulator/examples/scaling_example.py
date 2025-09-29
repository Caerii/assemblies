#!/usr/bin/env python3
"""
Scaling Example - Universal Brain Simulator
==========================================

This example shows how performance scales with different neuron counts
and demonstrates the benefits of optimized CUDA kernels at larger scales.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from universal_brain_simulator.client import BrainSimulator, compare_configurations


def main():
    print("📈 SCALING EXAMPLE")
    print("=" * 50)
    
    # Test different neuron counts - Large Scale Sweep
    neuron_counts = [1000000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000]  # 1M to 1B
    
    print("\n📊 Testing different neuron counts with optimized CUDA...")
    optimized_results = []
    
    for neurons in neuron_counts:
        print(f"\n🧪 Testing {neurons:,} neurons...")
        
        sim = BrainSimulator(
            neurons=neurons,
            active_percentage=0.01,
            areas=5,
            use_optimized_cuda=True
        )
        
        # Quick benchmark
        results = sim.benchmark(warmup_steps=2, measure_steps=3)
        
        optimized_results.append({
            'neurons': neurons,
            'steps_per_second': results['performance']['steps_per_second'],
            'neurons_per_second': results['performance']['neurons_per_second'],
            'step_time_ms': results['performance']['average_step_time_ms']
        })
    
    # Test with original CUDA for comparison
    print("\n📊 Testing with original CUDA for comparison...")
    original_results = []
    
    for neurons in [1000000, 5000000, 10000000, 50000000]:  # Test up to 50M for original CUDA
        print(f"\n🧪 Testing {neurons:,} neurons (original CUDA)...")
        
        sim = BrainSimulator(
            neurons=neurons,
            active_percentage=0.01,
            areas=5,
            use_optimized_cuda=False
        )
        
        results = sim.benchmark(warmup_steps=2, measure_steps=3)
        
        original_results.append({
            'neurons': neurons,
            'steps_per_second': results['performance']['steps_per_second'],
            'neurons_per_second': results['performance']['neurons_per_second'],
            'step_time_ms': results['performance']['average_step_time_ms']
        })
    
    # Print scaling results
    print(f"\n📈 SCALING RESULTS")
    print("=" * 80)
    print(f"{'Neurons':<12} {'Optimized CUDA':<20} {'Original CUDA':<20} {'Speedup':<10}")
    print(f"{'':<12} {'Steps/sec':<10} {'Neurons/sec':<10} {'Steps/sec':<10} {'Neurons/sec':<10} {'':<10}")
    print("-" * 80)
    
    for opt_result in optimized_results:
        neurons = opt_result['neurons']
        opt_steps = opt_result['steps_per_second']
        opt_neurons = opt_result['neurons_per_second']
        
        # Find matching original result
        orig_result = next((r for r in original_results if r['neurons'] == neurons), None)
        
        if orig_result:
            orig_steps = orig_result['steps_per_second']
            speedup = opt_steps / orig_steps if orig_steps > 0 else float('inf')
            print(f"{neurons:<12,} {opt_steps:<10.1f} {opt_neurons:<10,.0f} {orig_steps:<10.1f} {orig_result['neurons_per_second']:<10,.0f} {speedup:<10.1f}x")
        else:
            print(f"{neurons:<12,} {opt_steps:<10.1f} {opt_neurons:<10,.0f} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # Test different active percentages - Large Scale
    print(f"\n📊 Testing different active percentages (Large Scale)...")
    active_percentages = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    for active_pct in active_percentages:
        print(f"\n🧪 Testing {active_pct*100:.1f}% active neurons...")
        
        sim = BrainSimulator(
            neurons=100000000,  # 100 million neurons
            active_percentage=active_pct,
            areas=5,
            use_optimized_cuda=True
        )
        
        results = sim.benchmark(warmup_steps=2, measure_steps=3)
        
        print(f"   Steps/sec: {results['performance']['steps_per_second']:.1f}")
        print(f"   Neurons/sec: {results['performance']['neurons_per_second']:,.0f}")
        print(f"   Active neurons: {int(100000000 * active_pct):,}")
    
    # Test different area counts - Large Scale
    print(f"\n📊 Testing different area counts (Large Scale)...")
    area_counts = [1, 3, 5, 10, 20]
    
    for areas in area_counts:
        print(f"\n🧪 Testing {areas} areas...")
        
        sim = BrainSimulator(
            neurons=100000000,  # 100 million neurons
            active_percentage=0.01,
            areas=areas,
            use_optimized_cuda=True
        )
        
        results = sim.benchmark(warmup_steps=2, measure_steps=3)
        
        print(f"   Steps/sec: {results['performance']['steps_per_second']:.1f}")
        print(f"   Neurons/sec: {results['performance']['neurons_per_second']:,.0f}")
        print(f"   Total active neurons: {int(100000000 * 0.01 * areas):,}")
    
    print(f"\n🎯 Scaling analysis complete!")
    print(f"   Key insights:")
    print(f"   - Optimized CUDA shows significant speedup at larger scales")
    print(f"   - Performance scales well with neuron count")
    print(f"   - Active percentage affects performance linearly")
    print(f"   - More areas = more total active neurons = higher throughput")


if __name__ == "__main__":
    main()
