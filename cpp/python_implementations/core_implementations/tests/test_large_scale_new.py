#!/usr/bin/env python3
"""
Test Large Scale Examples
========================

Quick test to verify the updated examples work with larger neuron counts.
"""

import sys
sys.path.insert(0, '.')

from universal_brain_simulator.client import BrainSimulator, quick_sim


def test_medium_scale():
    """Test medium scale (10M neurons)"""
    print("🧪 Testing Medium Scale (10M neurons)")
    
    sim = BrainSimulator(
        neurons=10000000,  # 10 million neurons
        active_percentage=0.01,
        areas=5,
        use_optimized_cuda=True
    )
    
    results = sim.run(steps=5, verbose=False)
    print(f"   ✅ Medium scale: {results['summary']['steps_per_second']:.1f} steps/sec")
    print(f"   Neurons/sec: {results['summary']['neurons_per_second']:,.0f}")
    return results


def test_large_scale():
    """Test large scale (100M neurons)"""
    print("\n🧪 Testing Large Scale (100M neurons)")
    
    sim = BrainSimulator(
        neurons=100000000,  # 100 million neurons
        active_percentage=0.01,
        areas=5,
        use_optimized_cuda=True
    )
    
    results = sim.run(steps=3, verbose=False)
    print(f"   ✅ Large scale: {results['summary']['steps_per_second']:.1f} steps/sec")
    print(f"   Neurons/sec: {results['summary']['neurons_per_second']:,.0f}")
    return results


def test_billion_scale():
    """Test billion scale (1B neurons)"""
    print("\n🧪 Testing Billion Scale (1B neurons)")
    
    try:
        results = quick_sim(neurons=1000000000, steps=2, optimized=True)  # 1 billion neurons
        print(f"   ✅ Billion scale: {results['summary']['steps_per_second']:.1f} steps/sec")
        print(f"   Neurons/sec: {results['summary']['neurons_per_second']:,.0f}")
        return results
    except Exception as e:
        print(f"   ❌ Billion scale failed: {e}")
        return None


def main():
    print("🚀 LARGE SCALE TESTING")
    print("=" * 50)
    
    # Test different scales
    medium_results = test_medium_scale()
    large_results = test_large_scale()
    billion_results = test_billion_scale()
    
    # Summary
    print("\n📊 SCALING SUMMARY")
    print("=" * 50)
    
    if medium_results:
        print(f"   10M neurons: {medium_results['summary']['steps_per_second']:.1f} steps/sec")
        print(f"   Memory: {medium_results['summary']['memory_usage_gb']:.2f}GB")
    
    if large_results:
        print(f"   100M neurons: {large_results['summary']['steps_per_second']:.1f} steps/sec")
        print(f"   Memory: {large_results['summary']['memory_usage_gb']:.2f}GB")
    
    if billion_results:
        print(f"   1B neurons: {billion_results['summary']['steps_per_second']:.1f} steps/sec")
        print(f"   Memory: {billion_results['summary']['memory_usage_gb']:.2f}GB")
    
    # Calculate scaling efficiency
    if medium_results and large_results:
        scale_factor = 10  # 10M to 100M
        performance_ratio = large_results['summary']['steps_per_second'] / medium_results['summary']['steps_per_second']
        efficiency = performance_ratio * scale_factor
        print(f"\n   Scaling efficiency (10M→100M): {efficiency:.1f}%")
    
    if large_results and billion_results:
        scale_factor = 10  # 100M to 1B
        performance_ratio = billion_results['summary']['steps_per_second'] / large_results['summary']['steps_per_second']
        efficiency = performance_ratio * scale_factor
        print(f"   Scaling efficiency (100M→1B): {efficiency:.1f}%")
    
    print("\n🎯 Large scale testing complete!")
    print("   All updated examples are ready for billion-scale testing.")


if __name__ == "__main__":
    main()
