#!/usr/bin/env python3
"""
Billion-Scale Demo
=================

This script demonstrates the critical difference between O(N²) and O(N log K) algorithms
by showing the theoretical complexity and running a simple test.

The key insight: O(N²) algorithms require quadratically more operations as scale increases,
making them impractical for billion-scale simulation.
"""

import time
import sys
import os
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import the original implementation
try:
    from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig
    print("✅ Original implementation imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def calculate_complexity(n_neurons: int, active_percentage: float, n_areas: int = 5):
    """Calculate theoretical complexity for O(N²) algorithm"""
    k_active = int(n_neurons * active_percentage)
    
    # O(N²) complexity for top-k selection
    # Each area does k iterations × n² comparisons
    total_ops = n_areas * k_active * n_neurons * n_neurons
    
    return {
        'n_neurons': n_neurons,
        'k_active': k_active,
        'n_areas': n_areas,
        'total_ops': total_ops,
        'ops_per_second': 1e9,  # Assume 1 billion operations per second
        'estimated_time_seconds': total_ops / 1e9,
        'estimated_time_hours': total_ops / 1e9 / 3600,
        'estimated_time_days': total_ops / 1e9 / 3600 / 24
    }

def test_small_scale():
    """Test at a small scale to verify the algorithm works"""
    print("\n🧪 Testing Small Scale (100K neurons)")
    print("-" * 50)
    
    config = SimulationConfig(
        n_neurons=100_000,
        active_percentage=0.01,  # 1% active
        n_areas=5,
        use_gpu=True,
        use_cuda_kernels=True,
        memory_efficient=True,
        sparse_mode=True
    )
    
    try:
        start_time = time.time()
        simulator = UniversalBrainSimulator(config)
        init_time = time.time() - start_time
        
        print(f"   Initialization: {init_time:.2f}s")
        
        sim_start = time.time()
        total_time = simulator.simulate(n_steps=3, verbose=False)
        sim_time = time.time() - sim_start
        
        stats = simulator.get_performance_stats()
        
        print(f"   ✅ Simulation completed: {sim_time:.2f}s")
        print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
        print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
        print(f"   Memory: {stats['memory_usage_gb']:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def main():
    """Run the billion-scale complexity demonstration"""
    print("🧠 Billion-Scale Algorithmic Complexity Demo")
    print("=" * 60)
    print("This demo shows why O(N²) algorithms fail at large scales")
    print("and why O(N log K) optimization is critical for billion-scale simulation.")
    
    # Test scales
    scales = [
        {"n_neurons": 100_000, "active_percentage": 0.01, "name": "Small (100K)"},
        {"n_neurons": 1_000_000, "active_percentage": 0.01, "name": "Medium (1M)"},
        {"n_neurons": 10_000_000, "active_percentage": 0.001, "name": "Large (10M)"},
        {"n_neurons": 100_000_000, "active_percentage": 0.0001, "name": "Very Large (100M)"},
        {"n_neurons": 1_000_000_000, "active_percentage": 0.00001, "name": "Billion (1B)"},
    ]
    
    print("\n📊 THEORETICAL COMPLEXITY ANALYSIS")
    print("=" * 60)
    print(f"{'Scale':<20} {'Neurons':<15} {'Active':<10} {'Operations':<20} {'Time (est)':<15}")
    print("-" * 80)
    
    results = []
    
    for scale in scales:
        complexity = calculate_complexity(
            scale['n_neurons'], 
            scale['active_percentage']
        )
        
        # Format time estimate
        if complexity['estimated_time_days'] > 1:
            time_str = f"{complexity['estimated_time_days']:.1f} days"
        elif complexity['estimated_time_hours'] > 1:
            time_str = f"{complexity['estimated_time_hours']:.1f} hours"
        else:
            time_str = f"{complexity['estimated_time_seconds']:.1f} seconds"
        
        print(f"{scale['name']:<20} {complexity['n_neurons']:<15,} {complexity['k_active']:<10,} {complexity['total_ops']:<20,.0e} {time_str:<15}")
        
        results.append(complexity)
    
    print("\n🎯 KEY INSIGHTS:")
    print("  - At 100K neurons: ~50 trillion operations (manageable)")
    print("  - At 1M neurons: ~50 quadrillion operations (slow)")
    print("  - At 10M neurons: ~5 quintillion operations (very slow)")
    print("  - At 100M neurons: ~500 quintillion operations (impractical)")
    print("  - At 1B neurons: ~50 sextillion operations (impossible)")
    
    print("\n🚀 SOLUTION: O(N log K) Optimization")
    print("  - Original O(N²): k × n² operations")
    print("  - Optimized O(N log K): n × log₂(k) operations")
    print("  - Speedup: k × n² / (n × log₂(k)) = k × n / log₂(k)")
    print("  - At 1B neurons, k=10K: 10K × 1B / log₂(10K) ≈ 3.3 × 10¹²")
    print("  - Expected speedup: 3.3 trillion times faster!")
    
    # Test small scale to verify algorithm works
    print("\n🧪 VERIFICATION TEST")
    print("=" * 60)
    
    if test_small_scale():
        print("   ✅ Algorithm works at small scale")
        print("   ⚠️  But will fail at billion scale due to O(N²) complexity")
    else:
        print("   ❌ Algorithm failed even at small scale")
    
    print("\n📈 PERFORMANCE COMPARISON")
    print("=" * 60)
    print("Scale          | O(N²) Time    | O(N log K) Time | Speedup")
    print("-" * 60)
    
    for i, result in enumerate(results):
        scale_name = scales[i]['name']
        n = result['n_neurons']
        k = result['k_active']
        
        # O(N²) time
        o_n2_time = result['estimated_time_seconds']
        
        # O(N log K) time
        o_n_log_k_ops = n * (k.bit_length() - 1) if k > 0 else n
        o_n_log_k_time = o_n_log_k_ops / 1e9
        
        # Speedup
        speedup = o_n2_time / o_n_log_k_time if o_n_log_k_time > 0 else float('inf')
        
        print(f"{scale_name:<15} | {o_n2_time:>10.1f}s | {o_n_log_k_time:>12.1f}s | {speedup:>8.0f}x")
    
    print("\n🎯 CONCLUSION:")
    print("  - O(N²) algorithms become impractical at large scales")
    print("  - O(N log K) algorithms can handle billion-scale efficiently")
    print("  - Speedup increases dramatically with scale")
    print("  - This is why algorithmic optimization is critical!")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"billion_scale_demo_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
