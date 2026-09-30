#!/usr/bin/env python3
"""
Test ms per step metrics for Ultra Optimized CUDA Brain v2
"""

from ultra_optimized_cuda_brain_v2 import UltraOptimizedCUDABrainV2

def test_ms_per_step():
    """Test and display ms per step metrics"""
    print("⏱️  MS PER STEP METRICS TEST")
    print("=" * 50)
    
    # Test different scales
    test_cases = [
        {"n_neurons": 50000, "k_active": 500, "n_areas": 3, "name": "Tiny Scale"},
        {"n_neurons": 100000, "k_active": 1000, "n_areas": 3, "name": "Small Scale"},
        {"n_neurons": 500000, "k_active": 5000, "n_areas": 5, "name": "Medium Scale"},
        {"n_neurons": 1000000, "k_active": 10000, "n_areas": 5, "name": "Large Scale"},
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}:")
        print(f"   Neurons: {test_case['n_neurons']:,}")
        print(f"   Active: {test_case['k_active']:,}")
        print(f"   Areas: {test_case['n_areas']}")
        
        # Create brain
        brain = UltraOptimizedCUDABrainV2(
            n_neurons=test_case['n_neurons'],
            k_active=test_case['k_active'],
            n_areas=test_case['n_areas'],
            seed=42
        )
        
        # Simulate with verbose output to see ms per step
        print("\n   Running simulation...")
        total_time = brain.simulate(n_steps=20, verbose=True)
        
        # Get detailed stats
        stats = brain.get_performance_stats()
        
        print("\n   📊 DETAILED METRICS:")
        print(f"      Total time: {total_time:.3f}s")
        print(f"      Steps: {stats['total_steps']}")
        print(f"      Average step time: {stats['avg_step_time']*1000:.2f}ms")
        print(f"      Steps per second: {stats['steps_per_second']:.1f}")
        print(f"      Neurons per second: {stats['neurons_per_second']:,.0f}")
        print(f"      Active neurons per second: {stats['active_neurons_per_second']:,.0f}")
        
        # Calculate ms per step for different operations
        ms_per_step = stats['avg_step_time'] * 1000
        ms_per_neuron = ms_per_step / test_case['n_neurons'] * 1000  # microseconds per neuron
        ms_per_active = ms_per_step / test_case['k_active'] * 1000   # microseconds per active neuron
        
        print("\n   ⏱️  MS PER STEP BREAKDOWN:")
        print(f"      Total ms per step: {ms_per_step:.2f}ms")
        print(f"      μs per neuron: {ms_per_neuron:.3f}μs")
        print(f"      μs per active neuron: {ms_per_active:.3f}μs")
        
        # Area breakdown
        print("\n   📈 AREA TIMING BREAKDOWN:")
        for i, area_time in enumerate(stats['area_times']):
            area_ms = area_time * 1000
            print(f"      Area {i}: {area_ms:.2f}ms")
        
        print("   " + "="*50)

def test_single_step_timing():
    """Test individual step timing"""
    print("\n🔬 SINGLE STEP TIMING TEST")
    print("=" * 50)
    
    # Create a medium-scale brain
    brain = UltraOptimizedCUDABrainV2(
        n_neurons=100000,
        k_active=1000,
        n_areas=3,
        seed=42
    )
    
    print("Running 10 individual steps to measure timing...")
    
    step_times = []
    for i in range(10):
        step_time = brain.simulate_step()
        step_times.append(step_time)
        print(f"Step {i+1:2d}: {step_time*1000:.2f}ms")
    
    # Calculate statistics
    avg_time = sum(step_times) / len(step_times)
    min_time = min(step_times)
    max_time = max(step_times)
    
    print("\n📊 SINGLE STEP STATISTICS:")
    print(f"   Average: {avg_time*1000:.2f}ms")
    print(f"   Minimum: {min_time*1000:.2f}ms")
    print(f"   Maximum: {max_time*1000:.2f}ms")
    print(f"   Range: {(max_time-min_time)*1000:.2f}ms")
    print(f"   Std Dev: {(sum((t-avg_time)**2 for t in step_times)/len(step_times))**0.5*1000:.2f}ms")

if __name__ == "__main__":
    # Test ms per step metrics
    test_ms_per_step()
    
    # Test single step timing
    test_single_step_timing()
    
    print("\n🎯 SUMMARY")
    print("=" * 50)
    print("✅ ms per step metrics are available in the simulation output")
    print("✅ Individual step times are shown during verbose simulation")
    print("✅ Average step times are calculated and displayed")
    print("✅ Area-level timing breakdown is provided")
