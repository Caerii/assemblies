#!/usr/bin/env python3
"""
Test Extreme Limits
==================

Quick test to find the absolute limits of the system.
"""

import sys
sys.path.insert(0, '.')

from universal_brain_simulator.client import BrainSimulator


def test_extreme_limits():
    """Test extreme neuron counts to find limits"""
    print("🚀 TESTING EXTREME LIMITS")
    print("=" * 50)
    
    # Test progressively larger scales
    test_cases = [
        (2000000000, 0.001, 1),   # 2B neurons, 0.1% active, 1 area
        (3000000000, 0.001, 1),   # 3B neurons, 0.1% active, 1 area
        (5000000000, 0.0005, 1),  # 5B neurons, 0.05% active, 1 area
        (10000000000, 0.0001, 1), # 10B neurons, 0.01% active, 1 area
    ]
    
    for neurons, active_pct, areas in test_cases:
        print(f"\n🧪 Testing {neurons:,} neurons ({neurons/1000000000:.1f}B)...")
        print(f"   Active: {active_pct*100:.2f}% ({int(neurons*active_pct):,} active)")
        print(f"   Areas: {areas}")
        
        try:
            sim = BrainSimulator(
                neurons=neurons,
                active_percentage=active_pct,
                areas=areas,
                use_optimized_cuda=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            results = sim.run(steps=1, verbose=False)
            
            print("   ✅ SUCCESS!")
            print(f"   Steps/sec: {results['summary']['steps_per_sec']:.1f}")
            print(f"   Memory: {results['summary']['memory_usage_gb']:.2f}GB")
            print(f"   Neurons/sec: {results['summary']['neurons_per_second']:,.0f}")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                print(f"   🛑 Memory limit reached at {neurons:,} neurons")
                break
            else:
                print("   ⚠️  Other error, continuing...")


if __name__ == "__main__":
    test_extreme_limits()
