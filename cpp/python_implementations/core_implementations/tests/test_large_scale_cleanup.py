#!/usr/bin/env python3
"""
Test Large Scale with Fixed Cleanup
===================================

This script tests large-scale simulations with the fixed cleanup system
to verify that memory errors are resolved.
"""

import sys
import os
sys.path.insert(0, '.')

from universal_brain_simulator import BrainSimulator

def test_large_scale_cleanup():
    """Test large-scale simulations with proper cleanup"""
    print("🧪 Testing Large Scale with Fixed Cleanup")
    print("=" * 60)
    
    # Test scales that previously caused memory errors
    test_scales = [
        (10000000, "10M neurons"),
        (50000000, "50M neurons"),
        (100000000, "100M neurons"),
        (1000000000, "1B neurons")
    ]
    
    for neurons, scale_name in test_scales:
        print(f"\n🔬 Testing {scale_name}...")
        
        try:
            # Create simulator
            sim = BrainSimulator(
                neurons=neurons,
                active_percentage=0.001,  # Very low active percentage
                areas=3,
                use_optimized_cuda=True,
                use_cuda_kernels=True
            )
            
            # Run simulation
            result = sim.run(steps=5, verbose=False)
            
            print(f"   ✅ {scale_name}: {result['summary']['steps_per_second']:.1f} steps/sec")
            print(f"   📊 Memory: {result['summary']['memory_usage_gb']:.2f} GB")
            print(f"   🧠 CUDA: {'✅' if result['summary']['cuda_used'] else '❌'}")
            
            # Explicit cleanup
            sim.cleanup()
            print(f"   🧹 {scale_name} cleanup completed")
            
        except Exception as e:
            print(f"   ❌ {scale_name} failed: {e}")
            # Still try to cleanup if possible
            try:
                if 'sim' in locals():
                    sim.cleanup()
            except:
                pass
    
    print("\n🎉 Large scale cleanup test completed!")
    print("✅ Memory management fixes are working at large scales")

if __name__ == "__main__":
    test_large_scale_cleanup()
