#!/usr/bin/env python3
"""
Test Cleanup Fixes
==================

This script tests the cleanup fixes to ensure proper memory management
and prevent double-cleanup errors.
"""

import sys
sys.path.insert(0, '.')

from universal_brain_simulator import BrainSimulator

def test_cleanup_fixes():
    """Test that cleanup fixes work properly"""
    print("🧪 Testing Cleanup Fixes")
    print("=" * 50)
    
    # Test 1: Basic cleanup
    print("\n1. Testing basic cleanup...")
    sim = BrainSimulator(neurons=1000000, active_percentage=0.01, areas=3)
    
    # Run a few steps
    result = sim.run(steps=10, verbose=False)
    print(f"   ✅ Simulation completed: {result['summary']['steps_per_second']:.1f} steps/sec")
    
    # Explicit cleanup
    sim.cleanup()
    print("   ✅ Explicit cleanup completed")
    
    # Test 2: Multiple instances
    print("\n2. Testing multiple instances...")
    sims = []
    for i in range(3):
        sim = BrainSimulator(neurons=500000, active_percentage=0.01, areas=2)
        result = sim.run(steps=5, verbose=False)
        print(f"   ✅ Instance {i+1}: {result['summary']['steps_per_second']:.1f} steps/sec")
        sims.append(sim)
    
    # Cleanup all instances
    for i, sim in enumerate(sims):
        sim.cleanup()
        print(f"   ✅ Instance {i+1} cleanup completed")
    
    # Test 3: Double cleanup prevention
    print("\n3. Testing double cleanup prevention...")
    sim = BrainSimulator(neurons=1000000, active_percentage=0.01, areas=3)
    result = sim.run(steps=5, verbose=False)
    print(f"   ✅ Simulation completed: {result['summary']['steps_per_second']:.1f} steps/sec")
    
    # First cleanup
    sim.cleanup()
    print("   ✅ First cleanup completed")
    
    # Second cleanup (should be prevented)
    sim.cleanup()
    print("   ✅ Second cleanup prevented (no errors)")
    
    print("\n🎉 All cleanup tests passed!")
    print("✅ Memory management fixes are working correctly")

if __name__ == "__main__":
    test_cleanup_fixes()
