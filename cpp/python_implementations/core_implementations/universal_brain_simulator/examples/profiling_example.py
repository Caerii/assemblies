#!/usr/bin/env python3
"""
Profiling Example - Universal Brain Simulator
============================================

This example shows how to run detailed performance profiling
and save the results for analysis.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from universal_brain_simulator.client import BrainSimulator


def main():
    print("📊 PROFILING EXAMPLE")
    print("=" * 50)
    
    # Create simulator
    sim = BrainSimulator(
        neurons=1000000,
        active_percentage=0.01,
        areas=5,
        use_optimized_cuda=True,
        enable_profiling=True
    )
    
    # Run detailed profile
    print("\n📊 Running detailed profile...")
    profile_results = sim.profile(
        steps=100,
        save_to_file="profile_results.json"
    )
    
    # Get detailed information
    print("\n📋 Simulator Information:")
    info = sim.get_info()
    
    print(f"   Configuration: {info['configuration']}")
    print(f"   Memory info: {info['memory_info']}")
    print(f"   Areas info: {len(info['areas_info'])} areas")
    
    # Example with custom callback
    print("\n🔄 Running with custom callback...")
    
    def progress_callback(step, step_time):
        if step % 10 == 0:
            print(f"   Step {step}: {step_time*1000:.2f}ms")
    
    sim.run_with_callback(steps=50, callback=progress_callback, callback_interval=10)
    
    # Validate configuration
    print(f"\n✅ Configuration validation: {'PASSED' if sim.validate() else 'FAILED'}")
    
    # Reset and run again
    print(f"\n🔄 Resetting simulator...")
    sim.reset()
    
    # Quick run after reset
    results = sim.run(steps=10, verbose=False)
    print(f"   Post-reset performance: {results['summary']['steps_per_second']:.1f} steps/sec")


if __name__ == "__main__":
    main()
