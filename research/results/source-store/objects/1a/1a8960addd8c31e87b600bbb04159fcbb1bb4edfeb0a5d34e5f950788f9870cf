#!/usr/bin/env python3
"""
Quick Start Example - Universal Brain Simulator
==============================================

This example shows the simplest way to get started with the
Universal Brain Simulator using the lightweight client.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from universal_brain_simulator.client import BrainSimulator, quick_sim


def main():
    print("🚀 QUICK START EXAMPLE")
    print("=" * 50)
    
    # Method 1: Using the BrainSimulator class - Large Scale
    print("\n📋 Method 1: Using BrainSimulator class (Large Scale)")
    sim = BrainSimulator(
        neurons=10000000,     # 10 million neurons
        active_percentage=0.01,  # 1% active
        areas=5,              # 5 brain areas
        use_optimized_cuda=True  # Use optimized CUDA kernels
    )
    
    # Run simulation
    results = sim.run(steps=100, verbose=True)
    
    # Method 2: Using the quick_sim convenience function - Billion Scale
    print("\n📋 Method 2: Using quick_sim convenience function (Billion Scale)")
    results2 = quick_sim(neurons=1000000000, steps=50, optimized=True)  # 1 billion neurons
    
    print("\n🎯 Quick start complete!")
    print(f"   Method 1: {results['summary']['steps_per_second']:.1f} steps/sec")
    print(f"   Method 2: {results2['summary']['steps_per_second']:.1f} steps/sec")


if __name__ == "__main__":
    main()
