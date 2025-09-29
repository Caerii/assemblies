#!/usr/bin/env python3
"""
Test Ultra Optimized CUDA Brain
"""

import time
import numpy as np
from ultra_optimized_cuda_brain import UltraOptimizedCudaBrain, MegaOptimizedCudaBrain

def test_ultra_optimized():
    """Test ultra optimized brain"""
    print("🧠 TESTING ULTRA OPTIMIZED CUDA BRAIN")
    print("=" * 50)
    
    # Create ultra optimized brain
    brain = UltraOptimizedCudaBrain(p=0.1, beta=0.5, max_weight=1.0, seed=42)
    
    # Add test areas
    brain.AddArea("TestArea", n=1000000, k=10000)
    
    # Test single step
    print("Testing single step...")
    start_time = time.time()
    brain.SimulateOneStep()
    step_time = time.time() - start_time
    
    print(f"Single step time: {step_time*1000:.2f}ms")
    
    # Test multiple steps
    print("Testing 10 steps...")
    start_time = time.time()
    for _ in range(10):
        brain.SimulateOneStep()
    total_time = time.time() - start_time
    
    print(f"10 steps time: {total_time:.3f}s")
    print(f"Avg time per step: {total_time/10*1000:.2f}ms")
    print(f"Steps per second: {10/total_time:.1f}")
    
    return total_time

def test_mega_optimized():
    """Test mega optimized brain"""
    print("\n🚀 TESTING MEGA OPTIMIZED CUDA BRAIN")
    print("=" * 50)
    
    # Create mega optimized brain
    brain = MegaOptimizedCudaBrain(p=0.1, beta=0.5, max_weight=1.0, seed=42)
    
    # Add test areas
    brain.AddArea("SmallArea", n=100000, k=1000)
    brain.AddArea("LargeArea", n=1000000, k=10000)
    
    # Test single step
    print("Testing single step...")
    start_time = time.time()
    brain.SimulateOneStep()
    step_time = time.time() - start_time
    
    print(f"Single step time: {step_time*1000:.2f}ms")
    
    # Test multiple steps
    print("Testing 10 steps...")
    start_time = time.time()
    for _ in range(10):
        brain.SimulateOneStep()
    total_time = time.time() - start_time
    
    print(f"10 steps time: {total_time:.3f}s")
    print(f"Avg time per step: {total_time/10*1000:.2f}ms")
    print(f"Steps per second: {10/total_time:.1f}")
    
    return total_time

def test_numba_functions():
    """Test Numba JIT functions directly"""
    print("\n⚡ TESTING NUMBA JIT FUNCTIONS")
    print("=" * 50)
    
    from ultra_optimized_cuda_brain import fast_exponential_generation, fast_argsort_top_k
    
    # Test exponential generation
    print("Testing exponential generation...")
    start_time = time.time()
    for _ in range(100):
        _ = fast_exponential_generation(1000000)
    gen_time = time.time() - start_time
    print(f"100x 1M exponential generation: {gen_time:.3f}s")
    print(f"Per generation: {gen_time/100*1000:.2f}ms")
    
    # Test top-k selection
    print("Testing top-k selection...")
    start_time = time.time()
    for _ in range(100):
        candidates = np.random.exponential(1.0, 1000000)
        _ = fast_argsort_top_k(candidates, 10000)
    sel_time = time.time() - start_time
    print(f"100x 1M top-k selection: {sel_time:.3f}s")
    print(f"Per selection: {sel_time/100*1000:.2f}ms")
    
    return gen_time, sel_time

if __name__ == "__main__":
    try:
        # Test ultra optimized
        ultra_time = test_ultra_optimized()
        
        # Test mega optimized
        mega_time = test_mega_optimized()
        
        # Test Numba functions
        gen_time, sel_time = test_numba_functions()
        
        # Compare results
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("=" * 50)
        print(f"Ultra Optimized: {ultra_time/10*1000:.2f}ms per step")
        print(f"Mega Optimized:  {mega_time/10*1000:.2f}ms per step")
        print(f"Numba Generation: {gen_time/100*1000:.2f}ms per operation")
        print(f"Numba Selection:  {sel_time/100*1000:.2f}ms per operation")
        
        if mega_time < ultra_time:
            speedup = ultra_time / mega_time
            print(f"\n🏆 MEGA OPTIMIZED is {speedup:.2f}x faster!")
        else:
            print(f"\n✅ Both optimizations working well!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
