#!/usr/bin/env python3
"""
Performance test comparing C++ vs CUDA implementations
"""

import sys
import time
import numpy as np

# Add paths
sys.path.append('.')
sys.path.append('src/simulation')
sys.path.append('src/core')

def test_cuda_availability():
    """Test if CUDA extension is available"""
    try:
        import cuda_brain
        print("✓ CUDA Brain extension loaded successfully!")
        return True
    except ImportError as e:
        print(f"✗ CUDA Brain extension not available: {e}")
        print("Please run: python cpp/build_cuda.py")
        return False

def test_cpp_availability():
    """Test if C++ extension is available"""
    try:
        from src.core.brain_cpp import BrainCPP
        print("✓ C++ Brain extension loaded successfully!")
        return True
    except ImportError as e:
        print(f"✗ C++ Brain extension not available: {e}")
        return False

def run_cpp_benchmark(n, k, p, beta, overlap_iter):
    """Run C++ benchmark"""
    from src.simulation.association_simulator_cpp import association_sim_cpp
    
    print(f"Running C++ benchmark: n={n}, k={k}")
    start_time = time.time()
    
    try:
        brain, winners = association_sim_cpp(n, k, p, beta, overlap_iter, verbose=0)
        cpp_time = time.time() - start_time
        
        print(f"✓ C++ completed in {cpp_time:.4f} seconds")
        print(f"✓ Generated {len(winners)} winner sets")
        
        return cpp_time, len(winners), True
    except Exception as e:
        cpp_time = time.time() - start_time
        print(f"✗ C++ failed: {e}")
        return cpp_time, 0, False

def run_cuda_benchmark(n, k, p, beta, overlap_iter):
    """Run CUDA benchmark"""
    try:
        from cuda_brain_wrapper import BrainCUDA
        
        print(f"Running CUDA benchmark: n={n}, k={k}")
        start_time = time.time()
        
        # Create CUDA brain
        brain = BrainCUDA(p=p, beta=beta, max_weight=10000.0, seed=7777)
        
        # Add areas
        brain.add_area("A", n, k, beta=beta)
        brain.add_area("B", n, k, beta=beta)
        brain.add_area("C", n, k, beta=beta)
        
        # Add stimuli
        brain.add_stimulus("stimA", k)
        brain.add_stimulus("stimB", k)
        
        # Add fibers
        brain.add_fiber("stimA", "A")
        brain.add_fiber("stimB", "B")
        brain.add_fiber("A", "A")  # Recurrence
        brain.add_fiber("B", "B")  # Recurrence
        brain.add_fiber("C", "C")  # Recurrence
        brain.add_fiber("A", "C")  # A to C
        brain.add_fiber("B", "C")  # B to C
        
        # Run simulation phases
        print("  Phase 1: Stabilizing A/B...")
        for i in range(9):
            brain.project({"stimA": ["A"], "stimB": ["B"]}, {"A": ["A"], "B": ["B"]})
        
        print("  Phase 2: A->C...")
        for i in range(10):
            brain.project({"stimA": ["A"]}, {"A": ["A", "C"]})
        
        print("  Phase 3: B->C...")
        for i in range(10):
            brain.project({"stimB": ["B"]}, {"B": ["B", "C"]})
        
        print("  Phase 4: A,B->C overlap...")
        for i in range(overlap_iter):
            brain.project({"stimA": ["A"], "stimB": ["B"]}, {"A": ["A"], "B": ["B"], "C": ["C"]})
        
        print("  Phase 5: Final B-only...")
        for i in range(10):
            brain.project({"stimB": ["B"]}, {"B": ["B", "C"]})
        
        cuda_time = time.time() - start_time
        
        print(f"✓ CUDA completed in {cuda_time:.4f} seconds")
        print("✓ Generated simulation data")
        
        return cuda_time, 1, True
        
    except Exception as e:
        cuda_time = time.time() - start_time
        print(f"✗ CUDA failed: {e}")
        return cuda_time, 0, False

def run_performance_comparison():
    """Run comprehensive performance comparison"""
    print("🚀 CUDA vs C++ Performance Comparison")
    print("=" * 60)
    
    # Test availability
    cuda_available = test_cuda_availability()
    cpp_available = test_cpp_availability()
    
    if not cpp_available:
        print("\n❌ Cannot run comparison without C++ implementation")
        return
    
    # Test parameters
    test_cases = [
        {"n": 1000, "k": 50, "name": "Small"},
        {"n": 5000, "k": 100, "name": "Medium"},
        {"n": 10000, "k": 200, "name": "Large"},
        {"n": 50000, "k": 500, "name": "Very Large"},
    ]
    
    p = 0.05
    beta = 0.1
    overlap_iter = 2
    
    results = []
    
    for test_case in test_cases:
        n = test_case["n"]
        k = test_case["k"]
        name = test_case["name"]
        
        print(f"\n📊 {name} Test (n={n:,}, k={k})")
        print("-" * 40)
        
        # Run C++ benchmark
        cpp_time, cpp_winners, cpp_success = run_cpp_benchmark(n, k, p, beta, overlap_iter)
        
        # Run CUDA benchmark if available
        if cuda_available:
            cuda_time, cuda_winners, cuda_success = run_cuda_benchmark(n, k, p, beta, overlap_iter)
        else:
            cuda_time, cuda_winners, cuda_success = 0, 0, False
        
        # Store results
        results.append({
            "name": name,
            "n": n,
            "k": k,
            "cpp_time": cpp_time,
            "cpp_success": cpp_success,
            "cuda_time": cuda_time,
            "cuda_success": cuda_success,
            "speedup": cpp_time / cuda_time if cuda_success and cuda_time > 0 else 0
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Test':<12} {'C++ Time':<12} {'CUDA Time':<12} {'Speedup':<10} {'Status'}")
    print("-" * 60)
    
    for result in results:
        status = "✓ Both" if result["cpp_success"] and result["cuda_success"] else \
                "C++ Only" if result["cpp_success"] else \
                "CUDA Only" if result["cuda_success"] else "Failed"
        
        speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] > 0 else "N/A"
        
        print(f"{result['name']:<12} {result['cpp_time']:<12.4f} {result['cuda_time']:<12.4f} "
              f"{speedup_str:<10} {status}")
    
    # Calculate overall statistics
    successful_tests = [r for r in results if r["cpp_success"] and r["cuda_success"]]
    if successful_tests:
        avg_speedup = np.mean([r["speedup"] for r in successful_tests])
        max_speedup = max([r["speedup"] for r in successful_tests])
        min_speedup = min([r["speedup"] for r in successful_tests])
        
        print("\n📈 Overall Statistics:")
        print(f"  Average Speedup: {avg_speedup:.2f}x")
        print(f"  Maximum Speedup: {max_speedup:.2f}x")
        print(f"  Minimum Speedup: {min_speedup:.2f}x")
        print(f"  Successful Tests: {len(successful_tests)}/{len(results)}")
    
    print("\n🎯 Conclusion:")
    if cuda_available and successful_tests:
        print(f"  CUDA provides {avg_speedup:.1f}x average speedup over C++!")
        print(f"  This enables simulation of {max([r['n'] for r in successful_tests]):,} neurons efficiently!")
    elif cuda_available:
        print("  CUDA is available but had issues. Check CUDA installation.")
    else:
        print("  CUDA not available. C++ provides good performance for current tests.")
        print("  Install CUDA for even better performance!")

if __name__ == "__main__":
    run_performance_comparison()
