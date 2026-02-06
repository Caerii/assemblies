"""
High-performance C++ Association simulation module.

This module contains simulation functions for studying neural associations
using the high-performance C++ implementation.
"""

try:
    from src.core.brain_cpp import BrainCPP
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ brain not available, falling back to Python implementation")
    from src.core.brain import Brain as BrainCPP

import time


def associate_cpp(n=100000, k=317, p=0.05, beta=0.1, overlap_iter=10):
    """
    High-performance C++ version of association simulation.
    
    Simulates the association of two neural stimuli into a third neural area 
    through sequential and concurrent projections using C++ implementation.

    Parameters:
    n (int): Total number of neurons in each neural area.
    k (int): Number of neurons initially activated by each stimulus.
    p (float): Probability parameter for the brain setup.
    beta (float): Connectivity probability within the neural areas.
    overlap_iter (int): Number of iterations for the overlap phase.

    Returns:
    tuple: (brain, winners) where brain is the BrainCPP instance and winners is the list of saved winners.
    """
    if not CPP_AVAILABLE:
        raise ImportError("C++ brain not available. Install with: pip install -e cpp/")
    
    print(f"Running C++ association simulation: n={n}, k={k}, p={p}, beta={beta}")
    start_time = time.time()
    
    # Create C++ brain
    b = BrainCPP(p, beta, max_weight=10000.0, seed=7777, save_winners=True)
    
    # Add areas
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta) 
    b.add_area("C", n, k, beta)
    
    # Add stimuli
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    
    # Add connections
    b.add_fiber("stimA", "A")
    b.add_fiber("stimB", "B")
    b.add_fiber("A", "C")
    b.add_fiber("B", "C")
    b.add_fiber("C", "C")  # Self-recurrence
    
    # Stabilize A and B
    print("Stabilizing A/B...")
    for i in range(9):
        b.project({"stimA": ["A"], "stimB": ["B"]}, {})
        if i % 3 == 0:
            print(f"  Stabilization step {i+1}/9")
    
    # A->C phase
    print("A->C phase...")
    for i in range(10):
        b.project({"stimA": ["A"]}, {"A": ["A", "C"]})
        if i % 3 == 0:
            print(f"  A->C step {i+1}/10")
    
    # B->C phase  
    print("B->C phase...")
    for i in range(10):
        b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
        if i % 3 == 0:
            print(f"  B->C step {i+1}/10")
    
    # A,B->C overlap phase
    print("A,B->C overlap phase...")
    for i in range(overlap_iter):
        b.project({"stimA": ["A"], "stimB": ["B"]}, {"A": ["A"], "B": ["B"], "C": ["C"]})
        if i % 3 == 0:
            print(f"  Overlap step {i+1}/{overlap_iter}")
    
    # Final B-only phase
    print("Final B-only phase...")
    for i in range(10):
        b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
        if i % 3 == 0:
            print(f"  Final B step {i+1}/10")
    
    # Get saved winners
    winners = b.get_saved_winners("C")
    
    elapsed_time = time.time() - start_time
    print(f"C++ Association simulation completed in {elapsed_time:.2f} seconds")
    print(f"Total saved winners: {len(winners)}")
    
    return b, winners


def association_sim_cpp(n=100000, k=317, p=0.05, beta=0.1, overlap_iter=10):
    """
    High-performance C++ association simulation wrapper.
    
    Parameters:
    n (int): Total number of neurons in each neural area.
    k (int): Number of neurons initially activated by each stimulus.
    p (float): Probability parameter for the brain setup.
    beta (float): Connectivity probability within the neural areas.
    overlap_iter (int): Number of iterations for the overlap phase.

    Returns:
    tuple: (brain, winners) where brain is the BrainCPP instance and winners is the list of saved winners.
    """
    return associate_cpp(n, k, p, beta, overlap_iter)


def benchmark_comparison(n=100000, k=317, p=0.05, beta=0.1, overlap_iter=3):
    """
    Benchmark C++ vs Python implementation.
    
    Parameters:
    n (int): Total number of neurons in each neural area.
    k (int): Number of neurons initially activated by each stimulus.
    p (float): Probability parameter for the brain setup.
    beta (float): Connectivity probability within the neural areas.
    overlap_iter (int): Number of iterations for the overlap phase.

    Returns:
    dict: Benchmark results with timing and performance metrics.
    """
    results = {}
    
    # Test C++ implementation
    if CPP_AVAILABLE:
        print("Benchmarking C++ implementation...")
        start_time = time.time()
        try:
            _, cpp_winners = association_sim_cpp(n, k, p, beta, overlap_iter)
            cpp_time = time.time() - start_time
            results['cpp_time'] = cpp_time
            results['cpp_winners'] = len(cpp_winners)
            results['cpp_success'] = True
            print(f"C++ implementation: {cpp_time:.2f} seconds, {len(cpp_winners)} winners")
        except Exception as e:
            results['cpp_success'] = False
            results['cpp_error'] = str(e)
            print(f"C++ implementation failed: {e}")
    else:
        results['cpp_available'] = False
        print("C++ implementation not available")
    
    # Test Python implementation
    print("Benchmarking Python implementation...")
    try:
        from src.simulation.association_simulator import associate
        
        start_time = time.time()
        _, py_winners = associate(n, k, p, beta, overlap_iter)
        py_time = time.time() - start_time
        results['python_time'] = py_time
        results['python_winners'] = len(py_winners)
        results['python_success'] = True
        print(f"Python implementation: {py_time:.2f} seconds, {len(py_winners)} winners")
        
        # Calculate speedup
        if results.get('cpp_success') and results.get('python_success'):
            speedup = py_time / cpp_time
            results['speedup'] = speedup
            print(f"Speedup: {speedup:.2f}x faster with C++")
            
    except Exception as e:
        results['python_success'] = False
        results['python_error'] = str(e)
        print(f"Python implementation failed: {e}")
    
    return results


if __name__ == "__main__":
    # Run a quick test
    print("Testing C++ association simulation...")
    try:
        b, winners = association_sim_cpp(1000, 50, 0.05, 0.1, 3)
        print(f"Success! Generated {len(winners)} winner sets")
        
        # Run benchmark
        print("\nRunning benchmark comparison...")
        results = benchmark_comparison(10000, 100, 0.05, 0.1, 3)
        print(f"Benchmark results: {results}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
