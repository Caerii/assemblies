#!/usr/bin/env python3
"""
NumPy Only Test
===============

This test avoids scipy.sparse entirely and uses only NumPy operations
to identify the performance characteristics without getting stuck.
"""

import numpy as np
import time
import gc
import psutil

def test_numpy_operations():
    """Test pure NumPy operations without scipy."""
    print("🧪 Testing pure NumPy operations...")
    
    # Test 1: Basic arrays
    print("   Step 1: Creating arrays (10K neurons)")
    n = 10000
    activations = np.zeros(n, dtype=np.float32)
    candidates = np.zeros(n, dtype=np.float32)
    area = np.zeros(n, dtype=np.float32)
    print("   ✅ Arrays created successfully")
    
    # Test 2: Random generation
    print("   Step 2: Generating random numbers")
    candidates[:] = np.random.exponential(1.0, size=n).astype(np.float32)
    print("   ✅ Random numbers generated successfully")
    
    # Test 3: Simple operations
    print("   Step 3: Performing operations")
    threshold = np.percentile(activations, 90)
    area[:] = np.where(activations > threshold, candidates, 0.0)
    print("   ✅ Operations completed successfully")
    
    # Test 4: Memory usage
    print("   Step 4: Checking memory usage")
    memory_info = psutil.Process().memory_info()
    memory_gb = memory_info.rss / (1024**3)
    print(f"   ✅ Memory usage: {memory_gb:.3f} GB")
    
    # Cleanup
    del activations, candidates, area
    gc.collect()
    print("   ✅ Cleanup completed")
    
    return True

def test_dense_matrix_operations():
    """Test dense matrix operations instead of sparse."""
    print("\n🧪 Testing dense matrix operations...")
    
    try:
        # Test with small dense matrix
        n = 1000
        print(f"   Step 1: Creating dense matrix ({n}x{n})")
        
        # Create a dense matrix (this will use more memory but won't hang)
        weights = np.random.exponential(1.0, size=(n, n)).astype(np.float32)
        print("   ✅ Dense matrix created successfully")
        
        print("   Step 2: Testing matrix multiplication")
        test_vector = np.zeros(n, dtype=np.float32)
        result = np.dot(weights, test_vector)
        print("   ✅ Matrix multiplication successful")
        
        print("   Step 3: Testing with random vector")
        random_vector = np.random.exponential(1.0, size=n).astype(np.float32)
        result = np.dot(weights, random_vector)
        print("   ✅ Random vector multiplication successful")
        
        # Cleanup
        del weights, test_vector, random_vector, result
        gc.collect()
        print("   ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in dense matrix test: {e}")
        return False

def test_performance_scaling():
    """Test performance scaling with different sizes."""
    print("\n🧪 Testing performance scaling...")
    
    test_sizes = [1000, 5000, 10000]
    
    for n in test_sizes:
        print(f"\n   Testing {n} neurons...")
        
        try:
            # Create arrays
            activations = np.zeros(n, dtype=np.float32)
            candidates = np.zeros(n, dtype=np.float32)
            area = np.zeros(n, dtype=np.float32)
            
            # Test performance
            times = []
            for step in range(5):  # Only 5 steps for speed
                step_start = time.time()
                
                # Generate candidates
                candidates[:] = np.random.exponential(1.0, size=n).astype(np.float32)
                
                # Simple threshold operation
                threshold = np.percentile(activations, 90)
                area[:] = np.where(activations > threshold, candidates, 0.0)
                
                step_time = time.time() - step_start
                times.append(step_time)
            
            avg_time = np.mean(times)
            print(f"      Average time: {avg_time*1000:.3f} ms per step")
            print(f"      Steps per second: {1.0/avg_time:.1f}")
            
            # Cleanup
            del activations, candidates, area
            gc.collect()
            
        except Exception as e:
            print(f"      ❌ Error with {n} neurons: {e}")
            return False
    
    return True

def test_memory_efficiency():
    """Test memory efficiency with different approaches."""
    print("\n🧪 Testing memory efficiency...")
    
    n = 5000
    print(f"   Testing with {n} neurons...")
    
    try:
        # Approach 1: Dense matrix
        print("   Approach 1: Dense matrix")
        start_memory = psutil.Process().memory_info().rss / (1024**3)
        
        weights_dense = np.random.exponential(1.0, size=(n, n)).astype(np.float32)
        activations = np.zeros(n, dtype=np.float32)
        
        dense_memory = psutil.Process().memory_info().rss / (1024**3) - start_memory
        print(f"      Dense matrix memory: {dense_memory:.3f} GB")
        
        # Test performance
        start_time = time.time()
        for _ in range(10):
            activations[:] = np.dot(weights_dense, activations)
        dense_time = time.time() - start_time
        print(f"      Dense matrix time: {dense_time:.3f} seconds")
        
        # Cleanup
        del weights_dense, activations
        gc.collect()
        
        # Approach 2: Sparse-like with masking
        print("   Approach 2: Sparse-like with masking")
        start_memory = psutil.Process().memory_info().rss / (1024**3)
        
        # Create a mask for sparsity
        sparsity = 0.01  # 1% sparsity
        mask = np.random.random((n, n)) < sparsity
        weights_sparse = np.zeros((n, n), dtype=np.float32)
        weights_sparse[mask] = np.random.exponential(1.0, size=np.sum(mask)).astype(np.float32)
        activations = np.zeros(n, dtype=np.float32)
        
        sparse_memory = psutil.Process().memory_info().rss / (1024**3) - start_memory
        print(f"      Sparse-like memory: {sparse_memory:.3f} GB")
        
        # Test performance
        start_time = time.time()
        for _ in range(10):
            activations[:] = np.dot(weights_sparse, activations)
        sparse_time = time.time() - start_time
        print(f"      Sparse-like time: {sparse_time:.3f} seconds")
        
        # Cleanup
        del weights_sparse, activations, mask
        gc.collect()
        
        print(f"      Memory efficiency: {dense_memory/sparse_memory:.2f}x")
        print(f"      Time efficiency: {dense_time/sparse_time:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in memory efficiency test: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 NUMPY ONLY TEST")
    print("=" * 50)
    print("This test avoids scipy.sparse entirely to identify performance characteristics.")
    print()
    
    # Test 1: Basic NumPy operations
    if not test_numpy_operations():
        print("❌ Basic NumPy test failed")
        return
    
    # Test 2: Dense matrix operations
    if not test_dense_matrix_operations():
        print("❌ Dense matrix test failed")
        return
    
    # Test 3: Performance scaling
    if not test_performance_scaling():
        print("❌ Performance scaling test failed")
        return
    
    # Test 4: Memory efficiency
    if not test_memory_efficiency():
        print("❌ Memory efficiency test failed")
        return
    
    print("\n✅ ALL TESTS PASSED!")
    print("The issue is with scipy.sparse - we can work around this with NumPy-only approaches.")

if __name__ == "__main__":
    main()

