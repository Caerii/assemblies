#!/usr/bin/env python3
"""
Debug Memory Test
================

This test prints progress at every step to identify exactly where it gets stuck.
"""

import numpy as np
import time
import gc
import psutil

def test_basic_arrays():
    """Test basic array operations without sparse matrices."""
    print("🧪 Testing basic array operations...")
    
    # Test 1: Small arrays
    print("   Step 1: Creating small arrays (1K neurons)")
    activations = np.zeros(1000, dtype=np.float32)
    candidates = np.zeros(1000, dtype=np.float32)
    area = np.zeros(1000, dtype=np.float32)
    print("   ✅ Small arrays created successfully")
    
    # Test 2: Random generation
    print("   Step 2: Generating random numbers")
    candidates[:] = np.random.exponential(1.0, size=1000).astype(np.float32)
    print("   ✅ Random numbers generated successfully")
    
    # Test 3: Simple operations
    print("   Step 3: Performing simple operations")
    threshold = np.percentile(activations, 90)
    area[:] = np.where(activations > threshold, candidates, 0.0)
    print("   ✅ Simple operations completed successfully")
    
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

def test_medium_arrays():
    """Test medium-sized arrays."""
    print("\n🧪 Testing medium arrays (10K neurons)...")
    
    try:
        print("   Step 1: Creating medium arrays")
        activations = np.zeros(10000, dtype=np.float32)
        candidates = np.zeros(10000, dtype=np.float32)
        area = np.zeros(10000, dtype=np.float32)
        print("   ✅ Medium arrays created successfully")
        
        print("   Step 2: Generating random numbers")
        candidates[:] = np.random.exponential(1.0, size=10000).astype(np.float32)
        print("   ✅ Random numbers generated successfully")
        
        print("   Step 3: Performing operations")
        threshold = np.percentile(activations, 90)
        area[:] = np.where(activations > threshold, candidates, 0.0)
        print("   ✅ Operations completed successfully")
        
        print("   Step 4: Checking memory")
        memory_info = psutil.Process().memory_info()
        memory_gb = memory_info.rss / (1024**3)
        print(f"   ✅ Memory usage: {memory_gb:.3f} GB")
        
        # Cleanup
        del activations, candidates, area
        gc.collect()
        print("   ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in medium arrays: {e}")
        return False

def test_sparse_matrix_small():
    """Test small sparse matrix creation."""
    print("\n🧪 Testing small sparse matrix...")
    
    try:
        print("   Step 1: Creating small sparse matrix (100x100)")
        from scipy.sparse import coo_matrix
        
        # Create very small sparse matrix
        n = 100
        n_weights = 1000  # Only 1000 weights
        
        print("   Step 2: Generating indices and values")
        row_indices = np.random.randint(0, n, size=n_weights, dtype=np.int32)
        col_indices = np.random.randint(0, n, size=n_weights, dtype=np.int32)
        values = np.random.exponential(1.0, size=n_weights).astype(np.float32)
        print("   ✅ Indices and values generated")
        
        print("   Step 3: Creating COO matrix")
        coo_matrix_obj = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float32
        )
        print("   ✅ COO matrix created")
        
        print("   Step 4: Converting to CSR")
        csr_matrix_obj = coo_matrix_obj.tocsr()
        print("   ✅ CSR matrix created")
        
        print("   Step 5: Testing matrix multiplication")
        test_vector = np.zeros(n, dtype=np.float32)
        result = csr_matrix_obj.dot(test_vector)
        print("   ✅ Matrix multiplication successful")
        
        # Cleanup
        del coo_matrix_obj, csr_matrix_obj, row_indices, col_indices, values, test_vector, result
        gc.collect()
        print("   ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in sparse matrix test: {e}")
        return False

def test_performance_small():
    """Test performance on small scale."""
    print("\n🧪 Testing performance (1K neurons)...")
    
    try:
        n = 1000
        print(f"   Step 1: Setting up {n} neurons")
        
        activations = np.zeros(n, dtype=np.float32)
        candidates = np.zeros(n, dtype=np.float32)
        area = np.zeros(n, dtype=np.float32)
        print("   ✅ Arrays created")
        
        print("   Step 2: Running 10 simulation steps")
        times = []
        
        for step in range(10):
            step_start = time.time()
            
            # Generate candidates
            candidates[:] = np.random.exponential(1.0, size=n).astype(np.float32)
            
            # Simple threshold operation (no sparse matrix for now)
            threshold = np.percentile(activations, 90)
            area[:] = np.where(activations > threshold, candidates, 0.0)
            
            step_time = time.time() - step_start
            times.append(step_time)
            
            if step % 2 == 0:
                print(f"      Step {step}: {step_time*1000:.3f} ms")
        
        avg_time = np.mean(times)
        print(f"   ✅ Average time: {avg_time*1000:.3f} ms per step")
        print(f"   ✅ Steps per second: {1.0/avg_time:.1f}")
        
        # Cleanup
        del activations, candidates, area
        gc.collect()
        print("   ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in performance test: {e}")
        return False

def main():
    """Main test function with detailed progress reporting."""
    print("🔍 DEBUG MEMORY TEST")
    print("=" * 50)
    print("This test will print progress at every step to identify where it gets stuck.")
    print()
    
    # Test 1: Basic arrays
    if not test_basic_arrays():
        print("❌ Basic array test failed - stopping here")
        return
    
    # Test 2: Medium arrays
    if not test_medium_arrays():
        print("❌ Medium array test failed - stopping here")
        return
    
    # Test 3: Sparse matrix
    if not test_sparse_matrix_small():
        print("❌ Sparse matrix test failed - stopping here")
        return
    
    # Test 4: Performance
    if not test_performance_small():
        print("❌ Performance test failed - stopping here")
        return
    
    print("\n✅ ALL TESTS PASSED!")
    print("The issue might be with larger scales or specific operations.")

if __name__ == "__main__":
    main()

