#!/usr/bin/env python3
"""
Simple Sparse Test
=================

This script tests scipy.sparse with very small sizes and stops immediately
if it hangs to avoid terminal issues.
"""

import numpy as np
import time
import signal
import sys

# Set a timeout to prevent hanging
def timeout_handler(signum, frame):
    print("\n⏰ TIMEOUT! Operation took too long, stopping...")
    sys.exit(1)

# Set 5 second timeout
signal.signal(signal.SIGALRM, timeout_handler)

def test_very_small_sparse():
    """Test with very small sparse matrices."""
    print("🧪 Testing very small sparse matrices...")
    
    try:
        from scipy.sparse import coo_matrix, csr_matrix
        
        # Test 1: Tiny matrix
        print("   Test 1: 10x10 matrix")
        signal.alarm(5)  # 5 second timeout
        
        n = 10
        row_indices = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([0, 1, 2], dtype=np.int32)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        coo_matrix_obj = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float32
        )
        csr_matrix_obj = coo_matrix_obj.tocsr()
        result = csr_matrix_obj.dot(np.zeros(n, dtype=np.float32))
        
        signal.alarm(0)  # Cancel timeout
        print("   ✅ 10x10 matrix successful")
        
        # Test 2: Small matrix
        print("   Test 2: 100x100 matrix")
        signal.alarm(5)  # 5 second timeout
        
        n = 100
        n_weights = 50  # Very sparse
        row_indices = np.random.randint(0, n, size=n_weights, dtype=np.int32)
        col_indices = np.random.randint(0, n, size=n_weights, dtype=np.int32)
        values = np.random.exponential(1.0, size=n_weights).astype(np.float32)
        
        coo_matrix_obj = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float32
        )
        csr_matrix_obj = coo_matrix_obj.tocsr()
        result = csr_matrix_obj.dot(np.zeros(n, dtype=np.float32))
        
        signal.alarm(0)  # Cancel timeout
        print("   ✅ 100x100 matrix successful")
        
        # Test 3: Medium matrix
        print("   Test 3: 1000x1000 matrix")
        signal.alarm(10)  # 10 second timeout
        
        n = 1000
        n_weights = 500  # 0.05% sparsity
        row_indices = np.random.randint(0, n, size=n_weights, dtype=np.int32)
        col_indices = np.random.randint(0, n, size=n_weights, dtype=np.int32)
        values = np.random.exponential(1.0, size=n_weights).astype(np.float32)
        
        coo_matrix_obj = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float32
        )
        csr_matrix_obj = coo_matrix_obj.tocsr()
        result = csr_matrix_obj.dot(np.zeros(n, dtype=np.float32))
        
        signal.alarm(0)  # Cancel timeout
        print("   ✅ 1000x1000 matrix successful")
        
        return True
        
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"   ❌ Error: {e}")
        return False

def test_numpy_only_approach():
    """Test NumPy-only approach for comparison."""
    print("\n🧪 Testing NumPy-only approach...")
    
    try:
        # Test with dense matrices
        n = 1000
        print(f"   Testing {n}x{n} dense matrix")
        
        # Create dense matrix
        weights = np.random.exponential(1.0, size=(n, n)).astype(np.float32)
        test_vector = np.zeros(n, dtype=np.float32)
        
        # Test multiplication
        result = np.dot(weights, test_vector)
        
        print("   ✅ Dense matrix successful")
        
        # Test with sparse-like approach using masking
        print("   Testing sparse-like approach with masking")
        
        # Create sparse-like matrix
        sparsity = 0.01  # 1% sparsity
        mask = np.random.random((n, n)) < sparsity
        weights_sparse = np.zeros((n, n), dtype=np.float32)
        weights_sparse[mask] = np.random.exponential(1.0, size=np.sum(mask)).astype(np.float32)
        
        # Test multiplication
        result = np.dot(weights_sparse, test_vector)
        
        print("   ✅ Sparse-like approach successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 SIMPLE SPARSE TEST")
    print("=" * 50)
    print("This test uses timeouts to prevent hanging.")
    print()
    
    # Test 1: Very small sparse matrices
    if not test_very_small_sparse():
        print("❌ Sparse matrix test failed")
        return
    
    # Test 2: NumPy-only approach
    if not test_numpy_only_approach():
        print("❌ NumPy-only test failed")
        return
    
    print("\n✅ ALL TESTS PASSED!")
    print("The issue is likely with larger sparse matrices, not scipy.sparse itself.")

if __name__ == "__main__":
    main()

