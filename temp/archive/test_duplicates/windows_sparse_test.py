#!/usr/bin/env python3
"""
Windows Sparse Test
==================

This script tests scipy.sparse with very small sizes and uses Windows-compatible
timeout mechanisms to prevent hanging.
"""

import numpy as np
import time
import threading
import sys

def test_with_timeout(func, timeout_seconds=5):
    """Run a function with a timeout on Windows."""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        print(f"   ⏰ TIMEOUT! Operation took longer than {timeout_seconds} seconds")
        return None
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

def test_very_small_sparse():
    """Test with very small sparse matrices."""
    print("🧪 Testing very small sparse matrices...")
    
    def test_10x10():
        from scipy.sparse import coo_matrix, csr_matrix
        
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
        
        return True
    
    def test_100x100():
        from scipy.sparse import coo_matrix, csr_matrix
        
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
        
        return True
    
    def test_1000x1000():
        from scipy.sparse import coo_matrix, csr_matrix
        
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
        
        return True
    
    # Test 1: 10x10 matrix
    print("   Test 1: 10x10 matrix")
    result = test_with_timeout(test_10x10, 5)
    if result is None:
        print("   ❌ 10x10 matrix timed out")
        return False
    print("   ✅ 10x10 matrix successful")
    
    # Test 2: 100x100 matrix
    print("   Test 2: 100x100 matrix")
    result = test_with_timeout(test_100x100, 5)
    if result is None:
        print("   ❌ 100x100 matrix timed out")
        return False
    print("   ✅ 100x100 matrix successful")
    
    # Test 3: 1000x1000 matrix
    print("   Test 3: 1000x1000 matrix")
    result = test_with_timeout(test_1000x1000, 10)
    if result is None:
        print("   ❌ 1000x1000 matrix timed out")
        return False
    print("   ✅ 1000x1000 matrix successful")
    
    return True

def test_numpy_only_approach():
    """Test NumPy-only approach for comparison."""
    print("\n🧪 Testing NumPy-only approach...")
    
    def test_dense_matrix():
        n = 1000
        weights = np.random.exponential(1.0, size=(n, n)).astype(np.float32)
        test_vector = np.zeros(n, dtype=np.float32)
        result = np.dot(weights, test_vector)
        return True
    
    def test_sparse_like():
        n = 1000
        sparsity = 0.01  # 1% sparsity
        mask = np.random.random((n, n)) < sparsity
        weights_sparse = np.zeros((n, n), dtype=np.float32)
        weights_sparse[mask] = np.random.exponential(1.0, size=np.sum(mask)).astype(np.float32)
        test_vector = np.zeros(n, dtype=np.float32)
        result = np.dot(weights_sparse, test_vector)
        return True
    
    # Test dense matrix
    print("   Testing dense matrix")
    result = test_with_timeout(test_dense_matrix, 5)
    if result is None:
        print("   ❌ Dense matrix timed out")
        return False
    print("   ✅ Dense matrix successful")
    
    # Test sparse-like approach
    print("   Testing sparse-like approach")
    result = test_with_timeout(test_sparse_like, 5)
    if result is None:
        print("   ❌ Sparse-like approach timed out")
        return False
    print("   ✅ Sparse-like approach successful")
    
    return True

def test_larger_sparse_matrices():
    """Test larger sparse matrices to find the breaking point."""
    print("\n🧪 Testing larger sparse matrices...")
    
    def test_size(n, sparsity, description):
        from scipy.sparse import coo_matrix, csr_matrix
        
        n_weights = int(n * n * sparsity)
        print(f"      Testing {description}: {n}x{n}, {n_weights} weights")
        
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
        
        return True
    
    # Test different sizes
    test_configs = [
        (5000, 0.0001, "5K neurons, 0.01% sparsity"),
        (10000, 0.0001, "10K neurons, 0.01% sparsity"),
        (50000, 0.00001, "50K neurons, 0.001% sparsity"),
    ]
    
    for n, sparsity, description in test_configs:
        print(f"   {description}")
        result = test_with_timeout(lambda: test_size(n, sparsity, description), 15)
        if result is None:
            print(f"   ❌ {description} timed out - this is the breaking point!")
            break
        print(f"   ✅ {description} successful")

def main():
    """Main test function."""
    print("🔍 WINDOWS SPARSE TEST")
    print("=" * 50)
    print("This test uses threading timeouts to prevent hanging on Windows.")
    print()
    
    # Test 1: Very small sparse matrices
    if not test_very_small_sparse():
        print("❌ Small sparse matrix test failed")
        return
    
    # Test 2: NumPy-only approach
    if not test_numpy_only_approach():
        print("❌ NumPy-only test failed")
        return
    
    # Test 3: Larger sparse matrices
    test_larger_sparse_matrices()
    
    print("\n✅ ALL TESTS COMPLETED!")
    print("The issue is likely with larger sparse matrices, not scipy.sparse itself.")

if __name__ == "__main__":
    main()

