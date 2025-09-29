#!/usr/bin/env python3
"""
Debug SciPy Sparse
=================

This script specifically debugs what's wrong with scipy.sparse operations.
"""

import numpy as np
import time
import sys

def test_scipy_import():
    """Test if scipy can be imported."""
    print("🔍 Testing scipy import...")
    
    try:
        import scipy
        print(f"   ✅ SciPy version: {scipy.__version__}")
        return True
    except ImportError as e:
        print(f"   ❌ SciPy import failed: {e}")
        return False

def test_scipy_sparse_import():
    """Test if scipy.sparse can be imported."""
    print("\n🔍 Testing scipy.sparse import...")
    
    try:
        from scipy import sparse
        print(f"   ✅ scipy.sparse imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ scipy.sparse import failed: {e}")
        return False

def test_coo_matrix_creation():
    """Test COO matrix creation step by step."""
    print("\n🔍 Testing COO matrix creation...")
    
    try:
        from scipy.sparse import coo_matrix
        
        print("   Step 1: Creating small test data")
        n = 10  # Very small
        row_indices = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([0, 1, 2], dtype=np.int32)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        print("   ✅ Test data created")
        
        print("   Step 2: Creating COO matrix")
        coo_matrix_obj = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float32
        )
        print("   ✅ COO matrix created successfully")
        
        print("   Step 3: Testing matrix properties")
        print(f"      Shape: {coo_matrix_obj.shape}")
        print(f"      Data type: {coo_matrix_obj.dtype}")
        print(f"      Number of non-zero elements: {coo_matrix_obj.nnz}")
        print("   ✅ Matrix properties checked")
        
        # Cleanup
        del coo_matrix_obj
        print("   ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ COO matrix creation failed: {e}")
        print(f"      Error type: {type(e).__name__}")
        return False

def test_csr_matrix_creation():
    """Test CSR matrix creation step by step."""
    print("\n🔍 Testing CSR matrix creation...")
    
    try:
        from scipy.sparse import coo_matrix, csr_matrix
        
        print("   Step 1: Creating COO matrix")
        n = 10
        row_indices = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([0, 1, 2], dtype=np.int32)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        coo_matrix_obj = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float32
        )
        print("   ✅ COO matrix created")
        
        print("   Step 2: Converting to CSR")
        csr_matrix_obj = coo_matrix_obj.tocsr()
        print("   ✅ CSR matrix created successfully")
        
        print("   Step 3: Testing matrix properties")
        print(f"      Shape: {csr_matrix_obj.shape}")
        print(f"      Data type: {csr_matrix_obj.dtype}")
        print(f"      Number of non-zero elements: {csr_matrix_obj.nnz}")
        print("   ✅ Matrix properties checked")
        
        # Cleanup
        del coo_matrix_obj, csr_matrix_obj
        print("   ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CSR matrix creation failed: {e}")
        print(f"      Error type: {type(e).__name__}")
        return False

def test_matrix_multiplication():
    """Test matrix multiplication."""
    print("\n🔍 Testing matrix multiplication...")
    
    try:
        from scipy.sparse import coo_matrix, csr_matrix
        
        print("   Step 1: Creating test matrix")
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
        print("   ✅ Test matrix created")
        
        print("   Step 2: Creating test vector")
        test_vector = np.zeros(n, dtype=np.float32)
        test_vector[0] = 1.0
        print("   ✅ Test vector created")
        
        print("   Step 3: Performing matrix multiplication")
        result = csr_matrix_obj.dot(test_vector)
        print(f"   ✅ Matrix multiplication successful, result shape: {result.shape}")
        
        # Cleanup
        del coo_matrix_obj, csr_matrix_obj, test_vector, result
        print("   ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Matrix multiplication failed: {e}")
        print(f"      Error type: {type(e).__name__}")
        return False

def test_larger_matrix():
    """Test with a larger matrix to see where it fails."""
    print("\n🔍 Testing larger matrix...")
    
    try:
        from scipy.sparse import coo_matrix, csr_matrix
        
        # Test different sizes
        sizes = [100, 500, 1000]
        
        for n in sizes:
            print(f"   Testing {n}x{n} matrix...")
            
            # Create random sparse matrix
            n_weights = n // 10  # 10% sparsity
            row_indices = np.random.randint(0, n, size=n_weights, dtype=np.int32)
            col_indices = np.random.randint(0, n, size=n_weights, dtype=np.int32)
            values = np.random.exponential(1.0, size=n_weights).astype(np.float32)
            
            print(f"      Creating COO matrix with {n_weights} weights...")
            coo_matrix_obj = coo_matrix(
                (values, (row_indices, col_indices)),
                shape=(n, n),
                dtype=np.float32
            )
            print(f"      ✅ COO matrix created")
            
            print(f"      Converting to CSR...")
            csr_matrix_obj = coo_matrix_obj.tocsr()
            print(f"      ✅ CSR matrix created")
            
            print(f"      Testing multiplication...")
            test_vector = np.zeros(n, dtype=np.float32)
            result = csr_matrix_obj.dot(test_vector)
            print(f"      ✅ Multiplication successful")
            
            # Cleanup
            del coo_matrix_obj, csr_matrix_obj, test_vector, result
            print(f"      ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Larger matrix test failed at size {n}: {e}")
        print(f"      Error type: {type(e).__name__}")
        return False

def check_scipy_installation():
    """Check scipy installation details."""
    print("\n🔍 Checking SciPy installation...")
    
    try:
        import scipy
        print(f"   SciPy version: {scipy.__version__}")
        print(f"   SciPy location: {scipy.__file__}")
        
        # Check if it's a conda or pip installation
        try:
            import pkg_resources
            dist = pkg_resources.get_distribution("scipy")
            print(f"   Installation method: {dist.location}")
        except:
            print("   Could not determine installation method")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Could not check SciPy installation: {e}")
        return False

def main():
    """Main debug function."""
    print("🔍 DEBUGGING SCIPY.SPARSE")
    print("=" * 50)
    print("This script will identify exactly what's wrong with scipy.sparse operations.")
    print()
    
    # Test 1: Basic imports
    if not test_scipy_import():
        print("❌ SciPy not installed - install with: pip install scipy")
        return
    
    if not test_scipy_sparse_import():
        print("❌ scipy.sparse not available")
        return
    
    # Test 2: Check installation
    check_scipy_installation()
    
    # Test 3: Basic operations
    if not test_coo_matrix_creation():
        print("❌ COO matrix creation failed")
        return
    
    if not test_csr_matrix_creation():
        print("❌ CSR matrix creation failed")
        return
    
    if not test_matrix_multiplication():
        print("❌ Matrix multiplication failed")
        return
    
    # Test 4: Larger matrices
    if not test_larger_matrix():
        print("❌ Larger matrix test failed")
        return
    
    print("\n✅ ALL SCIPY.SPARSE TESTS PASSED!")
    print("The issue might be with specific operations or memory allocation patterns.")

if __name__ == "__main__":
    main()
