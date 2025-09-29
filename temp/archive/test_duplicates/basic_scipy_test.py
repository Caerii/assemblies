#!/usr/bin/env python3
"""
Basic SciPy Test
===============

This script tests the most basic scipy operations without any timeouts
to identify the exact issue.
"""

import numpy as np
import time

def test_basic_imports():
    """Test basic imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import scipy
        print(f"   ✅ SciPy version: {scipy.__version__}")
    except Exception as e:
        print(f"   ❌ SciPy import failed: {e}")
        return False
    
    try:
        from scipy import sparse
        print(f"   ✅ scipy.sparse imported")
    except Exception as e:
        print(f"   ❌ scipy.sparse import failed: {e}")
        return False
    
    try:
        from scipy.sparse import coo_matrix
        print(f"   ✅ coo_matrix imported")
    except Exception as e:
        print(f"   ❌ coo_matrix import failed: {e}")
        return False
    
    try:
        from scipy.sparse import csr_matrix
        print(f"   ✅ csr_matrix imported")
    except Exception as e:
        print(f"   ❌ csr_matrix import failed: {e}")
        return False
    
    return True

def test_manual_coo_creation():
    """Test manual COO matrix creation without scipy."""
    print("\n🔍 Testing manual COO matrix creation...")
    
    try:
        # Create data manually
        n = 5
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        row = np.array([0, 1, 2], dtype=np.int32)
        col = np.array([0, 1, 2], dtype=np.int32)
        
        print(f"   Data: {data}")
        print(f"   Row indices: {row}")
        print(f"   Col indices: {col}")
        print(f"   Shape: ({n}, {n})")
        
        # Create dense matrix manually
        dense_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(len(data)):
            dense_matrix[row[i], col[i]] = data[i]
        
        print(f"   Dense matrix:\n{dense_matrix}")
        
        # Test multiplication
        test_vector = np.zeros(n, dtype=np.float32)
        test_vector[0] = 1.0
        result = np.dot(dense_matrix, test_vector)
        print(f"   Test vector: {test_vector}")
        print(f"   Result: {result}")
        
        print("   ✅ Manual COO creation successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Manual COO creation failed: {e}")
        return False

def test_scipy_coo_creation():
    """Test scipy COO matrix creation step by step."""
    print("\n🔍 Testing scipy COO matrix creation...")
    
    try:
        from scipy.sparse import coo_matrix
        
        print("   Step 1: Creating data arrays")
        n = 5
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        row = np.array([0, 1, 2], dtype=np.int32)
        col = np.array([0, 1, 2], dtype=np.int32)
        print("   ✅ Data arrays created")
        
        print("   Step 2: Creating COO matrix")
        coo_matrix_obj = coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)
        print("   ✅ COO matrix created")
        
        print("   Step 3: Checking matrix properties")
        print(f"      Shape: {coo_matrix_obj.shape}")
        print(f"      Data type: {coo_matrix_obj.dtype}")
        print(f"      Number of non-zero elements: {coo_matrix_obj.nnz}")
        print("   ✅ Matrix properties checked")
        
        print("   Step 4: Converting to dense for verification")
        dense_matrix = coo_matrix_obj.toarray()
        print(f"      Dense matrix:\n{dense_matrix}")
        print("   ✅ Conversion to dense successful")
        
        print("   Step 5: Testing matrix multiplication")
        test_vector = np.zeros(n, dtype=np.float32)
        test_vector[0] = 1.0
        result = coo_matrix_obj.dot(test_vector)
        print(f"      Test vector: {test_vector}")
        print(f"      Result: {result}")
        print("   ✅ Matrix multiplication successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ SciPy COO creation failed: {e}")
        print(f"      Error type: {type(e).__name__}")
        return False

def test_scipy_csr_conversion():
    """Test scipy CSR conversion."""
    print("\n🔍 Testing scipy CSR conversion...")
    
    try:
        from scipy.sparse import coo_matrix, csr_matrix
        
        print("   Step 1: Creating COO matrix")
        n = 5
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        row = np.array([0, 1, 2], dtype=np.int32)
        col = np.array([0, 1, 2], dtype=np.int32)
        
        coo_matrix_obj = coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)
        print("   ✅ COO matrix created")
        
        print("   Step 2: Converting to CSR")
        csr_matrix_obj = coo_matrix_obj.tocsr()
        print("   ✅ CSR matrix created")
        
        print("   Step 3: Checking CSR properties")
        print(f"      Shape: {csr_matrix_obj.shape}")
        print(f"      Data type: {csr_matrix_obj.dtype}")
        print(f"      Number of non-zero elements: {csr_matrix_obj.nnz}")
        print("   ✅ CSR properties checked")
        
        print("   Step 4: Testing CSR multiplication")
        test_vector = np.zeros(n, dtype=np.float32)
        test_vector[0] = 1.0
        result = csr_matrix_obj.dot(test_vector)
        print(f"      Test vector: {test_vector}")
        print(f"      Result: {result}")
        print("   ✅ CSR multiplication successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ SciPy CSR conversion failed: {e}")
        print(f"      Error type: {type(e).__name__}")
        return False

def main():
    """Main test function."""
    print("🔍 BASIC SCIPY TEST")
    print("=" * 50)
    print("This test checks basic scipy operations step by step.")
    print()
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("❌ Basic imports failed")
        return
    
    # Test 2: Manual COO creation
    if not test_manual_coo_creation():
        print("❌ Manual COO creation failed")
        return
    
    # Test 3: SciPy COO creation
    if not test_scipy_coo_creation():
        print("❌ SciPy COO creation failed")
        return
    
    # Test 4: SciPy CSR conversion
    if not test_scipy_csr_conversion():
        print("❌ SciPy CSR conversion failed")
        return
    
    print("\n✅ ALL BASIC TESTS PASSED!")
    print("The issue might be with larger matrices or specific operations.")

if __name__ == "__main__":
    main()

