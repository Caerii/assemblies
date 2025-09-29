#!/usr/bin/env python3
"""
Immediate Output Test
====================

This test prints output immediately and catches hanging issues.
"""

import sys
import time

def print_immediate(message):
    """Print message immediately and flush output."""
    print(message)
    sys.stdout.flush()

def test_basic_imports():
    """Test basic imports with immediate output."""
    print_immediate("🔍 Testing basic imports...")
    
    try:
        print_immediate("   Importing scipy...")
        import scipy
        print_immediate(f"   ✅ SciPy version: {scipy.__version__}")
    except Exception as e:
        print_immediate(f"   ❌ SciPy import failed: {e}")
        return False
    
    try:
        print_immediate("   Importing scipy.sparse...")
        from scipy import sparse
        print_immediate("   ✅ scipy.sparse imported")
    except Exception as e:
        print_immediate(f"   ❌ scipy.sparse import failed: {e}")
        return False
    
    try:
        print_immediate("   Importing coo_matrix...")
        from scipy.sparse import coo_matrix
        print_immediate("   ✅ coo_matrix imported")
    except Exception as e:
        print_immediate(f"   ❌ coo_matrix import failed: {e}")
        return False
    
    try:
        print_immediate("   Importing csr_matrix...")
        from scipy.sparse import csr_matrix
        print_immediate("   ✅ csr_matrix imported")
    except Exception as e:
        print_immediate(f"   ❌ csr_matrix import failed: {e}")
        return False
    
    return True

def test_numpy_only():
    """Test NumPy operations only."""
    print_immediate("\n🔍 Testing NumPy operations...")
    
    try:
        print_immediate("   Importing numpy...")
        import numpy as np
        print_immediate("   ✅ NumPy imported")
        
        print_immediate("   Creating small array...")
        arr = np.zeros(10, dtype=np.float32)
        print_immediate("   ✅ Small array created")
        
        print_immediate("   Testing random generation...")
        random_arr = np.random.exponential(1.0, size=10).astype(np.float32)
        print_immediate("   ✅ Random generation successful")
        
        print_immediate("   Testing operations...")
        result = arr + random_arr
        print_immediate("   ✅ Operations successful")
        
        return True
        
    except Exception as e:
        print_immediate(f"   ❌ NumPy test failed: {e}")
        return False

def test_manual_matrix():
    """Test manual matrix operations without scipy."""
    print_immediate("\n🔍 Testing manual matrix operations...")
    
    try:
        import numpy as np
        
        print_immediate("   Creating 5x5 matrix...")
        n = 5
        matrix = np.zeros((n, n), dtype=np.float32)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 2.0
        matrix[2, 2] = 3.0
        print_immediate("   ✅ Matrix created")
        
        print_immediate("   Testing matrix multiplication...")
        vector = np.zeros(n, dtype=np.float32)
        vector[0] = 1.0
        result = np.dot(matrix, vector)
        print_immediate(f"   ✅ Multiplication result: {result}")
        
        return True
        
    except Exception as e:
        print_immediate(f"   ❌ Manual matrix test failed: {e}")
        return False

def test_scipy_simple():
    """Test very simple scipy operations."""
    print_immediate("\n🔍 Testing simple scipy operations...")
    
    try:
        print_immediate("   Creating COO matrix...")
        from scipy.sparse import coo_matrix
        import numpy as np
        
        # Very simple data
        data = np.array([1.0], dtype=np.float32)
        row = np.array([0], dtype=np.int32)
        col = np.array([0], dtype=np.int32)
        
        print_immediate("   Data created, creating matrix...")
        coo_matrix_obj = coo_matrix((data, (row, col)), shape=(2, 2), dtype=np.float32)
        print_immediate("   ✅ COO matrix created")
        
        print_immediate("   Converting to dense...")
        dense_matrix = coo_matrix_obj.toarray()
        print_immediate(f"   ✅ Dense matrix: {dense_matrix}")
        
        return True
        
    except Exception as e:
        print_immediate(f"   ❌ Simple scipy test failed: {e}")
        print_immediate(f"   Error type: {type(e).__name__}")
        return False

def main():
    """Main test function with immediate output."""
    print_immediate("🔍 IMMEDIATE OUTPUT TEST")
    print_immediate("=" * 50)
    print_immediate("This test prints output immediately to catch hanging issues.")
    print_immediate("")
    
    # Test 1: Basic imports
    print_immediate("Starting test 1: Basic imports")
    if not test_basic_imports():
        print_immediate("❌ Basic imports failed - stopping here")
        return
    
    # Test 2: NumPy only
    print_immediate("Starting test 2: NumPy operations")
    if not test_numpy_only():
        print_immediate("❌ NumPy test failed - stopping here")
        return
    
    # Test 3: Manual matrix
    print_immediate("Starting test 3: Manual matrix operations")
    if not test_manual_matrix():
        print_immediate("❌ Manual matrix test failed - stopping here")
        return
    
    # Test 4: Simple scipy
    print_immediate("Starting test 4: Simple scipy operations")
    if not test_scipy_simple():
        print_immediate("❌ Simple scipy test failed - stopping here")
        return
    
    print_immediate("\n✅ ALL TESTS PASSED!")
    print_immediate("The issue might be with larger operations or specific scipy functions.")

if __name__ == "__main__":
    main()

