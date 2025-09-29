#!/usr/bin/env python3
"""
Debug CuPy array assignment issue
"""

import cupy as cp
import numpy as np

print("🔍 Debugging CuPy array assignment...")

# Test 1: Basic array creation
print("\n1. Basic array creation:")
arr = cp.zeros(10, dtype=cp.float32)
print(f"   Array shape: {arr.shape}, dtype: {arr.dtype}")

# Test 2: Array slicing
print("\n2. Array slicing:")
arr_slice = arr[:5]
print(f"   Slice shape: {arr_slice.shape}, dtype: {arr_slice.dtype}")

# Test 3: Direct assignment to slice
print("\n3. Direct assignment to slice:")
try:
    arr_slice[:] = cp.random.exponential(1.0, size=5)
    print("   ✅ Direct assignment works")
    print(f"   Values: {arr_slice.get()}")
except Exception as e:
    print(f"   ❌ Direct assignment failed: {e}")

# Test 4: Assignment with intermediate variable
print("\n4. Assignment with intermediate variable:")
try:
    new_values = cp.random.exponential(1.0, size=5)
    arr_slice[:] = new_values
    print("   ✅ Intermediate variable assignment works")
    print(f"   Values: {arr_slice.get()}")
except Exception as e:
    print(f"   ❌ Intermediate variable assignment failed: {e}")

# Test 5: Using copyto
print("\n5. Using copyto:")
try:
    new_values = cp.random.exponential(1.0, size=5)
    cp.copyto(arr_slice, new_values)
    print("   ✅ copyto works")
    print(f"   Values: {arr_slice.get()}")
except Exception as e:
    print(f"   ❌ copyto failed: {e}")

# Test 6: Using advanced indexing
print("\n6. Using advanced indexing:")
try:
    new_values = cp.random.exponential(1.0, size=5)
    arr_slice[...] = new_values
    print("   ✅ Advanced indexing works")
    print(f"   Values: {arr_slice.get()}")
except Exception as e:
    print(f"   ❌ Advanced indexing failed: {e}")

print("\n🔍 Debug complete!")
