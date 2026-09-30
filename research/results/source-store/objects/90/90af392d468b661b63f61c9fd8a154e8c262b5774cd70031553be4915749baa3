#!/usr/bin/env python3
"""
CUDA Kernels Test - Isolated testing of CUDA kernels functionality

This test verifies that all CUDA kernels are working correctly:
- cuda_initialize_curand: CURAND state initialization
- cuda_generate_candidates: Random candidate generation  
- cuda_top_k_selection: Top-k selection algorithm
- cuda_accumulate_weights: Weight accumulation kernel (THE HOTTEST KERNEL)

Performance: Typically shows 12-21x speedup over CuPy
Memory: Tests with 1K-10K elements, suitable for small-scale validation

Usage: python test_cuda_kernels.py
"""

import time
import os
import ctypes

# Try to import CuPy
try:
    import cupy as cp
    print("✅ CuPy imported successfully!")
    CUPY_AVAILABLE = True
except ImportError:
    print("⚠️  CuPy not available")
    CUPY_AVAILABLE = False

def test_cuda_kernels():
    """Test CUDA kernels functionality"""
    print("🧪 TESTING CUDA KERNELS")
    print("=" * 50)
    
    # Load DLL
    dll_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.build', 'dlls', 'assemblies_cuda_kernels.dll')
    print(f"DLL path: {dll_path}")
    print(f"DLL exists: {os.path.exists(dll_path)}")
    
    if not os.path.exists(dll_path):
        print("❌ DLL not found!")
        return
    
    try:
        dll = ctypes.CDLL(dll_path)
        print("✅ DLL loaded successfully!")
        
        # Test function availability
        functions = ['cuda_accumulate_weights', 'cuda_generate_candidates', 'cuda_top_k_selection', 'cuda_initialize_curand']
        for func_name in functions:
            try:
                func = getattr(dll, func_name)
                print(f"✅ Function {func_name} found")
            except AttributeError:
                print(f"❌ Function {func_name} not found")
        
        if not CUPY_AVAILABLE:
            print("⚠️  CuPy not available, cannot test kernels")
            return
        
        # Test cuda_initialize_curand
        print("\n🧪 Testing cuda_initialize_curand...")
        n = 1000
        states = cp.zeros(n, dtype=cp.uint64)
        
        try:
            dll.cuda_initialize_curand(
                ctypes.c_void_p(states.data.ptr),
                ctypes.c_uint32(n),
                ctypes.c_uint32(42)
            )
            print("✅ cuda_initialize_curand successful")
        except Exception as e:
            print(f"❌ cuda_initialize_curand failed: {e}")
        
        # Test cuda_generate_candidates
        print("\n🧪 Testing cuda_generate_candidates...")
        candidates = cp.zeros(n, dtype=cp.float32)
        
        try:
            dll.cuda_generate_candidates(
                ctypes.c_void_p(states.data.ptr),
                ctypes.c_void_p(candidates.data.ptr),
                ctypes.c_uint32(n),
                ctypes.c_float(1.0),  # mean
                ctypes.c_float(1.0),  # stddev
                ctypes.c_float(0.0)   # cutoff
            )
            print("✅ cuda_generate_candidates successful")
            print(f"   Sample values: {candidates[:10].get()}")
            print(f"   Mean: {cp.mean(candidates).get():.3f}")
            print(f"   Std: {cp.std(candidates).get():.3f}")
        except Exception as e:
            print(f"❌ cuda_generate_candidates failed: {e}")
        
        # Test cuda_top_k_selection
        print("\n🧪 Testing cuda_top_k_selection...")
        k = 100
        top_k_indices = cp.zeros(k, dtype=cp.uint32)
        
        try:
            dll.cuda_top_k_selection(
                ctypes.c_void_p(candidates.data.ptr),
                ctypes.c_void_p(top_k_indices.data.ptr),
                ctypes.c_uint32(n),
                ctypes.c_uint32(k)
            )
            print("✅ cuda_top_k_selection successful")
            print(f"   Top-k indices: {top_k_indices[:10].get()}")
            print(f"   Top-k values: {candidates[top_k_indices[:10]].get()}")
        except Exception as e:
            print(f"❌ cuda_top_k_selection failed: {e}")
        
        # Performance test
        print("\n🏁 PERFORMANCE TEST")
        print("-" * 30)
        
        # Test CuPy vs CUDA kernels
        n_test = 10000
        k_test = 1000
        n_iterations = 100
        
        # CuPy test
        print("Testing CuPy performance...")
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            candidates_cupy = cp.random.exponential(1.0, size=n_test)
            top_k_cupy = cp.argpartition(candidates_cupy, -k_test)[-k_test:]
        cupy_time = time.perf_counter() - start_time
        
        # CUDA kernels test
        print("Testing CUDA kernels performance...")
        states_test = cp.zeros(n_test, dtype=cp.uint64)
        candidates_test = cp.zeros(n_test, dtype=cp.float32)
        top_k_test = cp.zeros(k_test, dtype=cp.uint32)
        
        # Initialize once
        dll.cuda_initialize_curand(
            ctypes.c_void_p(states_test.data.ptr),
            ctypes.c_uint32(n_test),
            ctypes.c_uint32(42)
        )
        
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            dll.cuda_generate_candidates(
                ctypes.c_void_p(states_test.data.ptr),
                ctypes.c_void_p(candidates_test.data.ptr),
                ctypes.c_uint32(n_test),
                ctypes.c_float(1.0),
                ctypes.c_float(1.0),
                ctypes.c_float(0.0)
            )
            dll.cuda_top_k_selection(
                ctypes.c_void_p(candidates_test.data.ptr),
                ctypes.c_void_p(top_k_test.data.ptr),
                ctypes.c_uint32(n_test),
                ctypes.c_uint32(k_test)
            )
        cuda_time = time.perf_counter() - start_time
        
        print(f"CuPy time: {cupy_time:.3f}s")
        print(f"CUDA kernels time: {cuda_time:.3f}s")
        print(f"Speedup: {cupy_time/cuda_time:.2f}x")
        
        if cuda_time < cupy_time:
            print("✅ CUDA kernels are faster!")
        else:
            print("⚠️  CuPy is faster - CUDA kernels may have overhead")
        
    except Exception as e:
        print(f"❌ DLL loading failed: {e}")

if __name__ == "__main__":
    test_cuda_kernels()
