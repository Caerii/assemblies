#!/usr/bin/env python3
"""
Custom CUDA Kernels Wrapper
===========================

Python wrapper for our custom dense assembly CUDA kernels.
"""

import ctypes
import numpy as np
import os
from ctypes import c_uint32, c_float, c_void_p, POINTER

# Load the DLL
DLL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'dlls', 'dense_assembly_kernels.dll')

class DenseAssemblyCUDA:
    """Wrapper for custom CUDA kernels"""
    
    def __init__(self):
        self.dll = None
        self.d_W = None
        self.d_active = None
        self.d_output = None
        self.n = 0
        self.k = 0
        
        self._load_dll()
    
    def _load_dll(self):
        """Load the CUDA DLL"""
        if not os.path.exists(DLL_PATH):
            raise FileNotFoundError(f"DLL not found: {DLL_PATH}")
        
        self.dll = ctypes.CDLL(DLL_PATH)
        
        # Define function signatures
        self.dll.dense_assembly_init.argtypes = []
        self.dll.dense_assembly_init.restype = ctypes.c_int
        
        self.dll.dense_assembly_alloc_weights.argtypes = [c_uint32]
        self.dll.dense_assembly_alloc_weights.restype = ctypes.POINTER(c_float)
        
        self.dll.dense_assembly_free.argtypes = [c_void_p]
        self.dll.dense_assembly_free.restype = None
        
        self.dll.dense_assembly_alloc_array.argtypes = [c_uint32, c_uint32]
        self.dll.dense_assembly_alloc_array.restype = c_void_p
        
        self.dll.dense_assembly_copy_to_gpu.argtypes = [c_void_p, c_void_p, c_uint32]
        self.dll.dense_assembly_copy_to_gpu.restype = None
        
        self.dll.dense_assembly_copy_from_gpu.argtypes = [c_void_p, c_void_p, c_uint32]
        self.dll.dense_assembly_copy_from_gpu.restype = None
        
        self.dll.dense_assembly_accumulate.argtypes = [
            ctypes.POINTER(c_float), ctypes.POINTER(c_uint32), 
            ctypes.POINTER(c_float), c_uint32, c_uint32
        ]
        self.dll.dense_assembly_accumulate.restype = None
        
        self.dll.dense_assembly_hebbian_update.argtypes = [
            ctypes.POINTER(c_float), ctypes.POINTER(c_uint32),
            c_float, c_uint32, c_uint32
        ]
        self.dll.dense_assembly_hebbian_update.restype = None
        
        # Initialize
        result = self.dll.dense_assembly_init()
        if result != 0:
            raise RuntimeError("Failed to initialize CUDA")
    
    def allocate(self, n: int, k: int):
        """Allocate GPU memory"""
        self.n = n
        self.k = k
        
        # Allocate weight matrix
        self.d_W = self.dll.dense_assembly_alloc_weights(n)
        if not self.d_W:
            raise MemoryError(f"Failed to allocate {n}x{n} weight matrix")
        
        # Allocate active indices array
        self.d_active = self.dll.dense_assembly_alloc_array(k, 4)  # uint32
        
        # Allocate output array
        self.d_output = self.dll.dense_assembly_alloc_array(n, 4)  # float32
    
    def set_active(self, active_indices: np.ndarray):
        """Copy active indices to GPU"""
        active = np.ascontiguousarray(active_indices, dtype=np.uint32)
        self.dll.dense_assembly_copy_to_gpu(
            self.d_active,
            active.ctypes.data_as(c_void_p),
            len(active) * 4
        )
    
    def accumulate(self) -> np.ndarray:
        """Run weight accumulation kernel"""
        self.dll.dense_assembly_accumulate(
            self.d_W,
            ctypes.cast(self.d_active, ctypes.POINTER(c_uint32)),
            ctypes.cast(self.d_output, ctypes.POINTER(c_float)),
            self.n, self.k
        )
        
        # Copy result back
        output = np.zeros(self.n, dtype=np.float32)
        self.dll.dense_assembly_copy_from_gpu(
            output.ctypes.data_as(c_void_p),
            self.d_output,
            self.n * 4
        )
        return output
    
    def hebbian_update(self, active_indices: np.ndarray, beta: float):
        """Run Hebbian update kernel"""
        active = np.ascontiguousarray(active_indices, dtype=np.uint32)
        self.dll.dense_assembly_copy_to_gpu(
            self.d_active,
            active.ctypes.data_as(c_void_p),
            len(active) * 4
        )
        
        self.dll.dense_assembly_hebbian_update(
            self.d_W,
            ctypes.cast(self.d_active, ctypes.POINTER(c_uint32)),
            beta, self.n, len(active)
        )
    
    def free(self):
        """Free GPU memory"""
        if self.d_W:
            self.dll.dense_assembly_free(self.d_W)
        if self.d_active:
            self.dll.dense_assembly_free(self.d_active)
        if self.d_output:
            self.dll.dense_assembly_free(self.d_output)


def test_custom_cuda():
    """Test the custom CUDA wrapper"""
    print("Testing Custom CUDA Kernels")
    print("=" * 50)
    
    try:
        cuda = DenseAssemblyCUDA()
        print("✅ CUDA initialized")
        
        n, k = 1000, 32
        cuda.allocate(n, k)
        print(f"✅ Allocated n={n}, k={k}")
        
        # Test accumulation
        active = np.random.choice(n, k, replace=False).astype(np.uint32)
        cuda.set_active(active)
        output = cuda.accumulate()
        print(f"✅ Accumulation: output shape={output.shape}, sum={output.sum():.4f}")
        
        # Test Hebbian update
        cuda.hebbian_update(active, 0.1)
        output2 = cuda.accumulate()
        print(f"✅ After Hebbian: output sum={output2.sum():.4f}")
        
        cuda.free()
        print("✅ Memory freed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_custom_cuda()

