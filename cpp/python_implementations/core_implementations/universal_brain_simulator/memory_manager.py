#!/usr/bin/env python3
"""
Memory Management for Universal Brain Simulator
===============================================

This module handles GPU/CPU memory allocation and pooling
for the universal brain simulator system.
"""

import ctypes
from typing import Tuple, Optional, Union
import numpy as np
from .config import SimulationConfig
from .cuda_manager import CUDAManager
from .utils import CUPY_AVAILABLE

# Import CuPy if available
if CUPY_AVAILABLE:
    import cupy as cp


class MemoryManager:
    """
    Handles GPU/CPU memory allocation and pooling
    
    This class manages memory allocation for both GPU and CPU operations,
    including CUDA memory pools and dynamic memory management.
    """
    
    def __init__(self, config: SimulationConfig, cuda_manager: CUDAManager):
        """
        Initialize memory manager with instance isolation
        
        Args:
            config: Simulation configuration
            cuda_manager: CUDA manager instance
        """
        self.config = config
        self.cuda_manager = cuda_manager
        
        # CUDA memory pools
        self._cuda_states = None
        self._cuda_candidates = None
        self._cuda_top_k = None
        self._cuda_max_k = 0
        
        # Memory tracking
        self._memory_initialized = False
        
        # Instance isolation - each manager gets unique ID
        self._instance_id = id(self)
        self._cleanup_called = False
    
    def initialize_cuda_pools(self):
        """Initialize CUDA memory pools for efficient reuse"""
        if not self.cuda_manager.is_loaded or not CUPY_AVAILABLE:
            return
        
        # Initialize memory pools as None - will allocate dynamically
        self._cuda_states = None
        self._cuda_candidates = None
        self._cuda_top_k = None
        self._cuda_max_k = 0
        
        self._memory_initialized = True
        print(f"   🔧 CUDA memory pools initialized (dynamic allocation)")
    
    def ensure_cuda_memory(self, required_k: int) -> bool:
        """
        Ensure CUDA memory is allocated for the required size
        
        Args:
            required_k: Required number of elements
            
        Returns:
            bool: True if memory is available, False otherwise
        """
        if not self.cuda_manager.is_loaded or not CUPY_AVAILABLE:
            return False
        
        if self._cuda_states is None or required_k > self._cuda_max_k:
            # Allocate new memory
            try:
                self._cuda_states = cp.zeros(required_k, dtype=cp.uint64)
                self._cuda_candidates = cp.zeros(required_k, dtype=cp.float32)
                self._cuda_top_k = cp.zeros(required_k, dtype=cp.uint32)
                self._cuda_max_k = required_k
                
                # Initialize curand states
                self.cuda_manager.cuda_kernels.cuda_initialize_curand(
                    ctypes.c_void_p(self._cuda_states.data.ptr),
                    ctypes.c_uint32(required_k),
                    ctypes.c_uint32(self.config.seed)
                )
                
                print(f"   🔧 CUDA memory reallocated for k={required_k:,}")
                return True
            except Exception as e:
                print(f"   ⚠️  Failed to allocate CUDA memory: {e}")
                return False
        
        return True
    
    def get_cuda_memory_arrays(self, required_k: int) -> Tuple[Optional[cp.ndarray], Optional[cp.ndarray], Optional[cp.ndarray]]:
        """
        Get CUDA memory arrays for the required size
        
        Args:
            required_k: Required number of elements
            
        Returns:
            Tuple of (states, candidates, top_k) arrays or (None, None, None) if failed
        """
        if not self.ensure_cuda_memory(required_k):
            return None, None, None
        
        return (
            self._cuda_states[:required_k],
            self._cuda_candidates[:required_k],
            self._cuda_top_k[:required_k]
        )
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Get current memory usage
        
        Returns:
            Tuple[float, float]: (used_gb, total_gb)
        """
        if self.config.use_gpu and CUPY_AVAILABLE:
            try:
                used, total = cp.cuda.Device().mem_info
                return used / 1024**3, total / 1024**3
            except:
                return 0.0, 0.0
        else:
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.used / 1024**3, memory.total / 1024**3
            except:
                return 0.0, 0.0
    
    def allocate_gpu_array(self, shape: Union[int, Tuple], dtype=np.float32) -> Union[np.ndarray, cp.ndarray]:
        """
        Allocate GPU array if available, otherwise CPU array
        
        Args:
            shape: Array shape
            dtype: Data type
            
        Returns:
            GPU array (CuPy) if available, otherwise CPU array (NumPy)
        """
        if self.config.use_gpu and CUPY_AVAILABLE:
            return cp.zeros(shape, dtype=dtype)
        else:
            return np.zeros(shape, dtype=dtype)
    
    def allocate_cpu_array(self, shape: Union[int, Tuple], dtype=np.float32) -> np.ndarray:
        """
        Allocate CPU array
        
        Args:
            shape: Array shape
            dtype: Data type
            
        Returns:
            CPU array (NumPy)
        """
        return np.zeros(shape, dtype=dtype)
    
    def transfer_to_gpu(self, array: np.ndarray) -> Union[np.ndarray, cp.ndarray]:
        """
        Transfer array to GPU if available
        
        Args:
            array: NumPy array to transfer
            
        Returns:
            GPU array (CuPy) if available, otherwise original array
        """
        if self.config.use_gpu and CUPY_AVAILABLE:
            return cp.asarray(array)
        else:
            return array
    
    def transfer_to_cpu(self, array: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        """
        Transfer array to CPU
        
        Args:
            array: Array to transfer (NumPy or CuPy)
            
        Returns:
            CPU array (NumPy)
        """
        if CUPY_AVAILABLE and hasattr(array, 'get'):
            return array.get()  # CuPy array
        else:
            return array  # Already NumPy
    
    def cleanup_memory(self):
        """Cleanup allocated memory with proper error handling and double-cleanup prevention"""
        # Prevent double-cleanup
        if self._cleanup_called:
            print(f"⚠️  Memory cleanup already called for instance {self._instance_id}")
            return
        
        self._cleanup_called = True
        
        try:
            if CUPY_AVAILABLE:
                # Clear CUDA memory pools
                self._cuda_states = None
                self._cuda_candidates = None
                self._cuda_top_k = None
                self._cuda_max_k = 0
                
                # Force garbage collection to free GPU memory
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    print(f"🧹 CuPy memory pool cleared for instance {self._instance_id}")
                except Exception as e:
                    print(f"⚠️  CuPy memory cleanup error for instance {self._instance_id}: {e}")
            
            self._memory_initialized = False
            print(f"🧹 Memory manager cleanup completed for instance {self._instance_id}")
            
        except Exception as e:
            print(f"⚠️  Memory cleanup error for instance {self._instance_id}: {e}")
            # Still reset the flag even if cleanup failed
            self._memory_initialized = False
    
    def get_memory_info(self) -> dict:
        """
        Get detailed memory information
        
        Returns:
            Dict containing memory information
        """
        used_gb, total_gb = self.get_memory_usage()
        
        info = {
            'used_gb': used_gb,
            'total_gb': total_gb,
            'utilization_percent': (used_gb / total_gb * 100) if total_gb > 0 else 0,
            'cuda_pools_initialized': self._memory_initialized,
            'cuda_max_k': self._cuda_max_k,
            'gpu_available': CUPY_AVAILABLE and self.config.use_gpu
        }
        
        if CUPY_AVAILABLE and self.config.use_gpu:
            try:
                info['gpu_device_id'] = cp.cuda.Device().id
                info['gpu_device_count'] = cp.cuda.runtime.getDeviceCount()
            except:
                pass
        
        return info
    
    @property
    def memory_initialized(self) -> bool:
        """Check if memory is initialized"""
        return self._memory_initialized
    
    @property
    def cuda_max_k(self) -> int:
        """Get maximum allocated CUDA memory size"""
        return self._cuda_max_k
