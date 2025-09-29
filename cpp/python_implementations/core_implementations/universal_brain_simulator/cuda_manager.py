#!/usr/bin/env python3
"""
CUDA Kernel Management for Universal Brain Simulator
====================================================

This module manages CUDA kernel loading and interface setup
for the universal brain simulator system.
"""

import os
import ctypes
from typing import Optional
from .config import SimulationConfig
from .utils import get_dll_path
from .cuda_signatures import (
    setup_optimized_brain_signatures,
    setup_individual_optimized_signatures,
    setup_original_kernel_signatures,
    validate_dll_interface
)


class CUDAManager:
    """
    Manages CUDA kernel loading and interface setup
    
    This class handles loading CUDA kernels, setting up function signatures,
    and managing the interface between Python and CUDA code.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize CUDA manager with instance isolation
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self._cuda_kernels = None
        self._optimized_brain_ptr = None
        self._using_optimized_kernels = False
        
        # Instance isolation - each manager gets unique ID
        self._instance_id = id(self)
        self._initialized = False
        self._cleanup_called = False
    
    def load_kernels(self) -> bool:
        """
        Load CUDA kernels with fallback logic
        
        Returns:
            bool: True if kernels were loaded successfully, False otherwise
        """
        if not self.config.use_cuda_kernels:
            return False
        
        # Try optimized kernels first if requested
        if self.config.use_optimized_kernels:
            if self._try_load_optimized_kernels():
                return True
            else:
                print("⚠️  Optimized CUDA kernels DLL not found, trying original...")
        
        # Fallback to original kernels
        return self._try_load_original_kernels()
    
    def _try_load_optimized_kernels(self) -> bool:
        """
        Try to load optimized CUDA kernels
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            optimized_dll_path = get_dll_path('assemblies_cuda_brain_optimized.dll')
            if os.path.exists(optimized_dll_path):
                self._cuda_kernels = ctypes.CDLL(optimized_dll_path)
                self._setup_optimized_kernel_signatures()
                print("✅ Optimized CUDA kernels loaded successfully!")
                self._using_optimized_kernels = True
                return True
            else:
                print(f"⚠️  Optimized DLL not found at: {optimized_dll_path}")
                return False
        except Exception as e:
            print(f"⚠️  Optimized CUDA kernels failed to load: {e}")
            return False
    
    def _try_load_original_kernels(self) -> bool:
        """
        Try to load original CUDA kernels
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            original_dll_path = get_dll_path('assemblies_cuda_kernels.dll')
            if os.path.exists(original_dll_path):
                self._cuda_kernels = ctypes.CDLL(original_dll_path)
                self._setup_original_kernel_signatures()
                print("✅ Original CUDA kernels loaded successfully!")
                self._using_optimized_kernels = False
                return True
            else:
                print(f"⚠️  Original DLL not found at: {original_dll_path}")
                return False
        except Exception as e:
            print(f"⚠️  Original CUDA kernels failed to load: {e}")
            return False
    
    def _setup_optimized_kernel_signatures(self):
        """Set up function signatures for optimized CUDA kernels"""
        if not self._cuda_kernels:
            return
        
        try:
            # Check if this is the optimized brain simulator DLL
            if hasattr(self._cuda_kernels, 'cuda_create_optimized_brain'):
                print("   🔧 Setting up optimized brain simulator interface")
                if setup_optimized_brain_signatures(self._cuda_kernels):
                    print("   ✅ Optimized brain simulator interface configured")
                    return
            
            # Individual optimized kernels (fallback)
            print("   🔧 Setting up individual optimized kernel signatures")
            if setup_individual_optimized_signatures(self._cuda_kernels):
                print("   ✅ Individual optimized kernel signatures configured")
            
        except Exception as e:
            print(f"   ⚠️  Failed to set up optimized kernel signatures: {e}")
    
    
    def _setup_original_kernel_signatures(self):
        """Set up function signatures for original CUDA kernels"""
        if not self._cuda_kernels:
            return
        
        try:
            print("   🔧 Setting up original kernel signatures")
            if setup_original_kernel_signatures(self._cuda_kernels):
                print("   ✅ Original kernel signatures configured")
            
        except Exception as e:
            print(f"   ⚠️  Failed to set up original kernel signatures: {e}")
    
    def create_optimized_brain(self, n_neurons: int, n_areas: int, k_active: int, seed: int) -> Optional[int]:
        """
        Create optimized brain instance
        
        Args:
            n_neurons: Number of neurons
            n_areas: Number of areas
            k_active: Number of active neurons
            seed: Random seed
            
        Returns:
            int: Brain instance pointer or None if failed
        """
        if not self._cuda_kernels or not self._using_optimized_kernels:
            return None
        
        try:
            self._optimized_brain_ptr = self._cuda_kernels.cuda_create_optimized_brain(
                ctypes.c_uint32(n_neurons),
                ctypes.c_uint32(n_areas),
                ctypes.c_uint32(k_active),
                ctypes.c_uint32(seed)
            )
            print(f"   🧠 Optimized brain instance created: {self._optimized_brain_ptr}")
            return self._optimized_brain_ptr
        except Exception as e:
            print(f"   ⚠️  Failed to create optimized brain: {e}")
            return None
    
    def simulate_step_optimized(self) -> bool:
        """
        Simulate one step using optimized brain
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._cuda_kernels or not self._optimized_brain_ptr:
            return False
        
        try:
            self._cuda_kernels.cuda_simulate_step_optimized(
                ctypes.c_void_p(self._optimized_brain_ptr)
            )
            return True
        except Exception as e:
            print(f"   ⚠️  Optimized brain simulation failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup CUDA resources with proper error handling and double-cleanup prevention"""
        # Prevent double-cleanup
        if self._cleanup_called:
            print(f"⚠️  Cleanup already called for instance {self._instance_id}")
            return
        
        self._cleanup_called = True
        
        if self._optimized_brain_ptr is not None and self._cuda_kernels is not None:
            try:
                # Only cleanup if we have a valid pointer
                if self._optimized_brain_ptr != 0:
                    self._cuda_kernels.cuda_destroy_optimized_brain(
                        ctypes.c_void_p(self._optimized_brain_ptr)
                    )
                    print(f"🧠 Optimized brain instance {self._instance_id} destroyed")
            except Exception as e:
                print(f"⚠️  CUDA cleanup error for instance {self._instance_id}: {e}")
                # Don't ignore - log the error for debugging
            finally:
                # Always reset the pointer to prevent double-cleanup
                self._optimized_brain_ptr = None
                self._initialized = False
    
    @property
    def cuda_kernels(self):
        """Get the CUDA kernels DLL"""
        return self._cuda_kernels
    
    @property
    def using_optimized_kernels(self) -> bool:
        """Check if using optimized kernels"""
        return self._using_optimized_kernels
    
    @property
    def optimized_brain_ptr(self):
        """Get the optimized brain pointer"""
        return self._optimized_brain_ptr
    
    @property
    def is_loaded(self) -> bool:
        """Check if CUDA kernels are loaded"""
        return self._cuda_kernels is not None
