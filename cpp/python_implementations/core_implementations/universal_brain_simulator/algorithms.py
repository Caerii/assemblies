#!/usr/bin/env python3
"""
Algorithm Strategies for Universal Brain Simulator
=================================================

This module contains different algorithm implementations for brain simulation,
using the Strategy pattern to eliminate complex if/else chains.
"""

import ctypes
from typing import Union, Optional, List
import numpy as np
from .config import SimulationConfig
from .cuda_manager import CUDAManager
from .utils import CUPY_AVAILABLE

# Import CuPy if available
if CUPY_AVAILABLE:
    import cupy as cp


class AlgorithmStrategy:
    """
    Abstract base class for algorithm strategies
    
    Each strategy implements the core brain simulation algorithms
    for a specific backend (CUDA, CuPy, NumPy).
    """
    
    def __init__(self, config: SimulationConfig, cuda_manager: CUDAManager):
        self.config = config
        self.cuda_manager = cuda_manager
    
    def generate_candidates(self, area: dict, area_idx: int) -> Optional[Union[np.ndarray, cp.ndarray]]:
        """Generate candidates for an area"""
        raise NotImplementedError
    
    def select_top_k(self, candidates: Union[np.ndarray, cp.ndarray], k: int) -> Union[np.ndarray, cp.ndarray]:
        """Select top-k winners from candidates"""
        raise NotImplementedError
    
    def update_weights(self, area: dict, winners: Union[np.ndarray, cp.ndarray]) -> None:
        """Update weights for an area"""
        raise NotImplementedError
    
    def get_strategy_name(self) -> str:
        """Get the name of this strategy"""
        raise NotImplementedError


class CUDAOptimizedStrategy(AlgorithmStrategy):
    """
    Strategy for CUDA optimized brain simulator
    
    This strategy delegates to the optimized brain simulator
    which handles all operations internally.
    """
    
    def generate_candidates(self, area: dict, area_idx: int) -> None:
        """Optimized brain simulator handles candidates internally"""
        return None  # Handled internally by optimized brain
    
    def select_top_k(self, candidates: Union[np.ndarray, cp.ndarray], k: int) -> None:
        """Optimized brain simulator handles top-k selection internally"""
        return None  # Handled internally by optimized brain
    
    def update_weights(self, area: dict, winners: Union[np.ndarray, cp.ndarray]) -> None:
        """Optimized brain simulator handles weight updates internally"""
        pass  # Handled internally by optimized brain
    
    def get_strategy_name(self) -> str:
        return "CUDA Optimized Brain"


class CUDARawStrategy(AlgorithmStrategy):
    """
    Strategy for raw CUDA kernels
    
    This strategy uses individual CUDA kernels for each operation.
    """
    
    def generate_candidates(self, area: dict, area_idx: int) -> Optional[Union[np.ndarray, cp.ndarray]]:
        """Generate candidates using CUDA kernels"""
        if not self.cuda_manager.is_loaded:
            return None
        
        try:
            # Get CUDA memory arrays
            states, candidates, _ = self.cuda_manager.memory_manager.get_cuda_memory_arrays(self.config.k_active)
            if states is None or candidates is None:
                return None
            
            # Call CUDA kernel
            self.cuda_manager.cuda_kernels.cuda_generate_candidates(
                ctypes.c_void_p(states.data.ptr),
                ctypes.c_void_p(candidates.data.ptr),
                ctypes.c_uint32(self.config.k_active),
                ctypes.c_float(0.0),  # mean
                ctypes.c_float(1.0),  # stddev
                ctypes.c_float(0.0)   # cutoff
            )
            
            return candidates
            
        except Exception as e:
            print(f"   ⚠️  CUDA candidate generation failed: {e}")
            return None
    
    def select_top_k(self, candidates: Union[np.ndarray, cp.ndarray], k: int) -> Optional[Union[np.ndarray, cp.ndarray]]:
        """Select top-k using CUDA kernels"""
        if not self.cuda_manager.is_loaded or candidates is None:
            return None
        
        try:
            # Get CUDA memory for top-k indices
            _, _, top_k_indices = self.cuda_manager.memory_manager.get_cuda_memory_arrays(k)
            if top_k_indices is None:
                return None
            
            # Call CUDA kernel
            self.cuda_manager.cuda_kernels.cuda_top_k_selection(
                ctypes.c_void_p(candidates.data.ptr),
                ctypes.c_void_p(top_k_indices.data.ptr),
                ctypes.c_uint32(len(candidates)),
                ctypes.c_uint32(k)
            )
            
            return top_k_indices
            
        except Exception as e:
            print(f"   ⚠️  CUDA top-k selection failed: {e}")
            return None
    
    def update_weights(self, area: dict, winners: Union[np.ndarray, cp.ndarray]) -> None:
        """Update weights using CUDA kernels"""
        if not self.cuda_manager.is_loaded or winners is None:
            return
        
        try:
            # Simple weight update - could be more sophisticated
            if CUPY_AVAILABLE and hasattr(winners, 'get'):
                # CuPy array
                area['weights'] = cp.ones_like(area['weights'], dtype=cp.float32)
            else:
                # NumPy array
                area['weights'] = np.ones_like(area['weights'], dtype=np.float32)
                
        except Exception as e:
            print(f"   ⚠️  CUDA weight update failed: {e}")
    
    def get_strategy_name(self) -> str:
        return "CUDA Raw Kernels"


class CuPyStrategy(AlgorithmStrategy):
    """
    Strategy for CuPy (GPU NumPy)
    
    This strategy uses CuPy for GPU-accelerated operations.
    """
    
    def generate_candidates(self, area: dict, area_idx: int) -> Optional[cp.ndarray]:
        """Generate candidates using CuPy"""
        if not CUPY_AVAILABLE:
            return None
        
        try:
            # Generate random candidates using CuPy
            candidates = cp.random.normal(0.0, 1.0, self.config.k_active, dtype=cp.float32)
            return candidates
            
        except Exception as e:
            print(f"   ⚠️  CuPy candidate generation failed: {e}")
            return None
    
    def select_top_k(self, candidates: cp.ndarray, k: int) -> Optional[cp.ndarray]:
        """Select top-k using CuPy"""
        if not CUPY_AVAILABLE or candidates is None:
            return None
        
        try:
            # Use CuPy's argsort for top-k selection
            top_k_indices = cp.argsort(candidates)[-k:][::-1]  # Top k in descending order
            return top_k_indices
            
        except Exception as e:
            print(f"   ⚠️  CuPy top-k selection failed: {e}")
            return None
    
    def update_weights(self, area: dict, winners: cp.ndarray) -> None:
        """Update weights using CuPy"""
        if not CUPY_AVAILABLE or winners is None:
            return
        
        try:
            # Simple weight update using CuPy
            area['weights'] = cp.ones_like(area['weights'], dtype=cp.float32)
            
        except Exception as e:
            print(f"   ⚠️  CuPy weight update failed: {e}")
    
    def get_strategy_name(self) -> str:
        return "CuPy GPU"


class NumPyStrategy(AlgorithmStrategy):
    """
    Strategy for NumPy (CPU)
    
    This strategy uses NumPy for CPU-based operations.
    """
    
    def generate_candidates(self, area: dict, area_idx: int) -> np.ndarray:
        """Generate candidates using NumPy"""
        try:
            # Generate random candidates using NumPy
            candidates = np.random.normal(0.0, 1.0, self.config.k_active, dtype=np.float32)
            return candidates
            
        except Exception as e:
            print(f"   ⚠️  NumPy candidate generation failed: {e}")
            return np.zeros(self.config.k_active, dtype=np.float32)
    
    def select_top_k(self, candidates: np.ndarray, k: int) -> np.ndarray:
        """Select top-k using NumPy"""
        if candidates is None:
            return np.zeros(k, dtype=np.int32)
        
        try:
            # Use NumPy's argsort for top-k selection
            top_k_indices = np.argsort(candidates)[-k:][::-1]  # Top k in descending order
            return top_k_indices
            
        except Exception as e:
            print(f"   ⚠️  NumPy top-k selection failed: {e}")
            return np.zeros(k, dtype=np.int32)
    
    def update_weights(self, area: dict, winners: np.ndarray) -> None:
        """Update weights using NumPy"""
        if winners is None:
            return
        
        try:
            # Simple weight update using NumPy
            area['weights'] = np.ones_like(area['weights'], dtype=np.float32)
            
        except Exception as e:
            print(f"   ⚠️  NumPy weight update failed: {e}")
    
    def get_strategy_name(self) -> str:
        return "NumPy CPU"


# =============================================================================
# STRATEGY FACTORY
# =============================================================================

def create_algorithm_strategy(config: SimulationConfig, cuda_manager: CUDAManager) -> AlgorithmStrategy:
    """
    Create the appropriate algorithm strategy based on configuration
    
    Args:
        config: Simulation configuration
        cuda_manager: CUDA manager instance
        
    Returns:
        AlgorithmStrategy: The appropriate strategy instance
    """
    # Priority order: CUDA Optimized > CUDA Raw > CuPy > NumPy
    
    if config.use_gpu and cuda_manager.using_optimized_kernels:
        return CUDAOptimizedStrategy(config, cuda_manager)
    
    elif config.use_gpu and cuda_manager.is_loaded:
        return CUDARawStrategy(config, cuda_manager)
    
    elif config.use_gpu and CUPY_AVAILABLE:
        return CuPyStrategy(config, cuda_manager)
    
    else:
        return NumPyStrategy(config, cuda_manager)


def get_available_strategies() -> List[str]:
    """
    Get list of available strategy names
    
    Returns:
        List[str]: List of available strategy names
    """
    strategies = ["CUDA Optimized Brain", "CUDA Raw Kernels"]
    
    if CUPY_AVAILABLE:
        strategies.append("CuPy GPU")
    
    strategies.append("NumPy CPU")
    
    return strategies


def print_strategy_info(strategy: AlgorithmStrategy):
    """
    Print information about a strategy
    
    Args:
        strategy: The strategy to print info for
    """
    print(f"   🧠 Using algorithm strategy: {strategy.get_strategy_name()}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_strategy_compatibility(config: SimulationConfig, cuda_manager: CUDAManager) -> bool:
    """
    Validate that the selected strategy is compatible with the configuration
    
    Args:
        config: Simulation configuration
        cuda_manager: CUDA manager instance
        
    Returns:
        bool: True if strategy is compatible, False otherwise
    """
    strategy = create_algorithm_strategy(config, cuda_manager)
    
    if isinstance(strategy, CUDAOptimizedStrategy):
        return cuda_manager.using_optimized_kernels and cuda_manager.optimized_brain_ptr is not None
    
    elif isinstance(strategy, CUDARawStrategy):
        return cuda_manager.is_loaded and not cuda_manager.using_optimized_kernels
    
    elif isinstance(strategy, CuPyStrategy):
        return CUPY_AVAILABLE and config.use_gpu
    
    elif isinstance(strategy, NumPyStrategy):
        return True  # NumPy is always available
    
    return False


def get_strategy_performance_estimate(strategy: AlgorithmStrategy) -> str:
    """
    Get a performance estimate for a strategy
    
    Args:
        strategy: The strategy to estimate performance for
        
    Returns:
        str: Performance estimate description
    """
    if isinstance(strategy, CUDAOptimizedStrategy):
        return "Highest performance - optimized CUDA kernels"
    elif isinstance(strategy, CUDARawStrategy):
        return "High performance - raw CUDA kernels"
    elif isinstance(strategy, CuPyStrategy):
        return "Medium performance - GPU NumPy"
    elif isinstance(strategy, NumPyStrategy):
        return "Lower performance - CPU NumPy"
    else:
        return "Unknown performance"
