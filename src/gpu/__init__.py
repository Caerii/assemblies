"""
GPU Acceleration Module for Neural Assembly Simulations

This module provides GPU-accelerated implementations of neural assembly
computations using both CuPy and PyTorch backends.

Performance Expectations:
- CuPy: 10-100x speedup for matrix operations
- PyTorch: 10-1000x speedup with advanced optimizations
- Memory efficiency: 2-10x larger networks possible
- Parallelization: 1000s of neurons processed simultaneously

Implementation Strategy:
1. Core operations: Matrix multiplications, winner selection
2. Statistical sampling: Parallel binomial and normal distributions
3. Connectome updates: Vectorized plasticity operations
4. Memory management: Efficient GPU memory usage
5. Multi-GPU support: Scale to massive networks

Backend Support:
- CuPy: Direct NumPy-like interface, easy migration
- PyTorch: Advanced features, custom kernels, JIT compilation
- Fallback: Automatic CPU fallback when GPU unavailable

Usage:
    from src.gpu import GPUBrain, CupyBrain, TorchBrain
    
    # CuPy implementation
    brain = CupyBrain(device='cuda:0')
    
    # PyTorch implementation  
    brain = TorchBrain(device='cuda:0')
    
    # Automatic backend selection
    brain = GPUBrain(backend='auto')
"""

from .cupy_brain import CupyBrain
from .torch_brain import TorchBrain
from .gpu_utils import GPUUtils, MemoryManager
from .performance import PerformanceProfiler

__all__ = [
    'CupyBrain',
    'TorchBrain', 
    'GPUUtils',
    'MemoryManager',
    'PerformanceProfiler'
]
