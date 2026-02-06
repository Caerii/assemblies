#!/usr/bin/env python3
"""
Utility Functions for Universal Brain Simulator
===============================================

This module contains utility functions used throughout the
universal brain simulator system.
"""

import os
from typing import List, Dict, Any, Tuple

# Try to import CuPy for GPU memory management
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def check_cupy_availability() -> bool:
    """
    Check if CuPy is available and print status
    
    Returns:
        bool: True if CuPy is available, False otherwise
    """
    if CUPY_AVAILABLE:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            current_device = cp.cuda.Device().id
            device_memory = cp.cuda.Device().mem_info[1] / 1024**3
            
            print("✅ CuPy imported successfully!")
            print(f"   CUDA devices: {device_count}")
            print(f"   Current device: {current_device}")
            print(f"   Device memory: {device_memory:.1f} GB")
            return True
        except Exception as e:
            print(f"⚠️  CuPy available but device error: {e}")
            return False
    else:
        print("⚠️  CuPy not available, using NumPy fallback")
        return False


def get_dll_path(dll_name: str) -> str:
    """
    Get the full path to a DLL file
    
    Args:
        dll_name: Name of the DLL file
        
    Returns:
        str: Full path to the DLL file
    """
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, '..', '..', '..', '.build', 'dlls', dll_name)


def create_test_configs() -> List[Dict[str, Any]]:
    """
    Create test configurations for benchmarking
    
    Returns:
        List[Dict]: List of test configuration dictionaries
    """
    from .config import SimulationConfig
    
    test_configs = [
        {
            "name": "GPU + Optimized CUDA (O(N log K))",
            "config": SimulationConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=True,
                use_optimized_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
        },
        {
            "name": "GPU + Original CUDA (O(N²))",
            "config": SimulationConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=True,
                use_optimized_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
        },
        {
            "name": "GPU Only (CuPy)",
            "config": SimulationConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=True,
                use_cuda_kernels=False,
                use_optimized_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
        },
        {
            "name": "CPU Only (NumPy)",
            "config": SimulationConfig(
                n_neurons=1000000,
                active_percentage=0.01,
                n_areas=5,
                use_gpu=False,
                use_cuda_kernels=False,
                use_optimized_kernels=False,
                memory_efficient=True,
                sparse_mode=True
            )
        }
    ]
    
    return test_configs


def print_performance_summary(results: List[Dict[str, Any]]):
    """
    Print formatted performance summary
    
    Args:
        results: List of performance result dictionaries
    """
    print("\n📊 UNIVERSAL BRAIN SIMULATOR SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Memory GB':<10} {'CUDA':<6} {'CuPy':<6} {'NumPy':<6}")
    print("-" * 80)
    
    for result in results:
        if result.get('steps_per_second', 0) > 0:
            print(f"{result['name']:<20} {result['steps_per_second']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_second']:<15,.0f} {result['memory_usage_gb']:<10.2f} {'✅' if result['cuda_kernels_used'] else '❌':<6} {'✅' if result['cupy_used'] else '❌':<6} {'✅' if result['numpy_fallback'] else '❌':<6}")
        else:
            print(f"{result['name']:<20} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<10} {'❌':<6} {'❌':<6} {'❌':<6}")


def find_best_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find the best performing configuration
    
    Args:
        results: List of performance result dictionaries
        
    Returns:
        Dict: Best performing configuration or empty dict if none found
    """
    successful_results = [r for r in results if r.get('steps_per_second', 0) > 0]
    if successful_results:
        return max(successful_results, key=lambda x: x['steps_per_second'])
    return {}


def format_number(num: float) -> str:
    """
    Format a number with appropriate units and precision
    
    Args:
        num: Number to format
        
    Returns:
        str: Formatted number string
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.1f}"


def get_memory_usage() -> Tuple[float, float]:
    """
    Get current memory usage (GPU if available, otherwise CPU)
    
    Returns:
        Tuple[float, float]: (used_gb, total_gb)
    """
    if CUPY_AVAILABLE:
        try:
            used, total = cp.cuda.Device().mem_info
            return used / 1024**3, total / 1024**3
        except:
            pass
    
    # Fallback to CPU memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.used / 1024**3, memory.total / 1024**3
    except:
        return 0.0, 0.0


def print_initialization_status(config, cuda_kernels_loaded: bool, kernel_type: str = None):
    """
    Print initialization status for the brain simulator
    
    Args:
        config: SimulationConfig object
        cuda_kernels_loaded: Whether CUDA kernels were loaded
        kernel_type: Type of kernel loaded (if any)
    """
    print("🚀 Universal Brain Simulator initialized:")
    print(f"   Neurons: {config.n_neurons:,}")
    print(f"   Active percentage: {config.active_percentage*100:.4f}%")
    print(f"   Active per area: {config.k_active:,}")
    print(f"   Areas: {config.n_areas}")
    print(f"   GPU mode: {'✅' if config.use_gpu and CUPY_AVAILABLE else '❌'}")
    print(f"   CUDA kernels: {'✅' if cuda_kernels_loaded else '❌'}")
    if cuda_kernels_loaded and kernel_type:
        print(f"   Kernel type: {kernel_type}")
    print(f"   Memory efficient: {'✅' if config.memory_efficient else '❌'}")
    print(f"   Sparse mode: {'✅' if config.sparse_mode else '❌'}")
    print("   ✅ Brain initialized successfully!")
