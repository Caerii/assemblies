#!/usr/bin/env python3
"""
Universal Brain Simulator Package
=================================

This package provides a modular, high-performance brain simulation system
that supports multiple backends (CUDA, CuPy, NumPy) and optimization levels.

Main Components:
- UniversalBrainSimulator: Main orchestrator class
- SimulationConfig: Configuration management
- PerformanceMonitor: Performance tracking
- CUDAManager: CUDA kernel management
- MemoryManager: Memory allocation and pooling
- AreaManager: Brain area management
- SimulationEngine: Core simulation logic

Usage:
    from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig
    
    config = SimulationConfig(
        n_neurons=1000000,
        active_percentage=0.01,
        use_optimized_kernels=True
    )
    
    simulator = UniversalBrainSimulator(config)
    simulator.simulate(n_steps=100)
"""

__version__ = "2.0.0"
__author__ = "Universal Brain Simulator Team"
__description__ = "Modular, high-performance brain simulation system"

# Import main classes
from .config import SimulationConfig, PerformanceMetrics
from .metrics import PerformanceMonitor
from .cuda_manager import CUDAManager
from .memory_manager import MemoryManager
from .area_manager import AreaManager
from .simulation_engine import SimulationEngine
from .universal_brain_simulator import UniversalBrainSimulator

# Import lightweight client
from .client import BrainSimulator, quick_sim, quick_benchmark, compare_configurations

# Import utility functions
from .utils import (
    check_cupy_availability,
    create_test_configs,
    print_performance_summary,
    find_best_performance,
    format_number,
    get_memory_usage,
    print_initialization_status
)

# Define what gets imported with "from universal_brain_simulator import *"
__all__ = [
    # Main classes
    'UniversalBrainSimulator',
    'SimulationConfig',
    'PerformanceMetrics',
    'PerformanceMonitor',
    'CUDAManager',
    'MemoryManager',
    'AreaManager',
    'SimulationEngine',
    
    # Lightweight client
    'BrainSimulator',
    'quick_sim',
    'quick_benchmark',
    'compare_configurations',
    
    # Utility functions
    'check_cupy_availability',
    'create_test_configs',
    'print_performance_summary',
    'find_best_performance',
    'format_number',
    'get_memory_usage',
    'print_initialization_status',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]

# Package initialization
def _initialize_package():
    """Initialize the package and check dependencies"""
    print(f"🚀 Universal Brain Simulator v{__version__}")
    print(f"   {__description__}")
    
    # Check CuPy availability
    cupy_available = check_cupy_availability()
    
    return {
        'version': __version__,
        'cupy_available': cupy_available,
        'components': len(__all__) - 3  # Exclude version, author, description
    }

# Initialize package when imported
_package_info = _initialize_package()
