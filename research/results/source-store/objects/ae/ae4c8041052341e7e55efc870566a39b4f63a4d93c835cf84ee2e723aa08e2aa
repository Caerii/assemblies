#!/usr/bin/env python3
"""
Configuration Classes for Universal Brain Simulator
==================================================

This module contains all configuration classes and data structures
used throughout the universal brain simulator system.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimulationConfig:
    """
    Configuration for brain simulation
    
    This class contains all the parameters needed to configure
    a brain simulation run.
    """
    # Core simulation parameters
    n_neurons: int = 1000000
    active_percentage: float = 0.01
    n_areas: int = 5
    seed: int = 42
    
    # Hardware and acceleration options
    use_gpu: bool = True
    use_cuda_kernels: bool = True
    use_optimized_kernels: bool = True  # Choose between original and optimized
    
    # Performance and memory options
    memory_efficient: bool = True
    sparse_mode: bool = True
    enable_profiling: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if not 0 < self.active_percentage <= 1:
            raise ValueError("active_percentage must be between 0 and 1")
        if self.n_areas <= 0:
            raise ValueError("n_areas must be positive")
    
    @property
    def k_active(self) -> int:
        """Calculate the number of active neurons per area"""
        return int(self.n_neurons * self.active_percentage)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'n_neurons': self.n_neurons,
            'active_percentage': self.active_percentage,
            'n_areas': self.n_areas,
            'seed': self.seed,
            'use_gpu': self.use_gpu,
            'use_cuda_kernels': self.use_cuda_kernels,
            'use_optimized_kernels': self.use_optimized_kernels,
            'memory_efficient': self.memory_efficient,
            'sparse_mode': self.sparse_mode,
            'enable_profiling': self.enable_profiling,
            'k_active': self.k_active
        }


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for monitoring simulation performance
    
    This class tracks various performance indicators during
    brain simulation runs.
    """
    # Timing metrics
    step_count: int = 0
    total_time: float = 0.0
    min_step_time: float = float('inf')
    max_step_time: float = 0.0
    
    # Memory metrics
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    
    # Technology usage flags
    cuda_kernels_used: bool = False
    cupy_used: bool = False
    numpy_fallback: bool = False
    
    def reset(self):
        """Reset all metrics to initial values"""
        self.step_count = 0
        self.total_time = 0.0
        self.min_step_time = float('inf')
        self.max_step_time = 0.0
        self.memory_usage_gb = 0.0
        self.gpu_utilization = 0.0
        self.cuda_kernels_used = False
        self.cupy_used = False
        self.numpy_fallback = False
    
    def update_timing(self, step_time: float):
        """Update timing metrics with a new step time"""
        self.step_count += 1
        self.total_time += step_time
        self.min_step_time = min(self.min_step_time, step_time)
        self.max_step_time = max(self.max_step_time, step_time)
    
    def update_memory(self, memory_gb: float, gpu_util: float):
        """Update memory metrics"""
        self.memory_usage_gb = memory_gb
        self.gpu_utilization = gpu_util
    
    def get_average_step_time(self) -> float:
        """Get average step time"""
        return self.total_time / self.step_count if self.step_count > 0 else 0.0
    
    def get_steps_per_second(self) -> float:
        """Get steps per second"""
        avg_time = self.get_average_step_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'step_count': self.step_count,
            'total_time': self.total_time,
            'min_step_time': self.min_step_time,
            'max_step_time': self.max_step_time,
            'average_step_time': self.get_average_step_time(),
            'steps_per_second': self.get_steps_per_second(),
            'memory_usage_gb': self.memory_usage_gb,
            'gpu_utilization': self.gpu_utilization,
            'cuda_kernels_used': self.cuda_kernels_used,
            'cupy_used': self.cupy_used,
            'numpy_fallback': self.numpy_fallback
        }
