#!/usr/bin/env python3
"""
Performance Monitoring for Universal Brain Simulator
====================================================

This module handles all performance monitoring and profiling
for the universal brain simulator system.
"""

import time
import json
from typing import Dict, Any, List
from .config import SimulationConfig, PerformanceMetrics
from .utils import get_memory_usage


class PerformanceMonitor:
    """
    Handles all performance monitoring and profiling
    
    This class tracks performance metrics during simulation runs
    and provides detailed statistics and profiling data.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize performance monitor
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.metrics = PerformanceMetrics()
        self.profile_data = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'cuda_kernel_usage': []
        }
        self.start_time = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.perf_counter()
        self.metrics.reset()
        self.profile_data = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'cuda_kernel_usage': []
        }
    
    def record_step(self, step_time: float, memory_usage: float = None, 
                   gpu_utilization: float = None, cuda_kernels_used: bool = None):
        """
        Record performance data for one step
        
        Args:
            step_time: Time taken for this step in seconds
            memory_usage: Memory usage in GB (if None, will be measured)
            gpu_utilization: GPU utilization percentage (if None, will be calculated)
            cuda_kernels_used: Whether CUDA kernels were used
        """
        # Update timing metrics
        self.metrics.update_timing(step_time)
        
        # Get memory usage if not provided
        if memory_usage is None:
            used_gb, total_gb = get_memory_usage()
            memory_usage = used_gb
            if gpu_utilization is None and total_gb > 0:
                gpu_utilization = (used_gb / total_gb) * 100
        
        # Update memory metrics
        if gpu_utilization is not None:
            self.metrics.update_memory(memory_usage, gpu_utilization)
        
        # Update technology flags
        if cuda_kernels_used is not None:
            self.metrics.cuda_kernels_used = cuda_kernels_used
        
        # Store profile data if profiling is enabled
        if self.config.enable_profiling:
            self.profile_data['step_times'].append(step_time)
            self.profile_data['memory_usage'].append(memory_usage)
            self.profile_data['gpu_utilization'].append(gpu_utilization or 0.0)
            self.profile_data['cuda_kernel_usage'].append(cuda_kernels_used or False)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance statistics
        
        Returns:
            Dict containing comprehensive performance statistics
        """
        if self.metrics.step_count == 0:
            return {}
        
        avg_step_time = self.metrics.get_average_step_time()
        steps_per_second = self.metrics.get_steps_per_second()
        
        return {
            'total_steps': self.metrics.step_count,
            'total_time': self.metrics.total_time,
            'avg_step_time': avg_step_time,
            'min_step_time': self.metrics.min_step_time,
            'max_step_time': self.metrics.max_step_time,
            'steps_per_second': steps_per_second,
            'neurons_per_second': self.config.n_neurons * steps_per_second,
            'active_neurons_per_second': self.config.k_active * self.config.n_areas * steps_per_second,
            'memory_usage_gb': self.metrics.memory_usage_gb,
            'gpu_utilization': self.metrics.gpu_utilization,
            'cuda_kernels_used': self.metrics.cuda_kernels_used,
            'cupy_used': self.metrics.cupy_used,
            'numpy_fallback': self.metrics.numpy_fallback
        }
    
    def get_profile_data(self) -> Dict[str, Any]:
        """
        Get detailed profiling data
        
        Returns:
            Dict containing raw profiling data
        """
        return self.profile_data.copy()
    
    def save_profile(self, filename: str):
        """
        Save profiling data to JSON file
        
        Args:
            filename: Path to save the profile data
        """
        profile_data = {
            'configuration': self.config.to_dict(),
            'performance': self.get_stats(),
            'profile_data': self.profile_data
        }
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        print(f"📊 Profile data saved to {filename}")
    
    def print_step_summary(self, step: int, step_time: float, profile_interval: int = 10):
        """
        Print summary for a single step
        
        Args:
            step: Current step number (1-indexed)
            step_time: Time for this step
            profile_interval: How often to print summaries
        """
        if step % profile_interval == 0:
            avg_time = self.metrics.get_average_step_time()
            memory_usage = self.metrics.memory_usage_gb
            gpu_util = self.metrics.gpu_utilization
            
            print(f"Step {step:3d}: {step_time*1000:.2f}ms | "
                  f"Avg: {avg_time*1000:.2f}ms | "
                  f"Memory: {memory_usage:.2f}GB ({gpu_util:.1f}%)")
    
    def print_simulation_summary(self, n_steps: int, total_time: float):
        """
        Print summary for a complete simulation run
        
        Args:
            n_steps: Total number of steps simulated
            total_time: Total time for the simulation
        """
        stats = self.get_stats()
        
        print(f"\n📊 UNIVERSAL SIMULATION COMPLETE")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average step time: {total_time/n_steps*1000:.2f}ms")
        print(f"   Min step time: {self.metrics.min_step_time*1000:.2f}ms")
        print(f"   Max step time: {self.metrics.max_step_time*1000:.2f}ms")
        print(f"   Steps per second: {n_steps/total_time:.1f}")
        print(f"   Final memory: {self.metrics.memory_usage_gb:.2f}GB")
        print(f"   GPU utilization: {self.metrics.gpu_utilization:.1f}%")
        print(f"   CUDA kernels: {'✅' if self.metrics.cuda_kernels_used else '❌'}")
        print(f"   CuPy used: {'✅' if self.metrics.cupy_used else '❌'}")
        print(f"   NumPy fallback: {'✅' if self.metrics.numpy_fallback else '❌'}")
    
    def set_technology_flags(self, cuda_kernels_used: bool = None, 
                           cupy_used: bool = None, numpy_fallback: bool = None):
        """
        Set technology usage flags
        
        Args:
            cuda_kernels_used: Whether CUDA kernels are being used
            cupy_used: Whether CuPy is being used
            numpy_fallback: Whether NumPy fallback is being used
        """
        if cuda_kernels_used is not None:
            self.metrics.cuda_kernels_used = cuda_kernels_used
        if cupy_used is not None:
            self.metrics.cupy_used = cupy_used
        if numpy_fallback is not None:
            self.metrics.numpy_fallback = numpy_fallback
    
    def reset(self):
        """Reset all performance metrics to initial state"""
        self.metrics = PerformanceMetrics()
        self.profile_data = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'cuda_kernel_usage': []
        }
