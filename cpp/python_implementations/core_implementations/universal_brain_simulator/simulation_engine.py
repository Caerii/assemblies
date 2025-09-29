#!/usr/bin/env python3
"""
Simulation Engine for Universal Brain Simulator
===============================================

This module contains the core simulation logic and orchestration
for the universal brain simulator system.
"""

import time
from typing import Union
import numpy as np
from .config import SimulationConfig
from .cuda_manager import CUDAManager
from .area_manager import AreaManager
from .metrics import PerformanceMonitor
from .utils import CUPY_AVAILABLE
from .simulation_commands import create_simulation_command, print_command_info

# Import CuPy if available
if CUPY_AVAILABLE:
    import cupy as cp


class SimulationEngine:
    """
    Core simulation logic and orchestration
    
    This class handles the main simulation loop and coordinates
    between different components of the brain simulator.
    """
    
    def __init__(self, config: SimulationConfig, cuda_manager: CUDAManager, 
                 area_manager: AreaManager, metrics: PerformanceMonitor):
        """
        Initialize simulation engine
        
        Args:
            config: Simulation configuration
            cuda_manager: CUDA manager instance
            area_manager: Area manager instance
            metrics: Performance monitor instance
        """
        self.config = config
        self.cuda_manager = cuda_manager
        self.area_manager = area_manager
        self.metrics = metrics
        
        # Create simulation command
        self.command = create_simulation_command(config, cuda_manager, area_manager, metrics)
        print_command_info(self.command)
    
    def simulate_step(self) -> float:
        """
        Simulate one step of the brain using the selected command
        
        Returns:
            float: Time taken for this step in seconds
        """
        # Execute the simulation step using the selected command
        success = self.command.execute()
        step_time = self.command.get_timing()
        
        if not success:
            print(f"   ⚠️  Simulation step failed")
        
        # Get memory usage for metrics
        used_gb, total_gb = self.area_manager.memory_manager.get_memory_usage()
        gpu_util = (used_gb / total_gb * 100) if total_gb > 0 else 0
        
        # Record performance metrics
        self.metrics.record_step(
            step_time=step_time,
            memory_usage=used_gb,
            gpu_utilization=gpu_util,
            cuda_kernels_used=self.cuda_manager.is_loaded
        )
        
        return step_time
    
    
    def simulate(self, n_steps: int = 100, verbose: bool = True, profile_interval: int = 10) -> float:
        """
        Simulate multiple steps
        
        Args:
            n_steps: Number of steps to simulate
            verbose: Whether to print progress information
            profile_interval: How often to print progress updates
            
        Returns:
            float: Total time for the simulation
        """
        if verbose:
            print(f"\n🧠 SIMULATING {n_steps} STEPS (Universal Mode)")
            print("=" * 60)
        
        # Start performance monitoring
        self.metrics.start_monitoring()
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            step_time = self.simulate_step()
            
            if verbose:
                self.metrics.print_step_summary(step + 1, step_time, profile_interval)
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            self.metrics.print_simulation_summary(n_steps, total_time)
        
        return total_time
    
    def simulate_with_callback(self, n_steps: int, callback_func, callback_interval: int = 1):
        """
        Simulate with a callback function for custom processing
        
        Args:
            n_steps: Number of steps to simulate
            callback_func: Function to call after each step (receives step number and step time)
            callback_interval: How often to call the callback function
        """
        self.metrics.start_monitoring()
        
        for step in range(n_steps):
            step_time = self.simulate_step()
            
            if (step + 1) % callback_interval == 0:
                callback_func(step + 1, step_time)
    
    def benchmark_step(self, n_warmup: int = 5, n_measure: int = 10) -> dict:
        """
        Benchmark a single simulation step
        
        Args:
            n_warmup: Number of warmup steps
            n_measure: Number of measurement steps
            
        Returns:
            Dict containing benchmark results
        """
        # Warmup
        for _ in range(n_warmup):
            self.simulate_step()
        
        # Measure
        step_times = []
        for _ in range(n_measure):
            step_time = self.simulate_step()
            step_times.append(step_time)
        
        # Calculate statistics
        avg_time = sum(step_times) / len(step_times)
        min_time = min(step_times)
        max_time = max(step_times)
        std_time = (sum((t - avg_time) ** 2 for t in step_times) / len(step_times)) ** 0.5
        
        return {
            'average_step_time': avg_time,
            'min_step_time': min_time,
            'max_step_time': max_time,
            'std_step_time': std_time,
            'steps_per_second': 1.0 / avg_time if avg_time > 0 else 0,
            'neurons_per_second': self.config.n_neurons / avg_time if avg_time > 0 else 0,
            'measurement_steps': n_measure,
            'warmup_steps': n_warmup
        }
    
    def get_simulation_info(self) -> dict:
        """
        Get information about the current simulation state
        
        Returns:
            Dict containing simulation information
        """
        return {
            'config': self.config.to_dict(),
            'cuda_loaded': self.cuda_manager.is_loaded,
            'using_optimized_kernels': self.cuda_manager.using_optimized_kernels,
            'optimized_brain_ptr': self.cuda_manager.optimized_brain_ptr,
            'num_areas': self.area_manager.num_areas,
            'total_neurons': self.area_manager.total_neurons,
            'total_active_neurons': self.area_manager.total_active_neurons,
            'memory_info': self.area_manager.memory_manager.get_memory_info(),
            'areas_info': self.area_manager.get_all_areas_info()
        }
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.area_manager.reset_areas()
        self.metrics.reset()
    
    def validate_simulation(self) -> bool:
        """
        Validate that the simulation is properly configured
        
        Returns:
            bool: True if simulation is valid, False otherwise
        """
        try:
            # Check configuration
            if self.config.n_neurons <= 0:
                print("❌ Invalid n_neurons")
                return False
            
            if not 0 < self.config.active_percentage <= 1:
                print("❌ Invalid active_percentage")
                return False
            
            if self.config.n_areas <= 0:
                print("❌ Invalid n_areas")
                return False
            
            # Check areas
            if len(self.area_manager.areas) != self.config.n_areas:
                print("❌ Area count mismatch")
                return False
            
            # Check CUDA if enabled
            if self.config.use_cuda_kernels and not self.cuda_manager.is_loaded:
                print("⚠️  CUDA kernels requested but not loaded")
                # This is a warning, not an error
            
            print("✅ Simulation validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Simulation validation failed: {e}")
            return False
