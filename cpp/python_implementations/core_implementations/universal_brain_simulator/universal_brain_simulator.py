#!/usr/bin/env python3
"""
Universal Brain Simulator - Main Orchestrator
=============================================

This is the main orchestrator class that coordinates all components
of the universal brain simulator system. Each component has a single,
clear responsibility, making the system highly maintainable and testable.

Architecture:
- Configuration: SimulationConfig
- Performance: PerformanceMonitor  
- CUDA: CUDAManager
- Memory: MemoryManager
- Areas: AreaManager
- Simulation: SimulationEngine
"""

from typing import Dict, Any
from .config import SimulationConfig
from .metrics import PerformanceMonitor
from .cuda_manager import CUDAManager
from .memory_manager import MemoryManager
from .area_manager import AreaManager
from .simulation_engine import SimulationEngine
from .utils import print_initialization_status


class UniversalBrainSimulator:
    """
    Universal Brain Simulator - Main Orchestrator
    
    This is the high-level interface that coordinates all components.
    Each component has a single, clear responsibility:
    
    - Configuration: Manages simulation parameters
    - Performance: Tracks metrics and profiling
    - CUDA: Handles CUDA kernel loading and management
    - Memory: Manages GPU/CPU memory allocation
    - Areas: Manages brain areas and data structures
    - Simulation: Orchestrates the simulation logic
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize all components of the universal brain simulator
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        
        # Initialize components in dependency order
        self.metrics = PerformanceMonitor(config)
        self.cuda_manager = CUDAManager(config)
        self.memory_manager = MemoryManager(config, self.cuda_manager)
        self.area_manager = AreaManager(config, self.memory_manager, self.cuda_manager)
        self.simulation_engine = SimulationEngine(
            config, self.cuda_manager, self.area_manager, self.metrics
        )
        
        # Initialize everything
        self._initialize()
    
    def _initialize(self):
        """Initialize all components in correct order"""
        # Load CUDA kernels first
        self.cuda_manager.load_kernels()
        
        # Create optimized brain instance if using optimized kernels
        if self.cuda_manager.using_optimized_kernels:
            self.cuda_manager.create_optimized_brain(
                self.config.n_neurons,
                self.config.n_areas,
                self.config.k_active,
                self.config.seed
            )
        
        # Initialize areas and memory
        self.area_manager.initialize_areas()
        self.memory_manager.initialize_cuda_pools()
        
        # Set technology flags in metrics
        memory_info = self.memory_manager.get_memory_info()
        self.metrics.set_technology_flags(
            cuda_kernels_used=self.cuda_manager.is_loaded,
            cupy_used=memory_info.get('gpu_available', False),
            numpy_fallback=not memory_info.get('gpu_available', False)
        )
        
        # Print initialization status
        kernel_type = None
        if self.cuda_manager.is_loaded:
            kernel_type = "Optimized (O(N log K))" if self.cuda_manager.using_optimized_kernels else "Original (O(N²))"
        
        print_initialization_status(
            self.config, 
            self.cuda_manager.is_loaded, 
            kernel_type
        )
    
    def simulate_step(self) -> float:
        """
        Simulate one step - delegate to simulation engine
        
        Returns:
            float: Time taken for this step in seconds
        """
        return self.simulation_engine.simulate_step()
    
    def simulate(self, n_steps: int = 100, verbose: bool = True, profile_interval: int = 10) -> float:
        """
        Simulate multiple steps - delegate to simulation engine
        
        Args:
            n_steps: Number of steps to simulate
            verbose: Whether to print progress information
            profile_interval: How often to print progress updates
            
        Returns:
            float: Total time for the simulation
        """
        return self.simulation_engine.simulate(n_steps, verbose, profile_interval)
    
    def simulate_with_callback(self, n_steps: int, callback_func, callback_interval: int = 1):
        """
        Simulate with a callback function - delegate to simulation engine
        
        Args:
            n_steps: Number of steps to simulate
            callback_func: Function to call after each step
            callback_interval: How often to call the callback function
        """
        self.simulation_engine.simulate_with_callback(n_steps, callback_func, callback_interval)
    
    def benchmark_step(self, n_warmup: int = 5, n_measure: int = 10) -> Dict[str, Any]:
        """
        Benchmark a single simulation step - delegate to simulation engine
        
        Args:
            n_warmup: Number of warmup steps
            n_measure: Number of measurement steps
            
        Returns:
            Dict containing benchmark results
        """
        return self.simulation_engine.benchmark_step(n_warmup, n_measure)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance stats - delegate to metrics
        
        Returns:
            Dict containing performance statistics
        """
        return self.metrics.get_stats()
    
    def get_profile_data(self) -> Dict[str, Any]:
        """
        Get profile data - delegate to metrics
        
        Returns:
            Dict containing raw profiling data
        """
        return self.metrics.get_profile_data()
    
    def save_profile_data(self, filename: str):
        """
        Save profile data - delegate to metrics
        
        Args:
            filename: Path to save the profile data
        """
        self.metrics.save_profile(filename)
    
    def get_simulation_info(self) -> Dict[str, Any]:
        """
        Get information about the current simulation state
        
        Returns:
            Dict containing simulation information
        """
        return self.simulation_engine.get_simulation_info()
    
    def validate_simulation(self) -> bool:
        """
        Validate that the simulation is properly configured
        
        Returns:
            bool: True if simulation is valid, False otherwise
        """
        return self.simulation_engine.validate_simulation()
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.simulation_engine.reset_simulation()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get detailed memory information
        
        Returns:
            Dict containing memory information
        """
        return self.memory_manager.get_memory_info()
    
    def get_areas_info(self) -> list:
        """
        Get information about all brain areas
        
        Returns:
            List of area information dictionaries
        """
        return self.area_manager.get_all_areas_info()
    
    def cleanup(self):
        """
        Explicitly cleanup all resources
        
        This should be called before destroying the simulator to ensure
        proper cleanup order and avoid memory errors.
        """
        print("🧹 Starting explicit cleanup...")
        try:
            # Cleanup in reverse order of initialization
            # 1. First cleanup CUDA resources (C++ objects)
            self.cuda_manager.cleanup()
            
            # 2. Then cleanup Python memory (CuPy arrays)
            self.memory_manager.cleanup_memory()
            
            print("✅ Explicit cleanup completed successfully")
            
        except Exception as e:
            print(f"⚠️  Explicit cleanup error: {e}")
            raise
    
    def __del__(self):
        """Cleanup - delegate to managers in proper order"""
        try:
            # Cleanup in reverse order of initialization
            # 1. First cleanup CUDA resources (C++ objects)
            if hasattr(self, 'cuda_manager'):
                self.cuda_manager.cleanup()
            
            # 2. Then cleanup Python memory (CuPy arrays)
            if hasattr(self, 'memory_manager'):
                self.memory_manager.cleanup_memory()
                
        except Exception as e:
            # Log cleanup errors but don't crash
            print(f"⚠️  Cleanup error in __del__: {e}")
            pass
    
    def __repr__(self) -> str:
        """String representation of the simulator"""
        return (f"UniversalBrainSimulator("
                f"neurons={self.config.n_neurons:,}, "
                f"areas={self.config.n_areas}, "
                f"active_pct={self.config.active_percentage:.3f}, "
                f"cuda={'✅' if self.cuda_manager.is_loaded else '❌'}, "
                f"optimized={'✅' if self.cuda_manager.using_optimized_kernels else '❌'})")
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        memory_info = self.memory_manager.get_memory_info()
        return (f"Universal Brain Simulator:\n"
                f"  Neurons: {self.config.n_neurons:,}\n"
                f"  Areas: {self.config.n_areas}\n"
                f"  Active: {self.config.active_percentage:.3f}\n"
                f"  CUDA: {'✅' if self.cuda_manager.is_loaded else '❌'}\n"
                f"  Optimized: {'✅' if self.cuda_manager.using_optimized_kernels else '❌'}\n"
                f"  Memory: {memory_info.get('used_gb', 0):.2f}GB")
