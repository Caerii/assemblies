#!/usr/bin/env python3
"""
Simulation Commands for Universal Brain Simulator
================================================

This module contains different simulation execution strategies,
using the Command pattern to simplify step execution logic.
"""

import time
from .config import SimulationConfig
from .cuda_manager import CUDAManager
from .area_manager import AreaManager
from .metrics import PerformanceMonitor


class SimulationCommand:
    """
    Abstract base class for simulation commands
    
    Each command implements a different way to execute a simulation step.
    """
    
    def __init__(self, config: SimulationConfig, cuda_manager: CUDAManager, 
                 area_manager: AreaManager, metrics: PerformanceMonitor):
        self.config = config
        self.cuda_manager = cuda_manager
        self.area_manager = area_manager
        self.metrics = metrics
    
    def execute(self) -> bool:
        """Execute a simulation step"""
        raise NotImplementedError
    
    def get_timing(self) -> float:
        """Get the timing for the last execution"""
        raise NotImplementedError
    
    def get_command_name(self) -> str:
        """Get the name of this command"""
        raise NotImplementedError


class OptimizedBrainCommand(SimulationCommand):
    """
    Command for optimized brain simulator
    
    This command delegates to the optimized brain simulator
    which handles all operations internally.
    """
    
    def __init__(self, config: SimulationConfig, cuda_manager: CUDAManager, 
                 area_manager: AreaManager, metrics: PerformanceMonitor):
        super().__init__(config, cuda_manager, area_manager, metrics)
        self._last_timing = 0.0
    
    def execute(self) -> bool:
        """Execute using optimized brain simulator"""
        try:
            start_time = time.perf_counter()
            success = self.cuda_manager.simulate_step_optimized()
            self._last_timing = time.perf_counter() - start_time
            
            if not success:
                print("   ⚠️  Optimized brain step failed, falling back to area-based simulation")
                return self._fallback_to_area_based()
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Optimized brain step error: {e}, falling back to area-based simulation")
            return self._fallback_to_area_based()
    
    def _fallback_to_area_based(self) -> bool:
        """Fallback to area-based simulation"""
        try:
            start_time = time.perf_counter()
            success = self._simulate_areas()
            self._last_timing = time.perf_counter() - start_time
            return success
        except Exception as e:
            print(f"   ⚠️  Area-based fallback failed: {e}")
            return False
    
    def _simulate_areas(self) -> bool:
        """Simulate using area-based approach"""
        try:
            for area_idx in range(self.area_manager.num_areas):
                # Generate candidates
                candidates = self.area_manager.generate_candidates(area_idx)
                
                # Select top-k winners
                winners = self.area_manager.select_top_k(candidates, self.config.k_active)
                
                # Update area state
                self.area_manager.update_area_state(area_idx, winners)
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Area simulation failed: {e}")
            return False
    
    def get_timing(self) -> float:
        return self._last_timing
    
    def get_command_name(self) -> str:
        return "Optimized Brain Simulator"


class AreaBasedCommand(SimulationCommand):
    """
    Command for area-based simulation
    
    This command uses the area manager to simulate each area individually.
    """
    
    def __init__(self, config: SimulationConfig, cuda_manager: CUDAManager, 
                 area_manager: AreaManager, metrics: PerformanceMonitor):
        super().__init__(config, cuda_manager, area_manager, metrics)
        self._last_timing = 0.0
    
    def execute(self) -> bool:
        """Execute using area-based simulation"""
        try:
            start_time = time.perf_counter()
            success = self._simulate_areas()
            self._last_timing = time.perf_counter() - start_time
            return success
            
        except Exception as e:
            print(f"   ⚠️  Area-based simulation failed: {e}")
            return False
    
    def _simulate_areas(self) -> bool:
        """Simulate using area-based approach"""
        try:
            for area_idx in range(self.area_manager.num_areas):
                # Generate candidates
                candidates = self.area_manager.generate_candidates(area_idx)
                
                # Select top-k winners
                winners = self.area_manager.select_top_k(candidates, self.config.k_active)
                
                # Update area state
                self.area_manager.update_area_state(area_idx, winners)
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Area simulation failed: {e}")
            return False
    
    def get_timing(self) -> float:
        return self._last_timing
    
    def get_command_name(self) -> str:
        return "Area-Based Simulation"


class HybridCommand(SimulationCommand):
    """
    Command for hybrid simulation
    
    This command tries optimized brain first, then falls back to area-based.
    """
    
    def __init__(self, config: SimulationConfig, cuda_manager: CUDAManager, 
                 area_manager: AreaManager, metrics: PerformanceMonitor):
        super().__init__(config, cuda_manager, area_manager, metrics)
        self._last_timing = 0.0
        self._using_optimized = False
    
    def execute(self) -> bool:
        """Execute using hybrid approach"""
        try:
            start_time = time.perf_counter()
            
            # Try optimized brain first if available
            if (self.cuda_manager.optimized_brain_ptr is not None and 
                self.cuda_manager.using_optimized_kernels):
                
                success = self.cuda_manager.simulate_step_optimized()
                if success:
                    self._using_optimized = True
                    self._last_timing = time.perf_counter() - start_time
                    return True
                else:
                    print("   ⚠️  Optimized brain failed, trying area-based")
            
            # Fallback to area-based
            self._using_optimized = False
            success = self._simulate_areas()
            self._last_timing = time.perf_counter() - start_time
            return success
            
        except Exception as e:
            print(f"   ⚠️  Hybrid simulation failed: {e}")
            return False
    
    def _simulate_areas(self) -> bool:
        """Simulate using area-based approach"""
        try:
            for area_idx in range(self.area_manager.num_areas):
                # Generate candidates
                candidates = self.area_manager.generate_candidates(area_idx)
                
                # Select top-k winners
                winners = self.area_manager.select_top_k(candidates, self.config.k_active)
                
                # Update area state
                self.area_manager.update_area_state(area_idx, winners)
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Area simulation failed: {e}")
            return False
    
    def get_timing(self) -> float:
        return self._last_timing
    
    def get_command_name(self) -> str:
        if self._using_optimized:
            return "Hybrid (Optimized Brain)"
        else:
            return "Hybrid (Area-Based)"


# =============================================================================
# COMMAND FACTORY
# =============================================================================

def create_simulation_command(config: SimulationConfig, cuda_manager: CUDAManager, 
                            area_manager: AreaManager, metrics: PerformanceMonitor) -> SimulationCommand:
    """
    Create the appropriate simulation command based on configuration
    
    Args:
        config: Simulation configuration
        cuda_manager: CUDA manager instance
        area_manager: Area manager instance
        metrics: Performance monitor instance
        
    Returns:
        SimulationCommand: The appropriate command instance
    """
    # Priority order: Optimized Brain > Hybrid > Area-Based
    
    if (config.use_cuda_kernels and 
        cuda_manager.using_optimized_kernels and 
        cuda_manager.optimized_brain_ptr is not None):
        return OptimizedBrainCommand(config, cuda_manager, area_manager, metrics)
    
    elif config.use_cuda_kernels and cuda_manager.is_loaded:
        return HybridCommand(config, cuda_manager, area_manager, metrics)
    
    else:
        return AreaBasedCommand(config, cuda_manager, area_manager, metrics)


def get_available_commands() -> list:
    """
    Get list of available command names
    
    Returns:
        list: List of available command names
    """
    return [
        "Optimized Brain Simulator",
        "Area-Based Simulation", 
        "Hybrid (Optimized Brain)",
        "Hybrid (Area-Based)"
    ]


def print_command_info(command: SimulationCommand):
    """
    Print information about a command
    
    Args:
        command: The command to print info for
    """
    print(f"   🎯 Using simulation command: {command.get_command_name()}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_command_compatibility(config: SimulationConfig, cuda_manager: CUDAManager) -> bool:
    """
    Validate that the selected command is compatible with the configuration
    
    Args:
        config: Simulation configuration
        cuda_manager: CUDA manager instance
        
    Returns:
        bool: True if command is compatible, False otherwise
    """
    if config.use_cuda_kernels and cuda_manager.using_optimized_kernels:
        return cuda_manager.optimized_brain_ptr is not None
    
    elif config.use_cuda_kernels:
        return cuda_manager.is_loaded
    
    else:
        return True  # Area-based is always available


def get_command_performance_estimate(command: SimulationCommand) -> str:
    """
    Get a performance estimate for a command
    
    Args:
        command: The command to estimate performance for
        
    Returns:
        str: Performance estimate description
    """
    if isinstance(command, OptimizedBrainCommand):
        return "Highest performance - optimized CUDA brain"
    elif isinstance(command, HybridCommand):
        return "High performance - hybrid approach with fallback"
    elif isinstance(command, AreaBasedCommand):
        return "Medium performance - area-based simulation"
    else:
        return "Unknown performance"


def benchmark_commands(config: SimulationConfig, cuda_manager: CUDAManager, 
                      area_manager: AreaManager, metrics: PerformanceMonitor, 
                      steps: int = 100) -> dict:
    """
    Benchmark different simulation commands
    
    Args:
        config: Simulation configuration
        cuda_manager: CUDA manager instance
        area_manager: Area manager instance
        metrics: Performance monitor instance
        steps: Number of steps to benchmark
        
    Returns:
        dict: Benchmark results for each command
    """
    results = {}
    
    # Test each command type
    command_types = [
        ("Optimized Brain", lambda: OptimizedBrainCommand(config, cuda_manager, area_manager, metrics)),
        ("Area-Based", lambda: AreaBasedCommand(config, cuda_manager, area_manager, metrics)),
        ("Hybrid", lambda: HybridCommand(config, cuda_manager, area_manager, metrics))
    ]
    
    for name, command_factory in command_types:
        try:
            command = command_factory()
            
            # Benchmark this command
            start_time = time.perf_counter()
            successful_steps = 0
            
            for _ in range(steps):
                if command.execute():
                    successful_steps += 1
            
            total_time = time.perf_counter() - start_time
            steps_per_sec = successful_steps / total_time if total_time > 0 else 0
            
            results[name] = {
                'steps_per_second': steps_per_sec,
                'successful_steps': successful_steps,
                'total_time': total_time,
                'command_name': command.get_command_name()
            }
            
        except Exception as e:
            results[name] = {
                'error': str(e),
                'steps_per_second': 0,
                'successful_steps': 0,
                'total_time': 0
            }
    
    return results
