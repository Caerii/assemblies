#!/usr/bin/env python3
"""
Universal Brain Simulator - Lightweight Thin Client
==================================================

This is a lightweight, easy-to-use client for the Universal Brain Simulator.
It provides simple, high-level interfaces for common use cases while hiding
the complexity of the underlying modular architecture.

Usage Examples:
    # Quick start
    from universal_brain_simulator.client import BrainSimulator
    
    sim = BrainSimulator(neurons=1000000, active_percentage=0.01)
    sim.run(steps=100)
    
    # Advanced usage
    sim = BrainSimulator(
        neurons=1000000,
        active_percentage=0.01,
        areas=5,
        use_optimized_cuda=True
    )
    results = sim.benchmark()
"""

from typing import Dict, Any, Optional, List, Callable
from .config import SimulationConfig
from .universal_brain_simulator import UniversalBrainSimulator


class BrainSimulator:
    """
    Lightweight thin client for the Universal Brain Simulator
    
    This class provides a simple, high-level interface that hides
    the complexity of the underlying modular architecture.
    """
    
    def __init__(self, 
                 neurons: int = 1000000,
                 active_percentage: float = 0.01,
                 areas: int = 5,
                 seed: int = 42,
                 use_gpu: bool = True,
                 use_optimized_cuda: bool = True,
                 use_cuda_kernels: bool = True,
                 memory_efficient: bool = True,
                 sparse_mode: bool = True,
                 enable_profiling: bool = True):
        """
        Initialize the brain simulator with simple parameters
        
        Args:
            neurons: Total number of neurons
            active_percentage: Percentage of neurons that are active (0.0-1.0)
            areas: Number of brain areas
            seed: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration
            use_optimized_cuda: Whether to use optimized CUDA kernels (O(N log K))
            use_cuda_kernels: Whether to use CUDA kernels (False = CuPy only)
            memory_efficient: Whether to use memory-efficient mode
            sparse_mode: Whether to use sparse memory representation
            enable_profiling: Whether to enable performance profiling
        """
        # Create configuration
        self.config = SimulationConfig(
            n_neurons=neurons,
            active_percentage=active_percentage,
            n_areas=areas,
            seed=seed,
            use_gpu=use_gpu,
            use_cuda_kernels=use_cuda_kernels,
            use_optimized_kernels=use_optimized_cuda,
            memory_efficient=memory_efficient,
            sparse_mode=sparse_mode,
            enable_profiling=enable_profiling
        )
        
        # Initialize the simulator
        self.simulator = UniversalBrainSimulator(self.config)
        
        # Store parameters for easy access
        self.neurons = neurons
        self.active_percentage = active_percentage
        self.areas = areas
        self.active_neurons = int(neurons * active_percentage)
    
    def run(self, steps: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a simulation for the specified number of steps
        
        Args:
            steps: Number of simulation steps to run
            verbose: Whether to print progress information
            
        Returns:
            Dict containing simulation results and performance metrics
        """
        print(f"🧠 Running {steps} simulation steps...")
        print(f"   Neurons: {self.neurons:,}")
        print(f"   Active: {self.active_percentage:.3f} ({self.active_neurons:,} per area)")
        print(f"   Areas: {self.areas}")
        
        # Run simulation
        total_time = self.simulator.simulate(n_steps=steps, verbose=verbose)
        
        # Get results
        stats = self.simulator.get_performance_stats()
        
        # Format results
        results = {
            'simulation': {
                'steps': steps,
                'total_time': total_time,
                'neurons': self.neurons,
                'active_percentage': self.active_percentage,
                'areas': self.areas
            },
            'performance': stats,
            'summary': {
                'steps_per_second': stats.get('steps_per_second', 0),
                'steps_per_sec': stats.get('steps_per_second', 0),  # Alias for compatibility
                'neurons_per_second': stats.get('neurons_per_second', 0),
                'memory_usage_gb': stats.get('memory_usage_gb', 0),
                'cuda_kernels_used': stats.get('cuda_kernels_used', False),
                'cuda_used': stats.get('cuda_kernels_used', False),  # Alias for compatibility
                'optimized': self.config.use_optimized_kernels
            }
        }
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def simulate_step(self) -> float:
        """
        Run a single simulation step
        
        Returns:
            Time taken for the step in seconds
        """
        return self.simulator.simulate_step()
    
    def benchmark(self, warmup_steps: int = 5, measure_steps: int = 10) -> Dict[str, Any]:
        """
        Run a benchmark to measure performance
        
        Args:
            warmup_steps: Number of warmup steps
            measure_steps: Number of measurement steps
            
        Returns:
            Dict containing benchmark results
        """
        print(f"⚡ Running benchmark ({warmup_steps} warmup + {measure_steps} measure)...")
        
        benchmark_results = self.simulator.benchmark_step(warmup_steps, measure_steps)
        
        # Format benchmark results
        results = {
            'benchmark': benchmark_results,
            'configuration': {
                'neurons': self.neurons,
                'active_percentage': self.active_percentage,
                'areas': self.areas,
                'optimized_cuda': self.config.use_optimized_kernels
            },
            'performance': {
                'steps_per_second': benchmark_results['steps_per_second'],
                'steps_per_sec': benchmark_results['steps_per_second'],  # Alias for compatibility
                'neurons_per_second': benchmark_results['neurons_per_second'],
                'average_step_time_ms': benchmark_results['average_step_time'] * 1000,
                'min_step_time_ms': benchmark_results['min_step_time'] * 1000,
                'max_step_time_ms': benchmark_results['max_step_time'] * 1000
            }
        }
        
        self._print_benchmark_summary(results)
        
        return results
    
    def profile(self, steps: int = 100, save_to_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a detailed performance profile
        
        Args:
            steps: Number of steps to profile
            save_to_file: Optional filename to save profile data
            
        Returns:
            Dict containing detailed profiling data
        """
        print(f"📊 Running detailed profile for {steps} steps...")
        
        # Run simulation with profiling
        self.simulator.simulate(n_steps=steps, verbose=False)
        
        # Get detailed data
        stats = self.simulator.get_performance_stats()
        profile_data = self.simulator.get_profile_data()
        simulation_info = self.simulator.get_simulation_info()
        
        # Combine results
        results = {
            'configuration': self.config.to_dict(),
            'performance_stats': stats,
            'profile_data': profile_data,
            'simulation_info': simulation_info,
            'memory_info': self.simulator.get_memory_info(),
            'areas_info': self.simulator.get_areas_info()
        }
        
        # Save to file if requested
        if save_to_file:
            self.simulator.save_profile_data(save_to_file)
            print(f"📁 Profile data saved to {save_to_file}")
        
        self._print_profile_summary(results)
        
        return results
    
    def run_with_callback(self, steps: int, callback: Callable[[int, float], None], 
                         callback_interval: int = 1):
        """
        Run simulation with a custom callback function
        
        Args:
            steps: Number of steps to run
            callback: Function to call after each step (receives step_number, step_time)
            callback_interval: How often to call the callback
        """
        print(f"🔄 Running {steps} steps with custom callback...")
        
        self.simulator.simulate_with_callback(steps, callback, callback_interval)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the current simulator configuration
        
        Returns:
            Dict containing simulator information
        """
        return {
            'configuration': self.config.to_dict(),
            'simulation_info': self.simulator.get_simulation_info(),
            'memory_info': self.simulator.get_memory_info(),
            'areas_info': self.simulator.get_areas_info()
        }
    
    def reset(self):
        """Reset the simulator to initial state"""
        self.simulator.reset_simulation()
        print("🔄 Simulator reset to initial state")
    
    def cleanup(self):
        """
        Explicitly cleanup all resources
        
        This should be called when done with the simulator to ensure
        proper cleanup and avoid memory errors.
        """
        if hasattr(self, 'simulator'):
            self.simulator.cleanup()
        print("🧹 Client cleanup completed")
    
    def __del__(self):
        """Cleanup resources when client is destroyed"""
        try:
            if hasattr(self, 'simulator'):
                self.simulator.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction
    
    def validate(self) -> bool:
        """
        Validate the simulator configuration
        
        Returns:
            bool: True if configuration is valid
        """
        return self.simulator.validate_simulation()
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of simulation results"""
        perf = results['summary']
        print(f"\n📊 SIMULATION SUMMARY")
        print(f"   Steps/sec: {perf['steps_per_second']:.1f}")
        print(f"   Neurons/sec: {perf['neurons_per_second']:,.0f}")
        print(f"   Memory: {perf['memory_usage_gb']:.2f}GB")
        print(f"   CUDA: {'✅' if perf['cuda_used'] else '❌'}")
        print(f"   Optimized: {'✅' if perf['optimized'] else '❌'}")
    
    def _print_benchmark_summary(self, results: Dict[str, Any]):
        """Print a summary of benchmark results"""
        perf = results['performance']
        config = results['configuration']
        
        print(f"\n⚡ BENCHMARK SUMMARY")
        print(f"   Configuration: {config['neurons']:,} neurons, {config['areas']} areas")
        print(f"   Optimized CUDA: {'✅' if config['optimized_cuda'] else '❌'}")
        print(f"   Steps/sec: {perf['steps_per_second']:.1f}")
        print(f"   Neurons/sec: {perf['neurons_per_second']:,.0f}")
        print(f"   Avg step time: {perf['average_step_time_ms']:.2f}ms")
        print(f"   Min step time: {perf['min_step_time_ms']:.2f}ms")
        print(f"   Max step time: {perf['max_step_time_ms']:.2f}ms")
    
    def _print_profile_summary(self, results: Dict[str, Any]):
        """Print a summary of profiling results"""
        stats = results['performance_stats']
        memory = results['memory_info']
        
        print(f"\n📊 PROFILE SUMMARY")
        print(f"   Steps/sec: {stats.get('steps_per_second', 0):.1f}")
        print(f"   Neurons/sec: {stats.get('neurons_per_second', 0):,.0f}")
        print(f"   Memory usage: {memory.get('used_gb', 0):.2f}GB")
        print(f"   Memory utilization: {memory.get('utilization_percent', 0):.1f}%")
        print(f"   CUDA kernels: {'✅' if stats.get('cuda_kernels_used', False) else '❌'}")
        print(f"   CuPy used: {'✅' if stats.get('cupy_used', False) else '❌'}")
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"BrainSimulator(neurons={self.neurons:,}, "
                f"active={self.active_percentage:.3f}, "
                f"areas={self.areas}, "
                f"optimized={'✅' if self.config.use_optimized_kernels else '❌'})")
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        return (f"Brain Simulator:\n"
                f"  Neurons: {self.neurons:,}\n"
                f"  Active: {self.active_percentage:.3f} ({self.active_neurons:,} per area)\n"
                f"  Areas: {self.areas}\n"
                f"  Optimized CUDA: {'✅' if self.config.use_optimized_kernels else '❌'}\n"
                f"  GPU: {'✅' if self.config.use_gpu else '❌'}")


# Convenience functions for quick access
def quick_sim(neurons: int = 1000000, steps: int = 100, optimized: bool = True) -> Dict[str, Any]:
    """
    Quick simulation with minimal parameters
    
    Args:
        neurons: Number of neurons
        steps: Number of steps
        optimized: Whether to use optimized CUDA kernels
        
    Returns:
        Dict containing simulation results
    """
    sim = BrainSimulator(neurons=neurons, use_optimized_cuda=optimized)
    return sim.run(steps=steps, verbose=True)


def quick_benchmark(neurons: int = 1000000, optimized: bool = True) -> Dict[str, Any]:
    """
    Quick benchmark with minimal parameters
    
    Args:
        neurons: Number of neurons
        optimized: Whether to use optimized CUDA kernels
        
    Returns:
        Dict containing benchmark results
    """
    sim = BrainSimulator(neurons=neurons, use_optimized_cuda=optimized)
    return sim.benchmark()


def compare_configurations(configs: List[Dict[str, Any]], steps: int = 100) -> List[Dict[str, Any]]:
    """
    Compare multiple simulator configurations
    
    Args:
        configs: List of configuration dictionaries
        steps: Number of steps to run for each configuration
        
    Returns:
        List of results for each configuration
    """
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Testing Configuration {i+1}/{len(configs)}")
        print(f"   {config}")
        
        sim = BrainSimulator(**config)
        result = sim.run(steps=steps, verbose=False)
        results.append({
            'config': config,
            'result': result
        })
    
    # Print comparison summary
    print(f"\n📊 CONFIGURATION COMPARISON")
    print("=" * 80)
    print(f"{'Config':<20} {'Steps/sec':<10} {'Neurons/sec':<15} {'Memory GB':<10} {'CUDA':<6} {'Optimized':<10}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        config = result['config']
        perf = result['result']['summary']
        
        config_name = f"Config {i+1}"
        print(f"{config_name:<20} {perf['steps_per_second']:<10.1f} {perf['neurons_per_second']:<15,.0f} {perf['memory_usage_gb']:<10.2f} {'✅' if perf['cuda_used'] else '❌':<6} {'✅' if perf['optimized'] else '❌':<10}")
    
    return results
