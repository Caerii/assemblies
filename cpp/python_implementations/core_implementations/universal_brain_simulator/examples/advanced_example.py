#!/usr/bin/env python3
"""
Advanced Example - Universal Brain Simulator
==========================================

This example shows advanced usage patterns including custom callbacks,
memory management, and integration with external systems.
"""

import time
import json
from typing import Dict, Any, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from universal_brain_simulator.client import BrainSimulator


class CustomSimulationMonitor:
    """Custom monitor class for advanced simulation tracking"""
    
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        self.callback_count = 0
    
    def step_callback(self, step: int, step_time: float):
        """Custom callback for each simulation step"""
        self.step_times.append(step_time)
        self.callback_count += 1
        
        if step % 20 == 0:
            avg_time = sum(self.step_times[-20:]) / min(20, len(self.step_times))
            print(f"   Step {step}: {step_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from the monitoring"""
        if not self.step_times:
            return {}
        
        return {
            'total_steps': len(self.step_times),
            'avg_step_time': sum(self.step_times) / len(self.step_times),
            'min_step_time': min(self.step_times),
            'max_step_time': max(self.step_times),
            'total_time': sum(self.step_times)
        }


def main():
    print("🔬 ADVANCED EXAMPLE")
    print("=" * 50)
    
    # Advanced configuration
    print("\n📋 Advanced Configuration")
    sim = BrainSimulator(
        neurons=2000000,           # 2 million neurons
        active_percentage=0.005,   # 0.5% active (sparse)
        areas=10,                  # 10 brain areas
        seed=12345,                # Fixed seed for reproducibility
        use_gpu=True,
        use_optimized_cuda=True,
        memory_efficient=True,
        sparse_mode=True,
        enable_profiling=True
    )
    
    print(f"   Configuration: {sim}")
    
    # Custom monitoring
    print("\n📊 Custom Monitoring")
    monitor = CustomSimulationMonitor()
    
    # Run with custom callback
    print("   Running with custom monitoring...")
    sim.run_with_callback(
        steps=100,
        callback=monitor.step_callback,
        callback_interval=1
    )
    
    # Get monitoring statistics
    monitor_stats = monitor.get_statistics()
    print(f"   Monitor stats: {monitor_stats}")
    
    # Detailed profiling
    print("\n📊 Detailed Profiling")
    profile_results = sim.profile(
        steps=50,
        save_to_file="advanced_profile.json"
    )
    
    # Memory analysis
    print("\n💾 Memory Analysis")
    memory_info = sim.get_info()['memory_info']
    print(f"   Memory usage: {memory_info.get('used_gb', 0):.2f}GB")
    print(f"   Memory utilization: {memory_info.get('utilization_percent', 0):.1f}%")
    
    # Performance analysis
    print("\n⚡ Performance Analysis")
    perf_stats = profile_results['performance_stats']
    print(f"   Steps/sec: {perf_stats.get('steps_per_second', 0):.1f}")
    print(f"   Neurons/sec: {perf_stats.get('neurons_per_second', 0):,.0f}")
    print(f"   CUDA kernels: {'✅' if perf_stats.get('cuda_kernels_used', False) else '❌'}")
    print(f"   CuPy used: {'✅' if perf_stats.get('cupy_used', False) else '❌'}")
    
    # Configuration validation
    print("\n✅ Configuration Validation")
    is_valid = sim.validate()
    print(f"   Configuration valid: {'✅' if is_valid else '❌'}")
    
    # Reset and re-run
    print("\n🔄 Reset and Re-run")
    sim.reset()
    
    # Quick benchmark after reset
    benchmark_results = sim.benchmark(warmup_steps=2, measure_steps=5)
    print(f"   Post-reset benchmark: {benchmark_results['performance']['steps_per_second']:.1f} steps/sec")
    
    # Multiple simulation runs
    print("\n🔄 Multiple Simulation Runs")
    run_results = []
    
    for i in range(3):
        print(f"   Run {i+1}/3...")
        results = sim.run(steps=20, verbose=False)
        run_results.append(results['summary']['steps_per_second'])
    
    avg_performance = sum(run_results) / len(run_results)
    print(f"   Average performance across runs: {avg_performance:.1f} steps/sec")
    print(f"   Performance variance: {max(run_results) - min(run_results):.1f} steps/sec")
    
    # Save results
    print("\n💾 Saving Results")
    final_results = {
        'configuration': sim.get_info()['configuration'],
        'monitor_stats': monitor_stats,
        'profile_results': profile_results,
        'run_results': run_results,
        'average_performance': avg_performance
    }
    
    with open('advanced_simulation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print("   Results saved to advanced_simulation_results.json")
    
    print(f"\n🎯 Advanced example complete!")
    print(f"   Total steps monitored: {monitor.callback_count}")
    print(f"   Average performance: {avg_performance:.1f} steps/sec")
    print(f"   Memory efficiency: {'✅' if memory_info.get('utilization_percent', 0) < 80 else '⚠️'}")


if __name__ == "__main__":
    main()
