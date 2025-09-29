#!/usr/bin/env python3
"""
Test Optimized vs Original Universal Brain Simulator
===================================================

This script compares the performance of:
1. Original universal_brain_simulator.py (O(N²) top-k selection)
2. Optimized universal_brain_simulator_optimized.py (O(N log K) top-k selection)

At different scales to demonstrate the algorithmic improvements.
"""

import time
import sys
import os
import json
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import both implementations
try:
    from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig as OriginalConfig
    from universal_brain_simulator_optimized import UniversalBrainSimulatorOptimized, SimulationConfig as OptimizedConfig
    print("✅ Both implementations imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class PerformanceComparison:
    """Compare performance between original and optimized implementations"""
    
    def __init__(self):
        self.results = []
    
    def test_scale(self, n_neurons: int, active_percentage: float, n_areas: int = 5, n_steps: int = 10):
        """Test both implementations at a specific scale"""
        print(f"\n🧪 Testing Scale: {n_neurons:,} neurons, {active_percentage*100:.4f}% active")
        print("=" * 70)
        
        k_active = int(n_neurons * active_percentage)
        print(f"   Active neurons per area: {k_active:,}")
        print(f"   Total active neurons: {k_active * n_areas:,}")
        print(f"   Simulation steps: {n_steps}")
        
        # Test configurations
        configs = [
            {
                "name": "Original (O(N²))",
                "class": UniversalBrainSimulator,
                "config": OriginalConfig(
                    n_neurons=n_neurons,
                    active_percentage=active_percentage,
                    n_areas=n_areas,
                    use_gpu=True,
                    use_cuda_kernels=True,
                    memory_efficient=True,
                    sparse_mode=True
                )
            },
            {
                "name": "Optimized (O(N log K))",
                "class": UniversalBrainSimulatorOptimized,
                "config": OptimizedConfig(
                    n_neurons=n_neurons,
                    active_percentage=active_percentage,
                    n_areas=n_areas,
                    use_gpu=True,
                    use_cuda_kernels=True,
                    use_optimized_kernels=True,
                    memory_efficient=True,
                    sparse_mode=True
                )
            }
        ]
        
        scale_results = {
            'n_neurons': n_neurons,
            'active_percentage': active_percentage,
            'k_active': k_active,
            'n_areas': n_areas,
            'n_steps': n_steps,
            'implementations': []
        }
        
        for config in configs:
            print(f"\n🔬 Testing: {config['name']}")
            print("-" * 40)
            
            try:
                # Create simulator
                start_time = time.time()
                simulator = config['class'](config['config'])
                init_time = time.time() - start_time
                
                print(f"   Initialization: {init_time*1000:.2f}ms")
                
                # Run simulation
                sim_start = time.time()
                total_time = simulator.simulate(n_steps=n_steps, verbose=False)
                sim_time = time.time() - sim_start
                
                # Get performance stats
                stats = simulator.get_performance_stats()
                
                print(f"   Simulation time: {sim_time:.2f}s")
                print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
                print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
                print(f"   Memory: {stats['memory_usage_gb']:.2f}GB")
                print(f"   CUDA kernels: {'✅' if stats['cuda_kernels_used'] else '❌'}")
                
                if hasattr(stats, 'optimized_kernels_used'):
                    print(f"   Optimized kernels: {'✅' if stats['optimized_kernels_used'] else '❌'}")
                
                result = {
                    'name': config['name'],
                    'init_time': init_time,
                    'sim_time': sim_time,
                    'total_time': init_time + sim_time,
                    'steps_per_second': stats['steps_per_second'],
                    'neurons_per_second': stats['neurons_per_second'],
                    'memory_usage_gb': stats['memory_usage_gb'],
                    'cuda_kernels_used': stats['cuda_kernels_used'],
                    'optimized_kernels_used': getattr(stats, 'optimized_kernels_used', False),
                    'success': True
                }
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                result = {
                    'name': config['name'],
                    'error': str(e),
                    'success': False
                }
            
            scale_results['implementations'].append(result)
        
        # Calculate speedup
        if len(scale_results['implementations']) == 2:
            orig = scale_results['implementations'][0]
            opt = scale_results['implementations'][1]
            
            if orig['success'] and opt['success']:
                speedup = orig['sim_time'] / opt['sim_time'] if opt['sim_time'] > 0 else float('inf')
                print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster")
                print(f"   Original: {orig['sim_time']:.2f}s")
                print(f"   Optimized: {opt['sim_time']:.2f}s")
                
                scale_results['speedup'] = speedup
            else:
                print(f"\n❌ Cannot calculate speedup - one or both implementations failed")
                scale_results['speedup'] = None
        
        self.results.append(scale_results)
        return scale_results
    
    def run_billion_scale_tests(self):
        """Run comprehensive billion-scale tests"""
        print("🚀 BILLION-SCALE PERFORMANCE COMPARISON")
        print("=" * 60)
        print("Testing Original (O(N²)) vs Optimized (O(N log K)) algorithms")
        print("Expected: Optimized should be 100-1000x faster at large scales!")
        
        # Test scales (increasing complexity)
        test_scales = [
            # Small scale (baseline)
            {"n_neurons": 100_000, "active_percentage": 0.01, "n_steps": 10},
            
            # Medium scale
            {"n_neurons": 1_000_000, "active_percentage": 0.01, "n_steps": 10},
            
            # Large scale
            {"n_neurons": 10_000_000, "active_percentage": 0.001, "n_steps": 5},
            
            # Very large scale (if memory allows)
            {"n_neurons": 100_000_000, "active_percentage": 0.0001, "n_steps": 3},
            
            # Billion scale (if memory allows)
            {"n_neurons": 1_000_000_000, "active_percentage": 0.00001, "n_steps": 1},
        ]
        
        for scale in test_scales:
            try:
                self.test_scale(**scale)
            except Exception as e:
                print(f"❌ Scale {scale['n_neurons']:,} failed: {e}")
                continue
        
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print performance summary"""
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            print(f"\nScale: {result['n_neurons']:,} neurons")
            print(f"  Active: {result['k_active']:,} per area")
            print(f"  Steps: {result['n_steps']}")
            
            if 'speedup' in result and result['speedup'] is not None:
                print(f"  🚀 Speedup: {result['speedup']:.2f}x")
            
            for impl in result['implementations']:
                if impl['success']:
                    print(f"    {impl['name']}: {impl['sim_time']:.2f}s, {impl['neurons_per_second']:,.0f} neurons/sec")
                else:
                    print(f"    {impl['name']}: ❌ {impl['error']}")
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_vs_original_comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Run the performance comparison"""
    print("🧠 Universal Brain Simulator: Optimized vs Original")
    print("=" * 60)
    print("This test compares the algorithmic improvements:")
    print("  - Original: O(N²) top-k selection")
    print("  - Optimized: O(N log K) top-k selection")
    print("  - Expected: 100-1000x speedup at large scales!")
    
    # Create comparison instance
    comparison = PerformanceComparison()
    
    # Run tests
    comparison.run_billion_scale_tests()
    
    print(f"\n🎯 Key Insights:")
    print(f"  - At small scales (< 1M neurons): Speedup may be modest")
    print(f"  - At large scales (> 10M neurons): Speedup should be dramatic")
    print(f"  - At billion scale: Original may fail, Optimized should work")
    print(f"  - Memory usage should be similar between implementations")

if __name__ == "__main__":
    main()
