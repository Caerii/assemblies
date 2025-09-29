#!/usr/bin/env python3
"""
Scaling Laws Analyzer for GPU Billion-Scale Brain
Tests higher activation ratios to understand scaling laws systematically
"""

import time
import numpy as np
import cupy as cp
import psutil
import os

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # GB

def analyze_gpu_memory():
    """Analyze GPU memory usage"""
    if cp.cuda.is_available():
        mempool = cp.get_default_memory_pool()
        return {
            'used_bytes': mempool.used_bytes(),
            'total_bytes': mempool.total_bytes(),
            'used_gb': mempool.used_bytes() / 1024**3,
            'total_gb': mempool.total_bytes() / 1024**3,
            'device_memory_gb': cp.cuda.Device().mem_info[1] / 1024**3
        }
    return None

class ScalingLawsAnalyzer:
    """
    Scaling Laws Analyzer for GPU Billion-Scale Brain
    Tests higher activation ratios to understand scaling laws
    """
    
    def __init__(self, n_neurons=1000000000, active_percentage=0.0001, n_areas=5, seed=42):
        """Initialize the scaling laws analyzer"""
        self.n_neurons = n_neurons
        self.active_percentage = active_percentage
        self.k_active = int(n_neurons * active_percentage)
        self.n_areas = n_areas
        self.seed = seed
        
        print(f"🧠 Scaling Laws Analysis:")
        print(f"   Total Neurons: {n_neurons:,}")
        print(f"   Active Percentage: {active_percentage*100:.4f}%")
        print(f"   Active Neurons: {self.k_active:,}")
        print(f"   Number of Areas: {n_areas}")
        print(f"   Active per Area: {self.k_active:,}")
        
        # Calculate theoretical memory requirements
        self.theoretical_memory = self._calculate_theoretical_memory()
        self.print_memory_analysis()
        
        # Initialize brain
        self.brain = self._initialize_brain()
        
    def _calculate_theoretical_memory(self):
        """Calculate theoretical memory requirements"""
        # Memory per area: winners + weights + support
        memory_per_area = self.k_active * 4 * 3  # 3 arrays, 4 bytes each (int32/float32)
        total_memory = memory_per_area * self.n_areas
        
        # Additional overhead for candidates and sorting arrays
        candidates_memory = self.k_active * 10 * 4  # 10x buffer for selection
        sorting_memory = self.k_active * 3 * 4  # 3 arrays for sorting
        
        total_with_overhead = total_memory + candidates_memory + sorting_memory
        
        return {
            'per_area_bytes': memory_per_area,
            'per_area_gb': memory_per_area / 1024**3,
            'total_bytes': total_memory,
            'total_gb': total_memory / 1024**3,
            'with_overhead_bytes': total_with_overhead,
            'with_overhead_gb': total_with_overhead / 1024**3,
            'candidates_gb': candidates_memory / 1024**3,
            'sorting_gb': sorting_memory / 1024**3
        }
    
    def print_memory_analysis(self):
        """Print detailed memory analysis"""
        print(f"\n💾 Memory Analysis:")
        print(f"   Per Area Memory: {self.theoretical_memory['per_area_gb']:.6f} GB")
        print(f"   Total Core Memory: {self.theoretical_memory['total_gb']:.6f} GB")
        print(f"   Candidates Buffer: {self.theoretical_memory['candidates_gb']:.6f} GB")
        print(f"   Sorting Arrays: {self.theoretical_memory['sorting_gb']:.6f} GB")
        print(f"   Total with Overhead: {self.theoretical_memory['with_overhead_gb']:.6f} GB")
        
        # Compare to total neuron count
        if self.n_neurons > 0:
            memory_per_neuron = self.theoretical_memory['with_overhead_gb'] / self.n_neurons
            print(f"   Memory per Neuron: {memory_per_neuron:.12f} GB")
            print(f"   Memory per Million Neurons: {memory_per_neuron * 1_000_000:.6f} GB")
    
    def _initialize_brain(self):
        """Initialize the brain with detailed monitoring"""
        print(f"\n🚀 Initializing Brain:")
        
        # Get initial memory
        initial_cpu_memory = get_memory_usage()
        initial_gpu_memory = analyze_gpu_memory()
        
        print(f"   Initial CPU Memory: {initial_cpu_memory:.3f} GB")
        if initial_gpu_memory:
            print(f"   Initial GPU Memory: {initial_gpu_memory['used_gb']:.3f} GB")
        
        # Initialize areas
        areas = []
        for i in range(self.n_areas):
            area = {
                'n': self.n_neurons,
                'k': self.k_active,
                'w': 0,
                'winners': cp.zeros(self.k_active, dtype=cp.int32),
                'weights': cp.zeros(self.k_active, dtype=cp.float32),
                'support': cp.zeros(self.k_active, dtype=cp.float32),
                'activated': False
            }
            areas.append(area)
        
        # Pre-allocated arrays
        candidates = cp.zeros(self.k_active * 10, dtype=cp.float32)
        top_k_indices = cp.zeros(self.k_active, dtype=cp.int32)
        top_k_values = cp.zeros(self.k_active, dtype=cp.float32)
        sorted_indices = cp.zeros(self.k_active, dtype=cp.int32)
        
        # Get final memory
        final_cpu_memory = get_memory_usage()
        final_gpu_memory = analyze_gpu_memory()
        
        print(f"   Final CPU Memory: {final_cpu_memory:.3f} GB")
        if final_gpu_memory:
            print(f"   Final GPU Memory: {final_gpu_memory['used_gb']:.3f} GB")
        
        print(f"   CPU Memory Increase: {final_cpu_memory - initial_cpu_memory:.3f} GB")
        if initial_gpu_memory and final_gpu_memory:
            print(f"   GPU Memory Increase: {final_gpu_memory['used_gb'] - initial_gpu_memory['used_gb']:.3f} GB")
        
        return {
            'areas': areas,
            'candidates': candidates,
            'top_k_indices': top_k_indices,
            'top_k_values': top_k_values,
            'sorted_indices': sorted_indices
        }
    
    def simulate_detailed(self, n_steps=10):
        """Simulate with detailed performance monitoring"""
        print(f"\n🧠 Detailed Simulation ({n_steps} steps):")
        
        # Get initial memory
        initial_cpu_memory = get_memory_usage()
        initial_gpu_memory = analyze_gpu_memory()
        
        step_times = []
        total_start = time.perf_counter()
        
        for step in range(n_steps):
            step_start = time.perf_counter()
            
            # Simulate one step
            self._simulate_step_detailed()
            
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
        
        total_time = time.perf_counter() - total_start
        
        # Final memory
        final_cpu_memory = get_memory_usage()
        final_gpu_memory = analyze_gpu_memory()
        
        # Calculate statistics
        avg_step_time = np.mean(step_times)
        min_step_time = np.min(step_times)
        max_step_time = np.max(step_times)
        std_step_time = np.std(step_times)
        
        print(f"\n📊 Performance Statistics:")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Average Step Time: {avg_step_time*1000:.2f}ms")
        print(f"   Min Step Time: {min_step_time*1000:.2f}ms")
        print(f"   Max Step Time: {max_step_time*1000:.2f}ms")
        print(f"   Std Dev: {std_step_time*1000:.2f}ms")
        print(f"   Steps per Second: {1/avg_step_time:.1f}")
        print(f"   Neurons per Second: {self.n_neurons / avg_step_time:,.0f}")
        print(f"   Active per Second: {self.k_active * self.n_areas / avg_step_time:,.0f}")
        
        print(f"\n💾 Memory Usage:")
        print(f"   CPU Memory Change: {final_cpu_memory - initial_cpu_memory:.3f} GB")
        if initial_gpu_memory and final_gpu_memory:
            print(f"   GPU Memory Change: {final_gpu_memory['used_gb'] - initial_gpu_memory['used_gb']:.3f} GB")
            print(f"   GPU Memory Efficiency: {self.theoretical_memory['with_overhead_gb'] / final_gpu_memory['used_gb'] * 100:.1f}%")
        
        return {
            'total_time': total_time,
            'avg_step_time': avg_step_time,
            'steps_per_second': 1/avg_step_time,
            'neurons_per_second': self.n_neurons / avg_step_time,
            'active_per_second': self.k_active * self.n_areas / avg_step_time,
            'memory_efficiency': final_gpu_memory['used_gb'] / self.theoretical_memory['with_overhead_gb'] * 100 if final_gpu_memory and self.theoretical_memory['with_overhead_gb'] > 0 else 0
        }
    
    def _simulate_step_detailed(self):
        """Simulate one step with detailed monitoring"""
        for area_idx in range(self.n_areas):
            area = self.brain['areas'][area_idx]
            
            # Generate candidates
            candidates = cp.random.exponential(1.0, size=area['k'])
            
            # Select top-k winners
            if area['k'] >= len(candidates):
                winners = cp.arange(len(candidates))
            else:
                top_k_indices = cp.argpartition(candidates, -area['k'])[-area['k']:]
                top_k_values = candidates[top_k_indices]
                sorted_indices = cp.argsort(top_k_values)[::-1]
                winners = top_k_indices[sorted_indices]
            
            # Update area state
            area['w'] = len(winners)
            area['winners'][:len(winners)] = winners
            area['activated'] = True
            
            # Update weights
            area['weights'][winners] += 0.1
            area['weights'] *= 0.99
            area['support'][winners] += 1.0

def test_scaling_laws():
    """Test scaling laws across different neuron counts and activation ratios"""
    print("🔬 SCALING LAWS ANALYSIS")
    print("=" * 80)
    
    # Test different scales with higher activation ratios
    test_cases = [
        # Small scales with high activation
        {"n_neurons": 1000000, "active_percentage": 0.01, "n_areas": 5, "name": "1M neurons (1%)"},
        {"n_neurons": 1000000, "active_percentage": 0.05, "n_areas": 5, "name": "1M neurons (5%)"},
        {"n_neurons": 1000000, "active_percentage": 0.10, "n_areas": 5, "name": "1M neurons (10%)"},
        
        # Medium scales with medium activation
        {"n_neurons": 10000000, "active_percentage": 0.01, "n_areas": 5, "name": "10M neurons (1%)"},
        {"n_neurons": 10000000, "active_percentage": 0.05, "n_areas": 5, "name": "10M neurons (5%)"},
        {"n_neurons": 10000000, "active_percentage": 0.10, "n_areas": 5, "name": "10M neurons (10%)"},
        
        # Large scales with low activation
        {"n_neurons": 100000000, "active_percentage": 0.001, "n_areas": 5, "name": "100M neurons (0.1%)"},
        {"n_neurons": 100000000, "active_percentage": 0.005, "n_areas": 5, "name": "100M neurons (0.5%)"},
        {"n_neurons": 100000000, "active_percentage": 0.01, "n_areas": 5, "name": "100M neurons (1%)"},
        
        # Billion scale with very low activation
        {"n_neurons": 1000000000, "active_percentage": 0.0001, "n_areas": 5, "name": "1B neurons (0.01%)"},
        {"n_neurons": 1000000000, "active_percentage": 0.0005, "n_areas": 5, "name": "1B neurons (0.05%)"},
        {"n_neurons": 1000000000, "active_percentage": 0.001, "n_areas": 5, "name": "1B neurons (0.1%)"},
        
        # Multi-billion scale
        {"n_neurons": 10000000000, "active_percentage": 0.00001, "n_areas": 5, "name": "10B neurons (0.001%)"},
        {"n_neurons": 10000000000, "active_percentage": 0.00005, "n_areas": 5, "name": "10B neurons (0.005%)"},
        {"n_neurons": 10000000000, "active_percentage": 0.0001, "n_areas": 5, "name": "10B neurons (0.01%)"},
        
        # Human brain scale
        {"n_neurons": 86000000000, "active_percentage": 0.000001, "n_areas": 5, "name": "86B neurons (0.0001%)"},
        {"n_neurons": 86000000000, "active_percentage": 0.000005, "n_areas": 5, "name": "86B neurons (0.0005%)"},
        {"n_neurons": 86000000000, "active_percentage": 0.00001, "n_areas": 5, "name": "86B neurons (0.001%)"},
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"🧪 TESTING: {test_case['name']}")
        print(f"{'='*80}")
        
        try:
            analyzer = ScalingLawsAnalyzer(
                n_neurons=test_case['n_neurons'],
                active_percentage=test_case['active_percentage'],
                n_areas=test_case['n_areas'],
                seed=42
            )
            
            stats = analyzer.simulate_detailed(n_steps=10)
            
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['n_neurons'],
                'active_percentage': test_case['active_percentage'],
                'k_active': int(test_case['n_neurons'] * test_case['active_percentage']),
                'n_areas': test_case['n_areas'],
                'total_time': stats['total_time'],
                'avg_step_time': stats['avg_step_time'],
                'steps_per_second': stats['steps_per_second'],
                'neurons_per_second': stats['neurons_per_second'],
                'active_neurons_per_second': stats['active_per_second'],
                'memory_efficiency': stats['memory_efficiency']
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['n_neurons'],
                'active_percentage': test_case['active_percentage'],
                'k_active': int(test_case['n_neurons'] * test_case['active_percentage']),
                'n_areas': test_case['n_areas'],
                'total_time': float('inf'),
                'avg_step_time': float('inf'),
                'steps_per_second': 0,
                'neurons_per_second': 0,
                'active_neurons_per_second': 0,
                'memory_efficiency': 0
            })
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SCALING LAWS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scale':<25} {'Neurons':<15} {'Active%':<8} {'Active':<12} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Active/sec':<15} {'Mem Eff%':<10}")
    print("-" * 80)
    
    for result in results:
        if result['steps_per_second'] > 0:
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {result['k_active']:<12,} {result['steps_per_second']:<10.1f} {result['avg_step_time']*1000:<10.2f} {result['neurons_per_second']:<15,.0f} {result['active_neurons_per_second']:<15,.0f} {result['memory_efficiency']:<10.1f}")
        else:
            print(f"{result['name']:<25} {result['n_neurons']:<15,} {result['active_percentage']*100:<8.4f} {result['k_active']:<12,} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15} {'FAILED':<10}")
    
    return results

if __name__ == "__main__":
    # Test scaling laws
    results = test_scaling_laws()
    
    # Find best performance
    successful_results = [r for r in results if r['steps_per_second'] > 0]
    if successful_results:
        best = max(successful_results, key=lambda x: x['steps_per_second'])
        print(f"\n🏆 BEST PERFORMANCE: {best['name']}")
        print(f"   Steps/sec: {best['steps_per_second']:.1f}")
        print(f"   ms/step: {best['avg_step_time']*1000:.2f}ms")
        print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
        print(f"   Active/sec: {best['active_neurons_per_second']:,.0f}")
        print(f"   Memory Efficiency: {best['memory_efficiency']:.1f}%")
    else:
        print(f"\n❌ No successful tests")
