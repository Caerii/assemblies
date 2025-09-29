#!/usr/bin/env python3
"""
Simple Memory Analyzer for Billion-Scale Neural Simulation
=========================================================

This tool provides focused analysis of memory usage patterns and performance
stability for billion-scale neural simulation.
"""

import cupy as cp
import numpy as np
import time
import gc
from typing import Dict, List, Tuple, Optional

class SimpleMemoryAnalyzer:
    def __init__(self):
        """Initialize the memory analyzer."""
        self.results = []
        
    def analyze_scale(self, n_neurons: int, n_areas: int = 5, 
                     active_percent: float = 0.001, sparsity: float = 0.0001,
                     n_steps: int = 10) -> Dict:
        """Analyze memory usage and performance for a specific scale."""
        print(f"\n🔍 Analyzing {n_neurons:,} neurons (sparsity: {sparsity*100:.3f}%)...")
        
        # Calculate theoretical memory requirements
        n_active_per_area = int(n_neurons * active_percent)
        n_weights = int(n_active_per_area * n_active_per_area * sparsity)
        
        # Memory calculations
        weight_memory_gb = n_weights * 4 / 1024**3  # 4 bytes per float32
        sparse_overhead = 4.0  # 4x overhead for ultra-sparse
        weight_memory_gb *= sparse_overhead
        
        activation_memory_gb = n_active_per_area * 4 / 1024**3
        candidate_memory_gb = n_active_per_area * 4 / 1024**3
        area_memory_gb = n_active_per_area * 4 / 1024**3
        
        per_area_memory_gb = weight_memory_gb + activation_memory_gb + candidate_memory_gb + area_memory_gb
        total_memory_gb = per_area_memory_gb * n_areas
        
        print(f"   Theoretical memory: {total_memory_gb:.3f} GB")
        print(f"   Active per area: {n_active_per_area:,}")
        print(f"   Weights per area: {n_weights:,}")
        
        try:
            # Initialize arrays
            areas = []
            weights = []
            activations = []
            candidates = []
            
            for i in range(n_areas):
                # Create area
                area = cp.zeros(n_active_per_area, dtype=cp.float32)
                areas.append(area)
                
                # Create sparse weight matrix
                if n_weights > 0:
                    row_indices = cp.random.randint(0, n_active_per_area, 
                                                  size=n_weights, dtype=cp.int32)
                    col_indices = cp.random.randint(0, n_active_per_area, 
                                                  size=n_weights, dtype=cp.int32)
                    values = cp.random.exponential(1.0, size=n_weights, dtype=cp.float32)
                    
                    coo_matrix = cp.sparse.coo_matrix(
                        (values, (row_indices, col_indices)),
                        shape=(n_active_per_area, n_active_per_area),
                        dtype=cp.float32
                    )
                    weights_sparse = coo_matrix.tocsr()
                    weights.append(weights_sparse)
                    del coo_matrix
                else:
                    weights.append(cp.sparse.csr_matrix(
                        (n_active_per_area, n_active_per_area),
                        dtype=cp.float32
                    ))
                
                # Create other arrays
                activations.append(cp.zeros(n_active_per_area, dtype=cp.float32))
                candidates.append(cp.zeros(n_active_per_area, dtype=cp.float32))
            
            # Get actual memory usage
            mempool = cp.get_default_memory_pool()
            actual_memory_gb = mempool.used_bytes() / 1024**3
            
            print(f"   Actual memory: {actual_memory_gb:.3f} GB")
            
            # Run benchmark with detailed timing
            times = []
            step_details = []
            
            for step in range(n_steps):
                step_start = time.time()
                step_times = {}
                
                for area_idx in range(n_areas):
                    area = areas[area_idx]
                    weights = weights[area_idx]
                    activations = activations[area_idx]
                    candidates = candidates[area_idx]
                    
                    # Time each operation
                    start = time.time()
                    candidates[:] = cp.random.exponential(1.0, size=n_active_per_area)
                    step_times['random_gen'] = time.time() - start
                    
                    start = time.time()
                    activations[:] = weights.dot(area)
                    step_times['matrix_mult'] = time.time() - start
                    
                    start = time.time()
                    threshold = cp.percentile(activations, 90)
                    step_times['percentile'] = time.time() - start
                    
                    start = time.time()
                    area[:] = cp.where(activations > threshold, candidates, 0.0)
                    step_times['threshold'] = time.time() - start
                
                step_time = time.time() - step_start
                times.append(step_time)
                step_details.append(step_times)
            
            # Calculate performance metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            steps_per_sec = 1.0 / avg_time
            neurons_per_sec = n_neurons * steps_per_sec
            active_per_sec = n_active_per_area * n_areas * steps_per_sec
            
            # Calculate operation timings
            avg_step_times = {}
            for op in step_details[0].keys():
                avg_step_times[op] = np.mean([step[op] for step in step_details])
            
            print(f"   Performance: {steps_per_sec:.1f} steps/sec")
            print(f"   Stability: {std_time:.3f}s std dev")
            print(f"   Neurons/sec: {neurons_per_sec:,.0f}")
            
            # Cleanup
            del areas, weights, activations, candidates
            gc.collect()
            mempool.free_all_blocks()
            
            result = {
                'n_neurons': n_neurons,
                'n_active_per_area': n_active_per_area,
                'n_areas': n_areas,
                'sparsity': sparsity,
                'theoretical_memory_gb': total_memory_gb,
                'actual_memory_gb': actual_memory_gb,
                'steps_per_sec': steps_per_sec,
                'ms_per_step': avg_time * 1000,
                'std_ms_per_step': std_time * 1000,
                'neurons_per_sec': neurons_per_sec,
                'active_per_sec': active_per_sec,
                'operation_times': avg_step_times,
                'memory_efficiency': actual_memory_gb / total_memory_gb if total_memory_gb > 0 else 0
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            return None
    
    def run_analysis(self):
        """Run comprehensive analysis across different scales and sparsity levels."""
        print("🔍 COMPREHENSIVE MEMORY ANALYSIS")
        print("=" * 60)
        
        # Test different scales and sparsity levels
        test_configs = [
            (1_000_000, 0.001, "Million Scale - 0.1%"),
            (1_000_000, 0.0001, "Million Scale - 0.01%"),
            (10_000_000, 0.0001, "Ten Million Scale - 0.01%"),
            (10_000_000, 0.00001, "Ten Million Scale - 0.001%"),
            (100_000_000, 0.0001, "Hundred Million Scale - 0.01%"),
            (100_000_000, 0.00001, "Hundred Million Scale - 0.001%"),
            (1_000_000_000, 0.00001, "Billion Scale - 0.001%"),
        ]
        
        for n_neurons, sparsity, description in test_configs:
            print(f"\n🧪 {description}")
            result = self.analyze_scale(n_neurons, sparsity=sparsity, n_steps=5)
            if result:
                print(f"   ✅ Analysis complete")
            else:
                print(f"   ❌ Analysis failed")
        
        # Analyze results
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyze the results and provide insights."""
        print(f"\n📊 ANALYSIS RESULTS")
        print("=" * 80)
        
        if not self.results:
            print("No results to analyze.")
            return
        
        # Print summary table
        print(f"{'Scale':<20} {'Sparsity':<10} {'Memory (GB)':<12} {'Steps/sec':<10} {'Stability':<10} {'Neurons/sec':<15}")
        print("-" * 80)
        
        for result in self.results:
            scale_name = f"{result['n_neurons']:,}"
            sparsity_pct = f"{result['sparsity']*100:.3f}%"
            memory_gb = f"{result['actual_memory_gb']:.3f}"
            steps_per_sec = f"{result['steps_per_sec']:.1f}"
            stability = f"{result['std_ms_per_step']:.1f}ms"
            neurons_per_sec = f"{result['neurons_per_sec']:,.0f}"
            
            print(f"{scale_name:<20} {sparsity_pct:<10} {memory_gb:<12} {steps_per_sec:<10} {stability:<10} {neurons_per_sec:<15}")
        
        # Find optimal configurations
        self._find_optimal_configurations()
        
        # Analyze performance stability
        self._analyze_stability()
    
    def _find_optimal_configurations(self):
        """Find optimal configurations for each scale."""
        print(f"\n🎯 OPTIMAL CONFIGURATIONS")
        print("=" * 60)
        
        # Group by scale
        scales = {}
        for result in self.results:
            if result['n_neurons'] not in scales:
                scales[result['n_neurons']] = []
            scales[result['n_neurons']].append(result)
        
        for n_neurons in sorted(scales.keys()):
            results = scales[n_neurons]
            
            # Find best performance
            best_performance = max(results, key=lambda r: r['steps_per_sec'])
            
            # Find most stable
            most_stable = min(results, key=lambda r: r['std_ms_per_step'])
            
            # Find best balance
            best_balance = max(results, key=lambda r: r['steps_per_sec'] / r['actual_memory_gb'])
            
            print(f"\nScale: {n_neurons:,} neurons")
            print(f"  Best Performance: {best_performance['sparsity']*100:.3f}% sparsity")
            print(f"    {best_performance['steps_per_sec']:.1f} steps/sec, {best_performance['actual_memory_gb']:.3f} GB")
            print(f"  Most Stable: {most_stable['sparsity']*100:.3f}% sparsity")
            print(f"    {most_stable['steps_per_sec']:.1f} steps/sec, {most_stable['std_ms_per_step']:.1f}ms std")
            print(f"  Best Balance: {best_balance['sparsity']*100:.3f}% sparsity")
            print(f"    {best_balance['steps_per_sec']:.1f} steps/sec, {best_balance['actual_memory_gb']:.3f} GB")
    
    def _analyze_stability(self):
        """Analyze performance stability across scales."""
        print(f"\n📈 PERFORMANCE STABILITY ANALYSIS")
        print("=" * 60)
        
        # Group by scale
        scales = {}
        for result in self.results:
            if result['n_neurons'] not in scales:
                scales[result['n_neurons']] = []
            scales[result['n_neurons']].append(result)
        
        for n_neurons in sorted(scales.keys()):
            results = scales[n_neurons]
            
            print(f"\nScale: {n_neurons:,} neurons")
            for result in results:
                stability_pct = (result['std_ms_per_step'] / result['ms_per_step']) * 100
                print(f"  Sparsity {result['sparsity']*100:.3f}%: {stability_pct:.1f}% variation")
    
    def generate_recommendations(self):
        """Generate recommendations for stable billion-scale simulation."""
        print(f"\n💡 RECOMMENDATIONS FOR STABLE BILLION-SCALE SIMULATION")
        print("=" * 70)
        
        if not self.results:
            print("No results available for recommendations.")
            return
        
        # Find the most stable billion-scale configuration
        billion_scale_results = [r for r in self.results if r['n_neurons'] == 1_000_000_000]
        
        if billion_scale_results:
            best_billion = min(billion_scale_results, key=lambda r: r['std_ms_per_step'])
            print(f"✅ Billion-scale is achievable with:")
            print(f"   Sparsity: {best_billion['sparsity']*100:.3f}%")
            print(f"   Memory: {best_billion['actual_memory_gb']:.3f} GB")
            print(f"   Performance: {best_billion['steps_per_sec']:.1f} steps/sec")
            print(f"   Stability: {best_billion['std_ms_per_step']:.1f}ms std dev")
        else:
            print("❌ Billion-scale not yet achievable with current configurations.")
            print("   Recommendations:")
            print("   1. Reduce sparsity further (0.0001% or lower)")
            print("   2. Reduce active neuron percentage")
            print("   3. Use more aggressive memory optimization")
        
        # General recommendations
        print(f"\n🔧 General Optimization Recommendations:")
        print(f"   1. Use ultra-sparse matrices (0.001% or lower)")
        print(f"   2. Implement memory pooling for frequent allocations")
        print(f"   3. Use COO format for matrix creation, CSR for operations")
        print(f"   4. Implement gradient-based sparsity adjustment")
        print(f"   5. Use memory-mapped files for very large scales")

def main():
    """Main function to run memory analysis."""
    analyzer = SimpleMemoryAnalyzer()
    analyzer.run_analysis()
    analyzer.generate_recommendations()

if __name__ == "__main__":
    main()

