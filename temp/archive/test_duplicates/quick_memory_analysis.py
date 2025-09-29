#!/usr/bin/env python3
"""
Quick Memory Analysis for Billion-Scale Neural Simulation
========================================================

This tool provides quick, focused analysis of memory usage patterns
without getting stuck on large allocations.
"""

import cupy as cp
import numpy as np
import time
import gc
from typing import Dict, List, Tuple, Optional

def analyze_memory_requirements(n_neurons: int, n_areas: int = 5, 
                               active_percent: float = 0.001, sparsity: float = 0.0001) -> Dict:
    """Analyze memory requirements without actually allocating."""
    print(f"\n🔍 Analyzing {n_neurons:,} neurons...")
    
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
    
    # Calculate memory efficiency
    dense_memory_gb = n_neurons * 4 / 1024**3  # Dense representation
    memory_efficiency = total_memory_gb / dense_memory_gb if dense_memory_gb > 0 else 0
    
    print(f"   Active per area: {n_active_per_area:,}")
    print(f"   Weights per area: {n_weights:,}")
    print(f"   Weight memory: {weight_memory_gb:.3f} GB")
    print(f"   Per area memory: {per_area_memory_gb:.3f} GB")
    print(f"   Total memory: {total_memory_gb:.3f} GB")
    print(f"   Memory efficiency: {memory_efficiency:.3f}")
    
    return {
        'n_neurons': n_neurons,
        'n_active_per_area': n_active_per_area,
        'n_areas': n_areas,
        'sparsity': sparsity,
        'n_weights': n_weights,
        'weight_memory_gb': weight_memory_gb,
        'per_area_memory_gb': per_area_memory_gb,
        'total_memory_gb': total_memory_gb,
        'memory_efficiency': memory_efficiency
    }

def test_small_scale_performance(n_neurons: int, sparsity: float = 0.0001, n_steps: int = 5) -> Dict:
    """Test performance on a small scale to understand patterns."""
    print(f"\n🧪 Testing {n_neurons:,} neurons (sparsity: {sparsity*100:.3f}%)...")
    
    try:
        # Use smaller active percentage for testing
        active_percent = 0.001
        n_active_per_area = int(n_neurons * active_percent)
        n_areas = 5
        
        # Create small test arrays
        area = cp.zeros(n_active_per_area, dtype=cp.float32)
        
        # Create small sparse matrix
        n_weights = int(n_active_per_area * n_active_per_area * sparsity)
        if n_weights > 0:
            row_indices = cp.random.randint(0, n_active_per_area, size=n_weights, dtype=cp.int32)
            col_indices = cp.random.randint(0, n_active_per_area, size=n_weights, dtype=cp.int32)
            values = cp.random.exponential(1.0, size=n_weights, dtype=cp.float32)
            
            coo_matrix = cp.sparse.coo_matrix(
                (values, (row_indices, col_indices)),
                shape=(n_active_per_area, n_active_per_area),
                dtype=cp.float32
            )
            weights = coo_matrix.tocsr()
            del coo_matrix
        else:
            weights = cp.sparse.csr_matrix(
                (n_active_per_area, n_active_per_area),
                dtype=cp.float32
            )
        
        activations = cp.zeros(n_active_per_area, dtype=cp.float32)
        candidates = cp.zeros(n_active_per_area, dtype=cp.float32)
        
        # Run benchmark
        times = []
        for step in range(n_steps):
            start = time.time()
            
            # Generate random candidates
            candidates[:] = cp.random.exponential(1.0, size=n_active_per_area)
            
            # Compute activations
            activations[:] = weights.dot(area)
            
            # Apply threshold
            threshold = cp.percentile(activations, 90)
            area[:] = cp.where(activations > threshold, candidates, 0.0)
            
            step_time = time.time() - start
            times.append(step_time)
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        steps_per_sec = 1.0 / avg_time
        neurons_per_sec = n_neurons * steps_per_sec
        
        print(f"   Performance: {steps_per_sec:.1f} steps/sec")
        print(f"   Stability: {std_time:.3f}s std dev")
        print(f"   Neurons/sec: {neurons_per_sec:,.0f}")
        
        # Cleanup
        del area, weights, activations, candidates
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        return {
            'n_neurons': n_neurons,
            'sparsity': sparsity,
            'steps_per_sec': steps_per_sec,
            'ms_per_step': avg_time * 1000,
            'std_ms_per_step': std_time * 1000,
            'neurons_per_sec': neurons_per_sec,
            'stability_pct': (std_time / avg_time) * 100
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return None

def main():
    """Main function to run quick memory analysis."""
    print("🔍 QUICK MEMORY ANALYSIS")
    print("=" * 50)
    
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
    
    memory_analysis = []
    performance_analysis = []
    
    for n_neurons, sparsity, description in test_configs:
        print(f"\n🧪 {description}")
        
        # Analyze memory requirements
        memory_req = analyze_memory_requirements(n_neurons, sparsity=sparsity)
        memory_analysis.append(memory_req)
        
        # Test performance on smaller scale if possible
        if n_neurons <= 10_000_000:  # Only test up to 10M to avoid hanging
            perf_result = test_small_scale_performance(n_neurons, sparsity)
            if perf_result:
                performance_analysis.append(perf_result)
    
    # Print summary
    print(f"\n📊 MEMORY REQUIREMENTS SUMMARY")
    print("=" * 80)
    print(f"{'Scale':<20} {'Sparsity':<10} {'Memory (GB)':<12} {'Efficiency':<12} {'Weights':<15}")
    print("-" * 80)
    
    for req in memory_analysis:
        scale_name = f"{req['n_neurons']:,}"
        sparsity_pct = f"{req['sparsity']*100:.3f}%"
        memory_gb = f"{req['total_memory_gb']:.3f}"
        efficiency = f"{req['memory_efficiency']:.3f}"
        weights = f"{req['n_weights']:,}"
        
        print(f"{scale_name:<20} {sparsity_pct:<10} {memory_gb:<12} {efficiency:<12} {weights:<15}")
    
    # Print performance summary
    if performance_analysis:
        print(f"\n📈 PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Scale':<20} {'Sparsity':<10} {'Steps/sec':<10} {'Stability':<10} {'Neurons/sec':<15}")
        print("-" * 80)
        
        for perf in performance_analysis:
            scale_name = f"{perf['n_neurons']:,}"
            sparsity_pct = f"{perf['sparsity']*100:.3f}%"
            steps_per_sec = f"{perf['steps_per_sec']:.1f}"
            stability = f"{perf['stability_pct']:.1f}%"
            neurons_per_sec = f"{perf['neurons_per_sec']:,.0f}"
            
            print(f"{scale_name:<20} {sparsity_pct:<10} {steps_per_sec:<10} {stability:<10} {neurons_per_sec:<15}")
    
    # Generate recommendations
    print(f"\n💡 RECOMMENDATIONS FOR STABLE BILLION-SCALE SIMULATION")
    print("=" * 70)
    
    # Find billion-scale memory requirements
    billion_req = next((req for req in memory_analysis if req['n_neurons'] == 1_000_000_000), None)
    
    if billion_req:
        print(f"✅ Billion-scale memory analysis:")
        print(f"   Required memory: {billion_req['total_memory_gb']:.3f} GB")
        print(f"   Memory efficiency: {billion_req['memory_efficiency']:.3f}")
        print(f"   Weights per area: {billion_req['n_weights']:,}")
        
        if billion_req['total_memory_gb'] <= 12.0:
            print(f"   ✅ Memory requirement is within RTX 4090 capacity!")
        else:
            print(f"   ❌ Memory requirement exceeds RTX 4090 capacity")
            print(f"   💡 Try reducing sparsity to {billion_req['sparsity']/2:.6f}%")
    
    # General recommendations
    print(f"\n🔧 Optimization Recommendations:")
    print(f"   1. Use ultra-sparse matrices (0.001% or lower)")
    print(f"   2. Implement memory pooling for frequent allocations")
    print(f"   3. Use COO format for matrix creation, CSR for operations")
    print(f"   4. Implement gradient-based sparsity adjustment")
    print(f"   5. Use memory-mapped files for very large scales")

if __name__ == "__main__":
    main()

