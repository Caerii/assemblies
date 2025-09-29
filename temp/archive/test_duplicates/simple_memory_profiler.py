#!/usr/bin/env python3
"""
Simple Memory Profiler for Existing Implementations
==================================================

This tool profiles memory usage patterns without getting stuck on large allocations.
"""

import numpy as np
import time
import psutil
import gc
import sys
from typing import Dict, List, Tuple, Optional

class SimpleMemoryProfiler:
    """Simple memory profiler that won't get stuck."""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            'rss_gb': memory_info.rss / (1024**3),  # Resident Set Size
            'vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size
            'percent': self.process.memory_percent()
        }
    
    def profile_small_scale(self, n_neurons: int, sparsity: float) -> Dict:
        """Profile memory usage on a small scale to understand patterns."""
        print(f"\n🔍 Profiling {n_neurons:,} neurons (sparsity: {sparsity*100:.3f}%)")
        
        # Calculate active neurons and weights
        n_active = min(n_neurons, 10000)  # Limit to 10K for safety
        n_weights = int(n_active * n_active * sparsity)
        
        print(f"   Active neurons: {n_active:,}")
        print(f"   Non-zero weights: {n_weights:,}")
        
        # Test array allocation
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        # Create arrays
        activations = np.zeros(n_active, dtype=np.float32)
        candidates = np.zeros(n_active, dtype=np.float32)
        area = np.zeros(n_active, dtype=np.float32)
        
        array_time = time.time() - start_time
        after_arrays = self.get_memory_usage()
        
        # Test sparse matrix creation (small scale)
        if n_weights > 0 and n_weights < 1000000:  # Only if reasonable size
            start_sparse = time.time()
            
            # Create small sparse matrix
            row_indices = np.random.randint(0, n_active, size=n_weights, dtype=np.int32)
            col_indices = np.random.randint(0, n_active, size=n_weights, dtype=np.int32)
            values = np.random.exponential(1.0, size=n_weights).astype(np.float32)
            
            sparse_time = time.time() - start_sparse
            after_sparse = self.get_memory_usage()
            
            # Calculate memory usage
            array_memory_gb = after_arrays['rss_gb'] - start_memory['rss_gb']
            sparse_memory_gb = after_sparse['rss_gb'] - after_arrays['rss_gb']
            total_memory_gb = after_sparse['rss_gb'] - start_memory['rss_gb']
            
            print(f"   Array allocation time: {array_time:.3f} seconds")
            print(f"   Sparse matrix time: {sparse_time:.3f} seconds")
            print(f"   Array memory: {array_memory_gb:.3f} GB")
            print(f"   Sparse memory: {sparse_memory_gb:.3f} GB")
            print(f"   Total memory: {total_memory_gb:.3f} GB")
            
            # Cleanup
            del row_indices, col_indices, values
        else:
            print(f"   Skipping sparse matrix (too large: {n_weights:,} weights)")
            array_memory_gb = after_arrays['rss_gb'] - start_memory['rss_gb']
            sparse_memory_gb = 0
            total_memory_gb = array_memory_gb
        
        # Cleanup
        del activations, candidates, area
        gc.collect()
        
        return {
            'n_neurons': n_neurons,
            'n_active': n_active,
            'sparsity': sparsity,
            'n_weights': n_weights,
            'array_memory_gb': array_memory_gb,
            'sparse_memory_gb': sparse_memory_gb,
            'total_memory_gb': total_memory_gb,
            'array_time': array_time,
            'sparse_time': sparse_time if 'sparse_time' in locals() else 0
        }

def analyze_memory_scaling():
    """Analyze how memory scales with different parameters."""
    print("🔍 MEMORY SCALING ANALYSIS")
    print("=" * 50)
    
    profiler = SimpleMemoryProfiler()
    
    # Test different scales and sparsity levels
    test_configs = [
        (100_000, 0.01, "100K neurons, 1% sparsity"),
        (100_000, 0.001, "100K neurons, 0.1% sparsity"),
        (1_000_000, 0.001, "1M neurons, 0.1% sparsity"),
        (1_000_000, 0.0001, "1M neurons, 0.01% sparsity"),
        (10_000_000, 0.0001, "10M neurons, 0.01% sparsity"),
        (10_000_000, 0.00001, "10M neurons, 0.001% sparsity"),
    ]
    
    results = []
    
    for n_neurons, sparsity, description in test_configs:
        print(f"\n📊 {description}")
        
        result = profiler.profile_small_scale(n_neurons, sparsity)
        results.append(result)
    
    return results

def analyze_existing_implementations():
    """Analyze memory patterns in existing implementations."""
    print("\n🔍 EXISTING IMPLEMENTATION ANALYSIS")
    print("=" * 50)
    
    # Read and analyze existing implementations
    implementations = [
        "working_cuda_brain_v14_39.py",
        "ultimate_billion_scale_brain.py",
        "final_billion_scale_brain.py"
    ]
    
    analysis_results = []
    
    for impl_file in implementations:
        try:
            print(f"\n📄 Analyzing {impl_file}")
            
            with open(impl_file, 'r') as f:
                content = f.read()
            
            # Count memory allocations
            array_allocations = content.count('np.zeros(') + content.count('np.ones(') + content.count('np.empty(')
            cupy_allocations = content.count('cp.zeros(') + content.count('cp.ones(') + content.count('cp.empty(')
            
            # Count sparse matrix operations
            sparse_operations = content.count('coo_matrix') + content.count('csr_matrix') + content.count('tocsr()')
            
            # Count memory cleanup
            cleanup_operations = content.count('del ') + content.count('gc.collect()')
            
            print(f"   Array allocations: {array_allocations}")
            print(f"   CuPy allocations: {cupy_allocations}")
            print(f"   Sparse operations: {sparse_operations}")
            print(f"   Cleanup operations: {cleanup_operations}")
            
            analysis_results.append({
                'file': impl_file,
                'array_allocations': array_allocations,
                'cupy_allocations': cupy_allocations,
                'sparse_operations': sparse_operations,
                'cleanup_operations': cleanup_operations
            })
            
        except FileNotFoundError:
            print(f"   ❌ File not found: {impl_file}")
        except Exception as e:
            print(f"   ❌ Error analyzing {impl_file}: {e}")
    
    return analysis_results

def generate_optimization_recommendations(scaling_results: List[Dict], impl_results: List[Dict]):
    """Generate optimization recommendations based on analysis."""
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    # Analyze memory scaling
    print("\n📊 Memory Scaling Analysis:")
    for result in scaling_results:
        if result['n_weights'] > 0:
            memory_per_weight = result['total_memory_gb'] / result['n_weights'] * 1024**3  # bytes per weight
            print(f"   {result['n_neurons']:,} neurons, {result['sparsity']*100:.3f}% sparsity:")
            print(f"      Memory per weight: {memory_per_weight:.2f} bytes")
            print(f"      Total memory: {result['total_memory_gb']:.3f} GB")
    
    # Analyze implementation patterns
    print("\n📊 Implementation Pattern Analysis:")
    for result in impl_results:
        print(f"   {result['file']}:")
        print(f"      Array allocations: {result['array_allocations']}")
        print(f"      CuPy allocations: {result['cupy_allocations']}")
        print(f"      Sparse operations: {result['sparse_operations']}")
        print(f"      Cleanup operations: {result['cleanup_operations']}")
    
    # Generate specific recommendations
    print("\n🎯 Specific Recommendations:")
    
    # Check for excessive allocations
    total_allocations = sum(r['array_allocations'] + r['cupy_allocations'] for r in impl_results)
    if total_allocations > 20:
        print("   🔧 High allocation count detected - implement memory pooling")
    
    # Check for sparse matrix usage
    total_sparse = sum(r['sparse_operations'] for r in impl_results)
    if total_sparse > 10:
        print("   🔧 Heavy sparse matrix usage - optimize COO to CSR conversion")
    
    # Check for cleanup
    total_cleanup = sum(r['cleanup_operations'] for r in impl_results)
    if total_cleanup < 5:
        print("   🔧 Insufficient cleanup - add explicit memory management")
    
    # General recommendations
    print("\n🔧 General Optimization Strategy:")
    print("   1. Use ultra-sparse matrices (0.001% or lower) for billion scale")
    print("   2. Pre-allocate all arrays at initialization")
    print("   3. Implement memory pooling for frequent allocations")
    print("   4. Use COO format for creation, CSR for operations")
    print("   5. Add explicit memory cleanup after each step")
    print("   6. Use gradient-based sparsity adjustment")
    print("   7. Implement memory-mapped files for very large scales")

def main():
    """Main profiling function."""
    print("🧠 SIMPLE MEMORY PROFILER FOR EXISTING IMPLEMENTATIONS")
    print("=" * 60)
    
    # Analyze memory scaling
    scaling_results = analyze_memory_scaling()
    
    # Analyze existing implementations
    impl_results = analyze_existing_implementations()
    
    # Generate recommendations
    generate_optimization_recommendations(scaling_results, impl_results)
    
    print(f"\n✅ Memory profiling complete!")

if __name__ == "__main__":
    main()

