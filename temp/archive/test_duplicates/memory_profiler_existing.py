#!/usr/bin/env python3
"""
Memory Profiler for Existing Implementations
===========================================

This tool profiles memory usage patterns of our existing neural simulation
implementations to identify bottlenecks and optimization opportunities.
"""

import numpy as np
import time
import psutil
import gc
import sys
from typing import Dict, List, Tuple, Optional
import tracemalloc
import os

class MemoryProfiler:
    """Memory profiler for neural simulation implementations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.tracemalloc_started = False
        
    def start_tracing(self):
        """Start memory tracing."""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            'rss_gb': memory_info.rss / (1024**3),  # Resident Set Size
            'vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size
            'percent': self.process.memory_percent()
        }
    
    def get_tracemalloc_stats(self) -> Dict:
        """Get tracemalloc statistics."""
        if not self.tracemalloc_started:
            return {}
        
        current, peak = tracemalloc.get_traced_memory()
        return {
            'current_mb': current / (1024**2),
            'peak_mb': peak / (1024**2)
        }
    
    def profile_array_allocation(self, size: int, dtype=np.float32) -> Dict:
        """Profile memory allocation for arrays of different sizes."""
        print(f"\n🔍 Profiling array allocation: {size:,} elements ({dtype})")
        
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        # Allocate array
        array = np.zeros(size, dtype=dtype)
        
        allocation_time = time.time() - start_time
        after_memory = self.get_memory_usage()
        
        # Calculate memory increase
        memory_increase_gb = after_memory['rss_gb'] - start_memory['rss_gb']
        expected_memory_gb = (size * np.dtype(dtype).itemsize) / (1024**3)
        
        print(f"   Expected memory: {expected_memory_gb:.3f} GB")
        print(f"   Actual increase: {memory_increase_gb:.3f} GB")
        print(f"   Allocation time: {allocation_time:.3f} seconds")
        print(f"   Memory efficiency: {expected_memory_gb/memory_increase_gb:.2f}x" if memory_increase_gb > 0 else "   Memory efficiency: N/A")
        
        # Cleanup
        del array
        gc.collect()
        
        return {
            'size': size,
            'dtype': str(dtype),
            'expected_memory_gb': expected_memory_gb,
            'actual_memory_gb': memory_increase_gb,
            'allocation_time': allocation_time,
            'efficiency': expected_memory_gb/memory_increase_gb if memory_increase_gb > 0 else 0
        }
    
    def profile_sparse_matrix_creation(self, n_active: int, sparsity: float) -> Dict:
        """Profile sparse matrix creation and memory usage."""
        print(f"\n🔍 Profiling sparse matrix: {n_active:,} x {n_active:,} (sparsity: {sparsity*100:.3f}%)")
        
        # Calculate number of non-zero elements
        n_weights = int(n_active * n_active * sparsity)
        
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        # Create COO matrix
        row_indices = np.random.randint(0, n_active, size=n_weights, dtype=np.int32)
        col_indices = np.random.randint(0, n_active, size=n_weights, dtype=np.int32)
        values = np.random.exponential(1.0, size=n_weights).astype(np.float32)
        
        coo_creation_time = time.time() - start_time
        coo_memory = self.get_memory_usage()
        
        # Convert to CSR
        start_csr = time.time()
        from scipy.sparse import coo_matrix
        coo_matrix_obj = coo_matrix((values, (row_indices, col_indices)), 
                                   shape=(n_active, n_active), dtype=np.float32)
        csr_matrix = coo_matrix_obj.tocsr()
        csr_conversion_time = time.time() - start_csr
        
        csr_memory = self.get_memory_usage()
        
        # Calculate memory usage
        coo_memory_gb = coo_memory['rss_gb'] - start_memory['rss_gb']
        csr_memory_gb = csr_memory['rss_gb'] - coo_memory['rss_gb']
        total_memory_gb = csr_memory['rss_gb'] - start_memory['rss_gb']
        
        # Theoretical memory usage
        coo_theoretical = (n_weights * 3 * 4) / (1024**3)  # 3 arrays * 4 bytes
        csr_theoretical = (n_weights * 2 * 4 + n_active * 4) / (1024**3)  # 2 arrays + row pointers
        
        print(f"   Non-zero weights: {n_weights:,}")
        print(f"   COO creation time: {coo_creation_time:.3f} seconds")
        print(f"   CSR conversion time: {csr_conversion_time:.3f} seconds")
        print(f"   COO memory: {coo_memory_gb:.3f} GB (theoretical: {coo_theoretical:.3f} GB)")
        print(f"   CSR memory: {csr_memory_gb:.3f} GB (theoretical: {csr_theoretical:.3f} GB)")
        print(f"   Total memory: {total_memory_gb:.3f} GB")
        
        # Cleanup
        del row_indices, col_indices, values, coo_matrix_obj, csr_matrix
        gc.collect()
        
        return {
            'n_active': n_active,
            'sparsity': sparsity,
            'n_weights': n_weights,
            'coo_creation_time': coo_creation_time,
            'csr_conversion_time': csr_conversion_time,
            'coo_memory_gb': coo_memory_gb,
            'csr_memory_gb': csr_memory_gb,
            'total_memory_gb': total_memory_gb,
            'coo_theoretical_gb': coo_theoretical,
            'csr_theoretical_gb': csr_theoretical
        }

def profile_working_cuda_brain():
    """Profile the working CUDA brain implementation."""
    print("🔍 PROFILING WORKING CUDA BRAIN")
    print("=" * 50)
    
    profiler = MemoryProfiler()
    profiler.start_tracing()
    
    # Test different scales
    scales = [100_000, 1_000_000, 10_000_000]
    results = []
    
    for scale in scales:
        print(f"\n📊 Testing scale: {scale:,} neurons")
        
        # Profile array allocations
        array_result = profiler.profile_array_allocation(scale, np.float32)
        
        # Profile sparse matrix creation
        sparsity = 0.001  # 0.1% sparsity
        sparse_result = profiler.profile_sparse_matrix_creation(scale, sparsity)
        
        results.append({
            'scale': scale,
            'array_result': array_result,
            'sparse_result': sparse_result
        })
    
    return results

def profile_ultimate_billion_scale():
    """Profile the ultimate billion scale brain implementation."""
    print("\n🔍 PROFILING ULTIMATE BILLION SCALE BRAIN")
    print("=" * 50)
    
    profiler = MemoryProfiler()
    profiler.start_tracing()
    
    # Test different scales and sparsity levels
    test_configs = [
        (1_000_000, 0.01, "1M neurons, 1% sparsity"),
        (1_000_000, 0.001, "1M neurons, 0.1% sparsity"),
        (10_000_000, 0.001, "10M neurons, 0.1% sparsity"),
        (10_000_000, 0.0001, "10M neurons, 0.01% sparsity"),
    ]
    
    results = []
    
    for scale, sparsity, description in test_configs:
        print(f"\n📊 Testing: {description}")
        
        # Profile sparse matrix creation
        sparse_result = profiler.profile_sparse_matrix_creation(scale, sparsity)
        
        results.append({
            'description': description,
            'scale': scale,
            'sparsity': sparsity,
            'sparse_result': sparse_result
        })
    
    return results

def analyze_memory_patterns(results: List[Dict]):
    """Analyze memory usage patterns from profiling results."""
    print("\n📈 MEMORY PATTERN ANALYSIS")
    print("=" * 50)
    
    # Analyze array allocation efficiency
    print("\n🔍 Array Allocation Efficiency:")
    for result in results:
        if 'array_result' in result:
            array_res = result['array_result']
            efficiency = array_res['efficiency']
            print(f"   {result['scale']:,} elements: {efficiency:.2f}x efficiency")
    
    # Analyze sparse matrix memory usage
    print("\n🔍 Sparse Matrix Memory Usage:")
    for result in results:
        if 'sparse_result' in result:
            sparse_res = result['sparse_result']
            print(f"   {sparse_res['n_active']:,} x {sparse_res['n_active']:,} (sparsity {sparse_res['sparsity']*100:.3f}%):")
            print(f"      COO: {sparse_res['coo_memory_gb']:.3f} GB (theoretical: {sparse_res['coo_theoretical_gb']:.3f} GB)")
            print(f"      CSR: {sparse_res['csr_memory_gb']:.3f} GB (theoretical: {sparse_res['csr_theoretical_gb']:.3f} GB)")
            print(f"      Total: {sparse_res['total_memory_gb']:.3f} GB")
    
    # Identify bottlenecks
    print("\n🚨 IDENTIFIED BOTTLENECKS:")
    
    # Check for memory inefficiency
    inefficient_allocations = [r for r in results if 'array_result' in r and r['array_result']['efficiency'] < 0.8]
    if inefficient_allocations:
        print("   ❌ Inefficient array allocations detected")
        for result in inefficient_allocations:
            print(f"      {result['scale']:,} elements: {result['array_result']['efficiency']:.2f}x efficiency")
    
    # Check for excessive memory usage
    excessive_memory = [r for r in results if 'sparse_result' in r and r['sparse_result']['total_memory_gb'] > 1.0]
    if excessive_memory:
        print("   ❌ Excessive memory usage detected")
        for result in excessive_memory:
            print(f"      {result['sparse_result']['n_active']:,} neurons: {result['sparse_result']['total_memory_gb']:.3f} GB")
    
    # Check for slow allocations
    slow_allocations = [r for r in results if 'array_result' in r and r['array_result']['allocation_time'] > 0.1]
    if slow_allocations:
        print("   ❌ Slow allocations detected")
        for result in slow_allocations:
            print(f"      {result['scale']:,} elements: {result['array_result']['allocation_time']:.3f} seconds")

def generate_optimization_recommendations(results: List[Dict]):
    """Generate optimization recommendations based on profiling results."""
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    # Analyze memory efficiency
    array_efficiencies = [r['array_result']['efficiency'] for r in results if 'array_result' in r]
    avg_efficiency = np.mean(array_efficiencies) if array_efficiencies else 0
    
    print(f"📊 Memory Efficiency Analysis:")
    print(f"   Average array allocation efficiency: {avg_efficiency:.2f}x")
    
    if avg_efficiency < 0.8:
        print("   🔧 Recommendation: Implement memory pooling for frequent allocations")
        print("   🔧 Recommendation: Pre-allocate arrays and reuse them")
    
    # Analyze sparse matrix overhead
    sparse_overheads = []
    for result in results:
        if 'sparse_result' in result:
            sparse_res = result['sparse_result']
            theoretical = sparse_res['coo_theoretical_gb'] + sparse_res['csr_theoretical_gb']
            actual = sparse_res['total_memory_gb']
            overhead = actual / theoretical if theoretical > 0 else 0
            sparse_overheads.append(overhead)
    
    avg_overhead = np.mean(sparse_overheads) if sparse_overheads else 0
    print(f"   Average sparse matrix overhead: {avg_overhead:.2f}x")
    
    if avg_overhead > 2.0:
        print("   🔧 Recommendation: Optimize sparse matrix storage format")
        print("   🔧 Recommendation: Use more efficient sparse matrix libraries")
    
    # General recommendations
    print(f"\n🎯 General Recommendations:")
    print(f"   1. Use memory mapping for large arrays")
    print(f"   2. Implement gradient-based sparsity adjustment")
    print(f"   3. Use COO format for creation, CSR for operations")
    print(f"   4. Implement memory pooling for frequent allocations")
    print(f"   5. Use ultra-sparse matrices (0.001% or lower) for billion scale")
    print(f"   6. Pre-allocate all arrays at initialization")
    print(f"   7. Use explicit memory cleanup after each step")

def main():
    """Main profiling function."""
    print("🧠 MEMORY PROFILER FOR EXISTING IMPLEMENTATIONS")
    print("=" * 60)
    
    # Profile working CUDA brain
    cuda_results = profile_working_cuda_brain()
    
    # Profile ultimate billion scale
    billion_results = profile_ultimate_billion_scale()
    
    # Combine results
    all_results = cuda_results + billion_results
    
    # Analyze patterns
    analyze_memory_patterns(all_results)
    
    # Generate recommendations
    generate_optimization_recommendations(all_results)
    
    print(f"\n✅ Memory profiling complete!")

if __name__ == "__main__":
    main()
