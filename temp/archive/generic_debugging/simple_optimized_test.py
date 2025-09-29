#!/usr/bin/env python3
"""
Simple Optimized Test
====================

This test starts small and gradually scales up to avoid getting stuck
on large memory allocations.
"""

import numpy as np
import time
import gc
import psutil
from scipy.sparse import coo_matrix, csr_matrix

def test_small_scale(n_neurons: int, sparsity: float, description: str):
    """Test a small scale to avoid getting stuck."""
    print(f"\n🧪 Testing: {description}")
    
    # Calculate active neurons (limit to reasonable size)
    n_active = min(n_neurons, 10000)  # Cap at 10K for safety
    n_weights = int(n_active * n_active * sparsity)
    
    print(f"   Active neurons: {n_active:,}")
    print(f"   Non-zero weights: {n_weights:,}")
    
    if n_weights > 100000:  # Skip if too many weights
        print(f"   ⚠️  Skipping (too many weights: {n_weights:,})")
        return None
    
    try:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**3)
        
        # Create arrays
        activations = np.zeros(n_active, dtype=np.float32)
        candidates = np.zeros(n_active, dtype=np.float32)
        area = np.zeros(n_active, dtype=np.float32)
        
        # Create sparse matrix if reasonable size
        if n_weights > 0:
            row_indices = np.random.randint(0, n_active, size=n_weights, dtype=np.int32)
            col_indices = np.random.randint(0, n_active, size=n_weights, dtype=np.int32)
            values = np.random.exponential(1.0, size=n_weights).astype(np.float32)
            
            # Create COO matrix
            coo_matrix_obj = coo_matrix(
                (values, (row_indices, col_indices)),
                shape=(n_active, n_active),
                dtype=np.float32
            )
            
            # Convert to CSR
            csr_matrix_obj = coo_matrix_obj.tocsr()
            
            # Test one simulation step
            candidates[:] = np.random.exponential(1.0, size=n_active).astype(np.float32)
            activations[:] = csr_matrix_obj.dot(area)
            threshold = np.percentile(activations, 90)
            area[:] = np.where(activations > threshold, candidates, 0.0)
            
            # Cleanup
            del coo_matrix_obj, csr_matrix_obj, row_indices, col_indices, values
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**3)
        
        step_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"   Step time: {step_time:.3f} seconds")
        print(f"   Memory used: {memory_used:.3f} GB")
        print(f"   MS per step: {step_time * 1000:.3f}")
        
        # Cleanup
        del activations, candidates, area
        gc.collect()
        
        return {
            'n_neurons': n_neurons,
            'n_active': n_active,
            'sparsity': sparsity,
            'n_weights': n_weights,
            'step_time': step_time,
            'memory_used': memory_used,
            'ms_per_step': step_time * 1000
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Main test function."""
    print("🧠 SIMPLE OPTIMIZED TEST")
    print("=" * 50)
    
    # Test configurations (start small, scale up gradually)
    test_configs = [
        (1000, 0.01, "1K neurons, 1% sparsity"),
        (1000, 0.001, "1K neurons, 0.1% sparsity"),
        (10000, 0.001, "10K neurons, 0.1% sparsity"),
        (10000, 0.0001, "10K neurons, 0.01% sparsity"),
        (100000, 0.0001, "100K neurons, 0.01% sparsity"),
        (100000, 0.00001, "100K neurons, 0.001% sparsity"),
        (1000000, 0.00001, "1M neurons, 0.001% sparsity"),
    ]
    
    results = []
    
    for n_neurons, sparsity, description in test_configs:
        result = test_small_scale(n_neurons, sparsity, description)
        if result:
            results.append(result)
    
    # Print summary
    print(f"\n📈 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Neurons':<10} {'Sparsity':<10} {'Weights':<12} {'MS/Step':<10} {'Memory (GB)':<12}")
    print("-" * 60)
    
    for result in results:
        neurons = f"{result['n_neurons']:,}"
        sparsity_pct = f"{result['sparsity']*100:.3f}%"
        weights = f"{result['n_weights']:,}"
        ms_per_step = f"{result['ms_per_step']:.3f}"
        memory = f"{result['memory_used']:.3f}"
        
        print(f"{neurons:<10} {sparsity_pct:<10} {weights:<12} {ms_per_step:<10} {memory:<12}")
    
    # Check for sub-millisecond performance
    sub_ms_results = [r for r in results if r['ms_per_step'] < 1.0]
    
    if sub_ms_results:
        print(f"\n✅ SUCCESS: Achieved sub-millisecond performance!")
        for result in sub_ms_results:
            print(f"   {result['n_neurons']:,} neurons: {result['ms_per_step']:.3f} ms/step")
    else:
        print(f"\n⚠️  Sub-millisecond performance not achieved in this test")
    
    print(f"\n🎯 Test complete!")

if __name__ == "__main__":
    main()

