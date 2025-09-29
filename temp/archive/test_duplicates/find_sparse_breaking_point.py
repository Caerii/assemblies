#!/usr/bin/env python3
"""
Find Sparse Breaking Point
=========================

This script finds the exact size where scipy.sparse operations start to hang.
"""

import numpy as np
import time
import gc
from scipy.sparse import coo_matrix, csr_matrix

def test_sparse_size(n_active, sparsity, description):
    """Test sparse matrix creation at a specific size."""
    print(f"\n🧪 Testing: {description}")
    print(f"   Active neurons: {n_active:,}")
    print(f"   Sparsity: {sparsity*100:.3f}%")
    
    n_weights = int(n_active * n_active * sparsity)
    print(f"   Non-zero weights: {n_weights:,}")
    
    if n_weights > 10_000_000:  # Skip if too many weights
        print(f"   ⚠️  Skipping (too many weights: {n_weights:,})")
        return None
    
    try:
        start_time = time.time()
        
        print("   Step 1: Creating random indices and values...")
        row_indices = np.random.randint(0, n_active, size=n_weights, dtype=np.int32)
        col_indices = np.random.randint(0, n_active, size=n_weights, dtype=np.int32)
        values = np.random.exponential(1.0, size=n_weights).astype(np.float32)
        print("   ✅ Random data created")
        
        print("   Step 2: Creating COO matrix...")
        coo_matrix_obj = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_active, n_active),
            dtype=np.float32
        )
        print("   ✅ COO matrix created")
        
        print("   Step 3: Converting to CSR...")
        csr_matrix_obj = coo_matrix_obj.tocsr()
        print("   ✅ CSR matrix created")
        
        print("   Step 4: Testing matrix multiplication...")
        test_vector = np.zeros(n_active, dtype=np.float32)
        result = csr_matrix_obj.dot(test_vector)
        print("   ✅ Matrix multiplication successful")
        
        total_time = time.time() - start_time
        print(f"   ✅ Total time: {total_time:.3f} seconds")
        
        # Cleanup
        del coo_matrix_obj, csr_matrix_obj, test_vector, result
        del row_indices, col_indices, values
        gc.collect()
        print("   ✅ Cleanup completed")
        
        return {
            'n_active': n_active,
            'sparsity': sparsity,
            'n_weights': n_weights,
            'time': total_time,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'n_active': n_active,
            'sparsity': sparsity,
            'n_weights': n_weights,
            'error': str(e),
            'success': False
        }

def find_breaking_point():
    """Find the exact breaking point for sparse matrix operations."""
    print("🔍 FINDING SPARSE MATRIX BREAKING POINT")
    print("=" * 50)
    
    # Test different sizes and sparsity levels
    test_configs = [
        # Small tests
        (100, 0.01, "100 neurons, 1% sparsity"),
        (100, 0.1, "100 neurons, 10% sparsity"),
        (1000, 0.01, "1K neurons, 1% sparsity"),
        (1000, 0.001, "1K neurons, 0.1% sparsity"),
        
        # Medium tests
        (5000, 0.001, "5K neurons, 0.1% sparsity"),
        (5000, 0.0001, "5K neurons, 0.01% sparsity"),
        (10000, 0.001, "10K neurons, 0.1% sparsity"),
        (10000, 0.0001, "10K neurons, 0.01% sparsity"),
        
        # Large tests
        (50000, 0.0001, "50K neurons, 0.01% sparsity"),
        (50000, 0.00001, "50K neurons, 0.001% sparsity"),
        (100000, 0.0001, "100K neurons, 0.01% sparsity"),
        (100000, 0.00001, "100K neurons, 0.001% sparsity"),
    ]
    
    results = []
    
    for n_active, sparsity, description in test_configs:
        result = test_sparse_size(n_active, sparsity, description)
        if result:
            results.append(result)
        
        # If we hit an error, stop testing larger sizes
        if result and not result['success']:
            print(f"\n❌ BREAKING POINT FOUND!")
            print(f"   Size: {n_active} neurons")
            print(f"   Sparsity: {sparsity*100:.3f}%")
            print(f"   Weights: {result['n_weights']:,}")
            print(f"   Error: {result['error']}")
            break
    
    return results

def analyze_results(results):
    """Analyze the results to find patterns."""
    print(f"\n📊 ANALYSIS RESULTS")
    print("=" * 50)
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"Successful tests: {len(successful_results)}")
    print(f"Failed tests: {len(failed_results)}")
    
    if successful_results:
        print(f"\n✅ SUCCESSFUL TESTS:")
        print(f"{'Neurons':<10} {'Sparsity':<10} {'Weights':<12} {'Time (s)':<10}")
        print("-" * 50)
        
        for result in successful_results:
            neurons = f"{result['n_active']:,}"
            sparsity = f"{result['sparsity']*100:.3f}%"
            weights = f"{result['n_weights']:,}"
            time_str = f"{result['time']:.3f}"
            
            print(f"{neurons:<10} {sparsity:<10} {weights:<12} {time_str:<10}")
    
    if failed_results:
        print(f"\n❌ FAILED TESTS:")
        print(f"{'Neurons':<10} {'Sparsity':<10} {'Weights':<12} {'Error':<20}")
        print("-" * 60)
        
        for result in failed_results:
            neurons = f"{result['n_active']:,}"
            sparsity = f"{result['sparsity']*100:.3f}%"
            weights = f"{result['n_weights']:,}"
            error = result['error'][:20] + "..." if len(result['error']) > 20 else result['error']
            
            print(f"{neurons:<10} {sparsity:<10} {weights:<12} {error:<20}")
    
    # Find the largest successful test
    if successful_results:
        largest_success = max(successful_results, key=lambda x: x['n_weights'])
        print(f"\n🎯 LARGEST SUCCESSFUL TEST:")
        print(f"   Neurons: {largest_success['n_active']:,}")
        print(f"   Sparsity: {largest_success['sparsity']*100:.3f}%")
        print(f"   Weights: {largest_success['n_weights']:,}")
        print(f"   Time: {largest_success['time']:.3f} seconds")
        
        # Calculate theoretical billion-scale requirements
        billion_neurons = 1_000_000_000
        billion_active = int(billion_neurons * 0.001)  # 0.1% active
        billion_weights = int(billion_active * billion_active * largest_success['sparsity'])
        
        print(f"\n🚀 BILLION-SCALE PROJECTION:")
        print(f"   Billion neurons: {billion_neurons:,}")
        print(f"   Billion active: {billion_active:,}")
        print(f"   Billion weights: {billion_weights:,}")
        print(f"   Estimated time: {largest_success['time'] * (billion_weights / largest_success['n_weights']):.3f} seconds")

def main():
    """Main function."""
    print("🔍 FINDING SPARSE MATRIX BREAKING POINT")
    print("=" * 60)
    
    # Find breaking point
    results = find_breaking_point()
    
    # Analyze results
    analyze_results(results)
    
    print(f"\n✅ Breaking point analysis complete!")

if __name__ == "__main__":
    main()

