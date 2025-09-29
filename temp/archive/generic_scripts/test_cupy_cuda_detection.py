#!/usr/bin/env python3
"""
Test CuPy with explicit CUDA detection
"""

import time
import numpy as np

def test_cupy_cuda_detection():
    """Test CuPy CUDA detection and basic functionality"""
    print("🧪 TESTING CUPY CUDA DETECTION")
    print("=" * 50)
    
    try:
        import cupy as cp
        print("✅ CuPy imported successfully!")
        
        # Test CUDA device detection
        print("Testing CUDA device detection...")
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"✅ CUDA devices detected: {device_count}")
            
            if device_count > 0:
                print("✅ CUDA is accessible!")
                
                # Test basic GPU operations
                print("Testing basic GPU operations...")
                
                # Test array creation on GPU
                gpu_array = cp.array([1, 2, 3, 4, 5])
                print(f"GPU array: {gpu_array}")
                
                # Test random generation
                start_time = time.time()
                gpu_random = cp.random.exponential(1.0, 1000000)
                gen_time = time.time() - start_time
                print(f"GPU random generation (1M): {gen_time*1000:.2f}ms")
                
                # Test top-k selection
                start_time = time.time()
                top_k = cp.argpartition(gpu_random, -1000)[-1000:]
                sel_time = time.time() - start_time
                print(f"GPU top-k selection: {sel_time*1000:.2f}ms")
                
                # Compare with NumPy
                print("\nComparing with NumPy...")
                
                # NumPy operations
                start_time = time.time()
                cpu_random = np.random.exponential(1.0, 1000000)
                cpu_gen_time = time.time() - start_time
                
                start_time = time.time()
                cpu_top_k = np.argpartition(cpu_random, -1000)[-1000:]
                cpu_sel_time = time.time() - start_time
                
                print(f"CPU random generation (1M): {cpu_gen_time*1000:.2f}ms")
                print(f"CPU top-k selection: {cpu_sel_time*1000:.2f}ms")
                
                # Calculate speedup
                gen_speedup = cpu_gen_time / gen_time
                sel_speedup = cpu_sel_time / sel_time
                
                print(f"\n📊 CUPY SPEEDUP RESULTS:")
                print(f"   Random generation: {gen_speedup:.2f}x faster")
                print(f"   Top-k selection: {sel_speedup:.2f}x faster")
                
                if gen_speedup > 1.5 or sel_speedup > 1.5:
                    print(f"   🚀 CuPy is providing significant GPU acceleration!")
                    return True, gen_speedup, sel_speedup
                else:
                    print(f"   ⚠️  CuPy speedup is minimal - may need larger arrays")
                    return True, gen_speedup, sel_speedup
            else:
                print("❌ No CUDA devices found")
                return False, 0, 0
                
        except Exception as e:
            print(f"❌ CUDA detection failed: {e}")
            return False, 0, 0
            
    except ImportError as e:
        print(f"❌ CuPy import failed: {e}")
        return False, 0, 0
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        return False, 0, 0

def test_cupy_large_scale():
    """Test CuPy with large-scale operations"""
    print(f"\n🚀 TESTING CUPY LARGE SCALE")
    print("=" * 50)
    
    try:
        import cupy as cp
        
        # Test different array sizes
        sizes = [1000000, 5000000, 10000000, 50000000]
        
        for size in sizes:
            print(f"\nTesting {size:,} elements...")
            
            # GPU operations
            start_time = time.time()
            gpu_array = cp.random.exponential(1.0, size)
            gpu_top_k = cp.argpartition(gpu_array, -min(10000, size//10))[-min(10000, size//10):]
            gpu_time = time.time() - start_time
            
            # CPU operations
            start_time = time.time()
            cpu_array = np.random.exponential(1.0, size)
            cpu_top_k = np.argpartition(cpu_array, -min(10000, size//10))[-min(10000, size//10):]
            cpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
            
            print(f"   GPU time: {gpu_time*1000:.2f}ms")
            print(f"   CPU time: {cpu_time*1000:.2f}ms")
            print(f"   Speedup: {speedup:.2f}x")
            
            if speedup > 2.0:
                print(f"   🚀 Excellent GPU acceleration!")
            elif speedup > 1.5:
                print(f"   ⚡ Good GPU acceleration!")
            else:
                print(f"   ⚠️  Minimal GPU acceleration")
        
        return True
        
    except Exception as e:
        print(f"❌ Large scale test failed: {e}")
        return False

def test_cupy_neural_simulation():
    """Test CuPy for neural simulation operations"""
    print(f"\n🧠 TESTING CUPY NEURAL SIMULATION")
    print("=" * 50)
    
    try:
        import cupy as cp
        
        # Simulate neural simulation operations
        n_neurons = 1000000
        k_active = 10000
        
        print(f"Simulating {n_neurons:,} neurons with {k_active:,} active...")
        
        # GPU neural simulation
        start_time = time.time()
        
        # Generate candidates
        gpu_candidates = cp.random.exponential(1.0, n_neurons)
        
        # Select top-k
        gpu_top_k = cp.argpartition(gpu_candidates, -k_active)[-k_active:]
        
        # Sort top-k
        gpu_top_k_values = gpu_candidates[gpu_top_k]
        gpu_sorted_indices = cp.argsort(gpu_top_k_values)[::-1]
        gpu_final_selection = gpu_top_k[gpu_sorted_indices]
        
        gpu_time = time.time() - start_time
        
        # CPU neural simulation
        start_time = time.time()
        
        # Generate candidates
        cpu_candidates = np.random.exponential(1.0, n_neurons)
        
        # Select top-k
        cpu_top_k = np.argpartition(cpu_candidates, -k_active)[-k_active:]
        
        # Sort top-k
        cpu_top_k_values = cpu_candidates[cpu_top_k]
        cpu_sorted_indices = np.argsort(cpu_top_k_values)[::-1]
        cpu_final_selection = cpu_top_k[cpu_sorted_indices]
        
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        
        print(f"   GPU neural simulation: {gpu_time*1000:.2f}ms")
        print(f"   CPU neural simulation: {cpu_time*1000:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 2.0:
            print(f"   🚀 Excellent neural simulation acceleration!")
            return True, speedup
        elif speedup > 1.5:
            print(f"   ⚡ Good neural simulation acceleration!")
            return True, speedup
        else:
            print(f"   ⚠️  Minimal neural simulation acceleration")
            return True, speedup
        
    except Exception as e:
        print(f"❌ Neural simulation test failed: {e}")
        return False, 0

if __name__ == "__main__":
    try:
        # Test CuPy CUDA detection
        success, gen_speedup, sel_speedup = test_cupy_cuda_detection()
        
        if success:
            print(f"\n🎉 CUPY CUDA DETECTION SUCCESSFUL!")
            
            # Test large scale
            large_success = test_cupy_large_scale()
            
            if large_success:
                # Test neural simulation
                neural_success, neural_speedup = test_cupy_neural_simulation()
                
                if neural_success:
                    print(f"\n🏆 CUPY TESTS COMPLETE!")
                    print(f"   Ready for GPU-accelerated neural simulation!")
                    print(f"   Neural simulation speedup: {neural_speedup:.2f}x")
                else:
                    print(f"\n⚠️  CuPy basic test passed, but neural simulation failed")
            else:
                print(f"\n⚠️  CuPy basic test passed, but large scale failed")
        else:
            print(f"\n❌ CuPy CUDA detection failed")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
