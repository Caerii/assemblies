#!/usr/bin/env python3
"""
Alternative CuPy test with different initialization approach
"""

import time
import numpy as np

def test_cupy_alternative():
    """Test CuPy with alternative initialization"""
    print("🧪 TESTING CUPY ALTERNATIVE APPROACH")
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
                
                # Try to set device explicitly
                try:
                    cp.cuda.Device(0).use()
                    print("✅ CUDA device 0 set successfully!")
                except Exception as e:
                    print(f"⚠️  Device setting failed: {e}")
                
                # Test basic operations without random generation
                print("Testing basic GPU operations...")
                
                # Test array creation and operations
                gpu_array = cp.array([1, 2, 3, 4, 5])
                print(f"GPU array: {gpu_array}")
                
                # Test mathematical operations
                gpu_array2 = gpu_array * 2
                print(f"GPU array * 2: {gpu_array2}")
                
                # Test array operations
                gpu_sum = cp.sum(gpu_array)
                print(f"GPU array sum: {gpu_sum}")
                
                # Test with NumPy arrays (no random generation)
                print("Testing with NumPy arrays...")
                
                # Create large array on CPU
                cpu_array = np.random.exponential(1.0, 1000000)
                
                # Transfer to GPU
                start_time = time.time()
                gpu_array = cp.asarray(cpu_array)
                transfer_time = time.time() - start_time
                print(f"CPU to GPU transfer: {transfer_time*1000:.2f}ms")
                
                # Test top-k selection on GPU
                start_time = time.time()
                gpu_top_k = cp.argpartition(gpu_array, -1000)[-1000:]
                gpu_sel_time = time.time() - start_time
                print(f"GPU top-k selection: {gpu_sel_time*1000:.2f}ms")
                
                # Transfer back to CPU
                start_time = time.time()
                cpu_result = cp.asnumpy(gpu_top_k)
                transfer_back_time = time.time() - start_time
                print(f"GPU to CPU transfer: {transfer_back_time*1000:.2f}ms")
                
                # Compare with CPU operations
                start_time = time.time()
                cpu_top_k = np.argpartition(cpu_array, -1000)[-1000:]
                cpu_sel_time = time.time() - start_time
                print(f"CPU top-k selection: {cpu_sel_time*1000:.2f}ms")
                
                # Calculate speedup
                total_gpu_time = transfer_time + gpu_sel_time + transfer_back_time
                speedup = cpu_sel_time / total_gpu_time
                
                print(f"\n📊 CUPY SPEEDUP RESULTS:")
                print(f"   Total GPU time: {total_gpu_time*1000:.2f}ms")
                print(f"   CPU time: {cpu_sel_time*1000:.2f}ms")
                print(f"   Speedup: {speedup:.2f}x")
                
                if speedup > 1.0:
                    print(f"   🚀 CuPy is providing GPU acceleration!")
                    return True, speedup
                else:
                    print(f"   ⚠️  Transfer overhead is too high for small arrays")
                    return True, speedup
            else:
                print("❌ No CUDA devices found")
                return False, 0
                
        except Exception as e:
            print(f"❌ CUDA detection failed: {e}")
            return False, 0
            
    except ImportError as e:
        print(f"❌ CuPy import failed: {e}")
        return False, 0
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        return False, 0

def test_cupy_large_arrays():
    """Test CuPy with large arrays to overcome transfer overhead"""
    print(f"\n🚀 TESTING CUPY LARGE ARRAYS")
    print("=" * 50)
    
    try:
        import cupy as cp
        
        # Test with very large arrays
        sizes = [1000000, 5000000, 10000000, 50000000]
        
        for size in sizes:
            print(f"\nTesting {size:,} elements...")
            
            # Create large array on CPU
            cpu_array = np.random.exponential(1.0, size)
            
            # GPU operations
            start_time = time.time()
            gpu_array = cp.asarray(cpu_array)
            gpu_top_k = cp.argpartition(gpu_array, -min(10000, size//10))[-min(10000, size//10):]
            cpu_result = cp.asnumpy(gpu_top_k)
            gpu_time = time.time() - start_time
            
            # CPU operations
            start_time = time.time()
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
        print(f"❌ Large arrays test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        # Test CuPy alternative approach
        success, speedup = test_cupy_alternative()
        
        if success:
            print(f"\n🎉 CUPY ALTERNATIVE APPROACH SUCCESSFUL!")
            
            # Test large arrays
            large_success = test_cupy_large_arrays()
            
            if large_success:
                print(f"\n🏆 CUPY TESTS COMPLETE!")
                print(f"   Ready for GPU-accelerated neural simulation!")
            else:
                print(f"\n⚠️  CuPy basic test passed, but large arrays failed")
        else:
            print(f"\n❌ CuPy alternative approach failed")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
