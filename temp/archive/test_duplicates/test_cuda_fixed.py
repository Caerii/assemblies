#!/usr/bin/env python3
"""
Test the fixed CUDA DLL
"""

import time
import numpy as np
import ctypes
from ctypes import c_int, c_float, c_uint32, c_void_p, POINTER

def test_cuda_fixed():
    """Test the fixed CUDA DLL"""
    print("🚀 TESTING FIXED CUDA DLL")
    print("=" * 50)
    
    try:
        # Load the fixed CUDA DLL
        dll = ctypes.CDLL('cuda_fixed.dll')
        
        # Define function signatures
        dll.create_cuda_brain.argtypes = [c_uint32, c_uint32, c_uint32]
        dll.create_cuda_brain.restype = c_void_p
        
        dll.destroy_cuda_brain.argtypes = [c_void_p]
        dll.destroy_cuda_brain.restype = None
        
        dll.simulate_step.argtypes = [c_void_p]
        dll.simulate_step.restype = None
        
        dll.get_candidates.argtypes = [c_void_p]
        dll.get_candidates.restype = POINTER(c_float)
        
        print("✅ Fixed CUDA DLL loaded successfully!")
        
        # Test different scales
        test_cases = [
            {"n_neurons": 100000, "k_active": 1000, "name": "Small Scale"},
            {"n_neurons": 1000000, "k_active": 10000, "name": "Million Scale"},
            {"n_neurons": 10000000, "k_active": 100000, "name": "Ten Million Scale"},
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing {test_case['name']}:")
            print(f"   Neurons: {test_case['n_neurons']:,}")
            print(f"   Active: {test_case['k_active']:,}")
            
            try:
                # Create CUDA brain
                brain_ptr = dll.create_cuda_brain(
                    test_case['n_neurons'],
                    test_case['k_active'],
                    42
                )
                
                if brain_ptr:
                    print("   ✅ CUDA brain created successfully!")
                    
                    # Test performance
                    step_times = []
                    for i in range(5):
                        start_time = time.perf_counter()
                        dll.simulate_step(brain_ptr)
                        step_time = time.perf_counter() - start_time
                        step_times.append(step_time)
                    
                    avg_time = sum(step_times) / len(step_times)
                    min_time = min(step_times)
                    max_time = max(step_times)
                    
                    print(f"   📊 CUDA PERFORMANCE:")
                    print(f"      Average step time: {avg_time*1000:.3f}ms")
                    print(f"      Minimum step time: {min_time*1000:.3f}ms")
                    print(f"      Maximum step time: {max_time*1000:.3f}ms")
                    print(f"      Steps per second: {1.0/avg_time:.1f}")
                    print(f"      Neurons per second: {test_case['n_neurons']/avg_time:,.0f}")
                    print(f"      Active per second: {test_case['k_active']/avg_time:,.0f}")
                    
                    # Test data access
                    candidates_ptr = dll.get_candidates(brain_ptr)
                    if candidates_ptr:
                        print(f"      ✅ Data access working!")
                        
                        # Convert to numpy array
                        candidates = np.ctypeslib.as_array(candidates_ptr, shape=(test_case['n_neurons'],))
                        print(f"      Candidates range: [{candidates.min():.3f}, {candidates.max():.3f}]")
                    else:
                        print(f"      ❌ Data access failed!")
                    
                    # Clean up
                    dll.destroy_cuda_brain(brain_ptr)
                    print(f"      ✅ CUDA brain destroyed")
                    
                else:
                    print(f"   ❌ Failed to create CUDA brain")
                    
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
        
        print(f"\n🎉 FIXED CUDA DLL TEST COMPLETE!")
        
    except Exception as e:
        print(f"❌ Fixed CUDA DLL test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cuda_fixed()
