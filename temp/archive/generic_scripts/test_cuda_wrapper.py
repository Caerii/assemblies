#!/usr/bin/env python3
"""
Test CUDA Brain Wrapper for maximum performance
"""

import time
import numpy as np
import ctypes
from ctypes import c_int, c_float, c_uint32, c_void_p, POINTER

def test_cuda_wrapper():
    """Test the CUDA brain wrapper"""
    print("🚀 TESTING CUDA BRAIN WRAPPER")
    print("=" * 50)
    
    try:
        # Load the CUDA wrapper DLL
        dll = ctypes.CDLL('cuda_brain_wrapper.dll')
        
        # Define function signatures
        dll.create_cuda_brain.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
        dll.create_cuda_brain.restype = c_void_p
        
        dll.destroy_cuda_brain.argtypes = [c_void_p]
        dll.destroy_cuda_brain.restype = None
        
        dll.simulate_step.argtypes = [c_void_p]
        dll.simulate_step.restype = None
        
        dll.get_candidates.argtypes = [c_void_p]
        dll.get_candidates.restype = POINTER(c_float)
        
        dll.get_top_k_indices.argtypes = [c_void_p]
        dll.get_top_k_indices.restype = POINTER(c_uint32)
        
        print("✅ CUDA wrapper DLL loaded successfully!")
        
        # Test different scales
        test_cases = [
            {"n_neurons": 50000, "k_active": 500, "n_areas": 3, "name": "Tiny Scale"},
            {"n_neurons": 100000, "k_active": 1000, "n_areas": 3, "name": "Small Scale"},
            {"n_neurons": 500000, "k_active": 5000, "n_areas": 5, "name": "Medium Scale"},
            {"n_neurons": 1000000, "k_active": 10000, "n_areas": 5, "name": "Large Scale"},
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing {test_case['name']}:")
            print(f"   Neurons: {test_case['n_neurons']:,}")
            print(f"   Active: {test_case['k_active']:,}")
            print(f"   Areas: {test_case['n_areas']}")
            
            # Create CUDA brain
            brain_ptr = dll.create_cuda_brain(
                test_case['n_neurons'],
                test_case['k_active'],
                test_case['n_areas'],
                42
            )
            
            if brain_ptr:
                print("   ✅ CUDA brain created successfully!")
                
                # Test single step timing
                step_times = []
                for i in range(10):
                    start_time = time.time()
                    dll.simulate_step(brain_ptr)
                    step_time = time.time() - start_time
                    step_times.append(step_time)
                
                # Calculate statistics
                avg_time = sum(step_times) / len(step_times)
                min_time = min(step_times)
                max_time = max(step_times)
                
                print(f"   📊 CUDA PERFORMANCE:")
                print(f"      Average step time: {avg_time*1000:.2f}ms")
                print(f"      Minimum step time: {min_time*1000:.2f}ms")
                print(f"      Maximum step time: {max_time*1000:.2f}ms")
                print(f"      Steps per second: {1.0/avg_time:.1f}")
                
                # Test data access
                candidates_ptr = dll.get_candidates(brain_ptr)
                top_k_ptr = dll.get_top_k_indices(brain_ptr)
                
                if candidates_ptr and top_k_ptr:
                    print(f"      ✅ Data access working!")
                    
                    # Convert to numpy arrays
                    candidates = np.ctypeslib.as_array(candidates_ptr, shape=(test_case['n_neurons'],))
                    top_k = np.ctypeslib.as_array(top_k_ptr, shape=(test_case['k_active'],))
                    
                    print(f"      Candidates range: [{candidates.min():.3f}, {candidates.max():.3f}]")
                    print(f"      Top-k indices range: [{top_k.min()}, {top_k.max()}]")
                else:
                    print(f"      ❌ Data access failed!")
                
                # Clean up
                dll.destroy_cuda_brain(brain_ptr)
                print(f"      ✅ CUDA brain destroyed")
                
            else:
                print(f"   ❌ Failed to create CUDA brain")
        
        print(f"\n🎉 CUDA WRAPPER TEST COMPLETE!")
        
    except Exception as e:
        print(f"❌ CUDA wrapper test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cuda_wrapper()
