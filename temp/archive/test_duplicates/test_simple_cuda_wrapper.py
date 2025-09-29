#!/usr/bin/env python3
"""
Test Simple CUDA Brain Wrapper for billion-neuron performance
"""

import time
import numpy as np
import ctypes
from ctypes import c_int, c_float, c_uint32, c_void_p, POINTER

def test_simple_cuda_wrapper():
    """Test the simple CUDA brain wrapper"""
    print("🚀 TESTING SIMPLE CUDA BRAIN WRAPPER")
    print("=" * 60)
    
    try:
        # Load the CUDA wrapper DLL
        dll = ctypes.CDLL('simple_cuda_wrapper.dll')
        
        # Define function signatures
        dll.create_simple_cuda_brain.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
        dll.create_simple_cuda_brain.restype = c_void_p
        
        dll.destroy_simple_cuda_brain.argtypes = [c_void_p]
        dll.destroy_simple_cuda_brain.restype = None
        
        dll.simulate_step.argtypes = [c_void_p]
        dll.simulate_step.restype = None
        
        dll.get_candidates.argtypes = [c_void_p]
        dll.get_candidates.restype = POINTER(c_float)
        
        dll.get_top_k_indices.argtypes = [c_void_p]
        dll.get_top_k_indices.restype = POINTER(c_uint32)
        
        print("✅ Simple CUDA wrapper DLL loaded successfully!")
        
        # Test different scales - including billion-neuron scale
        test_cases = [
            {"n_neurons": 50000, "k_active": 500, "n_areas": 3, "name": "Tiny Scale"},
            {"n_neurons": 100000, "k_active": 1000, "n_areas": 3, "name": "Small Scale"},
            {"n_neurons": 1000000, "k_active": 10000, "n_areas": 5, "name": "Million Scale"},
            {"n_neurons": 10000000, "k_active": 100000, "n_areas": 5, "name": "Ten Million Scale"},
            {"n_neurons": 100000000, "k_active": 1000000, "n_areas": 5, "name": "Hundred Million Scale"},
            {"n_neurons": 1000000000, "k_active": 10000000, "n_areas": 5, "name": "BILLION SCALE"},
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing {test_case['name']}:")
            print(f"   Neurons: {test_case['n_neurons']:,}")
            print(f"   Active: {test_case['k_active']:,}")
            print(f"   Areas: {test_case['n_areas']}")
            
            try:
                # Create CUDA brain
                brain_ptr = dll.create_simple_cuda_brain(
                    test_case['n_neurons'],
                    test_case['k_active'],
                    test_case['n_areas'],
                    42
                )
                
                if brain_ptr:
                    print("   ✅ CUDA brain created successfully!")
                    
                    # Test single step timing
                    step_times = []
                    for i in range(5):  # Test 5 steps
                        start_time = time.perf_counter()
                        dll.simulate_step(brain_ptr)
                        step_time = time.perf_counter() - start_time
                        step_times.append(step_time)
                    
                    # Calculate statistics
                    avg_time = sum(step_times) / len(step_times)
                    min_time = min(step_times)
                    max_time = max(step_times)
                    
                    print(f"   📊 CUDA PERFORMANCE:")
                    print(f"      Average step time: {avg_time*1000:.3f}ms")
                    print(f"      Minimum step time: {min_time*1000:.3f}ms")
                    print(f"      Maximum step time: {max_time*1000:.3f}ms")
                    print(f"      Steps per second: {1.0/avg_time:.1f}")
                    print(f"      Neurons per second: {test_case['n_neurons']/avg_time:,.0f}")
                    print(f"      Active per second: {test_case['k_active']*test_case['n_areas']/avg_time:,.0f}")
                    
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
                    dll.destroy_simple_cuda_brain(brain_ptr)
                    print(f"      ✅ CUDA brain destroyed")
                    
                else:
                    print(f"   ❌ Failed to create CUDA brain")
                    
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
        
        print(f"\n🎉 SIMPLE CUDA WRAPPER TEST COMPLETE!")
        
    except Exception as e:
        print(f"❌ Simple CUDA wrapper test failed: {e}")
        import traceback
        traceback.print_exc()

def test_billion_scale_performance():
    """Test billion-scale performance specifically"""
    print(f"\n🌍 BILLION-SCALE PERFORMANCE TEST")
    print("=" * 60)
    
    try:
        # Load the CUDA wrapper DLL
        dll = ctypes.CDLL('simple_cuda_wrapper.dll')
        
        # Define function signatures
        dll.create_simple_cuda_brain.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
        dll.create_simple_cuda_brain.restype = c_void_p
        
        dll.destroy_simple_cuda_brain.argtypes = [c_void_p]
        dll.destroy_simple_cuda_brain.restype = None
        
        dll.simulate_step.argtypes = [c_void_p]
        dll.simulate_step.restype = None
        
        # Test billion-scale
        n_neurons = 1000000000  # 1 billion
        k_active = 10000000     # 10 million active
        n_areas = 5
        
        print(f"Testing BILLION-SCALE simulation:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active: {k_active:,}")
        print(f"   Areas: {n_areas}")
        
        # Create CUDA brain
        brain_ptr = dll.create_simple_cuda_brain(n_neurons, k_active, n_areas, 42)
        
        if brain_ptr:
            print("   ✅ Billion-scale CUDA brain created!")
            
            # Test performance
            step_times = []
            for i in range(3):  # Test 3 steps
                start_time = time.perf_counter()
                dll.simulate_step(brain_ptr)
                step_time = time.perf_counter() - start_time
                step_times.append(step_time)
                print(f"   Step {i+1}: {step_time*1000:.2f}ms")
            
            avg_time = sum(step_times) / len(step_times)
            
            print(f"\n   📊 BILLION-SCALE PERFORMANCE:")
            print(f"      Average step time: {avg_time*1000:.2f}ms")
            print(f"      Steps per second: {1.0/avg_time:.1f}")
            print(f"      Neurons per second: {n_neurons/avg_time:,.0f}")
            print(f"      Active per second: {k_active*n_areas/avg_time:,.0f}")
            
            # Clean up
            dll.destroy_simple_cuda_brain(brain_ptr)
            print(f"      ✅ Billion-scale CUDA brain destroyed")
            
        else:
            print(f"   ❌ Failed to create billion-scale CUDA brain")
            
    except Exception as e:
        print(f"❌ Billion-scale test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test simple CUDA wrapper
    test_simple_cuda_wrapper()
    
    # Test billion-scale performance
    test_billion_scale_performance()
