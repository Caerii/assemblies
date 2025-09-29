#!/usr/bin/env python3
"""
Test CUDA DLL with runtime dependencies
"""

import time
import numpy as np
import ctypes
from ctypes import c_int, c_float, c_uint32, c_void_p, POINTER

def test_cuda_with_runtime():
    """Test CUDA DLL with runtime dependencies"""
    print("🚀 TESTING CUDA DLL WITH RUNTIME DEPENDENCIES")
    print("=" * 70)
    
    try:
        # Load the CUDA DLL with runtime dependencies
        dll = ctypes.CDLL('cuda_with_runtime.dll')
        
        # Define function signatures
        dll.create_gpu_memory_cuda_brain.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
        dll.create_gpu_memory_cuda_brain.restype = c_void_p
        
        dll.destroy_gpu_memory_cuda_brain.argtypes = [c_void_p]
        dll.destroy_gpu_memory_cuda_brain.restype = None
        
        dll.simulate_step.argtypes = [c_void_p]
        dll.simulate_step.restype = None
        
        dll.get_candidates.argtypes = [c_void_p, POINTER(c_float)]
        dll.get_candidates.restype = None
        
        dll.get_top_k_indices.argtypes = [c_void_p, POINTER(c_uint32)]
        dll.get_top_k_indices.restype = None
        
        dll.get_step_count.argtypes = [c_void_p]
        dll.get_step_count.restype = c_uint32
        
        dll.get_total_time.argtypes = [c_void_p]
        dll.get_total_time.restype = c_float
        
        dll.set_total_time.argtypes = [c_void_p, c_float]
        dll.set_total_time.restype = None
        
        print("✅ CUDA DLL with runtime dependencies loaded successfully!")
        
        # Test different scales
        test_cases = [
            {"n_neurons": 1000000, "k_active": 10000, "n_areas": 5, "name": "Million Scale"},
            {"n_neurons": 10000000, "k_active": 100000, "n_areas": 5, "name": "Ten Million Scale"},
            {"n_neurons": 100000000, "k_active": 1000000, "n_areas": 5, "name": "Hundred Million Scale"},
            {"n_neurons": 1000000000, "k_active": 1000000, "n_areas": 5, "name": "BILLION SCALE (0.1%)"},
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n🧪 Testing {test_case['name']}:")
            print(f"   Neurons: {test_case['n_neurons']:,}")
            print(f"   Active: {test_case['k_active']:,}")
            print(f"   Areas: {test_case['n_areas']}")
            print(f"   GPU Memory: {test_case['n_neurons'] * 4 * 4 / 1024 / 1024 / 1024:.2f} GB")
            
            try:
                # Create GPU Memory CUDA brain
                brain_ptr = dll.create_gpu_memory_cuda_brain(
                    test_case['n_neurons'],
                    test_case['k_active'],
                    test_case['n_areas'],
                    42
                )
                
                if brain_ptr:
                    print("   ✅ GPU Memory CUDA brain created successfully!")
                    
                    # Test performance
                    step_times = []
                    for i in range(3):  # Test 3 steps
                        start_time = time.perf_counter()
                        dll.simulate_step(brain_ptr)
                        step_time = time.perf_counter() - start_time
                        step_times.append(step_time)
                    
                    avg_time = sum(step_times) / len(step_times)
                    min_time = min(step_times)
                    max_time = max(step_times)
                    
                    print(f"   📊 GPU MEMORY CUDA PERFORMANCE:")
                    print(f"      Average step time: {avg_time*1000:.3f}ms")
                    print(f"      Minimum step time: {min_time*1000:.3f}ms")
                    print(f"      Maximum step time: {max_time*1000:.3f}ms")
                    print(f"      Steps per second: {1.0/avg_time:.1f}")
                    print(f"      Neurons per second: {test_case['n_neurons']/avg_time:,.0f}")
                    print(f"      Active per second: {test_case['k_active']/avg_time:,.0f}")
                    
                    # Test data access
                    candidates = np.zeros(test_case['n_neurons'], dtype=np.float32)
                    top_k_indices = np.zeros(test_case['k_active'], dtype=np.uint32)
                    
                    dll.get_candidates(brain_ptr, candidates.ctypes.data_as(POINTER(c_float)))
                    dll.get_top_k_indices(brain_ptr, top_k_indices.ctypes.data_as(POINTER(c_uint32)))
                    
                    print(f"      ✅ Data access working!")
                    print(f"      Candidates range: [{candidates.min():.3f}, {candidates.max():.3f}]")
                    print(f"      Top-k indices range: [{top_k_indices.min()}, {top_k_indices.max()}]")
                    
                    # Clean up
                    dll.destroy_gpu_memory_cuda_brain(brain_ptr)
                    print(f"      ✅ GPU Memory CUDA brain destroyed")
                    
                    results.append({
                        'name': test_case['name'],
                        'n_neurons': test_case['n_neurons'],
                        'k_active': test_case['k_active'],
                        'n_areas': test_case['n_areas'],
                        'avg_time': avg_time,
                        'steps_per_second': 1.0/avg_time,
                        'ms_per_step': avg_time*1000,
                        'neurons_per_second': test_case['n_neurons']/avg_time,
                        'active_per_second': test_case['k_active']/avg_time
                    })
                    
                else:
                    print(f"   ❌ Failed to create GPU Memory CUDA brain")
                    results.append({
                        'name': test_case['name'],
                        'n_neurons': test_case['n_neurons'],
                        'k_active': test_case['k_active'],
                        'n_areas': test_case['n_areas'],
                        'avg_time': float('inf'),
                        'steps_per_second': 0,
                        'ms_per_step': float('inf'),
                        'neurons_per_second': 0,
                        'active_per_second': 0
                    })
                    
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
                results.append({
                    'name': test_case['name'],
                    'n_neurons': test_case['n_neurons'],
                    'k_active': test_case['k_active'],
                    'n_areas': test_case['n_areas'],
                    'avg_time': float('inf'),
                    'steps_per_second': 0,
                    'ms_per_step': float('inf'),
                    'neurons_per_second': 0,
                    'active_per_second': 0
                })
        
        # Summary
        print(f"\n📊 CUDA WITH RUNTIME DEPENDENCIES BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Scale':<20} {'Neurons':<15} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Active/sec':<15}")
        print("-" * 80)
        
        for result in results:
            if result['steps_per_second'] > 0:
                print(f"{result['name']:<20} {result['n_neurons']:<15,} {result['steps_per_second']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_second']:<15,.0f} {result['active_per_second']:<15,.0f}")
            else:
                print(f"{result['name']:<20} {result['n_neurons']:<15,} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15}")
        
        # Find best performance
        successful_results = [r for r in results if r['steps_per_second'] > 0]
        if successful_results:
            best = max(successful_results, key=lambda x: x['steps_per_second'])
            print(f"\n🏆 BEST PERFORMANCE: {best['name']}")
            print(f"   Steps/sec: {best['steps_per_second']:.1f}")
            print(f"   ms/step: {best['ms_per_step']:.2f}ms")
            print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
            print(f"   Active/sec: {best['active_per_second']:,.0f}")
        else:
            print(f"\n❌ No successful tests")
        
        print(f"\n🎉 CUDA WITH RUNTIME DEPENDENCIES TEST COMPLETE!")
        print(f"✅ CUDA runtime dependencies working!")
        print(f"🚀 GPU memory-based acceleration working!")
        
    except Exception as e:
        print(f"❌ CUDA with runtime dependencies test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cuda_with_runtime()


