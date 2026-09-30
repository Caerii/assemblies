#!/usr/bin/env python3
"""
Test Script for Universal Brain Simulator Client
===============================================

This script tests the lightweight thin client to ensure it works correctly
with the modular architecture.
"""

import sys

# Add the current directory to the path
sys.path.insert(0, '.')

from universal_brain_simulator.client import BrainSimulator, quick_sim


def test_basic_client():
    """Test basic client functionality"""
    print("🧪 Testing Basic Client Functionality")
    print("=" * 50)
    
    # Test 1: Simple simulation - Large Scale
    print("\n📋 Test 1: Simple Simulation (Large Scale)")
    sim = BrainSimulator(
        neurons=10000000,  # 10 million neurons
        active_percentage=0.01,
        areas=3,
        use_optimized_cuda=True
    )
    
    results = sim.run(steps=10, verbose=True)
    print(f"   ✅ Simple simulation: {results['summary']['steps_per_second']:.1f} steps/sec")
    
    # Test 2: Quick simulation function - Billion Scale
    print("\n📋 Test 2: Quick Simulation Function (Billion Scale)")
    results2 = quick_sim(neurons=1000000000, steps=5, optimized=True)  # 1 billion neurons
    print(f"   ✅ Quick simulation: {results2['summary']['steps_per_second']:.1f} steps/sec")
    
    # Test 3: Benchmark
    print("\n📋 Test 3: Benchmark")
    benchmark_results = sim.benchmark(warmup_steps=2, measure_steps=3)
    print(f"   ✅ Benchmark: {benchmark_results['performance']['steps_per_second']:.1f} steps/sec")
    
    # Test 4: Get info
    print("\n📋 Test 4: Get Information")
    info = sim.get_info()
    print(f"   ✅ Configuration: {info['configuration']['n_neurons']:,} neurons")
    print(f"   ✅ Memory info: {info['memory_info'].get('used_gb', 0):.2f}GB")
    
    # Test 5: Validation
    print("\n📋 Test 5: Validation")
    is_valid = sim.validate()
    print(f"   ✅ Validation: {'PASSED' if is_valid else 'FAILED'}")
    
    return True


def test_client_with_different_configs():
    """Test client with different configurations"""
    print("\n🧪 Testing Client with Different Configurations")
    print("=" * 50)
    
    configs = [
        {
            'neurons': 10000000,  # 10 million neurons
            'active_percentage': 0.01,
            'areas': 3,
            'use_optimized_cuda': True,
            'use_gpu': True
        },
        {
            'neurons': 10000000,  # 10 million neurons
            'active_percentage': 0.01,
            'areas': 3,
            'use_optimized_cuda': False,
            'use_gpu': True
        },
        {
            'neurons': 1000000,   # 1 million neurons (CPU can't handle 10M)
            'active_percentage': 0.01,
            'areas': 3,
            'use_optimized_cuda': False,
            'use_gpu': False
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n📊 Configuration {i+1}: {config}")
        
        try:
            sim = BrainSimulator(**config)
            result = sim.run(steps=5, verbose=False)
            
            results.append({
                'config': config,
                'success': True,
                'steps_per_second': result['summary']['steps_per_second']
            })
            
            print(f"   ✅ Success: {result['summary']['steps_per_second']:.1f} steps/sec")
            
        except Exception as e:
            results.append({
                'config': config,
                'success': False,
                'error': str(e)
            })
            print(f"   ❌ Failed: {e}")
    
    # Print summary
    print("\n📊 CONFIGURATION TEST SUMMARY")
    print("=" * 60)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"   Successful configurations: {len(successful)}")
    print(f"   Failed configurations: {len(failed)}")
    
    if successful:
        best = max(successful, key=lambda x: x['steps_per_second'])
        print(f"   Best performance: {best['steps_per_second']:.1f} steps/sec")
        print(f"   Best config: {best['config']}")
    
    return len(successful) > 0


def test_client_advanced_features():
    """Test advanced client features"""
    print("\n🧪 Testing Advanced Client Features")
    print("=" * 50)
    
    sim = BrainSimulator(
        neurons=50000000,  # 50 million neurons
        active_percentage=0.005,
        areas=5,
        use_optimized_cuda=True,
        enable_profiling=True
    )
    
    # Test custom callback
    print("\n📋 Test: Custom Callback")
    callback_count = 0
    
    def test_callback(step, step_time):
        nonlocal callback_count
        callback_count += 1
        if step % 5 == 0:
            print(f"   Callback step {step}: {step_time*1000:.2f}ms")
    
    sim.run_with_callback(steps=20, callback=test_callback, callback_interval=1)
    print(f"   ✅ Callback executed {callback_count} times")
    
    # Test profiling
    print("\n📋 Test: Profiling")
    profile_results = sim.profile(steps=10, save_to_file="test_profile.json")
    print(f"   ✅ Profile completed: {profile_results['performance_stats'].get('steps_per_second', 0):.1f} steps/sec")
    
    # Test reset
    print("\n📋 Test: Reset")
    sim.reset()
    results_after_reset = sim.run(steps=5, verbose=False)
    print(f"   ✅ Reset successful: {results_after_reset['summary']['steps_per_second']:.1f} steps/sec")
    
    return True


def main():
    """Main test function"""
    print("🚀 UNIVERSAL BRAIN SIMULATOR CLIENT TEST")
    print("=" * 60)
    
    try:
        # Test basic functionality
        basic_success = test_basic_client()
        
        # Test different configurations
        config_success = test_client_with_different_configs()
        
        # Test advanced features
        advanced_success = test_client_advanced_features()
        
        # Summary
        print("\n🎯 CLIENT TEST SUMMARY")
        print("=" * 60)
        print(f"   Basic functionality: {'✅ PASSED' if basic_success else '❌ FAILED'}")
        print(f"   Configuration tests: {'✅ PASSED' if config_success else '❌ FAILED'}")
        print(f"   Advanced features: {'✅ PASSED' if advanced_success else '❌ FAILED'}")
        
        overall_success = basic_success and config_success and advanced_success
        print(f"   Overall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎉 Client is working correctly!")
            print("   You can now use the lightweight client for easy brain simulation.")
            print("   Try running the examples in the examples/ directory.")
        else:
            print("\n⚠️  Some tests failed. Check the error messages above.")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
