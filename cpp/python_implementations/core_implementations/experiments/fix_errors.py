#!/usr/bin/env python3
"""
Fix Identified Errors
====================

Fix the errors identified in the bottleneck analysis:
1. 'BrainSimulator' object has no attribute 'simulate_step'
2. BrainSimulator.__init__() got an unexpected keyword argument 'use_cuda_kernels'
"""

import sys
sys.path.insert(0, '.')

from universal_brain_simulator.client import BrainSimulator


def test_simulate_step_error():
    """Test and fix the simulate_step error"""
    print("🔧 TESTING SIMULATE_STEP ERROR")
    print("=" * 50)
    
    try:
        sim = BrainSimulator(
            neurons=1000000,
            active_percentage=0.01,
            areas=3,
            use_optimized_cuda=True
        )
        
        # Check if simulate_step method exists
        if hasattr(sim, 'simulate_step'):
            print("   ✅ simulate_step method exists")
            
            # Test the method
            step_time = sim.simulate_step()
            print(f"   ✅ simulate_step works: {step_time:.3f}s")
            
        else:
            print("   ❌ simulate_step method missing")
            print("   Available methods:", [method for method in dir(sim) if not method.startswith('_')])
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def test_use_cuda_kernels_error():
    """Test and fix the use_cuda_kernels error"""
    print("\n🔧 TESTING USE_CUDA_KERNELS ERROR")
    print("=" * 50)
    
    try:
        # Test with use_cuda_kernels parameter
        sim = BrainSimulator(
            neurons=1000000,
            active_percentage=0.01,
            areas=3,
            use_optimized_cuda=False,
            use_cuda_kernels=False  # This should work
        )
        
        print("   ✅ use_cuda_kernels parameter works")
        
        # Test simulation
        results = sim.run(steps=5, verbose=False)
        print(f"   ✅ Simulation works: {results['summary']['steps_per_sec']:.1f} steps/sec")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   This parameter might not be supported in the current client")


def test_client_interface():
    """Test the complete client interface"""
    print("\n🔧 TESTING CLIENT INTERFACE")
    print("=" * 50)
    
    try:
        # Test basic functionality
        sim = BrainSimulator(
            neurons=1000000,
            active_percentage=0.01,
            areas=3,
            use_optimized_cuda=True
        )
        
        print("   ✅ Basic initialization works")
        
        # Test all available methods
        methods_to_test = [
            'run',
            'benchmark', 
            'profile',
            'get_info',
            'validate',
            'reset'
        ]
        
        for method_name in methods_to_test:
            if hasattr(sim, method_name):
                print(f"   ✅ {method_name} method exists")
            else:
                print(f"   ❌ {method_name} method missing")
        
        # Test run method
        results = sim.run(steps=5, verbose=False)
        print(f"   ✅ run method works: {results['summary']['steps_per_sec']:.1f} steps/sec")
        
        # Test benchmark method
        benchmark_results = sim.benchmark(warmup_steps=2, measure_steps=3)
        print(f"   ✅ benchmark method works: {benchmark_results['performance']['steps_per_sec']:.1f} steps/sec")
        
        # Test get_info method
        info = sim.get_info()
        print(f"   ✅ get_info method works: {info['configuration']['n_neurons']:,} neurons")
        
        # Test validate method
        is_valid = sim.validate()
        print(f"   ✅ validate method works: {'PASSED' if is_valid else 'FAILED'}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def test_underlying_simulator():
    """Test the underlying UniversalBrainSimulator"""
    print("\n🔧 TESTING UNDERLYING SIMULATOR")
    print("=" * 50)
    
    try:
        from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig
        
        # Create configuration
        config = SimulationConfig(
            n_neurons=1000000,
            active_percentage=0.01,
            n_areas=3,
            use_optimized_kernels=True
        )
        
        # Create simulator
        sim = UniversalBrainSimulator(config)
        print("   ✅ UniversalBrainSimulator initialization works")
        
        # Test simulate_step method
        if hasattr(sim, 'simulate_step'):
            step_time = sim.simulate_step()
            print(f"   ✅ simulate_step works: {step_time:.3f}s")
        else:
            print("   ❌ simulate_step method missing")
        
        # Test simulate method
        if hasattr(sim, 'simulate'):
            total_time = sim.simulate(n_steps=5, verbose=False)
            print(f"   ✅ simulate method works: {total_time:.3f}s")
        else:
            print("   ❌ simulate method missing")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def main():
    """Run error fixing tests"""
    print("🔧 ERROR FIXING AND TESTING")
    print("=" * 80)
    print("Testing and fixing identified errors...")
    
    # Test all error scenarios
    test_simulate_step_error()
    test_use_cuda_kernels_error()
    test_client_interface()
    test_underlying_simulator()
    
    print("\n🎯 Error testing complete!")
    print("   Check the results above to see which errors are fixed.")
    print("   Any remaining errors need to be addressed in the code.")


if __name__ == "__main__":
    main()
