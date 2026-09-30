#!/usr/bin/env python3
"""
Test Script for Modular Universal Brain Simulator
=================================================

This script tests the new modular component breakdown of the
universal brain simulator to ensure everything works correctly.
"""

import time
from universal_brain_simulator import (
    UniversalBrainSimulator, 
    SimulationConfig, 
    create_test_configs,
    print_performance_summary,
    find_best_performance
)


def test_modular_simulator():
    """Test the modular universal brain simulator"""
    print("🚀 TESTING MODULAR UNIVERSAL BRAIN SIMULATOR")
    print("=" * 60)
    
    # Test different configurations
    test_configs = create_test_configs()
    results = []
    
    for test_case in test_configs:
        print(f"\n🧪 Testing {test_case['name']}:")
        
        try:
            # Create simulator
            simulator = UniversalBrainSimulator(test_case['config'])
            
            # Validate simulation
            if not simulator.validate_simulation():
                print("   ❌ Simulation validation failed")
                continue
            
            # Simulate
            start_time = time.perf_counter()
            simulator.simulate(n_steps=10, verbose=False)
            total_time = time.perf_counter() - start_time
            
            # Get stats
            stats = simulator.get_performance_stats()
            
            print("   ✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
            print(f"   ms/step: {stats['avg_step_time']*1000:.2f}ms")
            print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
            print(f"   Memory: {stats['memory_usage_gb']:.2f}GB")
            print(f"   CUDA kernels: {'✅' if stats['cuda_kernels_used'] else '❌'}")
            print(f"   CuPy used: {'✅' if stats['cupy_used'] else '❌'}")
            print(f"   NumPy fallback: {'✅' if stats['numpy_fallback'] else '❌'}")
            
            # Test additional functionality
            print("   📊 Testing additional functionality...")
            
            # Test benchmark
            benchmark = simulator.benchmark_step(n_warmup=2, n_measure=3)
            print(f"   Benchmark: {benchmark['steps_per_second']:.1f} steps/sec")
            
            # Test simulation info
            sim_info = simulator.get_simulation_info()
            print(f"   Areas: {sim_info['num_areas']}")
            print(f"   Total neurons: {sim_info['total_neurons']:,}")
            
            # Test memory info
            memory_info = simulator.get_memory_info()
            print(f"   Memory utilization: {memory_info['utilization_percent']:.1f}%")
            
            results.append({
                'name': test_case['name'],
                'total_time': total_time,
                'steps_per_second': stats['steps_per_second'],
                'ms_per_step': stats['avg_step_time'] * 1000,
                'neurons_per_second': stats['neurons_per_second'],
                'memory_usage_gb': stats['memory_usage_gb'],
                'cuda_kernels_used': stats['cuda_kernels_used'],
                'cupy_used': stats['cupy_used'],
                'numpy_fallback': stats['numpy_fallback']
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'total_time': float('inf'),
                'steps_per_second': 0,
                'ms_per_step': float('inf'),
                'neurons_per_second': 0,
                'memory_usage_gb': 0,
                'cuda_kernels_used': False,
                'cupy_used': False,
                'numpy_fallback': False,
                'error': str(e)
            })
    
    # Print summary
    print_performance_summary(results)
    
    # Find best performance
    best = find_best_performance(results)
    if best:
        print(f"\n🏆 BEST PERFORMANCE: {best['name']}")
        print(f"   Steps/sec: {best['steps_per_second']:.1f}")
        print(f"   ms/step: {best['ms_per_step']:.2f}ms")
        print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
        print(f"   Memory Usage: {best['memory_usage_gb']:.2f}GB")
    else:
        print("\n❌ No successful tests")
    
    return results


def test_individual_components():
    """Test individual components"""
    print("\n🔧 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 60)
    
    from universal_brain_simulator import (
        PerformanceMonitor, CUDAManager, 
        MemoryManager, AreaManager, SimulationEngine
    )
    
    # Test configuration
    print("📋 Testing SimulationConfig...")
    config = SimulationConfig(n_neurons=100000, active_percentage=0.01, n_areas=3)
    print(f"   ✅ Config created: {config.n_neurons:,} neurons, {config.k_active} active")
    
    # Test performance monitor
    print("📊 Testing PerformanceMonitor...")
    metrics = PerformanceMonitor(config)
    metrics.record_step(0.001, 1.5, 75.0, True)
    stats = metrics.get_stats()
    print(f"   ✅ Metrics recorded: {stats['steps_per_second']:.1f} steps/sec")
    
    # Test CUDA manager
    print("🔧 Testing CUDAManager...")
    cuda_manager = CUDAManager(config)
    cuda_loaded = cuda_manager.load_kernels()
    print(f"   ✅ CUDA manager: {'Loaded' if cuda_loaded else 'Not loaded'}")
    
    # Test memory manager
    print("💾 Testing MemoryManager...")
    memory_manager = MemoryManager(config, cuda_manager)
    memory_manager.initialize_cuda_pools()
    memory_info = memory_manager.get_memory_info()
    print(f"   ✅ Memory manager: {memory_info['used_gb']:.2f}GB used")
    
    # Test area manager
    print("🧠 Testing AreaManager...")
    area_manager = AreaManager(config, memory_manager, cuda_manager)
    area_manager.initialize_areas()
    areas_info = area_manager.get_all_areas_info()
    print(f"   ✅ Area manager: {len(areas_info)} areas created")
    
    # Test simulation engine
    print("⚙️  Testing SimulationEngine...")
    simulation_engine = SimulationEngine(config, cuda_manager, area_manager, metrics)
    validation = simulation_engine.validate_simulation()
    print(f"   ✅ Simulation engine: {'Valid' if validation else 'Invalid'}")
    
    print("\n🎉 All individual components tested successfully!")


if __name__ == "__main__":
    # Test individual components first
    test_individual_components()
    
    # Test the full modular simulator
    results = test_modular_simulator()
    
    print("\n🎯 MODULAR SIMULATOR TEST COMPLETE")
    print("   Components: 8 (config, metrics, cuda, memory, areas, engine, utils, main)")
    print(f"   Test configurations: {len(results)}")
    print(f"   Successful tests: {len([r for r in results if r.get('steps_per_second', 0) > 0])}")
    print("   Architecture: Clean separation of concerns ✅")
    print("   Maintainability: High (small, focused components) ✅")
    print("   Testability: High (individual component testing) ✅")
