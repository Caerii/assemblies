#!/usr/bin/env python3
"""
Performance Comparison Example - Universal Brain Simulator
========================================================

This example compares different configurations to help you find the
optimal setup for your specific hardware and use case.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from universal_brain_simulator.client import BrainSimulator


def main():
    print("⚡ PERFORMANCE COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Define test configurations - Large Scale
    configurations = [
        {
            'name': 'Optimized CUDA (O(N log K)) - 100M',
            'config': {
                'neurons': 100000000,  # 100 million neurons
                'active_percentage': 0.01,
                'areas': 5,
                'use_optimized_cuda': True,
                'use_gpu': True,
                'memory_efficient': True,
                'sparse_mode': True
            }
        },
        {
            'name': 'Optimized CUDA (O(N log K)) - 1B',
            'config': {
                'neurons': 1000000000,  # 1 billion neurons
                'active_percentage': 0.01,
                'areas': 5,
                'use_optimized_cuda': True,
                'use_gpu': True,
                'memory_efficient': True,
                'sparse_mode': True
            }
        },
        {
            'name': 'Original CUDA (O(N²)) - 50M',
            'config': {
                'neurons': 50000000,  # 50 million neurons (original CUDA can't handle 100M+)
                'active_percentage': 0.01,
                'areas': 5,
                'use_optimized_cuda': False,
                'use_gpu': True,
                'memory_efficient': True,
                'sparse_mode': True
            }
        },
        {
            'name': 'CuPy Only (No CUDA Kernels) - 50M',
            'config': {
                'neurons': 50000000,  # 50 million neurons
                'active_percentage': 0.01,
                'areas': 5,
                'use_optimized_cuda': False,
                'use_gpu': True,
                'use_cuda_kernels': False,
                'memory_efficient': True,
                'sparse_mode': True
            }
        },
        {
            'name': 'CPU Only (NumPy) - 10M',
            'config': {
                'neurons': 10000000,  # 10 million neurons (CPU can't handle more)
                'active_percentage': 0.01,
                'areas': 5,
                'use_optimized_cuda': False,
                'use_gpu': False,
                'use_cuda_kernels': False,
                'memory_efficient': True,
                'sparse_mode': True
            }
        }
    ]
    
    # Run comparisons
    print("\n🧪 Running performance comparisons...")
    results = []
    
    for config_info in configurations:
        name = config_info['name']
        config = config_info['config']
        
        print(f"\n📊 Testing: {name}")
        print(f"   Config: {config}")
        
        try:
            sim = BrainSimulator(**config)
            
            # Run benchmark
            benchmark_results = sim.benchmark(warmup_steps=3, measure_steps=5)
            
            # Get additional info
            info = sim.get_info()
            
            results.append({
                'name': name,
                'config': config,
                'benchmark': benchmark_results,
                'info': info,
                'success': True
            })
            
            print(f"   ✅ Success: {benchmark_results['performance']['steps_per_second']:.1f} steps/sec")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': name,
                'config': config,
                'error': str(e),
                'success': False
            })
    
    # Print detailed comparison
    print("\n📊 DETAILED PERFORMANCE COMPARISON")
    print("=" * 100)
    print(f"{'Configuration':<25} {'Steps/sec':<10} {'Neurons/sec':<15} {'Step Time (ms)':<15} {'Memory (GB)':<12} {'CUDA':<6} {'Status':<8}")
    print("-" * 100)
    
    successful_results = [r for r in results if r['success']]
    
    for result in results:
        name = result['name']
        if result['success']:
            benchmark = result['benchmark']['performance']
            info = result['info']
            memory_info = info['memory_info']
            
            print(f"{name:<25} {benchmark['steps_per_second']:<10.1f} {benchmark['neurons_per_second']:<15,.0f} {benchmark['average_step_time_ms']:<15.2f} {memory_info.get('used_gb', 0):<12.2f} {'✅' if result['config'].get('use_optimized_cuda', False) or result['config'].get('use_cuda_kernels', False) else '❌':<6} {'✅':<8}")
        else:
            print(f"{name:<25} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15} {'FAILED':<12} {'❌':<6} {'❌':<8}")
    
    # Find best configuration
    if successful_results:
        best_result = max(successful_results, 
                         key=lambda x: x['benchmark']['performance']['steps_per_second'])
        
        print(f"\n🏆 BEST CONFIGURATION: {best_result['name']}")
        print(f"   Steps/sec: {best_result['benchmark']['performance']['steps_per_second']:.1f}")
        print(f"   Neurons/sec: {best_result['benchmark']['performance']['neurons_per_second']:,.0f}")
        print(f"   Step time: {best_result['benchmark']['performance']['average_step_time_ms']:.2f}ms")
        print(f"   Memory usage: {best_result['info']['memory_info'].get('used_gb', 0):.2f}GB")
        
        # Calculate speedup vs other configurations
        print("\n📈 SPEEDUP COMPARISON")
        print("=" * 60)
        best_steps_per_sec = best_result['benchmark']['performance']['steps_per_second']
        
        for result in successful_results:
            if result != best_result:
                speedup = best_steps_per_sec / result['benchmark']['performance']['steps_per_second']
                print(f"   {best_result['name']} vs {result['name']}: {speedup:.1f}x faster")
    
    # Test scaling with best configuration
    if successful_results:
        print("\n📈 SCALING TEST WITH BEST CONFIGURATION")
        print("=" * 60)
        
        best_config = best_result['config']
        neuron_counts = [1000000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000]  # 1M to 1B
        
        for neurons in neuron_counts:
            print(f"\n🧪 Testing {neurons:,} neurons...")
            
            # Create config with different neuron count
            scaling_config = best_config.copy()
            scaling_config['neurons'] = neurons
            
            try:
                sim = BrainSimulator(**scaling_config)
                results = sim.benchmark(warmup_steps=2, measure_steps=3)
                
                print(f"   Steps/sec: {results['performance']['steps_per_second']:.1f}")
                print(f"   Neurons/sec: {results['performance']['neurons_per_second']:,.0f}")
                print(f"   Step time: {results['performance']['average_step_time_ms']:.2f}ms")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
    
    # Memory efficiency analysis
    print("\n💾 MEMORY EFFICIENCY ANALYSIS")
    print("=" * 60)
    
    for result in successful_results:
        name = result['name']
        memory_info = result['info']['memory_info']
        config = result['config']
        
        neurons = config['neurons']
        areas = config['areas']
        active_pct = config['active_percentage']
        total_active = int(neurons * active_pct * areas)
        
        memory_gb = memory_info.get('used_gb', 0)
        bytes_per_neuron = (memory_gb * 1024**3) / neurons if neurons > 0 else 0
        
        print(f"   {name}:")
        print(f"     Memory: {memory_gb:.2f}GB")
        print(f"     Bytes per neuron: {bytes_per_neuron:.1f}")
        print(f"     Total active neurons: {total_active:,}")
        print(f"     Memory per active neuron: {(memory_gb * 1024**3) / total_active:.1f} bytes" if total_active > 0 else "     Memory per active neuron: N/A")
    
    print("\n🎯 Performance comparison complete!")
    print(f"   Configurations tested: {len(configurations)}")
    print(f"   Successful tests: {len(successful_results)}")
    print(f"   Failed tests: {len(results) - len(successful_results)}")


if __name__ == "__main__":
    main()
