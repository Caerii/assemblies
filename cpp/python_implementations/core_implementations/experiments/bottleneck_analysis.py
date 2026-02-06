#!/usr/bin/env python3
"""
Performance Bottleneck Analysis
==============================

Comprehensive analysis to identify performance bottlenecks at different scales.
"""

import sys
import time
import json
from typing import Dict, List, Any
sys.path.insert(0, '.')

from universal_brain_simulator.client import BrainSimulator


def analyze_memory_bottleneck():
    """Analyze memory usage patterns across scales"""
    print("🔍 MEMORY BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Test different neuron counts with detailed memory tracking
    neuron_counts = [1000000, 5000000, 10000000, 25000000, 50000000, 75000000, 100000000, 250000000, 500000000, 1000000000]
    
    results = []
    
    for neurons in neuron_counts:
        print(f"\n🧪 Testing {neurons:,} neurons...")
        
        try:
            sim = BrainSimulator(
                neurons=neurons,
                active_percentage=0.01,
                areas=5,
                use_optimized_cuda=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            # Get memory info before simulation
            info = sim.get_info()
            initial_memory = info['memory_info']['used_gb']
            
            # Run simulation
            start_time = time.perf_counter()
            results_sim = sim.run(steps=3, verbose=False)
            end_time = time.perf_counter()
            
            # Get memory info after simulation
            info_after = sim.get_info()
            final_memory = info_after['memory_info']['used_gb']
            
            # Calculate metrics
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            neurons_per_sec = results_sim['summary']['neurons_per_second']
            memory_per_neuron = (final_memory * 1024**3) / neurons  # bytes per neuron
            
            result = {
                'neurons': neurons,
                'steps_per_sec': steps_per_sec,
                'neurons_per_sec': neurons_per_sec,
                'initial_memory_gb': initial_memory,
                'final_memory_gb': final_memory,
                'memory_per_neuron_bytes': memory_per_neuron,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Memory: {final_memory:.2f}GB ({memory_per_neuron:.1f} bytes/neuron)")
            print(f"   Time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'neurons': neurons,
                'error': str(e),
                'success': False
            })
    
    return results


def analyze_active_percentage_bottleneck():
    """Analyze how active percentage affects performance"""
    print("\n🔍 ACTIVE PERCENTAGE BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Test different active percentages with fixed neuron count
    base_neurons = 100000000  # 100M neurons
    active_percentages = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    results = []
    
    for active_pct in active_percentages:
        active_neurons = int(base_neurons * active_pct)
        print(f"\n🧪 Testing {active_pct*100:.1f}% active ({active_neurons:,} active neurons)...")
        
        try:
            sim = BrainSimulator(
                neurons=base_neurons,
                active_percentage=active_pct,
                areas=5,
                use_optimized_cuda=True
            )
            
            start_time = time.perf_counter()
            results_sim = sim.run(steps=3, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            neurons_per_sec = results_sim['summary']['neurons_per_second']
            
            result = {
                'active_percentage': active_pct,
                'active_neurons': active_neurons,
                'steps_per_sec': steps_per_sec,
                'neurons_per_sec': neurons_per_sec,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Neurons/sec: {neurons_per_sec:,.0f}")
            print(f"   Time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'active_percentage': active_pct,
                'active_neurons': active_neurons,
                'error': str(e),
                'success': False
            })
    
    return results


def analyze_area_bottleneck():
    """Analyze how number of areas affects performance"""
    print("\n🔍 AREA COUNT BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Test different area counts with fixed total neurons
    base_neurons = 100000000  # 100M neurons
    area_counts = [1, 2, 3, 5, 10, 20, 50, 100]
    
    results = []
    
    for areas in area_counts:
        neurons_per_area = base_neurons // areas
        total_active = int(base_neurons * 0.01)  # 1% active total
        
        print(f"\n🧪 Testing {areas} areas ({neurons_per_area:,} neurons/area)...")
        
        try:
            sim = BrainSimulator(
                neurons=base_neurons,
                active_percentage=0.01,
                areas=areas,
                use_optimized_cuda=True
            )
            
            start_time = time.perf_counter()
            results_sim = sim.run(steps=3, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            neurons_per_sec = results_sim['summary']['neurons_per_second']
            
            result = {
                'areas': areas,
                'neurons_per_area': neurons_per_area,
                'total_active': total_active,
                'steps_per_sec': steps_per_sec,
                'neurons_per_sec': neurons_per_sec,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Neurons/sec: {neurons_per_sec:,.0f}")
            print(f"   Time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'areas': areas,
                'neurons_per_area': neurons_per_area,
                'total_active': total_active,
                'error': str(e),
                'success': False
            })
    
    return results


def analyze_algorithm_bottleneck():
    """Compare different algorithms to identify bottlenecks"""
    print("\n🔍 ALGORITHM BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Test different configurations with same neuron count
    test_neurons = 50000000  # 50M neurons
    configurations = [
        {
            'name': 'Optimized CUDA (O(N log K))',
            'config': {
                'neurons': test_neurons,
                'active_percentage': 0.01,
                'areas': 5,
                'use_optimized_cuda': True,
                'use_gpu': True,
                'memory_efficient': True,
                'sparse_mode': True
            }
        },
        {
            'name': 'Original CUDA (O(N²))',
            'config': {
                'neurons': test_neurons,
                'active_percentage': 0.01,
                'areas': 5,
                'use_optimized_cuda': False,
                'use_gpu': True,
                'memory_efficient': True,
                'sparse_mode': True
            }
        },
        {
            'name': 'CuPy Only',
            'config': {
                'neurons': test_neurons,
                'active_percentage': 0.01,
                'areas': 5,
                'use_optimized_cuda': False,
                'use_gpu': True,
                'use_cuda_kernels': False,
                'memory_efficient': True,
                'sparse_mode': True
            }
        }
    ]
    
    results = []
    
    for config_info in configurations:
        name = config_info['name']
        config = config_info['config']
        
        print(f"\n🧪 Testing {name}...")
        
        try:
            sim = BrainSimulator(**config)
            
            start_time = time.perf_counter()
            results_sim = sim.run(steps=5, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            neurons_per_sec = results_sim['summary']['neurons_per_second']
            memory_gb = results_sim['summary']['memory_usage_gb']
            
            result = {
                'name': name,
                'config': config,
                'steps_per_sec': steps_per_sec,
                'neurons_per_sec': neurons_per_sec,
                'memory_gb': memory_gb,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Neurons/sec: {neurons_per_sec:,.0f}")
            print(f"   Memory: {memory_gb:.2f}GB")
            print(f"   Time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': name,
                'config': config,
                'error': str(e),
                'success': False
            })
    
    return results


def analyze_step_time_breakdown():
    """Analyze step time breakdown at different scales"""
    print("\n🔍 STEP TIME BREAKDOWN ANALYSIS")
    print("=" * 60)
    
    # Test step time at different scales
    neuron_counts = [1000000, 10000000, 50000000, 100000000, 500000000, 1000000000]
    
    results = []
    
    for neurons in neuron_counts:
        print(f"\n🧪 Analyzing step time for {neurons:,} neurons...")
        
        try:
            sim = BrainSimulator(
                neurons=neurons,
                active_percentage=0.01,
                areas=5,
                use_optimized_cuda=True
            )
            
            # Run multiple steps to get timing statistics
            step_times = []
            for step in range(10):
                start_time = time.perf_counter()
                sim.simulate_step()
                end_time = time.perf_counter()
                step_times.append(end_time - start_time)
            
            avg_step_time = sum(step_times) / len(step_times)
            min_step_time = min(step_times)
            max_step_time = max(step_times)
            steps_per_sec = 1.0 / avg_step_time
            
            result = {
                'neurons': neurons,
                'avg_step_time_ms': avg_step_time * 1000,
                'min_step_time_ms': min_step_time * 1000,
                'max_step_time_ms': max_step_time * 1000,
                'steps_per_sec': steps_per_sec,
                'step_times': step_times,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Avg step time: {avg_step_time*1000:.2f}ms")
            print(f"   Min step time: {min_step_time*1000:.2f}ms")
            print(f"   Max step time: {max_step_time*1000:.2f}ms")
            print(f"   Steps/sec: {steps_per_sec:.1f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'neurons': neurons,
                'error': str(e),
                'success': False
            })
    
    return results


def save_analysis_results(all_results: Dict[str, List[Any]]):
    """Save all analysis results to JSON file"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"bottleneck_analysis_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis results saved to {filename}")


def print_bottleneck_summary(all_results: Dict[str, List[Any]]):
    """Print summary of identified bottlenecks"""
    print("\n🎯 BOTTLENECK ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Memory bottleneck analysis
    memory_results = all_results.get('memory', [])
    if memory_results:
        print("\n📊 MEMORY BOTTLENECKS:")
        successful = [r for r in memory_results if r.get('success', False)]
        if successful:
            # Find where memory usage becomes problematic
            high_memory = [r for r in successful if r['memory_per_neuron_bytes'] > 100]
            if high_memory:
                worst = max(high_memory, key=lambda x: x['memory_per_neuron_bytes'])
                print(f"   ⚠️  High memory usage at {worst['neurons']:,} neurons: {worst['memory_per_neuron_bytes']:.1f} bytes/neuron")
            
            # Find performance drop
            if len(successful) >= 2:
                first = successful[0]
                last = successful[-1]
                scale_factor = last['neurons'] / first['neurons']
                perf_ratio = last['steps_per_sec'] / first['steps_per_sec']
                efficiency = (perf_ratio / scale_factor) * 100
                print(f"   📉 Scaling efficiency: {efficiency:.1f}% (from {first['neurons']:,} to {last['neurons']:,} neurons)")
    
    # Active percentage analysis
    active_results = all_results.get('active_percentage', [])
    if active_results:
        print("\n📊 ACTIVE PERCENTAGE BOTTLENECKS:")
        successful = [r for r in active_results if r.get('success', False)]
        if successful:
            # Find where performance drops significantly
            perf_by_active = [(r['active_percentage'], r['steps_per_sec']) for r in successful]
            perf_by_active.sort()
            
            if len(perf_by_active) >= 2:
                first_perf = perf_by_active[0][1]
                last_perf = perf_by_active[-1][1]
                perf_drop = (first_perf - last_perf) / first_perf * 100
                print(f"   📉 Performance drop with higher activity: {perf_drop:.1f}%")
    
    # Area count analysis
    area_results = all_results.get('areas', [])
    if area_results:
        print("\n📊 AREA COUNT BOTTLENECKS:")
        successful = [r for r in area_results if r.get('success', False)]
        if successful:
            # Find optimal area count
            best_area = max(successful, key=lambda x: x['steps_per_sec'])
            print(f"   🎯 Optimal area count: {best_area['areas']} areas ({best_area['steps_per_sec']:.1f} steps/sec)")
    
    # Algorithm comparison
    algo_results = all_results.get('algorithms', [])
    if algo_results:
        print("\n📊 ALGORITHM BOTTLENECKS:")
        successful = [r for r in algo_results if r.get('success', False)]
        if successful:
            best_algo = max(successful, key=lambda x: x['steps_per_sec'])
            worst_algo = min(successful, key=lambda x: x['steps_per_sec'])
            speedup = best_algo['steps_per_sec'] / worst_algo['steps_per_sec']
            print(f"   🚀 Best algorithm: {best_algo['name']} ({best_algo['steps_per_sec']:.1f} steps/sec)")
            print(f"   🐌 Worst algorithm: {worst_algo['name']} ({worst_algo['steps_per_sec']:.1f} steps/sec)")
            print(f"   📈 Speedup: {speedup:.1f}x")
    
    # Step time analysis
    step_results = all_results.get('step_time', [])
    if step_results:
        print("\n📊 STEP TIME BOTTLENECKS:")
        successful = [r for r in step_results if r.get('success', False)]
        if successful:
            # Find where step time becomes problematic
            slow_steps = [r for r in successful if r['avg_step_time_ms'] > 10]
            if slow_steps:
                worst = max(slow_steps, key=lambda x: x['avg_step_time_ms'])
                print(f"   ⚠️  Slow steps at {worst['neurons']:,} neurons: {worst['avg_step_time_ms']:.1f}ms/step")
            
            # Calculate step time scaling
            if len(successful) >= 2:
                first = successful[0]
                last = successful[-1]
                scale_factor = last['neurons'] / first['neurons']
                time_ratio = last['avg_step_time_ms'] / first['avg_step_time_ms']
                time_scaling = time_ratio / scale_factor
                print(f"   📈 Step time scaling factor: {time_scaling:.2f}x (linear scaling = 1.0x)")


def main():
    """Run comprehensive bottleneck analysis"""
    print("🔍 COMPREHENSIVE BOTTLENECK ANALYSIS")
    print("=" * 80)
    print("Analyzing performance bottlenecks across different scales and configurations...")
    
    all_results = {}
    
    # Run all analyses
    all_results['memory'] = analyze_memory_bottleneck()
    all_results['active_percentage'] = analyze_active_percentage_bottleneck()
    all_results['areas'] = analyze_area_bottleneck()
    all_results['algorithms'] = analyze_algorithm_bottleneck()
    all_results['step_time'] = analyze_step_time_breakdown()
    
    # Save and summarize results
    save_analysis_results(all_results)
    print_bottleneck_summary(all_results)
    
    print("\n🎯 Bottleneck analysis complete!")
    print("   Check the JSON file for detailed results.")
    print("   Use these insights to optimize performance.")


if __name__ == "__main__":
    main()
