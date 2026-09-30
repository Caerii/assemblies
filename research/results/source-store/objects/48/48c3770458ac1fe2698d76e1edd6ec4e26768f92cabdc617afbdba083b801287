#!/usr/bin/env python3
"""
Extreme Scale Sweep Analysis
===========================

Comprehensive sweep testing from million to multi-billion scale to identify
the absolute limits and bottlenecks of the system.
"""

import sys
import time
import json
from typing import Dict, List, Any
sys.path.insert(0, '.')

from universal_brain_simulator.client import BrainSimulator


def extreme_neuron_sweep():
    """Extreme neuron count sweep from 1M to 10B"""
    print("🚀 EXTREME NEURON COUNT SWEEP")
    print("=" * 80)
    
    # Extreme neuron counts - going beyond billion scale
    neuron_counts = [
        1000000,      # 1M
        5000000,      # 5M
        10000000,     # 10M
        50000000,     # 50M
        100000000,    # 100M
        250000000,    # 250M
        500000000,    # 500M
        750000000,    # 750M
        1000000000,   # 1B
        1500000000,   # 1.5B
        2000000000,   # 2B
        3000000000,   # 3B
        5000000000,   # 5B
        7500000000,   # 7.5B
        10000000000   # 10B
    ]
    
    results = []
    
    for neurons in neuron_counts:
        print(f"\n🧪 Testing {neurons:,} neurons ({neurons/1000000000:.1f}B)...")
        
        try:
            # Adjust active percentage for very large scales to avoid memory issues
            if neurons >= 5000000000:  # 5B+
                active_pct = 0.001  # 0.1% active
            elif neurons >= 1000000000:  # 1B+
                active_pct = 0.005  # 0.5% active
            else:
                active_pct = 0.01   # 1% active
            
            sim = BrainSimulator(
                neurons=neurons,
                active_percentage=active_pct,
                areas=5,
                use_optimized_cuda=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            # Get initial memory
            info = sim.get_info()
            initial_memory = info['memory_info']['used_gb']
            
            # Run simulation with fewer steps for very large scales
            steps = 1 if neurons >= 5000000000 else (2 if neurons >= 1000000000 else 3)
            
            start_time = time.perf_counter()
            results_sim = sim.run(steps=steps, verbose=False)
            end_time = time.perf_counter()
            
            # Get final memory
            info_after = sim.get_info()
            final_memory = info_after['memory_info']['used_gb']
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            neurons_per_sec = results_sim['summary']['neurons_per_second']
            memory_per_neuron = (final_memory * 1024**3) / neurons
            
            result = {
                'neurons': neurons,
                'neurons_billions': neurons / 1000000000,
                'active_percentage': active_pct,
                'active_neurons': int(neurons * active_pct),
                'steps': steps,
                'steps_per_sec': steps_per_sec,
                'neurons_per_sec': neurons_per_sec,
                'initial_memory_gb': initial_memory,
                'final_memory_gb': final_memory,
                'memory_per_neuron_bytes': memory_per_neuron,
                'total_time': total_time,
                'avg_step_time_ms': (total_time / steps) * 1000,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Neurons/sec: {neurons_per_sec:,.0f}")
            print(f"   Memory: {final_memory:.2f}GB ({memory_per_neuron:.1f} bytes/neuron)")
            print(f"   Step time: {(total_time/steps)*1000:.1f}ms")
            print(f"   Active: {active_pct*100:.1f}% ({int(neurons*active_pct):,} neurons)")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'neurons': neurons,
                'neurons_billions': neurons / 1000000000,
                'error': str(e),
                'success': False
            })
            
            # Stop if we hit a hard limit
            if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                print(f"   🛑 Memory limit reached at {neurons:,} neurons")
                break
    
    return results


def extreme_active_percentage_sweep():
    """Extreme active percentage sweep at billion scale"""
    print("\n🚀 EXTREME ACTIVE PERCENTAGE SWEEP (1B neurons)")
    print("=" * 80)
    
    base_neurons = 1000000000  # 1B neurons
    active_percentages = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    results = []
    
    for active_pct in active_percentages:
        active_neurons = int(base_neurons * active_pct)
        print(f"\n🧪 Testing {active_pct*100:.2f}% active ({active_neurons:,} active neurons)...")
        
        try:
            sim = BrainSimulator(
                neurons=base_neurons,
                active_percentage=active_pct,
                areas=5,
                use_optimized_cuda=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            start_time = time.perf_counter()
            results_sim = sim.run(steps=2, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            neurons_per_sec = results_sim['summary']['neurons_per_second']
            memory_gb = results_sim['summary']['memory_usage_gb']
            
            result = {
                'active_percentage': active_pct,
                'active_neurons': active_neurons,
                'steps_per_sec': steps_per_sec,
                'neurons_per_sec': neurons_per_sec,
                'memory_gb': memory_gb,
                'total_time': total_time,
                'avg_step_time_ms': (total_time / 2) * 1000,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Neurons/sec: {neurons_per_sec:,.0f}")
            print(f"   Memory: {memory_gb:.2f}GB")
            print(f"   Step time: {(total_time/2)*1000:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'active_percentage': active_pct,
                'active_neurons': active_neurons,
                'error': str(e),
                'success': False
            })
    
    return results


def extreme_area_sweep():
    """Extreme area count sweep at billion scale"""
    print("\n🚀 EXTREME AREA COUNT SWEEP (1B neurons)")
    print("=" * 80)
    
    base_neurons = 1000000000  # 1B neurons
    area_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
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
                use_optimized_cuda=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            start_time = time.perf_counter()
            results_sim = sim.run(steps=2, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            neurons_per_sec = results_sim['summary']['neurons_per_second']
            memory_gb = results_sim['summary']['memory_usage_gb']
            
            result = {
                'areas': areas,
                'neurons_per_area': neurons_per_area,
                'total_active': total_active,
                'steps_per_sec': steps_per_sec,
                'neurons_per_sec': neurons_per_sec,
                'memory_gb': memory_gb,
                'total_time': total_time,
                'avg_step_time_ms': (total_time / 2) * 1000,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Neurons/sec: {neurons_per_sec:,.0f}")
            print(f"   Memory: {memory_gb:.2f}GB")
            print(f"   Step time: {(total_time/2)*1000:.1f}ms")
            
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


def memory_limit_test():
    """Test the absolute memory limits of the system"""
    print("\n🚀 MEMORY LIMIT TEST")
    print("=" * 80)
    
    # Test progressively larger scales until we hit memory limits
    neuron_counts = [
        1000000000,   # 1B
        2000000000,   # 2B
        3000000000,   # 3B
        4000000000,   # 4B
        5000000000,   # 5B
        6000000000,   # 6B
        7000000000,   # 7B
        8000000000,   # 8B
        9000000000,   # 9B
        10000000000,  # 10B
        15000000000,  # 15B
        20000000000   # 20B
    ]
    
    results = []
    memory_limit_reached = False
    
    for neurons in neuron_counts:
        if memory_limit_reached:
            break
            
        print(f"\n🧪 Testing memory limit at {neurons:,} neurons ({neurons/1000000000:.1f}B)...")
        
        try:
            # Use very low active percentage for memory limit testing
            active_pct = 0.0001  # 0.01% active
            
            sim = BrainSimulator(
                neurons=neurons,
                active_percentage=active_pct,
                areas=1,  # Single area to minimize overhead
                use_optimized_cuda=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            # Just try to initialize and run one step
            info = sim.get_info()
            memory_gb = info['memory_info']['used_gb']
            
            start_time = time.perf_counter()
            sim.simulate_step()
            end_time = time.perf_counter()
            
            step_time = end_time - start_time
            
            result = {
                'neurons': neurons,
                'neurons_billions': neurons / 1000000000,
                'active_percentage': active_pct,
                'active_neurons': int(neurons * active_pct),
                'memory_gb': memory_gb,
                'step_time_ms': step_time * 1000,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Memory: {memory_gb:.2f}GB")
            print(f"   Step time: {step_time*1000:.1f}ms")
            print(f"   Active: {active_pct*100:.3f}% ({int(neurons*active_pct):,} neurons)")
            
        except Exception as e:
            print(f"   ❌ Memory limit reached: {e}")
            results.append({
                'neurons': neurons,
                'neurons_billions': neurons / 1000000000,
                'error': str(e),
                'success': False
            })
            memory_limit_reached = True
    
    return results


def save_extreme_results(all_results: Dict[str, List[Any]]):
    """Save extreme scale results to JSON file"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"extreme_scale_sweep_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Extreme scale results saved to {filename}")


def print_extreme_summary(all_results: Dict[str, List[Any]]):
    """Print summary of extreme scale testing"""
    print("\n🎯 EXTREME SCALE TESTING SUMMARY")
    print("=" * 80)
    
    # Neuron count sweep summary
    neuron_results = all_results.get('neuron_sweep', [])
    if neuron_results:
        successful = [r for r in neuron_results if r.get('success', False)]
        if successful:
            max_neurons = max(successful, key=lambda x: x['neurons'])
            print("\n📊 MAXIMUM SCALE ACHIEVED:")
            print(f"   🏆 Largest simulation: {max_neurons['neurons']:,} neurons ({max_neurons['neurons_billions']:.1f}B)")
            print(f"   Performance: {max_neurons['steps_per_sec']:.1f} steps/sec")
            print(f"   Memory usage: {max_neurons['final_memory_gb']:.2f}GB")
            print(f"   Memory efficiency: {max_neurons['memory_per_neuron_bytes']:.1f} bytes/neuron")
            
            # Find performance scaling
            if len(successful) >= 3:
                small = successful[0]
                large = successful[-1]
                scale_factor = large['neurons'] / small['neurons']
                perf_ratio = large['steps_per_sec'] / small['steps_per_sec']
                efficiency = (perf_ratio / scale_factor) * 100
                print(f"   📈 Scaling efficiency: {efficiency:.2f}% (from {small['neurons']:,} to {large['neurons']:,} neurons)")
    
    # Active percentage sweep summary
    active_results = all_results.get('active_percentage_sweep', [])
    if active_results:
        successful = [r for r in active_results if r.get('success', False)]
        if successful:
            best_active = max(successful, key=lambda x: x['steps_per_sec'])
            print("\n📊 OPTIMAL ACTIVE PERCENTAGE (1B neurons):")
            print(f"   🎯 Best performance: {best_active['active_percentage']*100:.2f}% active")
            print(f"   Steps/sec: {best_active['steps_per_sec']:.1f}")
            print(f"   Active neurons: {best_active['active_neurons']:,}")
    
    # Area count sweep summary
    area_results = all_results.get('area_sweep', [])
    if area_results:
        successful = [r for r in area_results if r.get('success', False)]
        if successful:
            best_areas = max(successful, key=lambda x: x['steps_per_sec'])
            print("\n📊 OPTIMAL AREA COUNT (1B neurons):")
            print(f"   🎯 Best performance: {best_areas['areas']} areas")
            print(f"   Steps/sec: {best_areas['steps_per_sec']:.1f}")
            print(f"   Neurons per area: {best_areas['neurons_per_area']:,}")
    
    # Memory limit summary
    memory_results = all_results.get('memory_limit', [])
    if memory_results:
        successful = [r for r in memory_results if r.get('success', False)]
        if successful:
            max_memory = max(successful, key=lambda x: x['neurons'])
            print("\n📊 MEMORY LIMITS:")
            print(f"   🏆 Maximum neurons: {max_memory['neurons']:,} ({max_memory['neurons_billions']:.1f}B)")
            print(f"   Memory usage: {max_memory['memory_gb']:.2f}GB")
            print(f"   Step time: {max_memory['step_time_ms']:.1f}ms")
        else:
            print("\n📊 MEMORY LIMITS:")
            print("   ❌ No successful memory limit tests")


def main():
    """Run extreme scale sweep analysis"""
    print("🚀 EXTREME SCALE SWEEP ANALYSIS")
    print("=" * 80)
    print("Testing performance limits from million to multi-billion scale...")
    
    all_results = {}
    
    # Run extreme scale tests
    all_results['neuron_sweep'] = extreme_neuron_sweep()
    all_results['active_percentage_sweep'] = extreme_active_percentage_sweep()
    all_results['area_sweep'] = extreme_area_sweep()
    all_results['memory_limit'] = memory_limit_test()
    
    # Save and summarize results
    save_extreme_results(all_results)
    print_extreme_summary(all_results)
    
    print("\n🎯 Extreme scale analysis complete!")
    print("   This reveals the absolute performance limits of the system.")
    print("   Use these results to understand scaling bottlenecks.")


if __name__ == "__main__":
    main()
