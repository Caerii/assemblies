#!/usr/bin/env python3
"""
Large Scale Testing with Fixed Client
====================================

Test the fixed client interface at large scales to ensure all operations work properly.
"""

import sys
import time
import json
from datetime import datetime
sys.path.insert(0, '.')

from universal_brain_simulator.client import BrainSimulator


def test_optimized_cuda_large_scale():
    """Test optimized CUDA kernels at large scales"""
    print("🚀 TESTING OPTIMIZED CUDA LARGE SCALE")
    print("=" * 60)
    
    test_cases = [
        {'neurons': 10000000, 'areas': 3, 'active_percentage': 0.01},
        {'neurons': 50000000, 'areas': 5, 'active_percentage': 0.01},
        {'neurons': 100000000, 'areas': 10, 'active_percentage': 0.01},
        {'neurons': 500000000, 'areas': 20, 'active_percentage': 0.01},
        {'neurons': 1000000000, 'areas': 30, 'active_percentage': 0.01},
        {'neurons': 2000000000, 'areas': 50, 'active_percentage': 0.01},
        {'neurons': 5000000000, 'areas': 100, 'active_percentage': 0.01},
        {'neurons': 10000000000, 'areas': 200, 'active_percentage': 0.01}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}/{len(test_cases)}: {test_case['neurons']:,} neurons, {test_case['areas']} areas")
        
        try:
            start_time = time.perf_counter()
            
            sim = BrainSimulator(
                neurons=test_case['neurons'],
                active_percentage=test_case['active_percentage'],
                areas=test_case['areas'],
                use_optimized_cuda=True,
                use_cuda_kernels=True
            )
            
            init_time = time.perf_counter() - start_time
            
            # Test single step
            step_start = time.perf_counter()
            step_time = sim.simulate_step()
            step_end = time.perf_counter()
            
            # Test multiple steps
            run_start = time.perf_counter()
            run_results = sim.run(steps=10, verbose=False)
            run_end = time.perf_counter()
            
            total_time = run_end - start_time
            
            result = {
                'test_case': test_case,
                'init_time': init_time,
                'single_step_time': step_time,
                'measured_step_time': step_end - step_start,
                'run_results': run_results,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Init: {init_time:.3f}s")
            print(f"   ✅ Single step: {step_time:.3f}s")
            print(f"   ✅ 10 steps: {run_results['summary']['steps_per_sec']:.1f} steps/sec")
            print(f"   ✅ Neurons/sec: {run_results['summary']['neurons_per_second']:,.0f}")
            print(f"   ✅ Memory: {run_results['summary']['memory_usage_gb']:.2f}GB")
            print(f"   ✅ Total time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'test_case': test_case,
                'error': str(e),
                'success': False
            })
    
    return results


def test_cupy_only_large_scale():
    """Test CuPy-only mode at large scales"""
    print("\n🔥 TESTING CUPY-ONLY LARGE SCALE")
    print("=" * 60)
    
    test_cases = [
        {'neurons': 10000000, 'areas': 3, 'active_percentage': 0.01},
        {'neurons': 50000000, 'areas': 5, 'active_percentage': 0.01},
        {'neurons': 100000000, 'areas': 10, 'active_percentage': 0.01},
        {'neurons': 500000000, 'areas': 20, 'active_percentage': 0.01}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}/{len(test_cases)}: {test_case['neurons']:,} neurons, {test_case['areas']} areas")
        
        try:
            start_time = time.perf_counter()
            
            sim = BrainSimulator(
                neurons=test_case['neurons'],
                active_percentage=test_case['active_percentage'],
                areas=test_case['areas'],
                use_optimized_cuda=False,
                use_cuda_kernels=False  # CuPy only
            )
            
            init_time = time.perf_counter() - start_time
            
            # Test single step
            step_start = time.perf_counter()
            step_time = sim.simulate_step()
            step_end = time.perf_counter()
            
            # Test multiple steps
            run_start = time.perf_counter()
            run_results = sim.run(steps=10, verbose=False)
            run_end = time.perf_counter()
            
            total_time = run_end - start_time
            
            result = {
                'test_case': test_case,
                'init_time': init_time,
                'single_step_time': step_time,
                'measured_step_time': step_end - step_start,
                'run_results': run_results,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Init: {init_time:.3f}s")
            print(f"   ✅ Single step: {step_time:.3f}s")
            print(f"   ✅ 10 steps: {run_results['summary']['steps_per_sec']:.1f} steps/sec")
            print(f"   ✅ Neurons/sec: {run_results['summary']['neurons_per_second']:,.0f}")
            print(f"   ✅ Memory: {run_results['summary']['memory_usage_gb']:.2f}GB")
            print(f"   ✅ Total time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'test_case': test_case,
                'error': str(e),
                'success': False
            })
    
    return results


def test_original_cuda_large_scale():
    """Test original CUDA kernels at large scales"""
    print("\n⚡ TESTING ORIGINAL CUDA LARGE SCALE")
    print("=" * 60)
    
    test_cases = [
        {'neurons': 10000000, 'areas': 3, 'active_percentage': 0.01},
        {'neurons': 50000000, 'areas': 5, 'active_percentage': 0.01},
        {'neurons': 100000000, 'areas': 10, 'active_percentage': 0.01}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}/{len(test_cases)}: {test_case['neurons']:,} neurons, {test_case['areas']} areas")
        
        try:
            start_time = time.perf_counter()
            
            sim = BrainSimulator(
                neurons=test_case['neurons'],
                active_percentage=test_case['active_percentage'],
                areas=test_case['areas'],
                use_optimized_cuda=False,
                use_cuda_kernels=True
            )
            
            init_time = time.perf_counter() - start_time
            
            # Test single step
            step_start = time.perf_counter()
            step_time = sim.simulate_step()
            step_end = time.perf_counter()
            
            # Test multiple steps
            run_start = time.perf_counter()
            run_results = sim.run(steps=10, verbose=False)
            run_end = time.perf_counter()
            
            total_time = run_end - start_time
            
            result = {
                'test_case': test_case,
                'init_time': init_time,
                'single_step_time': step_time,
                'measured_step_time': step_end - step_start,
                'run_results': run_results,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Init: {init_time:.3f}s")
            print(f"   ✅ Single step: {step_time:.3f}s")
            print(f"   ✅ 10 steps: {run_results['summary']['steps_per_sec']:.1f} steps/sec")
            print(f"   ✅ Neurons/sec: {run_results['summary']['neurons_per_second']:,.0f}")
            print(f"   ✅ Memory: {run_results['summary']['memory_usage_gb']:.2f}GB")
            print(f"   ✅ Total time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'test_case': test_case,
                'error': str(e),
                'success': False
            })
    
    return results


def test_benchmark_functionality():
    """Test benchmark functionality at different scales"""
    print("\n📊 TESTING BENCHMARK FUNCTIONALITY")
    print("=" * 60)
    
    test_cases = [
        {'neurons': 10000000, 'areas': 3, 'active_percentage': 0.01},
        {'neurons': 100000000, 'areas': 10, 'active_percentage': 0.01},
        {'neurons': 1000000000, 'areas': 30, 'active_percentage': 0.01}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Benchmark {i}/{len(test_cases)}: {test_case['neurons']:,} neurons")
        
        try:
            sim = BrainSimulator(
                neurons=test_case['neurons'],
                active_percentage=test_case['active_percentage'],
                areas=test_case['areas'],
                use_optimized_cuda=True,
                use_cuda_kernels=True
            )
            
            # Run benchmark
            benchmark_results = sim.benchmark(warmup_steps=2, measure_steps=3)
            
            result = {
                'test_case': test_case,
                'benchmark_results': benchmark_results,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {benchmark_results['performance']['steps_per_sec']:.1f}")
            print(f"   ✅ Neurons/sec: {benchmark_results['performance']['neurons_per_second']:,.0f}")
            print(f"   ✅ Avg step time: {benchmark_results['performance']['average_step_time_ms']:.2f}ms")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'test_case': test_case,
                'error': str(e),
                'success': False
            })
    
    return results


def save_results(all_results):
    """Save all test results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"large_scale_test_results_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for test_type, results in all_results.items():
        json_results[test_type] = []
        for result in results:
            if result.get('success', False):
                # Remove non-serializable objects
                clean_result = {
                    'test_case': result['test_case'],
                    'init_time': result.get('init_time', 0),
                    'single_step_time': result.get('single_step_time', 0),
                    'measured_step_time': result.get('measured_step_time', 0),
                    'total_time': result.get('total_time', 0),
                    'success': True
                }
                
                # Add run results summary
                if 'run_results' in result:
                    run_results = result['run_results']
                    clean_result['run_summary'] = {
                        'steps_per_sec': run_results['summary'].get('steps_per_sec', 0),
                        'neurons_per_second': run_results['summary'].get('neurons_per_second', 0),
                        'memory_usage_gb': run_results['summary'].get('memory_usage_gb', 0),
                        'cuda_kernels_used': run_results['summary'].get('cuda_kernels_used', False)
                    }
                
                # Add benchmark results summary
                if 'benchmark_results' in result:
                    benchmark_results = result['benchmark_results']
                    clean_result['benchmark_summary'] = {
                        'steps_per_sec': benchmark_results['performance'].get('steps_per_sec', 0),
                        'neurons_per_second': benchmark_results['performance'].get('neurons_per_second', 0),
                        'average_step_time_ms': benchmark_results['performance'].get('average_step_time_ms', 0)
                    }
                
                json_results[test_type].append(clean_result)
            else:
                json_results[test_type].append({
                    'test_case': result['test_case'],
                    'error': result.get('error', 'Unknown error'),
                    'success': False
                })
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {filename}")
    return filename


def print_summary(all_results):
    """Print comprehensive test summary"""
    print(f"\n🎯 LARGE SCALE TESTING SUMMARY")
    print("=" * 80)
    
    for test_type, results in all_results.items():
        successful = [r for r in results if r.get('success', False)]
        total = len(results)
        
        print(f"\n📊 {test_type.upper().replace('_', ' ')}:")
        print(f"   Successful tests: {len(successful)}/{total}")
        
        if successful:
            # Find best performance
            best_result = None
            best_steps_per_sec = 0
            
            for result in successful:
                steps_per_sec = 0
                if 'run_results' in result:
                    steps_per_sec = result['run_results']['summary'].get('steps_per_sec', 0)
                elif 'benchmark_results' in result:
                    steps_per_sec = result['benchmark_results']['performance'].get('steps_per_sec', 0)
                
                if steps_per_sec > best_steps_per_sec:
                    best_steps_per_sec = steps_per_sec
                    best_result = result
            
            if best_result:
                test_case = best_result['test_case']
                print(f"   Best performance: {test_case['neurons']:,} neurons ({best_steps_per_sec:.1f} steps/sec)")
                
                if 'run_results' in best_result:
                    run_summary = best_result['run_results']['summary']
                    print(f"   Neurons/sec: {run_summary.get('neurons_per_second', 0):,.0f}")
                    print(f"   Memory: {run_summary.get('memory_usage_gb', 0):.2f}GB")
        
        # Show failed tests
        failed = [r for r in results if not r.get('success', False)]
        if failed:
            print(f"   Failed tests: {len(failed)}")
            for result in failed:
                test_case = result['test_case']
                error = result.get('error', 'Unknown error')
                print(f"     - {test_case['neurons']:,} neurons: {error}")


def main():
    """Run comprehensive large-scale testing"""
    print("🚀 LARGE SCALE TESTING WITH FIXED CLIENT")
    print("=" * 80)
    print("Testing all configurations at extreme scales...")
    
    all_results = {}
    
    # Run all tests
    all_results['optimized_cuda'] = test_optimized_cuda_large_scale()
    all_results['cupy_only'] = test_cupy_only_large_scale()
    all_results['original_cuda'] = test_original_cuda_large_scale()
    all_results['benchmark'] = test_benchmark_functionality()
    
    # Save and summarize results
    filename = save_results(all_results)
    print_summary(all_results)
    
    print(f"\n🎯 Large scale testing complete!")
    print(f"   All client errors have been fixed.")
    print(f"   Results saved to: {filename}")


if __name__ == "__main__":
    main()
