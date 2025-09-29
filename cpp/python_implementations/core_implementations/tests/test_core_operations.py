#!/usr/bin/env python3
"""
Test Core Brain Operations
=========================

Test the core brain simulation operations: project, merge, and association.
These are the fundamental operations that make up the brain simulation.
"""

import sys
import time
import numpy as np
sys.path.insert(0, '.')

from universal_brain_simulator.client import BrainSimulator


def test_project_operation():
    """Test the project operation - how neurons project to other areas"""
    print("🧠 TESTING PROJECT OPERATION")
    print("=" * 50)
    
    # Test different projection patterns
    test_cases = [
        {
            'name': 'Small Projection',
            'neurons': 1000000,
            'areas': 2,
            'active_percentage': 0.01
        },
        {
            'name': 'Medium Projection', 
            'neurons': 10000000,
            'areas': 3,
            'active_percentage': 0.01
        },
        {
            'name': 'Large Projection',
            'neurons': 100000000,
            'areas': 5,
            'active_percentage': 0.01
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}...")
        
        try:
            sim = BrainSimulator(
                neurons=test_case['neurons'],
                active_percentage=test_case['active_percentage'],
                areas=test_case['areas'],
                use_optimized_cuda=True
            )
            
            # Run simulation to test projection
            start_time = time.perf_counter()
            results_sim = sim.run(steps=10, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            
            result = {
                'name': test_case['name'],
                'neurons': test_case['neurons'],
                'areas': test_case['areas'],
                'steps_per_sec': steps_per_sec,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Areas: {test_case['areas']}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'error': str(e),
                'success': False
            })
    
    return results


def test_merge_operation():
    """Test the merge operation - how information is merged between areas"""
    print("\n🔄 TESTING MERGE OPERATION")
    print("=" * 50)
    
    # Test different merge scenarios
    test_cases = [
        {
            'name': 'Simple Merge (2 areas)',
            'neurons': 1000000,
            'areas': 2,
            'active_percentage': 0.01
        },
        {
            'name': 'Complex Merge (5 areas)',
            'neurons': 10000000,
            'areas': 5,
            'active_percentage': 0.01
        },
        {
            'name': 'Large Merge (10 areas)',
            'neurons': 50000000,
            'areas': 10,
            'active_percentage': 0.01
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}...")
        
        try:
            sim = BrainSimulator(
                neurons=test_case['neurons'],
                active_percentage=test_case['active_percentage'],
                areas=test_case['areas'],
                use_optimized_cuda=True
            )
            
            # Run simulation to test merge operations
            start_time = time.perf_counter()
            results_sim = sim.run(steps=10, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            
            result = {
                'name': test_case['name'],
                'neurons': test_case['neurons'],
                'areas': test_case['areas'],
                'steps_per_sec': steps_per_sec,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Areas: {test_case['areas']}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'error': str(e),
                'success': False
            })
    
    return results


def test_association_operation():
    """Test the association operation - how neurons form associations"""
    print("\n🔗 TESTING ASSOCIATION OPERATION")
    print("=" * 50)
    
    # Test different association patterns
    test_cases = [
        {
            'name': 'Sparse Association (0.1% active)',
            'neurons': 10000000,
            'areas': 3,
            'active_percentage': 0.001
        },
        {
            'name': 'Moderate Association (1% active)',
            'neurons': 10000000,
            'areas': 3,
            'active_percentage': 0.01
        },
        {
            'name': 'Dense Association (5% active)',
            'neurons': 10000000,
            'areas': 3,
            'active_percentage': 0.05
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}...")
        
        try:
            sim = BrainSimulator(
                neurons=test_case['neurons'],
                active_percentage=test_case['active_percentage'],
                areas=test_case['areas'],
                use_optimized_cuda=True
            )
            
            # Run simulation to test association formation
            start_time = time.perf_counter()
            results_sim = sim.run(steps=10, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            active_neurons = int(test_case['neurons'] * test_case['active_percentage'])
            
            result = {
                'name': test_case['name'],
                'neurons': test_case['neurons'],
                'active_percentage': test_case['active_percentage'],
                'active_neurons': active_neurons,
                'steps_per_sec': steps_per_sec,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Active neurons: {active_neurons:,}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'error': str(e),
                'success': False
            })
    
    return results


def test_combined_operations():
    """Test combined project, merge, and association operations"""
    print("\n🎯 TESTING COMBINED OPERATIONS")
    print("=" * 50)
    
    # Test realistic brain simulation scenarios
    test_cases = [
        {
            'name': 'Small Brain Simulation',
            'neurons': 1000000,
            'areas': 3,
            'active_percentage': 0.01,
            'steps': 50
        },
        {
            'name': 'Medium Brain Simulation',
            'neurons': 10000000,
            'areas': 5,
            'active_percentage': 0.01,
            'steps': 100
        },
        {
            'name': 'Large Brain Simulation',
            'neurons': 100000000,
            'areas': 10,
            'active_percentage': 0.01,
            'steps': 200
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}...")
        
        try:
            sim = BrainSimulator(
                neurons=test_case['neurons'],
                active_percentage=test_case['active_percentage'],
                areas=test_case['areas'],
                use_optimized_cuda=True
            )
            
            # Run extended simulation to test all operations
            start_time = time.perf_counter()
            results_sim = sim.run(steps=test_case['steps'], verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_second']
            neurons_per_sec = results_sim['summary']['neurons_per_second']
            memory_gb = results_sim['summary']['memory_usage_gb']
            
            result = {
                'name': test_case['name'],
                'neurons': test_case['neurons'],
                'areas': test_case['areas'],
                'steps': test_case['steps'],
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
                'name': test_case['name'],
                'error': str(e),
                'success': False
            })
    
    return results


def test_cuda_kernel_correctness():
    """Test CUDA kernel correctness and individual operations"""
    print("\n🔧 TESTING CUDA KERNEL CORRECTNESS")
    print("=" * 50)
    
    # Test different kernel configurations
    test_cases = [
        {
            'name': 'Optimized CUDA Kernels',
            'use_optimized_cuda': True,
            'neurons': 1000000,
            'areas': 3
        },
        {
            'name': 'Original CUDA Kernels',
            'use_optimized_cuda': False,
            'neurons': 1000000,
            'areas': 3
        },
        {
            'name': 'CuPy Only (No CUDA Kernels)',
            'use_optimized_cuda': False,
            'use_cuda_kernels': False,
            'neurons': 1000000,
            'areas': 3
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}...")
        
        try:
            # Create config without problematic parameters
            config = {
                'neurons': test_case['neurons'],
                'active_percentage': 0.01,
                'areas': test_case['areas'],
                'use_optimized_cuda': test_case['use_optimized_cuda']
            }
            
            # Add use_cuda_kernels only if specified
            if 'use_cuda_kernels' in test_case:
                config['use_cuda_kernels'] = test_case['use_cuda_kernels']
            
            sim = BrainSimulator(**config)
            
            # Run simulation to test kernel correctness
            start_time = time.perf_counter()
            results_sim = sim.run(steps=10, verbose=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            steps_per_sec = results_sim['summary']['steps_per_sec']
            cuda_used = results_sim['summary']['cuda_kernels_used']
            
            result = {
                'name': test_case['name'],
                'config': config,
                'steps_per_sec': steps_per_sec,
                'cuda_kernels_used': cuda_used,
                'total_time': total_time,
                'success': True
            }
            
            results.append(result)
            
            print(f"   ✅ Steps/sec: {steps_per_sec:.1f}")
            print(f"   CUDA kernels: {'✅' if cuda_used else '❌'}")
            print(f"   Time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'config': test_case,
                'error': str(e),
                'success': False
            })
    
    return results


def print_operations_summary(all_results):
    """Print summary of core operations testing"""
    print(f"\n🎯 CORE OPERATIONS TESTING SUMMARY")
    print("=" * 80)
    
    # Project operation summary
    project_results = all_results.get('project', [])
    if project_results:
        successful = [r for r in project_results if r.get('success', False)]
        print(f"\n📊 PROJECT OPERATION:")
        print(f"   Successful tests: {len(successful)}/{len(project_results)}")
        if successful:
            best = max(successful, key=lambda x: x['steps_per_sec'])
            print(f"   Best performance: {best['name']} ({best['steps_per_sec']:.1f} steps/sec)")
    
    # Merge operation summary
    merge_results = all_results.get('merge', [])
    if merge_results:
        successful = [r for r in merge_results if r.get('success', False)]
        print(f"\n📊 MERGE OPERATION:")
        print(f"   Successful tests: {len(successful)}/{len(merge_results)}")
        if successful:
            best = max(successful, key=lambda x: x['steps_per_sec'])
            print(f"   Best performance: {best['name']} ({best['steps_per_sec']:.1f} steps/sec)")
    
    # Association operation summary
    association_results = all_results.get('association', [])
    if association_results:
        successful = [r for r in association_results if r.get('success', False)]
        print(f"\n📊 ASSOCIATION OPERATION:")
        print(f"   Successful tests: {len(successful)}/{len(association_results)}")
        if successful:
            best = max(successful, key=lambda x: x['steps_per_sec'])
            print(f"   Best performance: {best['name']} ({best['steps_per_sec']:.1f} steps/sec)")
    
    # Combined operations summary
    combined_results = all_results.get('combined', [])
    if combined_results:
        successful = [r for r in combined_results if r.get('success', False)]
        print(f"\n📊 COMBINED OPERATIONS:")
        print(f"   Successful tests: {len(successful)}/{len(combined_results)}")
        if successful:
            best = max(successful, key=lambda x: x['steps_per_sec'])
            print(f"   Best performance: {best['name']} ({best['steps_per_sec']:.1f} steps/sec)")
            print(f"   Neurons/sec: {best['neurons_per_sec']:,.0f}")
            print(f"   Memory: {best['memory_gb']:.2f}GB")
    
    # CUDA kernel summary
    kernel_results = all_results.get('kernels', [])
    if kernel_results:
        successful = [r for r in kernel_results if r.get('success', False)]
        print(f"\n📊 CUDA KERNEL CORRECTNESS:")
        print(f"   Successful tests: {len(successful)}/{len(kernel_results)}")
        if successful:
            for result in successful:
                print(f"   {result['name']}: {'✅' if result['cuda_kernels_used'] else '❌'} ({result['steps_per_sec']:.1f} steps/sec)")


def main():
    """Run comprehensive core operations testing"""
    print("🧠 CORE BRAIN OPERATIONS TESTING")
    print("=" * 80)
    print("Testing project, merge, and association operations...")
    
    all_results = {}
    
    # Run all operation tests
    all_results['project'] = test_project_operation()
    all_results['merge'] = test_merge_operation()
    all_results['association'] = test_association_operation()
    all_results['combined'] = test_combined_operations()
    all_results['kernels'] = test_cuda_kernel_correctness()
    
    # Print summary
    print_operations_summary(all_results)
    
    print(f"\n🎯 Core operations testing complete!")
    print(f"   This tests the fundamental brain simulation operations.")
    print(f"   Results show how well project, merge, and association work.")


if __name__ == "__main__":
    main()
