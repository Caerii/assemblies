#!/usr/bin/env python3
"""
Billion-Scale Optimization Test
==============================

This script specifically tests the performance difference between:
1. Original O(N²) top-k selection algorithm
2. Optimized O(N log K) top-k selection algorithm

At billion-scale to demonstrate the critical algorithmic improvements.

The O(N²) algorithm should fail or be extremely slow at large scales,
while the O(N log K) algorithm should handle billion-scale efficiently.
"""

import time
import sys
import os
import json
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import implementations
try:
    from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig as OriginalConfig
    from universal_brain_simulator_optimized import UniversalBrainSimulatorOptimized, SimulationConfig as OptimizedConfig
    print("✅ Both implementations imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class BillionScaleOptimizationTest:
    """Test billion-scale performance with algorithmic optimizations"""
    
    def __init__(self):
        self.results = []
        self.max_test_time = 300  # 5 minutes max per test
    
    def test_algorithmic_complexity(self, n_neurons: int, active_percentage: float, n_areas: int = 5):
        """Test algorithmic complexity at a specific scale"""
        k_active = int(n_neurons * active_percentage)
        
        print("\n🧪 Algorithmic Complexity Test")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active per area: {k_active:,}")
        print(f"   Total active: {k_active * n_areas:,}")
        print(f"   Areas: {n_areas}")
        
        # Calculate theoretical complexity
        n = n_neurons
        k = k_active
        
        # Original O(N²) complexity
        original_ops = k * n * n  # k iterations × n² comparisons each
        
        # Optimized O(N log K) complexity  
        optimized_ops = n * np.log2(max(k, 1))  # n elements × log₂(k) operations
        
        speedup_theoretical = original_ops / optimized_ops if optimized_ops > 0 else float('inf')
        
        print("\n📊 Theoretical Analysis:")
        print(f"   Original O(N²): {original_ops:,.0f} operations")
        print(f"   Optimized O(N log K): {optimized_ops:,.0f} operations")
        print(f"   Theoretical speedup: {speedup_theoretical:,.0f}x")
        
        # Test both implementations
        test_results = {
            'n_neurons': n_neurons,
            'k_active': k_active,
            'n_areas': n_areas,
            'theoretical_speedup': speedup_theoretical,
            'original_ops': original_ops,
            'optimized_ops': optimized_ops,
            'implementations': []
        }
        
        # Test Original Implementation
        print("\n🔬 Testing Original (O(N²)) Implementation:")
        print("-" * 50)
        
        try:
            config = OriginalConfig(
                n_neurons=n_neurons,
                active_percentage=active_percentage,
                n_areas=n_areas,
                use_gpu=True,
                use_cuda_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            start_time = time.time()
            simulator = UniversalBrainSimulator(config)
            init_time = time.time() - start_time
            
            print(f"   Initialization: {init_time:.2f}s")
            
            # Test with timeout
            sim_start = time.time()
            try:
                # Use a timer to limit test time
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Test timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.max_test_time)
                
                total_time = simulator.simulate(n_steps=1, verbose=False)
                
                signal.alarm(0)  # Cancel alarm
                
                sim_time = time.time() - sim_start
                stats = simulator.get_performance_stats()
                
                print(f"   ✅ Simulation completed: {sim_time:.2f}s")
                print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
                print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
                print(f"   Memory: {stats['memory_usage_gb']:.2f}GB")
                
                result = {
                    'name': 'Original (O(N²))',
                    'success': True,
                    'init_time': init_time,
                    'sim_time': sim_time,
                    'total_time': init_time + sim_time,
                    'steps_per_second': stats['steps_per_second'],
                    'neurons_per_second': stats['neurons_per_second'],
                    'memory_usage_gb': stats['memory_usage_gb'],
                    'timeout': False
                }
                
            except TimeoutError:
                signal.alarm(0)  # Cancel alarm
                sim_time = time.time() - sim_start
                print(f"   ⏰ TIMEOUT after {sim_time:.2f}s (>{self.max_test_time}s)")
                print("   ❌ O(N²) algorithm too slow at this scale!")
                
                result = {
                    'name': 'Original (O(N²))',
                    'success': False,
                    'init_time': init_time,
                    'sim_time': sim_time,
                    'total_time': init_time + sim_time,
                    'timeout': True,
                    'error': 'Algorithm too slow - O(N²) complexity'
                }
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            result = {
                'name': 'Original (O(N²))',
                'success': False,
                'error': str(e),
                'timeout': False
            }
        
        test_results['implementations'].append(result)
        
        # Test Optimized Implementation
        print("\n🔬 Testing Optimized (O(N log K)) Implementation:")
        print("-" * 50)
        
        try:
            config = OptimizedConfig(
                n_neurons=n_neurons,
                active_percentage=active_percentage,
                n_areas=n_areas,
                use_gpu=True,
                use_cuda_kernels=True,
                use_optimized_kernels=True,
                memory_efficient=True,
                sparse_mode=True
            )
            
            start_time = time.time()
            simulator = UniversalBrainSimulatorOptimized(config)
            init_time = time.time() - start_time
            
            print(f"   Initialization: {init_time:.2f}s")
            
            sim_start = time.time()
            total_time = simulator.simulate(n_steps=1, verbose=False)
            sim_time = time.time() - sim_start
            
            stats = simulator.get_performance_stats()
            
            print(f"   ✅ Simulation completed: {sim_time:.2f}s")
            print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
            print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
            print(f"   Memory: {stats['memory_usage_gb']:.2f}GB")
            print(f"   Optimized kernels: {'✅' if stats['optimized_kernels_used'] else '❌'}")
            
            result = {
                'name': 'Optimized (O(N log K))',
                'success': True,
                'init_time': init_time,
                'sim_time': sim_time,
                'total_time': init_time + sim_time,
                'steps_per_second': stats['steps_per_second'],
                'neurons_per_second': stats['neurons_per_second'],
                'memory_usage_gb': stats['memory_usage_gb'],
                'optimized_kernels_used': stats['optimized_kernels_used'],
                'timeout': False
            }
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            result = {
                'name': 'Optimized (O(N log K))',
                'success': False,
                'error': str(e),
                'timeout': False
            }
        
        test_results['implementations'].append(result)
        
        # Calculate actual speedup
        if len(test_results['implementations']) == 2:
            orig = test_results['implementations'][0]
            opt = test_results['implementations'][1]
            
            if orig['success'] and opt['success']:
                actual_speedup = orig['sim_time'] / opt['sim_time'] if opt['sim_time'] > 0 else float('inf')
                print(f"\n🚀 ACTUAL SPEEDUP: {actual_speedup:.2f}x")
                print(f"   Original: {orig['sim_time']:.2f}s")
                print(f"   Optimized: {opt['sim_time']:.2f}s")
                print(f"   Theoretical: {speedup_theoretical:,.0f}x")
                print(f"   Efficiency: {(actual_speedup / speedup_theoretical * 100):.1f}% of theoretical")
                
                test_results['actual_speedup'] = actual_speedup
                test_results['efficiency'] = actual_speedup / speedup_theoretical * 100 if speedup_theoretical > 0 else 0
            elif not orig['success'] and opt['success']:
                print("\n🚀 BREAKTHROUGH: Original failed, Optimized succeeded!")
                print("   This demonstrates the critical importance of O(N log K) optimization")
                test_results['breakthrough'] = True
            else:
                print("\n❌ Both implementations failed at this scale")
                test_results['both_failed'] = True
        
        self.results.append(test_results)
        return test_results
    
    def run_billion_scale_tests(self):
        """Run comprehensive billion-scale tests"""
        print("🚀 BILLION-SCALE ALGORITHMIC OPTIMIZATION TEST")
        print("=" * 70)
        print("Testing the critical difference between O(N²) and O(N log K) algorithms")
        print("Expected: O(N²) should fail at large scales, O(N log K) should succeed")
        
        # Test scales (exponentially increasing)
        test_scales = [
            # Small scale (both should work)
            {"n_neurons": 100_000, "active_percentage": 0.01},
            
            # Medium scale (both should work, but speedup visible)
            {"n_neurons": 1_000_000, "active_percentage": 0.01},
            
            # Large scale (speedup should be dramatic)
            {"n_neurons": 10_000_000, "active_percentage": 0.001},
            
            # Very large scale (original may timeout)
            {"n_neurons": 100_000_000, "active_percentage": 0.0001},
            
            # Billion scale (original should fail, optimized should work)
            {"n_neurons": 1_000_000_000, "active_percentage": 0.00001},
        ]
        
        for scale in test_scales:
            try:
                self.test_algorithmic_complexity(**scale)
            except KeyboardInterrupt:
                print("\n⏹️  Test interrupted by user")
                break
            except Exception as e:
                print(f"❌ Scale {scale['n_neurons']:,} failed: {e}")
                continue
        
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n📊 BILLION-SCALE OPTIMIZATION SUMMARY")
        print("=" * 70)
        
        for result in self.results:
            print(f"\nScale: {result['n_neurons']:,} neurons")
            print(f"  Active per area: {result['k_active']:,}")
            print(f"  Theoretical speedup: {result['theoretical_speedup']:,.0f}x")
            
            if 'actual_speedup' in result:
                print(f"  🚀 Actual speedup: {result['actual_speedup']:.2f}x")
                print(f"  📈 Efficiency: {result['efficiency']:.1f}% of theoretical")
            elif 'breakthrough' in result:
                print("  🎯 BREAKTHROUGH: Original failed, Optimized succeeded!")
            elif 'both_failed' in result:
                print("  ❌ Both implementations failed")
            
            for impl in result['implementations']:
                if impl['success']:
                    print(f"    {impl['name']}: {impl['sim_time']:.2f}s, {impl['neurons_per_second']:,.0f} neurons/sec")
                elif impl.get('timeout', False):
                    print(f"    {impl['name']}: ⏰ TIMEOUT (O(N²) too slow)")
                else:
                    print(f"    {impl['name']}: ❌ {impl['error']}")
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"billion_scale_optimization_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Run the billion-scale optimization test"""
    print("🧠 Billion-Scale Algorithmic Optimization Test")
    print("=" * 60)
    print("This test demonstrates the critical importance of O(N log K) optimization:")
    print("  - O(N²) algorithm: Fails at large scales (timeout or memory)")
    print("  - O(N log K) algorithm: Handles billion-scale efficiently")
    print("  - Expected: 100-1000x speedup at large scales")
    
    # Create test instance
    test = BillionScaleOptimizationTest()
    
    # Run tests
    test.run_billion_scale_tests()
    
    print("\n🎯 Key Insights:")
    print("  - At small scales: Both algorithms work, modest speedup")
    print("  - At medium scales: Speedup becomes significant")
    print("  - At large scales: O(N²) becomes impractical")
    print("  - At billion scale: Only O(N log K) can handle it")
    print("  - This demonstrates why algorithmic optimization is critical!")

if __name__ == "__main__":
    main()
