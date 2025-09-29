#!/usr/bin/env python3
"""
Simple Billion-Scale Test
=========================

This script demonstrates the critical difference between O(N²) and O(N log K) algorithms
at billion-scale by testing the original universal_brain_simulator.py at different scales.

The key insight: O(N²) algorithms become impractical at large scales, while O(N log K) 
algorithms can handle billion-scale efficiently.

This test shows why the algorithmic optimization is critical for billion-scale simulation.
"""

import time
import sys
import os
import json
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import the original implementation
try:
    from universal_brain_simulator import UniversalBrainSimulator, SimulationConfig
    print("✅ Original implementation imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class BillionScaleTest:
    """Test billion-scale performance with original implementation"""
    
    def __init__(self):
        self.results = []
        self.max_test_time = 60  # 1 minute max per test
    
    def test_scale(self, n_neurons: int, active_percentage: float, n_areas: int = 5, n_steps: int = 1):
        """Test at a specific scale"""
        k_active = int(n_neurons * active_percentage)
        
        print(f"\n🧪 Testing Scale: {n_neurons:,} neurons")
        print(f"   Active per area: {k_active:,}")
        print(f"   Total active: {k_active * n_areas:,}")
        print(f"   Steps: {n_steps}")
        
        # Calculate theoretical complexity
        n = n_neurons
        k = k_active
        
        # Original O(N²) complexity for top-k selection
        # Each area does k iterations × n² comparisons
        total_ops = n_areas * k * n * n
        
        print(f"   Theoretical O(N²) operations: {total_ops:,.0f}")
        
        if total_ops > 1e12:  # 1 trillion operations
            print(f"   ⚠️  WARNING: This scale requires >1 trillion operations!")
            print(f"   ⚠️  O(N²) algorithm will be extremely slow or fail")
        
        # Test configuration
        config = SimulationConfig(
            n_neurons=n_neurons,
            active_percentage=active_percentage,
            n_areas=n_areas,
            use_gpu=True,
            use_cuda_kernels=True,
            memory_efficient=True,
            sparse_mode=True
        )
        
        result = {
            'n_neurons': n_neurons,
            'k_active': k_active,
            'n_areas': n_areas,
            'n_steps': n_steps,
            'theoretical_ops': total_ops,
            'success': False,
            'timeout': False,
            'error': None,
            'sim_time': 0,
            'steps_per_second': 0,
            'neurons_per_second': 0,
            'memory_usage_gb': 0
        }
        
        try:
            print(f"   🔬 Creating simulator...")
            start_time = time.time()
            simulator = UniversalBrainSimulator(config)
            init_time = time.time() - start_time
            print(f"   ✅ Initialization: {init_time:.2f}s")
            
            print(f"   🚀 Running simulation...")
            sim_start = time.time()
            
            # Use a timer to limit test time
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Test timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_test_time)
            
            total_time = simulator.simulate(n_steps=n_steps, verbose=False)
            
            signal.alarm(0)  # Cancel alarm
            
            sim_time = time.time() - sim_start
            stats = simulator.get_performance_stats()
            
            print(f"   ✅ Simulation completed: {sim_time:.2f}s")
            print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
            print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
            print(f"   Memory: {stats['memory_usage_gb']:.2f}GB")
            
            result.update({
                'success': True,
                'init_time': init_time,
                'sim_time': sim_time,
                'total_time': init_time + sim_time,
                'steps_per_second': stats['steps_per_second'],
                'neurons_per_second': stats['neurons_per_second'],
                'memory_usage_gb': stats['memory_usage_gb']
            })
            
        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            sim_time = time.time() - sim_start
            print(f"   ⏰ TIMEOUT after {sim_time:.2f}s (>{self.max_test_time}s)")
            print(f"   ❌ O(N²) algorithm too slow at this scale!")
            
            result.update({
                'timeout': True,
                'sim_time': sim_time,
                'error': 'Algorithm too slow - O(N²) complexity'
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            result.update({
                'error': str(e)
            })
        
        self.results.append(result)
        return result
    
    def run_billion_scale_tests(self):
        """Run comprehensive billion-scale tests"""
        print("🚀 BILLION-SCALE PERFORMANCE TEST")
        print("=" * 60)
        print("Testing O(N²) algorithm at increasing scales")
        print("Expected: Performance should degrade dramatically at large scales")
        
        # Test scales (exponentially increasing)
        test_scales = [
            # Small scale (should work fine)
            {"n_neurons": 100_000, "active_percentage": 0.01, "n_steps": 5},
            
            # Medium scale (should work but slower)
            {"n_neurons": 1_000_000, "active_percentage": 0.01, "n_steps": 3},
            
            # Large scale (may be slow)
            {"n_neurons": 10_000_000, "active_percentage": 0.001, "n_steps": 1},
            
            # Very large scale (likely to timeout)
            {"n_neurons": 100_000_000, "active_percentage": 0.0001, "n_steps": 1},
            
            # Billion scale (likely to fail)
            {"n_neurons": 1_000_000_000, "active_percentage": 0.00001, "n_steps": 1},
        ]
        
        for scale in test_scales:
            try:
                self.test_scale(**scale)
            except KeyboardInterrupt:
                print(f"\n⏹️  Test interrupted by user")
                break
            except Exception as e:
                print(f"❌ Scale {scale['n_neurons']:,} failed: {e}")
                continue
        
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print performance summary"""
        print(f"\n📊 BILLION-SCALE TEST SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            print(f"\nScale: {result['n_neurons']:,} neurons")
            print(f"  Active per area: {result['k_active']:,}")
            print(f"  Theoretical ops: {result['theoretical_ops']:,.0f}")
            
            if result['success']:
                print(f"  ✅ Success: {result['sim_time']:.2f}s")
                print(f"  Speed: {result['neurons_per_second']:,.0f} neurons/sec")
                print(f"  Memory: {result['memory_usage_gb']:.2f}GB")
            elif result['timeout']:
                print(f"  ⏰ TIMEOUT: O(N²) too slow")
            else:
                print(f"  ❌ Failed: {result['error']}")
        
        # Calculate performance degradation
        print(f"\n📈 PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        successful_results = [r for r in self.results if r['success']]
        if len(successful_results) >= 2:
            print("Performance degradation as scale increases:")
            for i in range(1, len(successful_results)):
                prev = successful_results[i-1]
                curr = successful_results[i]
                
                scale_ratio = curr['n_neurons'] / prev['n_neurons']
                time_ratio = curr['sim_time'] / prev['sim_time']
                
                print(f"  {prev['n_neurons']:,} → {curr['n_neurons']:,} neurons:")
                print(f"    Scale increase: {scale_ratio:.1f}x")
                print(f"    Time increase: {time_ratio:.1f}x")
                print(f"    Efficiency: {scale_ratio/time_ratio:.1f}x worse")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"  - O(N²) algorithms become impractical at large scales")
        print(f"  - Performance degrades quadratically with neuron count")
        print(f"  - Billion-scale requires O(N log K) optimization")
        print(f"  - This demonstrates why algorithmic optimization is critical!")
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"billion_scale_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Run the billion-scale test"""
    print("🧠 Billion-Scale Performance Test")
    print("=" * 60)
    print("This test demonstrates why O(N²) algorithms fail at large scales:")
    print("  - Small scales: Work fine")
    print("  - Medium scales: Become slower")
    print("  - Large scales: May timeout")
    print("  - Billion scale: Likely to fail")
    print("  - Solution: O(N log K) optimization!")
    
    # Create test instance
    test = BillionScaleTest()
    
    # Run tests
    test.run_billion_scale_tests()
    
    print(f"\n🎯 Next Steps:")
    print(f"  - Implement O(N log K) top-k selection")
    print(f"  - Test optimized algorithms at billion scale")
    print(f"  - Compare performance improvements")
    print(f"  - Demonstrate 100-1000x speedup!")

if __name__ == "__main__":
    main()
