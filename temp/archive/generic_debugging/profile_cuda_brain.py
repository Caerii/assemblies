#!/usr/bin/env python3
"""
CUDA Brain Profiler - Identify computational bottlenecks for optimization
"""

import time
import cProfile
import pstats
import io
import sys
from cuda_brain_python import CudaBrainPython
import numpy as np

class CudaBrainProfiler:
    """Detailed profiler for CUDA brain performance"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.gpu_utilization = {}
        
    def profile_brain_creation(self, brain_params):
        """Profile brain creation and setup"""
        print("🔍 PROFILING BRAIN CREATION")
        print("=" * 50)
        
        start_time = time.time()
        
        # Profile brain initialization
        init_start = time.time()
        brain = CudaBrainPython(**brain_params)
        init_time = time.time() - init_start
        self.timings['brain_init'] = init_time
        
        print(f"✓ Brain initialization: {init_time*1000:.2f}ms")
        
        # Profile area addition
        area_start = time.time()
        brain.AddArea("TestArea", n=100000, k=1000)
        area_time = time.time() - area_start
        self.timings['area_add'] = area_time
        
        print(f"✓ Area addition: {area_time*1000:.2f}ms")
        
        # Profile stimulus addition
        stim_start = time.time()
        brain.AddStimulus("TestStim", k=500)
        stim_time = time.time() - stim_start
        self.timings['stimulus_add'] = stim_time
        
        print(f"✓ Stimulus addition: {stim_time*1000:.2f}ms")
        
        # Profile fiber addition
        fiber_start = time.time()
        brain.AddFiber("TestStim", "TestArea")
        fiber_time = time.time() - fiber_start
        self.timings['fiber_add'] = fiber_time
        
        print(f"✓ Fiber addition: {fiber_time*1000:.2f}ms")
        
        total_time = time.time() - start_time
        print(f"📊 Total setup time: {total_time*1000:.2f}ms")
        
        return brain
    
    def profile_simulation_step(self, brain, num_steps=100):
        """Profile individual simulation steps"""
        print(f"\n🔍 PROFILING SIMULATION STEPS ({num_steps} steps)")
        print("=" * 50)
        
        step_times = []
        area_times = {}
        
        for step in range(num_steps):
            step_start = time.time()
            
            # Profile each area individually
            for area_name, area in brain.areas.items():
                if area['is_explicit']:
                    continue
                
                area_start = time.time()
                
                # Profile candidate generation
                gen_start = time.time()
                candidates = brain._generate_candidates_cuda(area['n'], area['k'])
                gen_time = time.time() - gen_start
                
                # Profile top-k selection
                select_start = time.time()
                selected = brain._select_top_k_cuda(candidates, area['k'])
                select_time = time.time() - select_start
                
                # Profile state update
                update_start = time.time()
                area['activated'] = selected
                area['support'] = max(area['support'], len(selected))
                update_time = time.time() - update_start
                
                area_total = time.time() - area_start
                
                if area_name not in area_times:
                    area_times[area_name] = {
                        'generation': [],
                        'selection': [],
                        'update': [],
                        'total': []
                    }
                
                area_times[area_name]['generation'].append(gen_time)
                area_times[area_name]['selection'].append(select_time)
                area_times[area_name]['update'].append(update_time)
                area_times[area_name]['total'].append(area_total)
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            if step % 20 == 0:
                print(f"  Step {step:3d}: {step_time*1000:.2f}ms")
        
        # Calculate statistics
        avg_step_time = np.mean(step_times)
        std_step_time = np.std(step_times)
        min_step_time = np.min(step_times)
        max_step_time = np.max(step_times)
        
        print(f"\n📊 STEP TIMING STATISTICS:")
        print(f"   Average: {avg_step_time*1000:.2f}ms ± {std_step_time*1000:.2f}ms")
        print(f"   Range: {min_step_time*1000:.2f}ms - {max_step_time*1000:.2f}ms")
        
        # Analyze area-specific bottlenecks
        print(f"\n🔍 AREA-SPECIFIC BOTTLENECKS:")
        for area_name, times in area_times.items():
            avg_gen = np.mean(times['generation'])
            avg_sel = np.mean(times['selection'])
            avg_upd = np.mean(times['update'])
            avg_total = np.mean(times['total'])
            
            print(f"   {area_name:15}: {avg_total*1000:.2f}ms total")
            print(f"     Generation: {avg_gen*1000:.2f}ms ({avg_gen/avg_total*100:.1f}%)")
            print(f"     Selection:  {avg_sel*1000:.2f}ms ({avg_sel/avg_total*100:.1f}%)")
            print(f"     Update:     {avg_upd*1000:.2f}ms ({avg_upd/avg_total*100:.1f}%)")
        
        self.timings['step_avg'] = avg_step_time
        self.timings['step_std'] = std_step_time
        self.area_times = area_times
        
        return step_times, area_times
    
    def profile_memory_usage(self, brain):
        """Profile memory usage patterns"""
        print(f"\n🔍 PROFILING MEMORY USAGE")
        print("=" * 50)
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Profile memory during simulation
        memory_samples = []
        
        for step in range(10):
            brain.SimulateOneStep()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)
        
        avg_memory = np.mean(memory_samples)
        max_memory = np.max(memory_samples)
        memory_growth = max_memory - initial_memory
        
        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Average memory: {avg_memory:.1f} MB")
        print(f"   Peak memory: {max_memory:.1f} MB")
        print(f"   Memory growth: {memory_growth:.1f} MB")
        
        self.memory_usage = {
            'initial': initial_memory,
            'average': avg_memory,
            'peak': max_memory,
            'growth': memory_growth
        }
    
    def profile_cuda_kernels(self, brain):
        """Profile CUDA kernel performance"""
        print(f"\n🔍 PROFILING CUDA KERNELS")
        print("=" * 50)
        
        # Test different neuron counts
        test_sizes = [1000, 10000, 100000, 500000, 1000000]
        kernel_times = {}
        
        for size in test_sizes:
            print(f"   Testing {size:,} neurons...")
            
            # Profile candidate generation
            gen_times = []
            for _ in range(10):
                start = time.time()
                candidates = brain._generate_candidates_cuda(size, min(1000, size//10))
                gen_times.append(time.time() - start)
            
            # Profile top-k selection
            sel_times = []
            for _ in range(10):
                candidates = np.random.exponential(1.0, size)
                start = time.time()
                selected = brain._select_top_k_cuda(candidates, min(1000, size//10))
                sel_times.append(time.time() - start)
            
            kernel_times[size] = {
                'generation': np.mean(gen_times),
                'selection': np.mean(sel_times),
                'total': np.mean(gen_times) + np.mean(sel_times)
            }
            
            print(f"     Generation: {np.mean(gen_times)*1000:.2f}ms")
            print(f"     Selection:  {np.mean(sel_times)*1000:.2f}ms")
            print(f"     Total:      {kernel_times[size]['total']*1000:.2f}ms")
        
        self.kernel_times = kernel_times
        
        # Analyze scaling
        print(f"\n📊 KERNEL SCALING ANALYSIS:")
        for size in test_sizes[1:]:
            prev_size = test_sizes[test_sizes.index(size) - 1]
            scale_factor = size / prev_size
            time_factor = kernel_times[size]['total'] / kernel_times[prev_size]['total']
            efficiency = scale_factor / time_factor
            
            print(f"   {prev_size:,} → {size:,}: {time_factor:.2f}x time for {scale_factor:.1f}x neurons")
            print(f"     Efficiency: {efficiency:.2f} (1.0 = perfect scaling)")
    
    def identify_bottlenecks(self):
        """Identify the main computational bottlenecks"""
        print(f"\n🎯 BOTTLENECK ANALYSIS")
        print("=" * 50)
        
        # Find slowest operations
        if hasattr(self, 'area_times'):
            total_times = {}
            for area_name, times in self.area_times.items():
                total_times[area_name] = np.mean(times['total'])
            
            slowest_areas = sorted(total_times.items(), key=lambda x: x[1], reverse=True)
            
            print(f"🐌 SLOWEST AREAS:")
            for area_name, avg_time in slowest_areas[:5]:
                print(f"   {area_name:15}: {avg_time*1000:.2f}ms")
        
        # Find slowest operations within areas
        if hasattr(self, 'area_times'):
            print(f"\n🔍 SLOWEST OPERATIONS:")
            operation_times = {'generation': [], 'selection': [], 'update': []}
            
            for area_name, times in self.area_times.items():
                operation_times['generation'].extend(times['generation'])
                operation_times['selection'].extend(times['selection'])
                operation_times['update'].extend(times['update'])
            
            for op_name, times in operation_times.items():
                avg_time = np.mean(times)
                print(f"   {op_name:12}: {avg_time*1000:.2f}ms")
        
        # Memory bottlenecks
        if hasattr(self, 'memory_usage'):
            print(f"\n💾 MEMORY BOTTLENECKS:")
            print(f"   Memory growth: {self.memory_usage['growth']:.1f} MB")
            if self.memory_usage['growth'] > 100:
                print(f"   ⚠️  High memory growth - potential memory leak")
            else:
                print(f"   ✅ Memory usage looks good")
        
        # Kernel scaling bottlenecks
        if hasattr(self, 'kernel_times'):
            print(f"\n⚡ KERNEL SCALING BOTTLENECKS:")
            sizes = list(self.kernel_times.keys())
            for i in range(1, len(sizes)):
                prev_size = sizes[i-1]
                curr_size = sizes[i]
                scale_factor = curr_size / prev_size
                time_factor = self.kernel_times[curr_size]['total'] / self.kernel_times[prev_size]['total']
                
                if time_factor > scale_factor * 1.5:  # Poor scaling
                    print(f"   ⚠️  Poor scaling at {prev_size:,} → {curr_size:,} neurons")
                    print(f"       {time_factor:.2f}x time for {scale_factor:.1f}x neurons")
                else:
                    print(f"   ✅ Good scaling at {prev_size:,} → {curr_size:,} neurons")
    
    def generate_optimization_recommendations(self):
        """Generate specific optimization recommendations"""
        print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = []
        
        # Based on profiling results
        if hasattr(self, 'area_times'):
            # Check if generation is the bottleneck
            gen_times = []
            sel_times = []
            for times in self.area_times.values():
                gen_times.extend(times['generation'])
                sel_times.extend(times['selection'])
            
            avg_gen = np.mean(gen_times)
            avg_sel = np.mean(sel_times)
            
            if avg_gen > avg_sel * 2:
                recommendations.append("🔥 PRIORITY: Optimize candidate generation - it's the main bottleneck")
                recommendations.append("   - Use more efficient random number generation")
                recommendations.append("   - Implement vectorized operations")
                recommendations.append("   - Consider pre-computed candidate pools")
            
            if avg_sel > avg_gen * 2:
                recommendations.append("🔥 PRIORITY: Optimize top-k selection - it's the main bottleneck")
                recommendations.append("   - Implement faster sorting algorithms")
                recommendations.append("   - Use partial sort instead of full sort")
                recommendations.append("   - Consider approximate top-k selection")
        
        # Memory optimization
        if hasattr(self, 'memory_usage') and self.memory_usage['growth'] > 50:
            recommendations.append("💾 MEMORY: Reduce memory allocation overhead")
            recommendations.append("   - Reuse arrays instead of creating new ones")
            recommendations.append("   - Implement memory pooling")
            recommendations.append("   - Use in-place operations where possible")
        
        # Kernel optimization
        if hasattr(self, 'kernel_times'):
            recommendations.append("⚡ KERNELS: Optimize CUDA kernel performance")
            recommendations.append("   - Implement proper CUDA kernels (replace Python simulation)")
            recommendations.append("   - Use shared memory for frequently accessed data")
            recommendations.append("   - Optimize memory access patterns")
            recommendations.append("   - Consider using cuBLAS for matrix operations")
        
        # General recommendations
        recommendations.extend([
            "🔧 GENERAL: Implement parallel processing",
            "   - Process multiple areas simultaneously",
            "   - Use async operations where possible",
            "   - Consider multi-GPU support",
            "",
            "📊 MONITORING: Add real-time performance monitoring",
            "   - GPU utilization tracking",
            "   - Memory usage monitoring",
            "   - Performance counters"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
        
        return recommendations

def run_comprehensive_profiling():
    """Run comprehensive profiling analysis"""
    print("🔍 COMPREHENSIVE CUDA BRAIN PROFILING")
    print("=" * 60)
    
    profiler = CudaBrainProfiler()
    
    # Profile brain creation
    brain_params = {
        'p': 0.1,
        'beta': 0.5,
        'max_weight': 1.0,
        'seed': 42
    }
    
    brain = profiler.profile_brain_creation(brain_params)
    
    # Add test areas
    print(f"\n🏗️  Setting up test areas...")
    brain.AddArea("SmallArea", n=10000, k=100)
    brain.AddArea("MediumArea", n=100000, k=1000)
    brain.AddArea("LargeArea", n=500000, k=5000)
    brain.AddStimulus("TestStim", k=500)
    brain.AddFiber("TestStim", "SmallArea")
    brain.AddFiber("SmallArea", "MediumArea")
    brain.AddFiber("MediumArea", "LargeArea")
    
    # Profile simulation steps
    step_times, area_times = profiler.profile_simulation_step(brain, num_steps=50)
    
    # Profile memory usage
    profiler.profile_memory_usage(brain)
    
    # Profile CUDA kernels
    profiler.profile_cuda_kernels(brain)
    
    # Identify bottlenecks
    profiler.identify_bottlenecks()
    
    # Generate recommendations
    recommendations = profiler.generate_optimization_recommendations()
    
    print(f"\n🎯 PROFILING COMPLETE!")
    print(f"   Ready for optimization based on data-driven insights!")
    
    return profiler, recommendations

if __name__ == "__main__":
    try:
        profiler, recommendations = run_comprehensive_profiling()
    except KeyboardInterrupt:
        print("\n⏹️  Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        sys.exit(1)
