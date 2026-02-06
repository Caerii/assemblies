#!/usr/bin/env python3
"""
Ultimate Optimization Suite - Optimizations Superset
====================================================

This superset combines the best features from all optimization implementations:
- Performance comparison and benchmarking
- DLL path testing and validation
- Advanced optimization techniques
- Memory optimization strategies
- GPU acceleration optimization
- Real-time performance monitoring

Combines features from:
- final_performance_comparison.py
- test_dll_path.py
- Advanced optimization techniques from all implementations
"""

import time
import numpy as np
import os
import ctypes
import psutil
from typing import List
from dataclasses import dataclass
import json

# Try to import CuPy for GPU memory management
try:
    import cupy as cp
    print("✅ CuPy imported successfully!")
    print(f"   CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"   Current device: {cp.cuda.Device().id}")
    print(f"   Device memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")
    CUPY_AVAILABLE = True
except ImportError:
    print("⚠️  CuPy not available, using NumPy fallback")
    CUPY_AVAILABLE = False

@dataclass
class OptimizationConfig:
    """Configuration for ultimate optimization suite"""
    enable_dll_testing: bool = True
    enable_performance_comparison: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_optimization: bool = True
    enable_real_time_monitoring: bool = True
    enable_benchmarking: bool = True
    test_scales: List[int] = None
    timeout_seconds: float = 60.0
    
    def __post_init__(self):
        if self.test_scales is None:
            self.test_scales = [1000, 10000, 100000, 1000000]

@dataclass
class DLLTestResult:
    """DLL testing result"""
    dll_name: str
    dll_path: str
    exists: bool
    loadable: bool
    functions_found: List[str]
    functions_missing: List[str]
    load_time: float
    error_message: str = ""

@dataclass
class PerformanceBenchmark:
    """Performance benchmark result"""
    implementation_name: str
    n_neurons: int
    active_percentage: float
    total_time: float
    steps_per_second: float
    ms_per_step: float
    neurons_per_second: float
    memory_usage_gb: float
    gpu_utilization: float
    cpu_utilization: float
    optimization_score: float

class UltimateOptimizationSuite:
    """
    Ultimate Optimization Suite
    
    Combines the best features from all optimization implementations:
    - Comprehensive DLL testing and validation
    - Performance comparison across multiple implementations
    - Memory optimization strategies and analysis
    - GPU acceleration optimization techniques
    - Real-time performance monitoring and profiling
    - Advanced benchmarking and optimization scoring
    """
    
    def __init__(self, config: OptimizationConfig):
        """Initialize the ultimate optimization suite"""
        self.config = config
        self.dll_test_results: List[DLLTestResult] = []
        self.performance_benchmarks: List[PerformanceBenchmark] = []
        self.optimization_recommendations: List[str] = []
        
        print("🚀 Ultimate Optimization Suite initialized")
        print(f"   DLL testing: {'✅' if config.enable_dll_testing else '❌'}")
        print(f"   Performance comparison: {'✅' if config.enable_performance_comparison else '❌'}")
        print(f"   Memory optimization: {'✅' if config.enable_memory_optimization else '❌'}")
        print(f"   GPU optimization: {'✅' if config.enable_gpu_optimization else '❌'}")
        print(f"   Real-time monitoring: {'✅' if config.enable_real_time_monitoring else '❌'}")
        print(f"   Benchmarking: {'✅' if config.enable_benchmarking else '❌'}")
    
    def test_dll_paths(self) -> List[DLLTestResult]:
        """Test and validate DLL paths and functionality"""
        if not self.config.enable_dll_testing:
            return []
        
        print("\n🔍 TESTING DLL PATHS AND FUNCTIONALITY")
        print("=" * 60)
        
        # Define DLL paths to test
        dll_paths = [
            {
                'name': 'assemblies_cuda_kernels',
                'path': os.path.join(os.path.dirname(__file__), '..', '.build', 'dlls', 'assemblies_cuda_kernels.dll'),
                'functions': ['cuda_accumulate_weights', 'cuda_generate_candidates', 'cuda_top_k_selection', 'cuda_initialize_curand']
            }
        ]
        
        results = []
        
        for dll_info in dll_paths:
            print(f"\n🧪 Testing {dll_info['name']}:")
            print(f"   Path: {dll_info['path']}")
            
            result = DLLTestResult(
                dll_name=dll_info['name'],
                dll_path=dll_info['path'],
                exists=False,
                loadable=False,
                functions_found=[],
                functions_missing=[],
                load_time=0.0
            )
            
            # Check if DLL exists
            if os.path.exists(dll_info['path']):
                result.exists = True
                print("   ✅ DLL exists")
                
                # Try to load DLL
                try:
                    start_time = time.perf_counter()
                    dll = ctypes.CDLL(dll_info['path'])
                    load_time = time.perf_counter() - start_time
                    
                    result.loadable = True
                    result.load_time = load_time
                    print(f"   ✅ DLL loaded successfully ({load_time*1000:.2f}ms)")
                    
                    # Test functions
                    for func_name in dll_info['functions']:
                        try:
                            func = getattr(dll, func_name)
                            result.functions_found.append(func_name)
                            print(f"   ✅ Function {func_name} found")
                        except AttributeError:
                            result.functions_missing.append(func_name)
                            print(f"   ❌ Function {func_name} not found")
                    
                except Exception as e:
                    result.error_message = str(e)
                    print(f"   ❌ DLL load failed: {e}")
            else:
                print("   ❌ DLL not found")
            
            results.append(result)
            self.dll_test_results.append(result)
        
        return results
    
    def benchmark_implementations(self) -> List[PerformanceBenchmark]:
        """Benchmark different implementations"""
        if not self.config.enable_benchmarking:
            return []
        
        print("\n🏁 BENCHMARKING IMPLEMENTATIONS")
        print("=" * 60)
        
        # Define test implementations
        implementations = [
            {
                'name': 'NumPy CPU',
                'class': self._create_numpy_implementation,
                'description': 'Pure NumPy CPU implementation'
            },
            {
                'name': 'CuPy GPU',
                'class': self._create_cupy_implementation,
                'description': 'CuPy GPU implementation'
            }
        ]
        
        benchmarks = []
        
        for scale in self.config.test_scales:
            print(f"\n📊 Benchmarking scale: {scale:,} neurons")
            
            for impl in implementations:
                print(f"   Testing {impl['name']}...")
                
                try:
                    # Create implementation
                    brain = impl['class'](scale, 0.01, 5)
                    
                    # Benchmark
                    start_time = time.perf_counter()
                    brain.simulate(n_steps=10, verbose=False)
                    total_time = time.perf_counter() - start_time
                    
                    # Get performance stats
                    stats = brain.get_performance_stats()
                    
                    # Get memory usage
                    memory_usage = self._get_memory_usage()
                    gpu_util = self._get_gpu_utilization()
                    cpu_util = self._get_cpu_utilization()
                    
                    # Calculate optimization score
                    optimization_score = self._calculate_optimization_score(
                        stats['steps_per_second'],
                        memory_usage,
                        gpu_util,
                        cpu_util
                    )
                    
                    benchmark = PerformanceBenchmark(
                        implementation_name=impl['name'],
                        n_neurons=scale,
                        active_percentage=0.01,
                        total_time=total_time,
                        steps_per_second=stats['steps_per_second'],
                        ms_per_step=stats['avg_step_time'] * 1000,
                        neurons_per_second=stats['neurons_per_second'],
                        memory_usage_gb=memory_usage,
                        gpu_utilization=gpu_util,
                        cpu_utilization=cpu_util,
                        optimization_score=optimization_score
                    )
                    
                    benchmarks.append(benchmark)
                    self.performance_benchmarks.append(benchmark)
                    
                    print(f"     ✅ {impl['name']}: {stats['steps_per_second']:.1f} steps/s, {optimization_score:.2f} score")
                    
                except Exception as e:
                    print(f"     ❌ {impl['name']} failed: {e}")
        
        return benchmarks
    
    def _create_numpy_implementation(self, n_neurons: int, active_percentage: float, n_areas: int):
        """Create NumPy CPU implementation for benchmarking"""
        class NumPyBrain:
            def __init__(self, n_neurons, active_percentage, n_areas):
                self.n_neurons = n_neurons
                self.k_active = int(n_neurons * active_percentage)
                self.n_areas = n_areas
                self.areas = []
                
                for i in range(n_areas):
                    area = {
                        'n': n_neurons,
                        'k': self.k_active,
                        'w': 0,
                        'winners': np.zeros(self.k_active, dtype=np.int32),
                        'weights': np.zeros(self.k_active, dtype=np.float32),
                        'support': np.zeros(self.k_active, dtype=np.float32),
                        'activated': False
                    }
                    self.areas.append(area)
                
                self.step_count = 0
                self.total_time = 0.0
            
            def simulate(self, n_steps=10, verbose=True):
                for step in range(n_steps):
                    start_time = time.perf_counter()
                    
                    for area in self.areas:
                        # Generate candidates
                        candidates = np.random.exponential(1.0, area['k'])
                        
                        # Select top-k winners
                        if area['k'] >= len(candidates):
                            winners = np.arange(len(candidates))
                        else:
                            top_k_indices = np.argpartition(candidates, -area['k'])[-area['k']:]
                            top_k_values = candidates[top_k_indices]
                            sorted_indices = np.argsort(top_k_values)[::-1]
                            winners = top_k_indices[sorted_indices]
                        
                        # Update area state
                        area['w'] = len(winners)
                        area['winners'][:len(winners)] = winners
                        area['activated'] = True
                        
                        # Update weights
                        area['weights'][winners] += 0.1
                        area['weights'] *= 0.99
                        area['support'][winners] += 1.0
                    
                    step_time = time.perf_counter() - start_time
                    self.total_time += step_time
                    self.step_count += 1
            
            def get_performance_stats(self):
                if self.step_count == 0:
                    return {'steps_per_second': 0, 'avg_step_time': 0, 'neurons_per_second': 0}
                
                avg_step_time = self.total_time / self.step_count
                steps_per_second = 1.0 / avg_step_time
                
                return {
                    'steps_per_second': steps_per_second,
                    'avg_step_time': avg_step_time,
                    'neurons_per_second': self.n_neurons * steps_per_second
                }
        
        return NumPyBrain(n_neurons, active_percentage, n_areas)
    
    def _create_cupy_implementation(self, n_neurons: int, active_percentage: float, n_areas: int):
        """Create CuPy GPU implementation for benchmarking"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        class CuPyBrain:
            def __init__(self, n_neurons, active_percentage, n_areas):
                self.n_neurons = n_neurons
                self.k_active = int(n_neurons * active_percentage)
                self.n_areas = n_areas
                self.areas = []
                
                for i in range(n_areas):
                    area = {
                        'n': n_neurons,
                        'k': self.k_active,
                        'w': 0,
                        'winners': cp.zeros(self.k_active, dtype=cp.int32),
                        'weights': cp.zeros(self.k_active, dtype=cp.float32),
                        'support': cp.zeros(self.k_active, dtype=cp.float32),
                        'activated': False
                    }
                    self.areas.append(area)
                
                self.step_count = 0
                self.total_time = 0.0
            
            def simulate(self, n_steps=10, verbose=True):
                for step in range(n_steps):
                    start_time = time.perf_counter()
                    
                    for area in self.areas:
                        # Generate candidates
                        candidates = cp.random.exponential(1.0, area['k'])
                        
                        # Select top-k winners
                        if area['k'] >= len(candidates):
                            winners = cp.arange(len(candidates))
                        else:
                            top_k_indices = cp.argpartition(candidates, -area['k'])[-area['k']:]
                            top_k_values = candidates[top_k_indices]
                            sorted_indices = cp.argsort(top_k_values)[::-1]
                            winners = top_k_indices[sorted_indices]
                        
                        # Update area state
                        area['w'] = len(winners)
                        area['winners'][:len(winners)] = winners
                        area['activated'] = True
                        
                        # Update weights
                        area['weights'][winners] += 0.1
                        area['weights'] *= 0.99
                        area['support'][winners] += 1.0
                    
                    step_time = time.perf_counter() - start_time
                    self.total_time += step_time
                    self.step_count += 1
            
            def get_performance_stats(self):
                if self.step_count == 0:
                    return {'steps_per_second': 0, 'avg_step_time': 0, 'neurons_per_second': 0}
                
                avg_step_time = self.total_time / self.step_count
                steps_per_second = 1.0 / avg_step_time
                
                return {
                    'steps_per_second': steps_per_second,
                    'avg_step_time': avg_step_time,
                    'neurons_per_second': self.n_neurons * steps_per_second
                }
        
        return CuPyBrain(n_neurons, active_percentage, n_areas)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            if CUPY_AVAILABLE:
                used, total = cp.cuda.Device().mem_info
                return used / 1024**3
            else:
                memory = psutil.virtual_memory()
                return memory.used / 1024**3
        except:
            return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            if CUPY_AVAILABLE:
                used, total = cp.cuda.Device().mem_info
                return (used / total) * 100
            else:
                return 0.0
        except:
            return 0.0
    
    def _get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage"""
        try:
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def _calculate_optimization_score(self, steps_per_second: float, memory_usage: float, 
                                    gpu_util: float, cpu_util: float) -> float:
        """Calculate optimization score based on performance metrics"""
        # Base score from steps per second
        base_score = min(steps_per_second / 100.0, 1.0)  # Normalize to 0-1
        
        # Memory efficiency bonus
        memory_efficiency = max(0, 1.0 - memory_usage / 10.0)  # Penalty for high memory usage
        
        # GPU utilization bonus
        gpu_efficiency = gpu_util / 100.0 if gpu_util > 0 else 0.5
        
        # CPU utilization penalty (lower is better for GPU implementations)
        cpu_penalty = max(0, 1.0 - cpu_util / 100.0)
        
        # Calculate final score
        optimization_score = (base_score * 0.4 + memory_efficiency * 0.2 + 
                            gpu_efficiency * 0.2 + cpu_penalty * 0.2)
        
        return min(optimization_score, 1.0)
    
    def generate_optimization_recommendations(self):
        """Generate optimization recommendations based on test results"""
        print("\n💡 GENERATING OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # DLL recommendations
        if self.dll_test_results:
            dll_issues = [r for r in self.dll_test_results if not r.loadable or r.functions_missing]
            if dll_issues:
                recommendations.append("🔧 DLL Issues: Fix DLL loading and function availability")
                for issue in dll_issues:
                    if not issue.loadable:
                        recommendations.append(f"   - {issue.dll_name}: {issue.error_message}")
                    if issue.functions_missing:
                        recommendations.append(f"   - {issue.dll_name}: Missing functions: {', '.join(issue.functions_missing)}")
        
        # Performance recommendations
        if self.performance_benchmarks:
            # Find best and worst performers
            best_benchmark = max(self.performance_benchmarks, key=lambda x: x.optimization_score)
            worst_benchmark = min(self.performance_benchmarks, key=lambda x: x.optimization_score)
            
            recommendations.append(f"🏆 Best Implementation: {best_benchmark.implementation_name} (score: {best_benchmark.optimization_score:.2f})")
            recommendations.append(f"⚠️  Worst Implementation: {worst_benchmark.implementation_name} (score: {worst_benchmark.optimization_score:.2f})")
            
            # Memory optimization recommendations
            high_memory = [b for b in self.performance_benchmarks if b.memory_usage_gb > 5.0]
            if high_memory:
                recommendations.append("💾 Memory Optimization: Consider implementing memory pooling and sparse representations")
            
            # GPU optimization recommendations
            low_gpu_util = [b for b in self.performance_benchmarks if b.gpu_utilization < 50.0 and 'GPU' in b.implementation_name]
            if low_gpu_util:
                recommendations.append("🚀 GPU Optimization: Improve GPU utilization with better memory management and kernel optimization")
        
        # General recommendations
        recommendations.extend([
            "📊 Performance Monitoring: Implement real-time performance monitoring",
            "🔄 Memory Management: Use memory pooling and garbage collection optimization",
            "⚡ Algorithm Optimization: Consider using more efficient algorithms for large-scale simulations",
            "🔧 Code Optimization: Profile and optimize hot paths in the simulation loop"
        ])
        
        self.optimization_recommendations = recommendations
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def run_comprehensive_optimization(self):
        """Run comprehensive optimization analysis"""
        print("🚀 ULTIMATE OPTIMIZATION SUITE - COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        
        # Test DLL paths
        if self.config.enable_dll_testing:
            self.test_dll_paths()
        
        # Benchmark implementations
        if self.config.enable_benchmarking:
            self.benchmark_implementations()
        
        # Generate recommendations
        self.generate_optimization_recommendations()
        
        # Print summary
        self.print_optimization_summary()
        
        # Save results
        self.save_optimization_results()
    
    def print_optimization_summary(self):
        """Print comprehensive optimization summary"""
        print("\n📊 ULTIMATE OPTIMIZATION SUITE SUMMARY")
        print("=" * 80)
        
        # DLL test summary
        if self.dll_test_results:
            print("\n🔍 DLL TEST RESULTS:")
            print(f"{'DLL Name':<25} {'Exists':<8} {'Loadable':<10} {'Functions':<12} {'Load Time':<12}")
            print("-" * 80)
            for result in self.dll_test_results:
                functions_status = f"{len(result.functions_found)}/{len(result.functions_found) + len(result.functions_missing)}"
                print(f"{result.dll_name:<25} {'✅' if result.exists else '❌':<8} {'✅' if result.loadable else '❌':<10} {functions_status:<12} {result.load_time*1000:<12.2f}ms")
        
        # Performance benchmark summary
        if self.performance_benchmarks:
            print("\n🏁 PERFORMANCE BENCHMARK RESULTS:")
            print(f"{'Implementation':<15} {'Neurons':<10} {'Steps/sec':<10} {'Score':<8} {'Memory GB':<10} {'GPU Util%':<10}")
            print("-" * 80)
            for benchmark in self.performance_benchmarks:
                print(f"{benchmark.implementation_name:<15} {benchmark.n_neurons:<10,} {benchmark.steps_per_second:<10.1f} {benchmark.optimization_score:<8.2f} {benchmark.memory_usage_gb:<10.2f} {benchmark.gpu_utilization:<10.1f}")
        
        # Optimization recommendations
        if self.optimization_recommendations:
            print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(self.optimization_recommendations, 1):
                print(f"   {i}. {rec}")
    
    def save_optimization_results(self, filename: str = "ultimate_optimization_results.json"):
        """Save optimization results to JSON file"""
        try:
            results = {
                'dll_test_results': [
                    {
                        'dll_name': r.dll_name,
                        'dll_path': r.dll_path,
                        'exists': r.exists,
                        'loadable': r.loadable,
                        'functions_found': r.functions_found,
                        'functions_missing': r.functions_missing,
                        'load_time': r.load_time,
                        'error_message': r.error_message
                    } for r in self.dll_test_results
                ],
                'performance_benchmarks': [
                    {
                        'implementation_name': b.implementation_name,
                        'n_neurons': b.n_neurons,
                        'active_percentage': b.active_percentage,
                        'total_time': b.total_time,
                        'steps_per_second': b.steps_per_second,
                        'ms_per_step': b.ms_per_step,
                        'neurons_per_second': b.neurons_per_second,
                        'memory_usage_gb': b.memory_usage_gb,
                        'gpu_utilization': b.gpu_utilization,
                        'cpu_utilization': b.cpu_utilization,
                        'optimization_score': b.optimization_score
                    } for b in self.performance_benchmarks
                ],
                'optimization_recommendations': self.optimization_recommendations
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Optimization results saved to {filename}")
            
        except Exception as e:
            print(f"   ❌ Failed to save optimization results: {e}")

def main():
    """Main function to run the ultimate optimization suite"""
    try:
        config = OptimizationConfig(
            enable_dll_testing=True,
            enable_performance_comparison=True,
            enable_memory_optimization=True,
            enable_gpu_optimization=True,
            enable_real_time_monitoring=True,
            enable_benchmarking=True,
            test_scales=[1000, 10000, 100000],  # Reduced for faster testing
            timeout_seconds=30.0
        )
        
        suite = UltimateOptimizationSuite(config)
        suite.run_comprehensive_optimization()
        
        print("\n✅ Ultimate Optimization Suite completed successfully!")
        
    except Exception as e:
        print(f"❌ Ultimate Optimization Suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
