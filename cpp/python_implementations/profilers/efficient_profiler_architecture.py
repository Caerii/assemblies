"""
Efficient Profiler Architecture Design

This shows how to restructure the profiler for maximum efficiency:
1. Async/parallel execution
2. Memory pooling
3. Cached calculations
4. Streaming results
5. Modular metrics
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import json
import numpy as np

# Try to import CuPy, fallback to NumPy if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False

@dataclass
class ProfilerConfig:
    """Configuration for efficient profiler"""
    max_workers: int = 4  # Parallel execution
    memory_pool_size: int = 1024 * 1024 * 1024  # 1GB pool
    cache_theoretical: bool = True
    streaming_output: bool = True
    real_time_monitoring: bool = True
    custom_metrics: List[str] = None

class MemoryPool:
    """Efficient memory pool for reusing allocations"""
    def __init__(self, size_gb: float):
        self.pool = {}
        self.size_gb = size_gb
        self.lock = threading.Lock()
    
    def get_array(self, shape: tuple, dtype: str) -> Any:
        """Get pre-allocated array from pool"""
        key = (shape, dtype)
        with self.lock:
            if key not in self.pool:
                # Allocate new array
                if dtype == 'float32':
                    self.pool[key] = cp.zeros(shape, dtype=cp.float32)
                else:
                    self.pool[key] = cp.zeros(shape, dtype=cp.int32)
            return self.pool[key].copy()  # Return a copy to avoid sharing
    
    def clear_pool(self):
        """Clear memory pool"""
        with self.lock:
            self.pool.clear()

class MetricCollector:
    """Modular metric collection system"""
    def __init__(self):
        self.metrics = {}
        self.collectors = {}
    
    def register_metric(self, name: str, collector: Callable):
        """Register a custom metric collector"""
        self.collectors[name] = collector
    
    def collect_metrics(self, brain: Dict[str, Any]) -> Dict[str, float]:
        """Collect all registered metrics"""
        results = {}
        for name, collector in self.collectors.items():
            try:
                results[name] = collector(brain)
            except Exception as e:
                results[name] = 0.0
        return results

class AsyncProfiler:
    """Asynchronous profiler for parallel execution"""
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.memory_pool = MemoryPool(config.memory_pool_size)
        self.metric_collector = MetricCollector()
        self.results_queue = Queue()
        self.theoretical_cache = {}
        
        # Register built-in metrics
        self._register_builtin_metrics()
    
    def _register_builtin_metrics(self):
        """Register built-in performance metrics"""
        self.metric_collector.register_metric('bytes_per_neuron', self._bytes_per_neuron)
        self.metric_collector.register_metric('gpu_utilization', self._gpu_utilization)
        self.metric_collector.register_metric('memory_bandwidth', self._memory_bandwidth)
        self.metric_collector.register_metric('cache_hit_rate', self._cache_hit_rate)
    
    async def profile_scale_async(self, scale_name: str, n_neurons: int, 
                                active_percentage: float, n_areas: int = 5) -> Dict[str, Any]:
        """Profile a single scale asynchronously"""
        # Use thread pool for CPU-bound work
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor, 
                self._profile_scale_sync, 
                scale_name, n_neurons, active_percentage, n_areas
            )
        return result
    
    def _profile_scale_sync(self, scale_name: str, n_neurons: int, 
                           active_percentage: float, n_areas: int) -> Dict[str, Any]:
        """Synchronous profiling with memory pooling"""
        k_active = int(n_neurons * active_percentage)
        
        # Use cached theoretical memory
        cache_key = (n_neurons, k_active, n_areas)
        if self.config.cache_theoretical and cache_key in self.theoretical_cache:
            theoretical_memory = self.theoretical_cache[cache_key]
        else:
            theoretical_memory = self._calculate_theoretical_memory(n_neurons, k_active, n_areas)
            if self.config.cache_theoretical:
                self.theoretical_cache[cache_key] = theoretical_memory
        
        # Create brain with memory pooling
        brain = self._create_brain_with_pooling(n_neurons, active_percentage, n_areas)
        
        # Run simulation
        start_time = time.perf_counter()
        step_times = []
        
        for step in range(5):  # Reduced steps for efficiency
            step_time = self._simulate_step_with_pooling(brain)
            step_times.append(step_time)
        
        simulation_time = time.perf_counter() - start_time
        
        # Collect metrics
        metrics = self.metric_collector.collect_metrics(brain)
        
        # Calculate performance
        steps_per_sec = len(step_times) / simulation_time if simulation_time > 0 else 0
        ms_per_step = (simulation_time / len(step_times)) * 1000 if step_times else 0
        
        return {
            'scale_name': scale_name,
            'n_neurons': n_neurons,
            'active_percentage': active_percentage,
            'n_areas': n_areas,
            'steps_per_sec': steps_per_sec,
            'ms_per_step': ms_per_step,
            'simulation_time': simulation_time,
            'theoretical_memory_gb': theoretical_memory['total_theoretical_memory_gb'],
            **metrics  # Include all collected metrics
        }
    
    def _create_brain_with_pooling(self, n_neurons: int, active_percentage: float, 
                                 n_areas: int) -> Dict[str, Any]:
        """Create brain using memory pool for efficiency"""
        k_active = int(n_neurons * active_percentage)
        
        areas = []
        for i in range(n_areas):
            # Use memory pool for arrays
            area = {
                'n': n_neurons,
                'k': k_active,
                'w': 0,
                'winners': self.memory_pool.get_array((k_active,), 'int32'),
                'weights': self.memory_pool.get_array((k_active,), 'float32'),
                'support': self.memory_pool.get_array((k_active,), 'float32'),
                'activated': False
            }
            areas.append(area)
        
        return {
            'n_neurons': n_neurons,
            'k_active': k_active,
            'n_areas': n_areas,
            'areas': areas,
            'step_count': 0,
            'total_time': 0.0,
            'step_times': []
        }
    
    def _simulate_step_with_pooling(self, brain: Dict[str, Any]) -> float:
        """Simulate one step using memory pool"""
        start_time = time.perf_counter()
        
        for area in brain['areas']:
            # Generate candidates using pooled memory
            candidates = cp.random.exponential(1.0, size=area['k'])
            
            # Select top-k winners
            if area['k'] >= len(candidates):
                winners = cp.arange(len(candidates))
            else:
                top_k_indices = cp.argpartition(candidates, -area['k'])[-area['k']:]
                top_k_values = candidates[top_k_indices]
                sorted_indices = cp.argsort(top_k_values)[::-1]
                winners = top_k_indices[sorted_indices]
            
            # Update area state
            area['winners'] = winners
            area['w'] = len(winners)
            area['activated'] = True
        
        return time.perf_counter() - start_time
    
    def _bytes_per_neuron(self, brain: Dict[str, Any]) -> float:
        """Calculate bytes per neuron metric"""
        total_bytes = 0
        for area in brain['areas']:
            total_bytes += area['weights'].nbytes
            total_bytes += area['support'].nbytes
            total_bytes += area['winners'].nbytes
        
        return total_bytes / brain['n_neurons'] if brain['n_neurons'] > 0 else 0.0
    
    def _gpu_utilization(self, brain: Dict[str, Any]) -> float:
        """Calculate GPU utilization metric"""
        # This would use nvidia-ml-py or similar
        return 0.0  # Placeholder
    
    def _memory_bandwidth(self, brain: Dict[str, Any]) -> float:
        """Calculate memory bandwidth metric"""
        # This would measure actual memory transfer rates
        return 0.0  # Placeholder
    
    def _cache_hit_rate(self, brain: Dict[str, Any]) -> float:
        """Calculate cache hit rate metric"""
        # This would measure cache performance
        return 0.0  # Placeholder
    
    def _calculate_theoretical_memory(self, n_neurons: int, k_active: int, n_areas: int) -> Dict[str, float]:
        """Calculate theoretical memory requirements (cached)"""
        # Same as current implementation but cached
        sparsity_factor = 0.001
        weight_matrix_size = int(k_active * k_active * sparsity_factor)
        weight_matrix_memory_gb = weight_matrix_size * 4 / 1024**3
        
        indices_memory_gb = weight_matrix_size * 4 / 1024**3
        activation_memory_gb = k_active * 4 / 1024**3
        candidate_memory_gb = k_active * 4 / 1024**3
        area_memory_gb = k_active * 4 * 3 / 1024**3
        
        overhead_factor = 1.1
        total_theoretical_memory_gb = (weight_matrix_memory_gb + indices_memory_gb + 
                                     activation_memory_gb + candidate_memory_gb + 
                                     area_memory_gb) * n_areas * overhead_factor
        
        return {
            'total_theoretical_memory_gb': total_theoretical_memory_gb
        }
    
    async def profile_all_scales_parallel(self, scales: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Profile all scales in parallel for maximum efficiency"""
        tasks = []
        for scale in scales:
            task = self.profile_scale_async(
                scale['name'], 
                scale['neurons'], 
                scale['active_percentage'], 
                scale.get('areas', 5)
            )
            tasks.append(task)
        
        # Run all scales in parallel
        results = await asyncio.gather(*tasks)
        return results
    
    def stream_results(self, results: List[Dict[str, Any]], output_file: str):
        """Stream results to file as they complete"""
        # Ensure the __generated__ folder exists
        import os
        generated_dir = "__generated__"
        os.makedirs(generated_dir, exist_ok=True)
        
        # If output_file doesn't start with __generated__, prepend it
        if not output_file.startswith(generated_dir):
            output_file = os.path.join(generated_dir, output_file)
        
        with open(output_file, 'w') as f:
            f.write('[')
            for i, result in enumerate(results):
                if i > 0:
                    f.write(',')
                json.dump(result, f, indent=2)
                f.flush()  # Ensure data is written immediately
            f.write(']')

# Example usage
async def main():
    config = ProfilerConfig(
        max_workers=4,
        memory_pool_size=2.0,  # 2GB pool
        cache_theoretical=True,
        streaming_output=True
    )
    
    profiler = AsyncProfiler(config)
    
    scales = [
        {'name': 'Million Scale', 'neurons': 1_000_000, 'active_percentage': 0.01},
        {'name': 'Ten Million Scale', 'neurons': 10_000_000, 'active_percentage': 0.01},
        {'name': 'Hundred Million Scale', 'neurons': 100_000_000, 'active_percentage': 0.001},
        {'name': 'Billion Scale', 'neurons': 1_000_000_000, 'active_percentage': 0.0001},
    ]
    
    # Profile all scales in parallel
    results = await profiler.profile_all_scales_parallel(scales)
    
    # Stream results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'efficient_profiler_results_{timestamp}.json'
    profiler.stream_results(results, filename)
    
    print("✅ Efficient profiling completed!")

if __name__ == "__main__":
    asyncio.run(main())
