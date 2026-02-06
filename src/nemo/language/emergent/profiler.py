"""
NEMO Profiler
=============

Version: 1.0.0
Date: 2025-11-30

Comprehensive profiling infrastructure for the NEMO emergent language system.
Identifies bottlenecks and provides optimization recommendations.

Usage:
    from nemo.language.emergent.profiler import NEMOProfiler, profile_training
    
    # Profile a training run
    results = profile_training(num_sentences=100, epochs=1)
    results.print_summary()
    
    # Or use context manager for custom profiling
    with NEMOProfiler() as profiler:
        # ... your code ...
    profiler.print_summary()
"""

import time
import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager
import functools

__all__ = [
    'NEMOProfiler', 
    'ProfileResults', 
    'profile_function',
    'profile_training',
    'profile_parsing',
    'run_full_benchmark'
]


@dataclass
class TimingRecord:
    """Single timing measurement."""
    name: str
    duration_ms: float
    count: int = 1
    memory_mb: float = 0.0
    
    @property
    def avg_ms(self) -> float:
        return self.duration_ms / self.count if self.count > 0 else 0.0


@dataclass
class ProfileResults:
    """Aggregated profiling results."""
    total_time_ms: float = 0.0
    timings: Dict[str, TimingRecord] = field(default_factory=dict)
    call_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    memory_peak_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    
    def add_timing(self, name: str, duration_ms: float, memory_mb: float = 0.0):
        """Add a timing measurement."""
        if name in self.timings:
            self.timings[name].duration_ms += duration_ms
            self.timings[name].count += 1
            self.timings[name].memory_mb = max(self.timings[name].memory_mb, memory_mb)
        else:
            self.timings[name] = TimingRecord(name, duration_ms, 1, memory_mb)
        self.call_counts[name] += 1
    
    def get_sorted_timings(self) -> List[TimingRecord]:
        """Get timings sorted by total time descending."""
        return sorted(self.timings.values(), key=lambda x: x.duration_ms, reverse=True)
    
    def get_hotspots(self, top_n: int = 10) -> List[TimingRecord]:
        """Get the top N time-consuming operations."""
        return self.get_sorted_timings()[:top_n]
    
    def print_summary(self):
        """Print a formatted summary of profiling results."""
        print("\n" + "="*70)
        print("PROFILING RESULTS")
        print("="*70)
        
        print(f"\nTotal Time: {self.total_time_ms:.2f} ms ({self.total_time_ms/1000:.2f} s)")
        print(f"GPU Memory Peak: {self.gpu_memory_peak_mb:.2f} MB")
        
        print("\n" + "-"*70)
        print("HOTSPOTS (Top Time Consumers)")
        print("-"*70)
        print(f"{'Operation':<35} {'Total (ms)':>10} {'Calls':>8} {'Avg (ms)':>10} {'%':>6}")
        print("-"*70)
        
        for timing in self.get_hotspots(15):
            pct = (timing.duration_ms / self.total_time_ms * 100) if self.total_time_ms > 0 else 0
            print(f"{timing.name:<35} {timing.duration_ms:>10.2f} {timing.count:>8} {timing.avg_ms:>10.3f} {pct:>5.1f}%")
        
        # Categorized summary
        print("\n" + "-"*70)
        print("BY CATEGORY")
        print("-"*70)
        
        categories = defaultdict(float)
        for timing in self.timings.values():
            if 'projection' in timing.name.lower():
                categories['Projection'] += timing.duration_ms
            elif 'hebbian' in timing.name.lower():
                categories['Hebbian Learning'] += timing.duration_ms
            elif 'topk' in timing.name.lower() or 'torch' in timing.name.lower():
                categories['TopK Selection'] += timing.duration_ms
            elif 'assembly' in timing.name.lower():
                categories['Assembly Ops'] += timing.duration_ms
            elif 'parse' in timing.name.lower():
                categories['Parsing'] += timing.duration_ms
            else:
                categories['Other'] += timing.duration_ms
        
        for cat, time_ms in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            pct = (time_ms / self.total_time_ms * 100) if self.total_time_ms > 0 else 0
            print(f"  {cat:<30} {time_ms:>10.2f} ms ({pct:>5.1f}%)")
        
        # Recommendations
        print("\n" + "-"*70)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("-"*70)
        
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("="*70)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        # Check projection time
        proj_time = sum(t.duration_ms for t in self.timings.values() 
                       if 'projection' in t.name.lower())
        if proj_time > self.total_time_ms * 0.3:
            recommendations.append(
                f"Projection takes {proj_time/self.total_time_ms*100:.1f}% of time. "
                "Consider batching multiple projections together."
            )
        
        # Check hebbian time
        hebb_time = sum(t.duration_ms for t in self.timings.values() 
                       if 'hebbian' in t.name.lower())
        if hebb_time > self.total_time_ms * 0.2:
            recommendations.append(
                f"Hebbian learning takes {hebb_time/self.total_time_ms*100:.1f}% of time. "
                "Consider reducing learning frequency or using sparse updates."
            )
        
        # Check topk time
        topk_time = sum(t.duration_ms for t in self.timings.values() 
                       if 'topk' in t.name.lower() or 'torch' in t.name.lower())
        if topk_time > self.total_time_ms * 0.15:
            recommendations.append(
                f"TopK selection takes {topk_time/self.total_time_ms*100:.1f}% of time. "
                "Consider using CuPy's argpartition instead of torch.topk."
            )
        
        # Check assembly storage
        assembly_time = sum(t.duration_ms for t in self.timings.values() 
                          if 'store' in t.name.lower() or 'assembly' in t.name.lower())
        if assembly_time > self.total_time_ms * 0.1:
            recommendations.append(
                f"Assembly storage takes {assembly_time/self.total_time_ms*100:.1f}% of time. "
                "Consider caching assemblies or reducing storage frequency."
            )
        
        # Check call counts
        high_call_ops = [(name, count) for name, count in self.call_counts.items() 
                        if count > 1000]
        if high_call_ops:
            ops_str = ", ".join(f"{n} ({c}x)" for n, c in high_call_ops[:3])
            recommendations.append(
                f"High call count operations: {ops_str}. "
                "Consider batching or caching."
            )
        
        if not recommendations:
            recommendations.append("No major bottlenecks detected. System is well-optimized.")
        
        return recommendations


class NEMOProfiler:
    """
    Context manager for profiling NEMO operations.
    
    Usage:
        with NEMOProfiler() as profiler:
            # ... operations ...
        profiler.results.print_summary()
    """
    
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.results = ProfileResults()
        self._start_time = None
        self._timing_stack = []
    
    def __enter__(self):
        if self.sync_cuda:
            cp.cuda.Stream.null.synchronize()
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda:
            cp.cuda.Stream.null.synchronize()
        self.results.total_time_ms = (time.perf_counter() - self._start_time) * 1000
        
        # Get GPU memory
        try:
            mempool = cp.get_default_memory_pool()
            self.results.gpu_memory_peak_mb = mempool.total_bytes() / 1024 / 1024
        except:
            pass
    
    @contextmanager
    def measure(self, name: str):
        """Measure time for a specific operation."""
        if self.sync_cuda:
            cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        
        try:
            yield
        finally:
            if self.sync_cuda:
                cp.cuda.Stream.null.synchronize()
            duration_ms = (time.perf_counter() - start) * 1000
            self.results.add_timing(name, duration_ms)
    
    def record(self, name: str, duration_ms: float):
        """Manually record a timing."""
        self.results.add_timing(name, duration_ms)


def profile_function(name: str = None):
    """
    Decorator to profile a function.
    
    Usage:
        @profile_function("my_operation")
        def my_function():
            ...
    """
    def decorator(func):
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Look for profiler in kwargs or global
            profiler = kwargs.pop('_profiler', None)
            
            if profiler is None:
                return func(*args, **kwargs)
            
            with profiler.measure(func_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# PROFILED BRAIN OPERATIONS
# =============================================================================

def create_profiled_brain(brain, profiler: NEMOProfiler):
    """
    Wrap brain operations with profiling.
    
    Returns a wrapper that profiles all brain operations.
    """
    original_project = brain._project
    original_store = brain.store_learned_assembly
    original_get = brain.get_learned_assembly
    original_overlap = brain.get_assembly_overlap
    
    def profiled_project(area, inp, learn=True):
        with profiler.measure(f"projection.{area.name}"):
            return original_project(area, inp, learn)
    
    def profiled_store(area, word, assembly):
        with profiler.measure(f"store_assembly.{area.name}"):
            return original_store(area, word, assembly)
    
    def profiled_get(area, word):
        with profiler.measure(f"get_assembly.{area.name}"):
            return original_get(area, word)
    
    def profiled_overlap(a1, a2):
        with profiler.measure("assembly_overlap"):
            return original_overlap(a1, a2)
    
    brain._project = profiled_project
    brain.store_learned_assembly = profiled_store
    brain.get_learned_assembly = profiled_get
    brain.get_assembly_overlap = profiled_overlap
    
    return brain


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def profile_training(num_sentences: int = 100, epochs: int = 1, 
                    verbose: bool = False) -> ProfileResults:
    """
    Profile the training pipeline.
    
    Returns ProfileResults with detailed timing breakdown.
    """
    from .params import EmergentParams
    from .learner import EmergentLanguageLearner
    from .training_data import create_training_data
    
    print(f"\nProfiling training: {num_sentences} sentences × {epochs} epochs")
    print("-" * 50)
    
    with NEMOProfiler() as profiler:
        # Initialization
        with profiler.measure("init.params"):
            params = EmergentParams()
        
        with profiler.measure("init.learner"):
            learner = EmergentLanguageLearner(params, verbose=False)
        
        with profiler.measure("init.training_data"):
            training_data = create_training_data()[:num_sentences]
        
        # Training
        for epoch in range(epochs):
            with profiler.measure(f"epoch.{epoch}"):
                for sentence in training_data:
                    with profiler.measure("train.sentence"):
                        with profiler.measure("train.present_grounded"):
                            learner.present_grounded_sentence(
                                sentence.words,
                                sentence.contexts,
                                sentence.roles,
                                sentence.mood,
                                learn=True
                            )
    
    return profiler.results


def profile_parsing(learner, sentences: List[List[str]]) -> ProfileResults:
    """
    Profile sentence parsing.
    """
    from .parser import SentenceParser
    
    print(f"\nProfiling parsing: {len(sentences)} sentences")
    print("-" * 50)
    
    with NEMOProfiler() as profiler:
        parser = SentenceParser(learner)
        
        for words in sentences:
            with profiler.measure("parse.sentence"):
                parser.parse(words)
    
    return profiler.results


def profile_projection_kernel(n: int = 10000, k: int = 100, 
                              iterations: int = 100) -> ProfileResults:
    """
    Profile the core projection kernel in isolation.
    """
    from .brain import EmergentNemoBrain
    from .params import EmergentParams
    from .areas import Area
    
    print(f"\nProfiling projection kernel: n={n}, k={k}, {iterations} iterations")
    print("-" * 50)
    
    params = EmergentParams(n=n, k=k)
    brain = EmergentNemoBrain(params, verbose=False)
    
    # Create test input
    test_input = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    with NEMOProfiler() as profiler:
        for i in range(iterations):
            with profiler.measure("projection_only"):
                brain._project(Area.NOUN_CORE, test_input, learn=False)
            
            brain._clear_area(Area.NOUN_CORE)
    
    return profiler.results


def profile_hebbian_kernel(n: int = 10000, k: int = 100,
                           iterations: int = 100) -> ProfileResults:
    """
    Profile the Hebbian learning kernel in isolation.
    """
    from .brain import EmergentNemoBrain
    from .params import EmergentParams
    from .areas import Area
    
    print(f"\nProfiling Hebbian kernel: n={n}, k={k}, {iterations} iterations")
    print("-" * 50)
    
    params = EmergentParams(n=n, k=k)
    brain = EmergentNemoBrain(params, verbose=False)
    
    # Create test inputs
    test_input1 = cp.random.randint(0, n, k, dtype=cp.uint32)
    test_input2 = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    with NEMOProfiler() as profiler:
        for i in range(iterations):
            # First projection sets prev
            brain._project(Area.NOUN_CORE, test_input1, learn=False)
            
            # Second projection triggers Hebbian
            with profiler.measure("hebbian_only"):
                brain._project(Area.NOUN_CORE, test_input2, learn=True)
            
            brain._clear_area(Area.NOUN_CORE)
    
    return profiler.results


def run_full_benchmark(verbose: bool = True) -> Dict[str, ProfileResults]:
    """
    Run comprehensive benchmarks on all components.
    
    Returns dict of component name -> ProfileResults
    """
    print("="*70)
    print("NEMO FULL BENCHMARK SUITE")
    print("="*70)
    
    results = {}
    
    # 1. Projection kernel
    results['projection'] = profile_projection_kernel(
        n=10000, k=100, iterations=100
    )
    
    # 2. Hebbian kernel
    results['hebbian'] = profile_hebbian_kernel(
        n=10000, k=100, iterations=100
    )
    
    # 3. Training pipeline
    results['training'] = profile_training(
        num_sentences=50, epochs=1, verbose=False
    )
    
    # Print combined summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Total: {result.total_time_ms:.2f} ms")
        hotspots = result.get_hotspots(3)
        for h in hotspots:
            print(f"  - {h.name}: {h.duration_ms:.2f} ms ({h.count} calls)")
    
    return results


# =============================================================================
# MICRO-BENCHMARKS
# =============================================================================

def benchmark_topk_methods(n: int = 10000, k: int = 100, 
                          iterations: int = 100) -> Dict[str, float]:
    """
    Compare different topk implementations.
    """
    import torch
    
    print(f"\nBenchmarking topk methods: n={n}, k={k}")
    print("-" * 50)
    
    results = {}
    
    # Create test data
    data_cp = cp.random.randn(n).astype(cp.float32)
    data_torch = torch.as_tensor(data_cp, device='cuda')
    
    # Method 1: torch.topk
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _, idx = torch.topk(data_torch, k, sorted=False)
    cp.cuda.Stream.null.synchronize()
    results['torch.topk'] = (time.perf_counter() - start) * 1000 / iterations
    
    # Method 2: cupy argpartition
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        idx = cp.argpartition(data_cp, -k)[-k:]
    cp.cuda.Stream.null.synchronize()
    results['cupy.argpartition'] = (time.perf_counter() - start) * 1000 / iterations
    
    # Method 3: cupy argsort (baseline)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        idx = cp.argsort(data_cp)[-k:]
    cp.cuda.Stream.null.synchronize()
    results['cupy.argsort'] = (time.perf_counter() - start) * 1000 / iterations
    
    print("Results (ms per call):")
    for method, time_ms in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {method:<25} {time_ms:.4f} ms")
    
    return results


def benchmark_assembly_storage(k: int = 100, 
                               iterations: int = 1000) -> Dict[str, float]:
    """
    Benchmark different assembly storage methods.
    """
    print(f"\nBenchmarking assembly storage: k={k}")
    print("-" * 50)
    
    results = {}
    
    # Create test assemblies
    assemblies = [cp.random.randint(0, 10000, k, dtype=cp.uint32) 
                  for _ in range(100)]
    
    # Method 1: Dict with copy
    storage1 = {}
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for i in range(iterations):
        storage1[f"word_{i}"] = assemblies[i % 100].copy()
    cp.cuda.Stream.null.synchronize()
    results['dict_copy'] = (time.perf_counter() - start) * 1000 / iterations
    
    # Method 2: Dict without copy (reference)
    storage2 = {}
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for i in range(iterations):
        storage2[f"word_{i}"] = assemblies[i % 100]
    cp.cuda.Stream.null.synchronize()
    results['dict_ref'] = (time.perf_counter() - start) * 1000 / iterations
    
    # Method 3: Pre-allocated array
    storage3 = cp.zeros((iterations, k), dtype=cp.uint32)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for i in range(iterations):
        storage3[i] = assemblies[i % 100]
    cp.cuda.Stream.null.synchronize()
    results['preallocated'] = (time.perf_counter() - start) * 1000 / iterations
    
    print("Results (ms per call):")
    for method, time_ms in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {method:<25} {time_ms:.6f} ms")
    
    return results


if __name__ == "__main__":
    # Run benchmarks when executed directly
    print("Running NEMO Profiler Benchmarks...")
    
    # Micro-benchmarks
    benchmark_topk_methods()
    benchmark_assembly_storage()
    
    # Full benchmark
    results = run_full_benchmark()
    
    # Print detailed results for training
    print("\n" + "="*70)
    print("DETAILED TRAINING PROFILE")
    print("="*70)
    results['training'].print_summary()

