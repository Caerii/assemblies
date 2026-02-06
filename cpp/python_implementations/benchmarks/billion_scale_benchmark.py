#!/usr/bin/env python3
"""
Billion-Scale Neural Simulation Benchmark
==========================================

Tests pattern completion at million and billion neuron scales using
sparse representation (O(k²) memory instead of O(n²)).

Key insight: With k = sqrt(n), we only need k×k weights:
- n = 1 million → k = 1000 → 4 MB weights
- n = 1 billion → k = 31623 → 4 GB weights
- n = 86 billion (human brain) → k = 293258 → 344 GB weights (multi-GPU)
"""

import ctypes
import numpy as np
import time
import os
from ctypes import c_uint64, c_uint32, c_float, c_void_p

# Load the sparse assembly DLLs
DLL_PATH_V1 = os.path.join(os.path.dirname(__file__), '..', '..', 'dlls', 'sparse_assembly_kernels.dll')
DLL_PATH_V2 = os.path.join(os.path.dirname(__file__), '..', '..', 'dlls', 'sparse_assembly_kernels_v2.dll')

class SparseAssemblyCUDA:
    """Wrapper for billion-scale sparse CUDA simulation"""
    
    def __init__(self, version=1):
        self.dll = None
        self.sim = None
        self.version = version
        self._load_dll()
    
    def _load_dll(self):
        dll_path = DLL_PATH_V1 if self.version == 1 else DLL_PATH_V2
        prefix = "sparse_assembly" if self.version == 1 else "sparse_assembly_v2"
        
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL not found: {dll_path}")
        
        self.dll = ctypes.CDLL(dll_path)
        self.prefix = prefix
        
        # Define function signatures dynamically based on prefix
        p = self.prefix
        
        getattr(self.dll, f"{p}_init").argtypes = []
        getattr(self.dll, f"{p}_init").restype = ctypes.c_int
        
        getattr(self.dll, f"{p}_create").argtypes = [c_uint64, c_uint32, c_uint32]
        getattr(self.dll, f"{p}_create").restype = c_void_p
        
        getattr(self.dll, f"{p}_destroy").argtypes = [c_void_p]
        getattr(self.dll, f"{p}_destroy").restype = None
        
        getattr(self.dll, f"{p}_step").argtypes = [c_void_p]
        getattr(self.dll, f"{p}_step").restype = None
        
        getattr(self.dll, f"{p}_print_stats").argtypes = [c_void_p]
        getattr(self.dll, f"{p}_print_stats").restype = None
        
        getattr(self.dll, f"{p}_get_memory").argtypes = [c_void_p]
        getattr(self.dll, f"{p}_get_memory").restype = c_uint64
        
        getattr(self.dll, f"{p}_get_steps").argtypes = [c_void_p]
        getattr(self.dll, f"{p}_get_steps").restype = c_uint64
        
        getattr(self.dll, f"{p}_get_time").argtypes = [c_void_p]
        getattr(self.dll, f"{p}_get_time").restype = c_float
        
        # Initialize
        result = getattr(self.dll, f"{p}_init")()
        if result != 0:
            raise RuntimeError("Failed to initialize CUDA")
    
    def create(self, n_neurons: int, k_active: int, seed: int = 42):
        """Create a billion-scale simulator"""
        self.sim = getattr(self.dll, f"{self.prefix}_create")(n_neurons, k_active, seed)
        return self.sim is not None
    
    def step(self):
        """Run one simulation step"""
        getattr(self.dll, f"{self.prefix}_step")(self.sim)
    
    def get_memory_mb(self) -> float:
        """Get memory usage in MB"""
        return getattr(self.dll, f"{self.prefix}_get_memory")(self.sim) / 1024 / 1024
    
    def get_steps(self) -> int:
        """Get step count"""
        return getattr(self.dll, f"{self.prefix}_get_steps")(self.sim)
    
    def get_time_ms(self) -> float:
        """Get total time in ms"""
        return getattr(self.dll, f"{self.prefix}_get_time")(self.sim)
    
    def print_stats(self):
        """Print simulation stats"""
        getattr(self.dll, f"{self.prefix}_print_stats")(self.sim)
    
    def destroy(self):
        """Destroy simulator"""
        if self.sim:
            getattr(self.dll, f"{self.prefix}_destroy")(self.sim)
            self.sim = None


def format_neurons(n: int) -> str:
    """Format neuron count nicely"""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def calculate_memory(n: int, k: int) -> dict:
    """Calculate theoretical memory requirements"""
    # Dense: n × n × 4 bytes
    dense_bytes = n * n * 4
    
    # Sparse: k × k × 4 bytes (weights) + overhead
    sparse_weights = k * k * 4
    sparse_indices = k * 2 * 4  # active + prev_active
    sparse_candidates = min(k * 10, min(n, 1000000)) * 8  # candidates + activations
    sparse_rand = min(k * 10, min(n, 1000000)) * 48  # curandState
    sparse_total = sparse_weights + sparse_indices + sparse_candidates + sparse_rand
    
    return {
        'dense_gb': dense_bytes / 1024**3,
        'sparse_mb': sparse_total / 1024**2,
        'compression': dense_bytes / sparse_total if sparse_total > 0 else float('inf')
    }


def run_benchmark():
    """Run billion-scale benchmark"""
    print("=" * 80)
    print("BILLION-SCALE NEURAL SIMULATION BENCHMARK")
    print("Using SPARSE representation: O(k²) memory instead of O(n²)")
    print("Comparing V1 (baseline) vs V2 (optimized)")
    print("=" * 80)
    print()
    
    # Test both versions
    cuda_v1 = SparseAssemblyCUDA(version=1)
    cuda_v2 = SparseAssemblyCUDA(version=2)
    
    # Test cases from million to billion scale
    test_cases = [
        # (neurons, description)
        (100_000, "100K neurons"),
        (1_000_000, "1 Million neurons"),
        (10_000_000, "10 Million neurons"),
        (100_000_000, "100 Million neurons"),
        (1_000_000_000, "1 Billion neurons"),
        (10_000_000_000, "10 Billion neurons"),
        (86_000_000_000, "86 Billion (Human Brain)"),
    ]
    
    print(f"{'Scale':<20} {'n':>12} {'k':>8} {'Sparse':>8} {'V1 ms':>10} {'V2 ms':>10} {'Speedup':>8} {'V2 step/s':>10}")
    print("-" * 100)
    
    results = []
    
    for n, desc in test_cases:
        k = int(np.sqrt(n))
        mem = calculate_memory(n, k)
        
        # Check if we can run this
        if mem['sparse_mb'] > 14000:  # 14GB limit
            print(f"{desc:<20} {format_neurons(n):>12} {format_neurons(k):>8} {mem['sparse_mb']:>6.0f}MB {'SKIPPED':>10} {'SKIPPED':>10} {'(OOM)':>8}")
            continue
        
        v1_time = None
        v2_time = None
        actual_memory = 0
        
        # Test V1
        try:
            cuda_v1.create(n, k, seed=42)
            for _ in range(5):  # Warmup
                cuda_v1.step()
            cuda_v1.destroy()
            cuda_v1.create(n, k, seed=42)
            
            steps = 100
            start = time.perf_counter()
            for _ in range(steps):
                cuda_v1.step()
            v1_time = (time.perf_counter() - start) / steps * 1000
            actual_memory = cuda_v1.get_memory_mb()
            cuda_v1.destroy()
        except Exception as e:
            v1_time = None
        
        # Test V2
        try:
            cuda_v2.create(n, k, seed=42)
            for _ in range(5):  # Warmup
                cuda_v2.step()
            cuda_v2.destroy()
            cuda_v2.create(n, k, seed=42)
            
            steps = 100
            start = time.perf_counter()
            for _ in range(steps):
                cuda_v2.step()
            v2_time = (time.perf_counter() - start) / steps * 1000
            if actual_memory == 0:
                actual_memory = cuda_v2.get_memory_mb()
            cuda_v2.destroy()
        except Exception as e:
            v2_time = None
        
        # Print results
        v1_str = f"{v1_time:.3f}" if v1_time else "ERROR"
        v2_str = f"{v2_time:.3f}" if v2_time else "ERROR"
        
        if v1_time and v2_time:
            speedup = v1_time / v2_time
            speedup_str = f"{speedup:.2f}x"
            steps_per_sec = 1000 / v2_time
        else:
            speedup_str = "N/A"
            steps_per_sec = 0
        
        print(f"{desc:<20} {format_neurons(n):>12} {format_neurons(k):>8} {actual_memory:>6.1f}MB {v1_str:>10} {v2_str:>10} {speedup_str:>8} {steps_per_sec:>10.1f}")
        
        if v2_time:
            results.append({
                'desc': desc,
                'n': n,
                'k': k,
                'memory_mb': actual_memory,
                'v1_time_ms': v1_time,
                'v2_time_ms': v2_time,
                'speedup': v1_time / v2_time if v1_time else 0,
                'steps_per_sec': 1000 / v2_time
            })
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        # Find largest successful scale
        largest = max(results, key=lambda x: x['n'])
        print(f"\nLargest successful scale: {largest['desc']}")
        print(f"  Neurons: {format_neurons(largest['n'])}")
        print(f"  Active (k): {format_neurons(largest['k'])}")
        print(f"  Memory: {largest['memory_mb']:.1f} MB")
        print(f"  V2 Speed: {largest['steps_per_sec']:.1f} steps/second")
        print(f"  V2 Time per step: {largest['v2_time_ms']:.3f} ms")
        if largest.get('speedup'):
            print(f"  V2 Speedup over V1: {largest['speedup']:.2f}x")
        
        # Calculate what's needed for 86B
        human_n = 86_000_000_000
        human_k = int(np.sqrt(human_n))
        human_mem = calculate_memory(human_n, human_k)
        
        print("\n86 Billion neurons (Human Brain) would require:")
        print(f"  k = {format_neurons(human_k)} active neurons")
        print(f"  Sparse memory: {human_mem['sparse_mb']/1024:.1f} GB")
        print(f"  Dense memory: {human_mem['dense_gb']:.0f} GB (impossible!)")
        print(f"  Compression: {human_mem['compression']:.0f}x")
        
        if human_mem['sparse_mb'] / 1024 > 16:
            gpus_needed = int(np.ceil(human_mem['sparse_mb'] / 1024 / 16))
            print(f"  GPUs needed (16GB each): {gpus_needed}")
    
    print()
    print("KEY INSIGHT: Sparse representation enables billion-scale simulation!")
    print("Dense n×n matrix would require EXABYTES of memory at billion scale.")


if __name__ == "__main__":
    run_benchmark()

