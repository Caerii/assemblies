#!/usr/bin/env python3
"""
Dense Pattern Completion Benchmark
===================================

Compares performance of:
1. NumPy (CPU baseline)
2. PyTorch GPU
3. Custom CUDA kernels (if available)

For pattern completion with k = sqrt(n)
"""

import numpy as np
import time
import os
from typing import Dict, Any
from dataclasses import dataclass

# Try importing GPU libraries
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available")

# Try loading custom CUDA kernels
CUSTOM_CUDA_AVAILABLE = False
try:
    import ctypes
    dll_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dlls', 'dense_assembly_kernels.dll')
    if os.path.exists(dll_path):
        cuda_dll = ctypes.CDLL(dll_path)
        CUSTOM_CUDA_AVAILABLE = True
        print(f"✅ Custom CUDA kernels loaded from {dll_path}")
except Exception as e:
    print(f"❌ Custom CUDA kernels not available: {e}")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    method: str
    n: int
    k: int
    train_time: float
    completion_time: float
    recovery_rate: float
    total_time: float


def pattern_completion_numpy(n: int, k: int, beta: float = 0.5, 
                             train_rounds: int = 50, recovery_rounds: int = 10) -> BenchmarkResult:
    """Pattern completion using NumPy (CPU baseline)"""
    start = time.perf_counter()
    
    # Initialize
    W = np.zeros((n, n), dtype=np.float32)
    rng = np.random.default_rng(42)
    stim = rng.choice(n, k, replace=False)
    stim_input = np.zeros(n, dtype=np.float32)
    stim_input[stim] = 1.0
    winners = stim.copy()
    
    # Training
    train_start = time.perf_counter()
    for _ in range(train_rounds):
        recurrent = W[:, winners].sum(axis=1)
        total = recurrent + stim_input
        new_winners = np.argpartition(total, -k)[-k:]
        np.add.at(W, (new_winners[:, None], new_winners), beta)
        winners = new_winners
    train_time = time.perf_counter() - train_start
    
    original = set(winners)
    
    # Pattern completion
    comp_start = time.perf_counter()
    partial = np.array(list(original)[:k//2])
    for _ in range(recovery_rounds):
        recurrent = W[:, partial].sum(axis=1)
        partial = np.argpartition(recurrent, -k)[-k:]
    comp_time = time.perf_counter() - comp_start
    
    recovered = set(partial)
    recovery = len(recovered & original) / k
    
    total_time = time.perf_counter() - start
    
    return BenchmarkResult(
        method="NumPy (CPU)",
        n=n, k=k,
        train_time=train_time,
        completion_time=comp_time,
        recovery_rate=recovery,
        total_time=total_time
    )


def pattern_completion_pytorch(n: int, k: int, beta: float = 0.5,
                               train_rounds: int = 50, recovery_rounds: int = 10) -> BenchmarkResult:
    """Pattern completion using PyTorch GPU"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch CUDA not available")
    
    device = torch.device('cuda')
    
    # Warmup
    _ = torch.zeros(100, 100, device=device)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    # Initialize on GPU
    W = torch.zeros((n, n), dtype=torch.float32, device=device)
    
    torch.manual_seed(42)
    stim = torch.randperm(n, device=device)[:k]
    stim_input = torch.zeros(n, dtype=torch.float32, device=device)
    stim_input[stim] = 1.0
    winners = stim.clone()
    
    # Training
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    
    for _ in range(train_rounds):
        # Matrix-vector multiply (uses cuBLAS)
        recurrent = W[:, winners].sum(dim=1)
        total = recurrent + stim_input
        
        # Top-k selection
        _, new_winners = torch.topk(total, k)
        
        # Hebbian update using index_add (outer product)
        # W[new_winners][:, new_winners] += beta
        idx = new_winners.unsqueeze(0).expand(k, k)
        W.index_put_((new_winners.unsqueeze(1).expand(k, k), idx), 
                     torch.full((k, k), beta, device=device), accumulate=True)
        
        winners = new_winners
    
    torch.cuda.synchronize()
    train_time = time.perf_counter() - train_start
    
    original = set(winners.cpu().numpy())
    
    # Pattern completion
    torch.cuda.synchronize()
    comp_start = time.perf_counter()
    
    partial_list = list(original)[:k//2]
    partial = torch.tensor(partial_list, dtype=torch.long, device=device)
    
    for _ in range(recovery_rounds):
        recurrent = W[:, partial].sum(dim=1)
        _, partial = torch.topk(recurrent, k)
    
    torch.cuda.synchronize()
    comp_time = time.perf_counter() - comp_start
    
    recovered = set(partial.cpu().numpy())
    recovery = len(recovered & original) / k
    
    total_time = time.perf_counter() - start
    
    # Clean up GPU memory
    del W, stim, stim_input, winners, partial
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        method="PyTorch (GPU)",
        n=n, k=k,
        train_time=train_time,
        completion_time=comp_time,
        recovery_rate=recovery,
        total_time=total_time
    )


def pattern_completion_pytorch_optimized(n: int, k: int, beta: float = 0.5,
                                         train_rounds: int = 50, recovery_rounds: int = 10) -> BenchmarkResult:
    """Optimized PyTorch GPU with better memory access patterns"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch CUDA not available")
    
    device = torch.device('cuda')
    
    # Warmup
    _ = torch.zeros(100, 100, device=device)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    # Initialize on GPU
    W = torch.zeros((n, n), dtype=torch.float32, device=device)
    
    torch.manual_seed(42)
    stim = torch.randperm(n, device=device)[:k]
    stim_input = torch.zeros(n, dtype=torch.float32, device=device)
    stim_input[stim] = 1.0
    winners = stim.clone()
    
    # Pre-allocate tensors for efficiency
    recurrent = torch.zeros(n, dtype=torch.float32, device=device)
    total = torch.zeros(n, dtype=torch.float32, device=device)
    
    # Training
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    
    for _ in range(train_rounds):
        # Use index_select + sum (more efficient for sparse access)
        selected = torch.index_select(W, 1, winners)
        torch.sum(selected, dim=1, out=recurrent)
        torch.add(recurrent, stim_input, out=total)
        
        # Top-k
        _, new_winners = torch.topk(total, k)
        
        # Efficient Hebbian update using scatter_add
        # Create k x k update matrix and scatter
        update_rows = new_winners.unsqueeze(1).expand(k, k).reshape(-1)
        update_cols = new_winners.unsqueeze(0).expand(k, k).reshape(-1)
        flat_idx = update_rows * n + update_cols
        W.view(-1).scatter_add_(0, flat_idx, torch.full((k*k,), beta, device=device))
        
        winners = new_winners
    
    torch.cuda.synchronize()
    train_time = time.perf_counter() - train_start
    
    original = set(winners.cpu().numpy())
    
    # Pattern completion
    torch.cuda.synchronize()
    comp_start = time.perf_counter()
    
    partial_list = list(original)[:k//2]
    partial = torch.tensor(partial_list, dtype=torch.long, device=device)
    
    for _ in range(recovery_rounds):
        selected = torch.index_select(W, 1, partial)
        torch.sum(selected, dim=1, out=recurrent)
        _, partial = torch.topk(recurrent, k)
    
    torch.cuda.synchronize()
    comp_time = time.perf_counter() - comp_start
    
    recovered = set(partial.cpu().numpy())
    recovery = len(recovered & original) / k
    
    total_time = time.perf_counter() - start
    
    # Clean up
    del W, stim, stim_input, winners, partial, recurrent, total, selected
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        method="PyTorch Optimized (GPU)",
        n=n, k=k,
        train_time=train_time,
        completion_time=comp_time,
        recovery_rate=recovery,
        total_time=total_time
    )


def run_benchmark(sizes: list = None) -> Dict[str, Any]:
    """Run full benchmark suite"""
    if sizes is None:
        sizes = [1000, 5000, 10000, 20000, 30000]
    
    results = []
    
    print("\n" + "="*80)
    print("DENSE PATTERN COMPLETION BENCHMARK")
    print("="*80)
    print(f"\n{'n':>8} {'k':>5} {'Method':<25} {'Train':>8} {'Complete':>8} {'Recovery':>8} {'Total':>8}")
    print("-"*80)
    
    for n in sizes:
        k = int(np.sqrt(n))
        
        # Check memory requirements
        mem_gb = n * n * 4 / 1024**3
        if mem_gb > 15:  # 16GB GPU limit with some headroom
            print(f"{n:>8} {k:>5} {'SKIPPED - too large for GPU memory':>60}")
            continue
        
        # NumPy baseline
        try:
            result = pattern_completion_numpy(n, k)
            results.append(result)
            status = "OK" if result.recovery_rate > 0.9 else "FAIL"
            print(f"{n:>8} {k:>5} {result.method:<25} {result.train_time:>7.3f}s {result.completion_time:>7.4f}s {result.recovery_rate:>7.0%} {result.total_time:>7.3f}s [{status}]")
        except Exception as e:
            print(f"{n:>8} {k:>5} {'NumPy (CPU)':<25} ERROR: {e}")
        
        # PyTorch GPU
        if TORCH_AVAILABLE:
            try:
                result = pattern_completion_pytorch(n, k)
                results.append(result)
                status = "OK" if result.recovery_rate > 0.9 else "FAIL"
                print(f"{n:>8} {k:>5} {result.method:<25} {result.train_time:>7.3f}s {result.completion_time:>7.4f}s {result.recovery_rate:>7.0%} {result.total_time:>7.3f}s [{status}]")
            except Exception as e:
                print(f"{n:>8} {k:>5} {'PyTorch (GPU)':<25} ERROR: {e}")
            
            try:
                result = pattern_completion_pytorch_optimized(n, k)
                results.append(result)
                status = "OK" if result.recovery_rate > 0.9 else "FAIL"
                print(f"{n:>8} {k:>5} {result.method:<25} {result.train_time:>7.3f}s {result.completion_time:>7.4f}s {result.recovery_rate:>7.0%} {result.total_time:>7.3f}s [{status}]")
            except Exception as e:
                print(f"{n:>8} {k:>5} {'PyTorch Optimized (GPU)':<25} ERROR: {e}")
        
        print()  # Blank line between sizes
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Group by size and find speedups
    for n in sizes:
        k = int(np.sqrt(n))
        size_results = [r for r in results if r.n == n]
        
        if len(size_results) >= 2:
            numpy_result = next((r for r in size_results if "NumPy" in r.method), None)
            pytorch_result = next((r for r in size_results if "PyTorch Optimized" in r.method), None)
            
            if numpy_result and pytorch_result:
                speedup = numpy_result.total_time / pytorch_result.total_time
                print(f"n={n:,}, k={k}: PyTorch GPU is {speedup:.1f}x faster than NumPy CPU")
    
    return {"results": results}


if __name__ == "__main__":
    # Run with default sizes
    run_benchmark([1000, 5000, 10000, 20000, 30000, 50000])

