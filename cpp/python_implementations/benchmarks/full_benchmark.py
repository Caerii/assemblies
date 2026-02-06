#!/usr/bin/env python3
"""
Full Benchmark: NumPy vs PyTorch vs Custom CUDA
===============================================

Compares pattern completion performance across:
1. NumPy (CPU baseline)
2. PyTorch GPU
3. Custom CUDA kernels
"""

import numpy as np
import time
import os
import sys

# Add parent path
sys.path.insert(0, os.path.dirname(__file__))

# Import PyTorch
import torch
TORCH_AVAILABLE = torch.cuda.is_available()

# Import custom CUDA
try:
    from custom_cuda_wrapper import DenseAssemblyCUDA
    CUSTOM_CUDA_AVAILABLE = True
except Exception as e:
    print(f"Custom CUDA not available: {e}")
    CUSTOM_CUDA_AVAILABLE = False


def pattern_completion_numpy(n: int, k: int, beta: float = 0.5, 
                             train_rounds: int = 50, recovery_rounds: int = 10):
    """NumPy CPU baseline"""
    start = time.perf_counter()
    
    W = np.zeros((n, n), dtype=np.float32)
    rng = np.random.default_rng(42)
    stim = rng.choice(n, k, replace=False)
    stim_input = np.zeros(n, dtype=np.float32)
    stim_input[stim] = 1.0
    winners = stim.copy()
    
    train_start = time.perf_counter()
    for _ in range(train_rounds):
        recurrent = W[:, winners].sum(axis=1)
        total = recurrent + stim_input
        new_winners = np.argpartition(total, -k)[-k:]
        np.add.at(W, (new_winners[:, None], new_winners), beta)
        winners = new_winners
    train_time = time.perf_counter() - train_start
    
    original = set(winners)
    partial = np.array(list(original)[:k//2])
    
    comp_start = time.perf_counter()
    for _ in range(recovery_rounds):
        recurrent = W[:, partial].sum(axis=1)
        partial = np.argpartition(recurrent, -k)[-k:]
    comp_time = time.perf_counter() - comp_start
    
    recovered = set(partial)
    recovery = len(recovered & original) / k
    total_time = time.perf_counter() - start
    
    return {
        'method': 'NumPy (CPU)',
        'train_time': train_time,
        'completion_time': comp_time,
        'recovery': recovery,
        'total_time': total_time
    }


def pattern_completion_pytorch(n: int, k: int, beta: float = 0.5,
                               train_rounds: int = 50, recovery_rounds: int = 10):
    """PyTorch GPU"""
    device = torch.device('cuda')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    W = torch.zeros((n, n), dtype=torch.float32, device=device)
    torch.manual_seed(42)
    stim = torch.randperm(n, device=device)[:k]
    stim_input = torch.zeros(n, dtype=torch.float32, device=device)
    stim_input[stim] = 1.0
    winners = stim.clone()
    
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    
    for _ in range(train_rounds):
        selected = torch.index_select(W, 1, winners)
        recurrent = selected.sum(dim=1)
        total = recurrent + stim_input
        _, new_winners = torch.topk(total, k)
        
        update_rows = new_winners.unsqueeze(1).expand(k, k).reshape(-1)
        update_cols = new_winners.unsqueeze(0).expand(k, k).reshape(-1)
        flat_idx = update_rows * n + update_cols
        W.view(-1).scatter_add_(0, flat_idx, torch.full((k*k,), beta, device=device))
        winners = new_winners
    
    torch.cuda.synchronize()
    train_time = time.perf_counter() - train_start
    
    original = set(winners.cpu().numpy())
    partial_list = list(original)[:k//2]
    partial = torch.tensor(partial_list, dtype=torch.long, device=device)
    
    torch.cuda.synchronize()
    comp_start = time.perf_counter()
    
    for _ in range(recovery_rounds):
        recurrent = torch.index_select(W, 1, partial).sum(dim=1)
        _, partial = torch.topk(recurrent, k)
    
    torch.cuda.synchronize()
    comp_time = time.perf_counter() - comp_start
    
    recovered = set(partial.cpu().numpy())
    recovery = len(recovered & original) / k
    total_time = time.perf_counter() - start
    
    del W
    torch.cuda.empty_cache()
    
    return {
        'method': 'PyTorch (GPU)',
        'train_time': train_time,
        'completion_time': comp_time,
        'recovery': recovery,
        'total_time': total_time
    }


def pattern_completion_custom_cuda(n: int, k: int, beta: float = 0.5,
                                   train_rounds: int = 50, recovery_rounds: int = 10):
    """Custom CUDA kernels"""
    cuda = DenseAssemblyCUDA()
    
    start = time.perf_counter()
    
    cuda.allocate(n, k)
    
    rng = np.random.default_rng(42)
    stim = rng.choice(n, k, replace=False).astype(np.uint32)
    stim_input = np.zeros(n, dtype=np.float32)
    stim_input[stim] = 1.0
    winners = stim.copy()
    
    train_start = time.perf_counter()
    
    for _ in range(train_rounds):
        cuda.set_active(winners)
        recurrent = cuda.accumulate()
        total = recurrent + stim_input
        new_winners = np.argpartition(total, -k)[-k:].astype(np.uint32)
        cuda.hebbian_update(new_winners, beta)
        winners = new_winners
    
    train_time = time.perf_counter() - train_start
    
    original = set(winners)
    partial = np.array(list(original)[:k//2], dtype=np.uint32)
    
    comp_start = time.perf_counter()
    
    for _ in range(recovery_rounds):
        cuda.set_active(partial)
        recurrent = cuda.accumulate()
        partial = np.argpartition(recurrent, -k)[-k:].astype(np.uint32)
    
    comp_time = time.perf_counter() - comp_start
    
    recovered = set(partial)
    recovery = len(recovered & original) / k
    total_time = time.perf_counter() - start
    
    cuda.free()
    
    return {
        'method': 'Custom CUDA',
        'train_time': train_time,
        'completion_time': comp_time,
        'recovery': recovery,
        'total_time': total_time
    }


def run_benchmark():
    """Run full benchmark"""
    print("=" * 80)
    print("FULL BENCHMARK: Pattern Completion with k = sqrt(n)")
    print("=" * 80)
    print()
    
    if TORCH_AVAILABLE:
        print(f"PyTorch: {torch.__version__} with CUDA {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Custom CUDA: {'Available' if CUSTOM_CUDA_AVAILABLE else 'Not available'}")
    print()
    
    sizes = [1000, 5000, 10000, 20000, 30000]
    
    print(f"{'n':>8} {'k':>5} {'Method':<20} {'Train':>10} {'Complete':>10} {'Recovery':>8} {'Total':>10} {'Speedup':>8}")
    print("-" * 90)
    
    for n in sizes:
        k = int(np.sqrt(n))
        mem_gb = n * n * 4 / 1024**3
        
        if mem_gb > 14:
            print(f"{n:>8} {k:>5} SKIPPED (needs {mem_gb:.1f}GB)")
            continue
        
        results = []
        
        # NumPy baseline
        try:
            r = pattern_completion_numpy(n, k)
            results.append(r)
            baseline_time = r['total_time']
            status = "OK" if r['recovery'] > 0.9 else "FAIL"
            print(f"{n:>8} {k:>5} {r['method']:<20} {r['train_time']:>9.4f}s {r['completion_time']:>9.5f}s {r['recovery']:>7.0%} {r['total_time']:>9.4f}s {'1.0x':>8} [{status}]")
        except Exception as e:
            print(f"{n:>8} {k:>5} {'NumPy (CPU)':<20} ERROR: {e}")
            baseline_time = 1.0
        
        # PyTorch GPU
        if TORCH_AVAILABLE:
            try:
                r = pattern_completion_pytorch(n, k)
                results.append(r)
                speedup = baseline_time / r['total_time']
                status = "OK" if r['recovery'] > 0.9 else "FAIL"
                print(f"{n:>8} {k:>5} {r['method']:<20} {r['train_time']:>9.4f}s {r['completion_time']:>9.5f}s {r['recovery']:>7.0%} {r['total_time']:>9.4f}s {speedup:>7.1f}x [{status}]")
            except Exception as e:
                print(f"{n:>8} {k:>5} {'PyTorch (GPU)':<20} ERROR: {e}")
        
        # Custom CUDA
        if CUSTOM_CUDA_AVAILABLE:
            try:
                r = pattern_completion_custom_cuda(n, k)
                results.append(r)
                speedup = baseline_time / r['total_time']
                status = "OK" if r['recovery'] > 0.9 else "FAIL"
                print(f"{n:>8} {k:>5} {r['method']:<20} {r['train_time']:>9.4f}s {r['completion_time']:>9.5f}s {r['recovery']:>7.0%} {r['total_time']:>9.4f}s {speedup:>7.1f}x [{status}]")
            except Exception as e:
                print(f"{n:>8} {k:>5} {'Custom CUDA':<20} ERROR: {e}")
        
        print()  # Blank line between sizes
    
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

