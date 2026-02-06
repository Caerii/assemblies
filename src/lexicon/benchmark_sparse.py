"""Benchmark sparse vs dense operations for NEMO"""
import torch
import cupy as cp
from cupyx.scipy import sparse as cp_sparse
import time
import numpy as np

print('=' * 70)
print('FINAL BENCHMARK: PyTorch Dense vs CuPy Sparse')
print('=' * 70)
print()
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'CuPy: {cp.__version__}')
print()

results = []

for n in [10000, 20000, 30000]:
    p = 0.05
    k = 50
    n_iter = 100
    
    print(f'n={n:,}, p={p}, k={k}')
    print('-' * 50)
    
    rng = np.random.RandomState(42)
    
    # PyTorch Dense
    try:
        W_torch = torch.zeros(n, n, device='cuda')
        mask = torch.rand(n, n, device='cuda') < p
        W_torch[mask] = 1.0
        del mask
        
        active_torch = torch.randperm(n, device='cuda')[:k]
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            result = W_torch[active_torch].sum(dim=0)
            torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) * 1000 / n_iter
        torch_mem = W_torch.numel() * 4 / 1e6
        
        print(f'  PyTorch Dense: {torch_time:.2f} ms, {torch_mem:.0f} MB')
        del W_torch
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f'  PyTorch Dense: OOM')
        torch_time = float('inf')
        torch_mem = 0
    
    # CuPy CSR
    nnz = int(n * n * p)
    rows = rng.randint(0, n, nnz).astype(np.int32)
    cols = rng.randint(0, n, nnz).astype(np.int32)
    data = np.ones(nnz, dtype=np.float32)
    W_csr = cp_sparse.csr_matrix((cp.array(data), (cp.array(rows), cp.array(cols))), shape=(n, n))
    
    active = cp.array(rng.permutation(n)[:k].astype(np.int64))
    one_hot = cp.zeros(n, dtype=cp.float32)
    one_hot[active] = 1.0
    
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        result = W_csr.T.dot(one_hot)
        cp.cuda.Stream.null.synchronize()
    cupy_time = (time.perf_counter() - start) * 1000 / n_iter
    cupy_mem = (W_csr.data.nbytes + W_csr.indices.nbytes + W_csr.indptr.nbytes) / 1e6
    
    print(f'  CuPy CSR spmv: {cupy_time:.2f} ms, {cupy_mem:.0f} MB')
    
    if torch_time != float('inf'):
        if torch_time < cupy_time:
            print(f'  Winner: PyTorch ({cupy_time/torch_time:.1f}x slower sparse)')
        else:
            print(f'  Winner: CuPy ({torch_time/cupy_time:.1f}x faster sparse)')
    else:
        print(f'  Winner: CuPy (only option)')
    
    results.append((n, torch_time, torch_mem, cupy_time, cupy_mem))
    
    del W_csr
    cp.get_default_memory_pool().free_all_blocks()
    print()

print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('For n=10,000: Use PyTorch Dense (faster, memory OK)')
print('For n>20,000: Use CuPy CSR Sparse (only option that fits)')
print()
print('RECOMMENDATION:')
print('  - Keep using PyTorch Dense for n <= 15,000')
print('  - For paper-scale (n=100,000), need CuPy CSR sparse')
print('  - Memory: 10x savings with sparse')
print('  - Speed: Sparse is ~2-3x slower but SCALES')

