# default_params.py

DEFAULT_BETA = 0.05
DEFAULT_P = 0.01

# Weight ceiling for Hebbian plasticity.
# Prevents unbounded weight growth over long simulations.
# See: Dabagia et al. "Coin-Flipping in the Brain" (2024), Section 3.
DEFAULT_W_MAX = 20.0

# CUDA kernel constants (must match #define values in core/kernels/implicit.py)
CUDA_HASH_TABLE_SIZE = 1 << 20          # 1M entries, ~4 MB per pair
CUDA_BLOCK_SIZE = 512                   # threads per block (optimised for RTX 4090)
CUDA_MAX_LEARNED_PER_PAIR = 100_000     # COO buffer size per (source, target) pair
