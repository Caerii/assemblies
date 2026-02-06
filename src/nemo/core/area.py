"""
NEMO Core Area
==============

A single brain area that can:
1. Receive input (assembly of k neurons)
2. Project to output (new assembly of k neurons)
3. Learn associations via Hebbian plasticity

No language-specific logic here - just the neural mechanics.
"""

import torch
import cupy as cp
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .kernel import projection_kernel


@dataclass
class AreaParams:
    """Parameters for a brain area."""
    n: int = 10000          # Number of neurons
    k: int = None           # Assembly size (default: sqrt(n))
    p: float = 0.1          # Connection probability
    seed: int = 0           # Random seed for this area
    
    def __post_init__(self):
        if self.k is None:
            self.k = int(np.sqrt(self.n))


class Area:
    """
    A single brain area.
    
    Core operations:
    - project(input) -> output assembly
    - The projection uses hash-based random connectivity
    - No explicit weight storage (implicit connectivity)
    """
    
    def __init__(self, params: AreaParams = None, name: str = "area"):
        self.p = params or AreaParams()
        self.name = name
        
        n, k = self.p.n, self.p.k
        
        # Pre-allocated buffers (on GPU)
        self.active = torch.zeros(k, device='cuda', dtype=torch.int64)
        self.result = torch.zeros(n, device='cuda', dtype=torch.float16)
        
        # CuPy views for kernel (zero-copy)
        self.active_cp = cp.from_dlpack(self.active)
        self.result_cp = cp.from_dlpack(self.result)
        
        # Pre-created scalars for kernel
        self.k_u32 = cp.uint32(k)
        self.n_u32 = cp.uint32(n)
        self.p_f32 = cp.float32(self.p.p)
        self.seed_u32 = cp.uint32(self.p.seed)
        self.shared_mem = k * 4
        
        # Kernel grid config
        self.bs = 512
        self.gx = (n + self.bs - 1) // self.bs
        
        # Current assembly (last output)
        self.current: Optional[torch.Tensor] = None
    
    def project(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Project input assembly to output assembly.
        
        Args:
            inp: Input assembly (tensor of neuron indices)
            
        Returns:
            Output assembly (tensor of k neuron indices)
        """
        # Copy input to buffer
        inp_len = min(len(inp), self.p.k)
        self.active[:inp_len] = inp[:inp_len]
        
        # Run projection kernel
        projection_kernel(
            (self.gx,), (self.bs,),
            (self.active_cp.astype(cp.uint32), self.result_cp,
             self.k_u32, self.n_u32, self.seed_u32, self.p_f32),
            shared_mem=self.shared_mem
        )
        
        # Top-k selection
        _, winners = torch.topk(self.result, self.p.k, sorted=False)
        
        self.current = winners
        return winners
    
    def overlap(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute overlap between two assemblies."""
        set_a = set(a.cpu().numpy())
        set_b = set(b.cpu().numpy())
        return len(set_a & set_b) / self.p.k

