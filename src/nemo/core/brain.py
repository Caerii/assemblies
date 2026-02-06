"""
NEMO Core Brain
===============

A collection of brain areas that can be connected.

The Brain class provides:
1. Multiple areas with different seeds
2. Assembly storage (word -> assembly mapping)
3. Projection between areas

No language-specific logic - that belongs in language modules.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from .area import Area, AreaParams


@dataclass  
class BrainParams:
    """Parameters for the brain."""
    n: int = 10000          # Neurons per area
    k: int = None           # Assembly size
    p: float = 0.1          # Connection probability
    
    def __post_init__(self):
        if self.k is None:
            self.k = int(np.sqrt(self.n))


class Brain:
    """
    A brain with multiple areas.
    
    Usage:
        brain = Brain(params)
        brain.add_area("LEX")
        brain.add_area("SUBJ")
        
        # Store an assembly
        brain.store("dog", brain.random_assembly())
        
        # Project between areas
        output = brain.project("LEX", input_assembly)
    """
    
    def __init__(self, params: BrainParams = None):
        self.p = params or BrainParams()
        self.areas: Dict[str, Area] = {}
        self.assemblies: Dict[str, torch.Tensor] = {}
        self._next_seed = 0
    
    def add_area(self, name: str) -> Area:
        """Add a new brain area."""
        area_params = AreaParams(
            n=self.p.n,
            k=self.p.k,
            p=self.p.p,
            seed=self._next_seed * 1000
        )
        self._next_seed += 1
        
        area = Area(area_params, name=name)
        self.areas[name] = area
        return area
    
    def project(self, area_name: str, inp: torch.Tensor) -> torch.Tensor:
        """Project to an area."""
        if area_name not in self.areas:
            self.add_area(area_name)
        return self.areas[area_name].project(inp)
    
    def random_assembly(self) -> torch.Tensor:
        """Create a random assembly."""
        return torch.randint(0, self.p.n, (self.p.k,), device='cuda')
    
    def store(self, name: str, assembly: torch.Tensor = None) -> torch.Tensor:
        """Store an assembly by name."""
        if assembly is None:
            assembly = self.random_assembly()
        self.assemblies[name] = assembly
        return assembly
    
    def get(self, name: str) -> Optional[torch.Tensor]:
        """Get a stored assembly."""
        return self.assemblies.get(name)
    
    def get_or_create(self, name: str) -> torch.Tensor:
        """Get or create an assembly."""
        if name not in self.assemblies:
            self.store(name)
        return self.assemblies[name]
    
    def combine(self, *assemblies: torch.Tensor) -> torch.Tensor:
        """Combine multiple assemblies (union, then truncate to k)."""
        combined = torch.unique(torch.cat(assemblies))
        return combined[:self.p.k]
    
    def overlap(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute overlap between assemblies."""
        set_a = set(a.cpu().numpy())
        set_b = set(b.cpu().numpy())
        return len(set_a & set_b) / self.p.k

