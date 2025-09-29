"""
High-performance C++ Brain implementation wrapper.

This module provides a Python interface to the C++ Brain implementation
for much faster neural assembly simulations.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Try to import the C++ extension
try:
    import brain_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ brain extension not available. Install with: pip install -e cpp/")

from .area import Area
from .connectome import Connectome


class BrainCPP:
    """
    High-performance C++ Brain implementation wrapper.
    
    This class provides a Python interface to the C++ Brain implementation,
    offering significant performance improvements over the pure Python version.
    """
    
    def __init__(self, p: float, beta: float = 0.1, max_weight: float = 10000.0, 
                 seed: int = 0, save_size: bool = True, save_winners: bool = False):
        """
        Initialize the C++ Brain.
        
        Args:
            p: Connection probability
            beta: Learning rate
            max_weight: Maximum synaptic weight
            seed: Random seed
            save_size: Whether to save area sizes
            save_winners: Whether to save winner neurons
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ brain extension not available. Install with: pip install -e cpp/")
        
        self.p = p
        self.beta = beta
        self.max_weight = max_weight
        self.seed = seed
        self.save_size = save_size
        self.save_winners = save_winners
        
        # Initialize C++ brain
        self._brain = brain_cpp.Brain(p, beta, max_weight, seed)
        
        # Python-side tracking for compatibility
        self.areas = {}
        self.saved_w = {}  # area_name -> list of w values
        self.saved_winners = {}  # area_name -> list of winner lists
        
    def add_area(self, name: str, n: int, k: int, beta: float = 0.1, 
                 explicit: bool = False) -> Area:
        """Add a neural area."""
        self._brain.add_area(name, n, k, recurrent=True, is_explicit=explicit)
        
        # Create Python Area object for compatibility
        area = Area(name, n, k, beta, explicit)
        self.areas[name] = area
        
        # Initialize tracking
        if self.save_size:
            self.saved_w[name] = []
        if self.save_winners:
            self.saved_winners[name] = []
            
        return area
    
    def add_stimulus(self, name: str, size: int) -> None:
        """Add a stimulus."""
        self._brain.add_stimulus(name, size)
    
    def add_fiber(self, from_name: str, to_name: str, bidirectional: bool = False) -> None:
        """Add a connection between areas."""
        self._brain.add_fiber(from_name, to_name, bidirectional)
    
    def project(self, areas_by_stim: Dict[str, List[str]], 
                dst_areas_by_src_area: Dict[str, List[str]], 
                verbose: int = 0) -> None:
        """
        Project activity through the network.
        
        Args:
            areas_by_stim: Mapping from stimulus names to target area names
            dst_areas_by_src_area: Mapping from source area names to target area names
            verbose: Verbosity level
        """
        # Convert to C++ format
        graph = {}
        
        # Add stimulus projections
        for stim, areas in areas_by_stim.items():
            graph[stim] = areas
        
        # Add area-to-area projections
        for src_area, dst_areas in dst_areas_by_src_area.items():
            if src_area not in graph:
                graph[src_area] = []
            graph[src_area].extend(dst_areas)
        
        # Execute projection
        self._brain.project(graph, 1, update_plasticity=True)
        
        # Update Python-side tracking
        self._update_python_tracking()
    
    def _update_python_tracking(self) -> None:
        """Update Python-side area tracking for compatibility."""
        for name, area in self.areas.items():
            # Get activated neurons from C++
            activated = self._brain.get_activated(name)
            
            # Update area state
            area.winners = activated
            area.w = len(activated)
            
            # Save if requested
            if self.save_size:
                self.saved_w[name].append(area.w)
            
            if self.save_winners:
                self.saved_winners[name].append(activated.copy())
    
    def get_area(self, name: str) -> Area:
        """Get an area by name."""
        return self.areas[name]
    
    def get_saved_winners(self, name: str) -> List[List[int]]:
        """Get saved winners for an area."""
        return self.saved_winners.get(name, [])
    
    def get_saved_w(self, name: str) -> List[int]:
        """Get saved w values for an area."""
        return self.saved_w.get(name, [])
    
    def inhibit_all(self) -> None:
        """Inhibit all connections."""
        self._brain.inhibit_all()
    
    def inhibit_fiber(self, from_name: str, to_name: str) -> None:
        """Inhibit a specific connection."""
        self._brain.inhibit_fiber(from_name, to_name)
    
    def activate_fiber(self, from_name: str, to_name: str) -> None:
        """Activate a specific connection."""
        self._brain.activate_fiber(from_name, to_name)
    
    def set_log_level(self, level: int) -> None:
        """Set logging level."""
        self._brain.set_log_level(level)
    
    def log_graph_stats(self) -> None:
        """Log graph statistics."""
        self._brain.log_graph_stats()
    
    def log_activated(self, area_name: str) -> None:
        """Log activated neurons for an area."""
        self._brain.log_activated(area_name)


def create_high_performance_brain(p: float, beta: float = 0.1, 
                                 max_weight: float = 10000.0, seed: int = 0,
                                 save_size: bool = True, save_winners: bool = False) -> BrainCPP:
    """
    Create a high-performance C++ Brain instance.
    
    Args:
        p: Connection probability
        beta: Learning rate  
        max_weight: Maximum synaptic weight
        seed: Random seed
        save_size: Whether to save area sizes
        save_winners: Whether to save winner neurons
        
    Returns:
        BrainCPP instance
    """
    return BrainCPP(p, beta, max_weight, seed, save_size, save_winners)


# Convenience function for backward compatibility
def Brain(p: float, beta: float = 0.1, max_weight: float = 10000.0, 
          seed: int = 0, save_size: bool = True, save_winners: bool = False) -> BrainCPP:
    """Create a Brain instance (alias for create_high_performance_brain)."""
    return create_high_performance_brain(p, beta, max_weight, seed, save_size, save_winners)
