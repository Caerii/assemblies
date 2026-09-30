#!/usr/bin/env python3
"""
Brain Area Management for Universal Brain Simulator
===================================================

This module manages brain areas and their data structures
for the universal brain simulator system.
"""

from typing import List, Union
import numpy as np
from .config import SimulationConfig
from .memory_manager import MemoryManager
from .cuda_manager import CUDAManager
from .utils import CUPY_AVAILABLE
from .algorithms import create_algorithm_strategy, print_strategy_info

# Import CuPy if available
if CUPY_AVAILABLE:
    import cupy as cp


class AreaManager:
    """
    Manages brain areas and their data structures
    
    This class handles the creation and management of brain areas,
    including candidate generation, top-k selection, and weight updates.
    """
    
    def __init__(self, config: SimulationConfig, memory_manager: MemoryManager, 
                 cuda_manager: CUDAManager):
        """
        Initialize area manager
        
        Args:
            config: Simulation configuration
            memory_manager: Memory manager instance
            cuda_manager: CUDA manager instance
        """
        self.config = config
        self.memory_manager = memory_manager
        self.cuda_manager = cuda_manager
        self.areas = []
        
        # Initialize random number generator
        self._rng = np.random.default_rng(config.seed)
        
        # Create algorithm strategy
        self.strategy = create_algorithm_strategy(config, cuda_manager)
        print_strategy_info(self.strategy)
    
    def initialize_areas(self):
        """Initialize brain areas with appropriate memory management"""
        self.areas = []
        
        for i in range(self.config.n_areas):
            if self.config.use_gpu and CUPY_AVAILABLE:
                # GPU memory allocation
                area = {
                    'n': self.config.n_neurons,
                    'k': self.config.k_active,
                    'w': 0,
                    'winners': self.memory_manager.allocate_gpu_array(self.config.k_active, dtype=np.int32),
                    'weights': self.memory_manager.allocate_gpu_array(self.config.k_active, dtype=np.float32),
                    'support': self.memory_manager.allocate_gpu_array(self.config.k_active, dtype=np.float32),
                    'activated': False,
                    'area_id': i
                }
            else:
                # CPU memory allocation
                area = {
                    'n': self.config.n_neurons,
                    'k': self.config.k_active,
                    'w': 0,
                    'winners': self.memory_manager.allocate_cpu_array(self.config.k_active, dtype=np.int32),
                    'weights': self.memory_manager.allocate_cpu_array(self.config.k_active, dtype=np.float32),
                    'support': self.memory_manager.allocate_cpu_array(self.config.k_active, dtype=np.float32),
                    'activated': False,
                    'area_id': i
                }
            
            self.areas.append(area)
    
    def generate_candidates(self, area_idx: int) -> Union[np.ndarray, cp.ndarray]:
        """
        Generate candidates using the selected algorithm strategy
        
        Args:
            area_idx: Index of the area to generate candidates for
            
        Returns:
            Array of candidate values
        """
        area = self.areas[area_idx]
        return self.strategy.generate_candidates(area, area_idx)
    
    def select_top_k(self, candidates: Union[np.ndarray, cp.ndarray], k: int) -> Union[np.ndarray, cp.ndarray]:
        """
        Select top-k using the selected algorithm strategy
        
        Args:
            candidates: Array of candidate values
            k: Number of top elements to select
            
        Returns:
            Array of top-k indices
        """
        return self.strategy.select_top_k(candidates, k)
    
    def update_weights(self, area_idx: int, winners: Union[np.ndarray, cp.ndarray]):
        """
        Update weights using the selected algorithm strategy
        
        Args:
            area_idx: Index of the area to update
            winners: Array of winner indices
        """
        area = self.areas[area_idx]
        self.strategy.update_weights(area, winners)
    
    def update_area_state(self, area_idx: int, winners: Union[np.ndarray, cp.ndarray]):
        """
        Update area state after simulation step
        
        Args:
            area_idx: Index of the area to update
            winners: Array of winner indices
        """
        area = self.areas[area_idx]
        
        # Update area state
        area['w'] = len(winners)
        area['winners'][:len(winners)] = winners
        area['activated'] = True
        
        # Update weights
        self.update_weights(area_idx, winners)
    
    def get_area_info(self, area_idx: int) -> dict:
        """
        Get information about a specific area
        
        Args:
            area_idx: Index of the area
            
        Returns:
            Dict containing area information
        """
        if area_idx >= len(self.areas):
            return {}
        
        area = self.areas[area_idx]
        return {
            'area_id': area['area_id'],
            'n_neurons': area['n'],
            'k_active': area['k'],
            'winners_count': area['w'],
            'activated': area['activated'],
            'memory_type': 'GPU' if CUPY_AVAILABLE and self.config.use_gpu else 'CPU'
        }
    
    def get_all_areas_info(self) -> List[dict]:
        """
        Get information about all areas
        
        Returns:
            List of area information dictionaries
        """
        return [self.get_area_info(i) for i in range(len(self.areas))]
    
    def reset_areas(self):
        """Reset all areas to initial state"""
        for area in self.areas:
            area['w'] = 0
            area['activated'] = False
            # Reset arrays to zero
            if CUPY_AVAILABLE and self.config.use_gpu:
                area['winners'].fill(0)
                area['weights'].fill(0.0)
                area['support'].fill(0.0)
            else:
                area['winners'].fill(0)
                area['weights'].fill(0.0)
                area['support'].fill(0.0)
    
    @property
    def num_areas(self) -> int:
        """Get number of areas"""
        return len(self.areas)
    
    @property
    def total_neurons(self) -> int:
        """Get total number of neurons across all areas"""
        return sum(area['n'] for area in self.areas)
    
    @property
    def total_active_neurons(self) -> int:
        """Get total number of active neurons across all areas"""
        return sum(area['k'] for area in self.areas)
