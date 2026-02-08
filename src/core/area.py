# area.py
"""
Neural Area Simulation

This module implements the Area class for simulating individual brain areas
within the neural assembly framework. Each area represents a distinct brain
region with its own neural population and assembly dynamics.

Biological Context:
- Models cortical columns, brain regions, or functional areas
- Implements sparse neural coding: only k neurons fire per timestep
- Tracks neural activity patterns and assembly formation
- Supports both explicit (full simulation) and sparse (statistical) modes

Assembly Calculus Context:
- Each area can contain multiple neural assemblies
- Assemblies are sets of k co-active neurons representing concepts
- Assembly formation follows winner-take-all competition
- Supports assembly fixation for stable representations

Mathematical Foundation:
- Sparse coding: k/n ratio determines representation sparsity
- Winner-take-all: Only top-k neurons with highest inputs fire
- Hebbian plasticity: Synaptic weights strengthen with co-activation
- Statistical efficiency: Sparse simulation for large populations
"""

import numpy as np
from typing import Dict, List, Optional

from .backend import get_xp, to_cpu


class Area:
    """
    Neural Area for Assembly Simulation
    
    Represents a brain area containing a population of neurons that can form
    neural assemblies through co-activation and learning. Each area implements
    sparse neural coding where only a small fraction of neurons fire per timestep.
    
    This class models the fundamental unit of neural computation in the Assembly
    Calculus framework, where assemblies of co-active neurons represent concepts
    and perform computations through their interactions.
    
    Biological Principles:
    - Sparse coding: Only k neurons fire per timestep (k << n)
    - Assembly formation: Co-active neurons form stable assemblies
    - Plasticity: Synaptic weights adapt through Hebbian learning
    - Hierarchical processing: Areas can project to other areas
    
    Assembly Calculus Operations:
    - Assembly creation: Winner-take-all selection forms new assemblies
    - Assembly fixation: Stable assemblies can be frozen for reuse
    - Assembly projection: Assemblies can project to other areas
    - Assembly association: Assemblies can become more similar through co-activation
    
    References:
    - Papadimitriou, C. H., et al. "Brain Computation by Assemblies of Neurons." 
      Proceedings of the National Academy of Sciences 117.25 (2020): 14464-14472.
    - Mitropolsky, D., et al. "The Architecture of a Biologically Plausible 
      Language Organ." 2023.
    """

    def __init__(
        self,
        name: str,
        n: int,
        k: int,
        beta: float = 0.05,
        explicit: bool = False,
    ):
        """
        Initializes the Area.

        Args:
            name (str): Name of the area.
            n (int): Number of neurons in the area.
            k (int): Number of neurons that can fire at any time step.
            beta (float): Default synaptic plasticity parameter.
            explicit (bool): Whether the area is fully simulated (explicit).
        """
        self.name = name
        self.n = n
        self.k = k
        self.beta = beta
        self.explicit = explicit

        xp = get_xp()
        self._winners = xp.array([], dtype=xp.uint32)
        self.w = 0  # Number of neurons that have ever fired
        self.fixed_assembly = False

        # Temporary state for projection updates (matches brain.py)
        self._new_winners = xp.array([], dtype=xp.uint32)
        self._new_w = 0
        self.num_first_winners = -1

        if explicit:
            self.ever_fired = xp.zeros(self.n, dtype=bool)
            self.num_ever_fired = 0

        self.beta_by_stimulus: Dict[str, float] = {}
        self.beta_by_area: Dict[str, float] = {}
        self.saved_winners: List[np.ndarray] = []
        self.saved_w: List[int] = []

        # Sparse mapping: compact winner index -> actual neuron id
        # And a pre-shuffled pool of neuron ids for assigning to new winners
        self.compact_to_neuron_id: List[int] = []
        self.neuron_id_pool: Optional[np.ndarray] = None  # set by Brain.add_area
        self.neuron_id_pool_ptr: int = 0

    @property
    def winners(self) -> np.ndarray:
        return self._winners

    @winners.setter
    def winners(self, value):
        xp = get_xp()
        self._winners = xp.asarray(value, dtype=xp.uint32)
        self.w = len(self._winners)

    def fix_assembly(self):
        """Freezes the current assembly, preventing it from changing."""
        if len(self.winners) == 0:
            raise ValueError(f"Area {self.name} has no winners to fix.")
        self.fixed_assembly = True

    def unfix_assembly(self):
        """Allows the assembly to change in future simulations."""
        self.fixed_assembly = False

    def update_beta_by_stimulus(self, stimulus_name: str, new_beta: float):
        """Updates synaptic plasticity parameter for a specific stimulus."""
        self.beta_by_stimulus[stimulus_name] = new_beta

    def update_beta_by_area(self, area_name: str, new_beta: float):
        """Updates synaptic plasticity parameter for a specific area."""
        self.beta_by_area[area_name] = new_beta

    def get_num_ever_fired(self) -> int:
        """
        Returns the total number of neurons that have ever fired in this area.

        Returns:
            int: The number of neurons that have ever fired.
        """
        if self.explicit:
            return self.num_ever_fired
        else:
            return self.w

    def _update_winners(self, new_winners):
        """
        Updates the winners and records the state.

        Args:
            new_winners: The new winners to set.
        """
        xp = get_xp()
        self.winners = new_winners
        if self.explicit:
            self.ever_fired[new_winners] = True
            self.num_ever_fired = int(xp.sum(self.ever_fired))
        if self.saved_winners is not None:
            self.saved_winners.append(new_winners)
        if self.saved_w is not None:
            self.saved_w.append(self.w)
