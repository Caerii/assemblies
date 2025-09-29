# brain.py
"""
Neural Assembly Brain Simulation

This module implements the core Brain class for simulating neural assemblies
based on the Assembly Calculus framework introduced by Papadimitriou et al.
in "Brain Computation by Assemblies of Neurons" (PNAS, 2020).

The Brain class orchestrates the fundamental operations of the Assembly Calculus:
- Projection: Creating new assemblies in downstream areas
- Association: Increasing overlap between assemblies
- Merge: Combining assemblies to form new representations

Biological Context:
- Implements the NEMO model (Mitropolsky et al., 2023) for biological realism
- Models Hebbian plasticity: "neurons that fire together, wire together"
- Simulates sparse neural activity patterns found in biological brains
- Supports both explicit (full simulation) and sparse (statistical) modes

Mathematical Foundation:
- Assembly Calculus operations preserve overlap properties
- Winner-take-all selection implements sparse coding principles
- Synaptic plasticity follows Hebbian learning rules
- Statistical approximations enable scalable simulations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys
import os
import heapq
import math
from scipy.stats import binom, truncnorm

# We implement our own modularized brain logic using our extracted primitives

from .area import Area
from .stimulus import Stimulus
from .connectome import Connectome

# Import extracted math primitives
try:
    from math_primitives.statistics import StatisticalEngine
    from math_primitives.sparse_simulation import SparseSimulationEngine
    from math_primitives.winner_selection import WinnerSelector
    from math_primitives.plasticity import PlasticityEngine
    from math_primitives.explicit_projection import ExplicitProjectionEngine
    from math_primitives.image_activation import ImageActivationEngine
except ImportError:
    # Fallback for when running from root directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from math_primitives.statistics import StatisticalEngine
    from math_primitives.sparse_simulation import SparseSimulationEngine
    from math_primitives.winner_selection import WinnerSelector
    from math_primitives.plasticity import PlasticityEngine
    from math_primitives.explicit_projection import ExplicitProjectionEngine
    from math_primitives.image_activation import ImageActivationEngine

try:
    from ..utils.math_utils import select_top_k_indices
    from ..constants.default_params import DEFAULT_P, DEFAULT_BETA
except ImportError:
    # Fallback for when running as script
    from utils.math_utils import select_top_k_indices
    from constants.default_params import DEFAULT_P, DEFAULT_BETA

class Brain:
    """
    Neural Assembly Brain Orchestrator
    
    Manages the simulation of neural assemblies across multiple brain areas,
    implementing the Assembly Calculus framework for neural computation.
    
    This class serves as the central coordinator for:
    - Neural area management and connectivity
    - Assembly projection operations (Assembly Calculus)
    - Synaptic plasticity and learning
    - Stimulus processing and integration
    
    Biological Principles:
    - Sparse neural coding: Only k neurons fire per area per timestep
    - Hebbian plasticity: Synaptic weights strengthen with co-activation
    - Hierarchical processing: Information flows through area hierarchies
    - Statistical efficiency: Sparse simulation for large-scale networks
    
    Assembly Calculus Operations:
    - Projection: A → B (assembly A projects to create assembly B)
    - Association: A + B → A' + B' (assemblies A and B become more similar)
    - Merge: A + B → C (assemblies A and B combine to form assembly C)
    
    References:
    - Papadimitriou, C. H., et al. "Brain Computation by Assemblies of Neurons." 
      Proceedings of the National Academy of Sciences 117.25 (2020): 14464-14472.
    - Mitropolsky, D., et al. "The Architecture of a Biologically Plausible 
      Language Organ." 2023.
    """

    def __init__(self, p: float = DEFAULT_P, save_size: bool = True, save_winners: bool = False, seed: int = 0):
        """
        Initialize a neural assembly brain simulation.
        
        Creates a new brain instance with the specified connectivity parameters.
        The connection probability p determines the sparsity of the neural network,
        which is crucial for biological realism and computational efficiency.

        Args:
            p (float): Connection probability between neurons (0 < p < 1).
                      Lower values create sparser, more biologically realistic networks.
                      Typical values: 0.01-0.1 for large networks.
            seed (int): Random seed for reproducible simulations.
                       Essential for scientific reproducibility and debugging.
                       
        Biological Context:
            The connection probability p models the sparse connectivity found in
            biological neural networks, where each neuron connects to only a small
            fraction of other neurons, enabling efficient computation.
            
        Mathematical Context:
            The connection probability follows a binomial distribution B(n, p),
            where n is the number of potential connections and p is the probability
            of each connection existing.
        """
        self.p = p
        self.save_size = save_size
        self.save_winners = save_winners
        self.areas: Dict[str, Area] = {}
        self.stimuli: Dict[str, Stimulus] = {}
        self.connectomes_by_stimulus: Dict[str, Dict[str, Connectome]] = {}
        self.connectomes: Dict[str, Dict[str, Connectome]] = {}
        self.rng = np.random.default_rng(seed)
        self.disable_plasticity = False
        
        # Initialize extracted math primitives engines
        self.statistical_engine = StatisticalEngine(self.rng)
        self.sparse_simulation_engine = SparseSimulationEngine(self.rng)
        self.winner_selector = WinnerSelector(self.rng)
        self.plasticity_engine = PlasticityEngine(self.rng)
        self.explicit_projection_engine = ExplicitProjectionEngine(self.rng)
        self.image_activation_engine = ImageActivationEngine()

    def add_area(self, area_name: str, n: int, k: int, beta: float = DEFAULT_BETA, explicit: bool = False):
        """
        Add a neural area to the brain simulation.
        
        Creates a new brain area with specified neural population and assembly parameters.
        Each area represents a distinct brain region (e.g., visual cortex, hippocampus)
        with its own neural population and assembly dynamics.

        Args:
            area_name (str): Unique identifier for the brain area.
            n (int): Total number of neurons in the area (population size).
            k (int): Assembly size - number of neurons that fire per timestep.
            beta (float): Synaptic plasticity parameter (0 < beta < 1).
                        Controls the rate of Hebbian learning.
            explicit (bool): Whether to use explicit (full) or sparse simulation.
                           Explicit: tracks every neuron individually.
                           Sparse: uses statistical approximations for efficiency.
                           
        Biological Context:
            - n: Models the total neural population in a brain region
            - k: Implements sparse coding - only a small fraction of neurons fire
            - beta: Controls synaptic plasticity strength (Hebbian learning rate)
            - explicit: Determines simulation fidelity vs. computational efficiency
            
        Assembly Calculus Context:
            Each area can contain multiple assemblies, where an assembly is a set
            of k co-active neurons representing a specific concept or computation.
            The k parameter determines the sparsity of neural representations.
            
        Mathematical Context:
            - Assembly sparsity: k/n ratio (typically 0.01-0.1)
            - Hebbian learning: Δw = β * pre_activity * post_activity
            - Winner-take-all: Only top-k neurons fire per timestep
        """
        area = Area(area_name, n, k, beta, explicit)
        self.areas[area_name] = area
        # Initialize neuron id pool for sparse areas (permute 0..n-1)
        if not explicit:
            area.neuron_id_pool = self.rng.permutation(np.arange(n, dtype=np.uint32))
            area.neuron_id_pool_ptr = 0
        self.connectomes[area_name] = {}
        # Initialize connectomes for the new area
        self._initialize_connectomes_for_area(area)

    def add_stimulus(self, stimulus_name: str, size: int):
        """
        Adds a stimulus to the brain.

        Args:
            stimulus_name (str): Name of the stimulus.
            size (int): Number of firing neurons in the stimulus.
        """
        stimulus = Stimulus(stimulus_name, size)
        self.stimuli[stimulus_name] = stimulus
        self.connectomes_by_stimulus[stimulus_name] = {}
        # Initialize connectomes for the new stimulus
        self._initialize_connectomes_for_stimulus(stimulus)

    def project(
        self,
        areas_by_stim: Dict[str, List[str]] = None,
        dst_areas_by_src_area: Dict[str, List[str]] = None,
        external_inputs: Dict[str, np.ndarray] = None,
        projections: Dict[str, List[str]] = None,
        verbose: int = 0,
    ):
        """
        Execute Assembly Calculus Projection operations using our modular primitives.
        
        This method supports two APIs for maximum compatibility:
        1. Legacy API (areas_by_stim, dst_areas_by_src_area) - for existing simulation functions
        2. New API (external_inputs, projections) - for future sophisticated usage
        
        Args:
            areas_by_stim (Dict[str, List[str]], optional): Legacy API - maps stimulus names to target areas
            dst_areas_by_src_area (Dict[str, List[str]], optional): Legacy API - maps source areas to target areas
            external_inputs (Dict[str, np.ndarray], optional): New API - external activations to areas
            projections (Dict[str, List[str]], optional): New API - projection mapping
            verbose (int): Verbosity level for debugging (0=silent, 1=basic, 2=detailed)
        """
        # Determine which API is being used
        if areas_by_stim is not None or dst_areas_by_src_area is not None:
            # Legacy API - use the same logic as project_legacy but with our modular primitives
            self._project_legacy_compatible(areas_by_stim or {}, dst_areas_by_src_area or {}, verbose)
        elif external_inputs is not None or projections is not None:
            # New API - use sophisticated projection logic
            self._project_new_api(external_inputs or {}, projections or {}, verbose)
        else:
            raise ValueError("Must provide either legacy API parameters or new API parameters")

    def _project_legacy_compatible(self, areas_by_stim, dst_areas_by_src_area, verbose=0):
        """
        Legacy-compatible projection using our modular primitives.
        
        This method implements the same logic as project_legacy but uses our
        extracted engines for better maintainability and testing.
        """
        # Build input mappings exactly like original brain.py
        stim_in = defaultdict(list)
        area_in = defaultdict(list)

        # Validate and build stimulus inputs
        for stim, areas in areas_by_stim.items():
            if stim not in self.stimuli:
                raise IndexError(f"Not in brain.stimuli: {stim}")
            for area_name in areas:
                if area_name not in self.areas:
                    raise IndexError(f"Not in brain.areas: {area_name}")
                stim_in[area_name].append(stim)
        
        # Validate and build area inputs
        for from_area_name, to_area_names in dst_areas_by_src_area.items():
            if from_area_name not in self.areas:
                raise IndexError(f"Not in brain.areas: {from_area_name}")
            for to_area_name in to_area_names:
                if to_area_name not in self.areas:
                    raise IndexError(f"Not in brain.areas: {to_area_name}")
                area_in[to_area_name].append(from_area_name)

        # Process each target area using our modular primitives
        to_update_area_names = stim_in.keys() | area_in.keys()
        
        for area_name in to_update_area_names:
            area = self.areas[area_name]
            # Ensure each area gets a different random seed for proper randomization
            area_rng = np.random.default_rng(self.rng.integers(0, 2**32))
            num_first_winners, had_inputs = self._project_into_legacy(
                area, stim_in[area_name], area_in[area_name], verbose, area_rng)
            area.num_first_winners = num_first_winners
            
            # Save winners if requested and area received actual inputs (not no-signal preserve)
            if self.save_winners and had_inputs:
                if area.explicit or not hasattr(area, 'compact_to_neuron_id'):
                    saved = np.array(area._new_winners, dtype=np.uint32)
                else:
                    # Map compact indices to stable neuron ids for saving
                    mapping = area.compact_to_neuron_id
                    saved = np.array([mapping[idx] if idx < len(mapping) else np.uint32(idx)
                                      for idx in area._new_winners], dtype=np.uint32)
                area.saved_winners.append(saved)
                if verbose >= 1 and area_name == 'C':
                    phase = 'unknown'
                    if 'A' in area_in[area_name] and 'B' in area_in[area_name]:
                        phase = 'A,B->C'
                    elif 'A' in area_in[area_name]:
                        phase = 'A->C'
                    elif 'B' in area_in[area_name]:
                        phase = 'B->C' if len(area_in[area_name]) == 1 else 'Final B-only'
            
            # Save size if requested  
            if self.save_size:
                area.saved_w.append(area._new_w)
            
            # Update area state
            area.winners = area._new_winners
            area.w = area._new_w

    def _project_new_api(self, external_inputs, projections, verbose=0):
        """
        New API projection using sophisticated modular primitives.
        
        This method implements the advanced projection logic for future use cases.
        """
        # Process external inputs
        for area_name, input_winners in external_inputs.items():
            area = self.areas[area_name]
            area.winners = input_winners

        # Prepare mappings for projections
        stim_in = defaultdict(list)
        area_in = defaultdict(list)

        # Map stimuli to target areas
        for stim_name, stim in self.stimuli.items():
            for area_name in self.areas:
                stim_in[area_name].append(stim_name)

        # Map areas to target areas
        for from_area_name, to_area_names in projections.items():
            for to_area_name in to_area_names:
                area_in[to_area_name].append(from_area_name)

        # Perform projections into each target area using our modular primitives
        for area_name in set(stim_in.keys()) | set(area_in.keys()):
            target_area = self.areas[area_name]
            from_stimuli = stim_in[area_name]
            from_areas = area_in[area_name]
            self._project_into(target_area, from_stimuli, from_areas, verbose)

    def project_legacy(self, areas_by_stim, dst_areas_by_src_area, verbose=0):
        """
        Legacy compatibility method for the original brain.py API.
        
        This method directly implements the original brain.py projection logic
        to ensure 100% behavioral parity.
        
        Args:
            areas_by_stim: Dict mapping stimulus names to lists of target area names
            dst_areas_by_src_area: Dict mapping source area names to lists of target area names
            verbose: Verbosity level
        """
        # Build input mappings exactly like original brain.py
        stim_in = defaultdict(list)
        area_in = defaultdict(list)

        # Validate and build stimulus inputs
        for stim, areas in areas_by_stim.items():
            if stim not in self.stimuli:
                raise IndexError(f"Not in brain.stimuli: {stim}")
            for area_name in areas:
                if area_name not in self.areas:
                    raise IndexError(f"Not in brain.areas: {area_name}")
                stim_in[area_name].append(stim)
        
        # Validate and build area inputs
        for from_area_name, to_area_names in dst_areas_by_src_area.items():
            if from_area_name not in self.areas:
                raise IndexError(f"Not in brain.areas: {from_area_name}")
            for to_area_name in to_area_names:
                if to_area_name not in self.areas:
                    raise IndexError(f"Not in brain.areas: {to_area_name}")
                area_in[to_area_name].append(from_area_name)

        # Process each target area exactly like original brain.py
        to_update_area_names = stim_in.keys() | area_in.keys()
        
        for area_name in to_update_area_names:
            area = self.areas[area_name]
            num_first_winners, had_inputs = self._project_into_legacy(
                area, stim_in[area_name], area_in[area_name], verbose, None)
            area.num_first_winners = num_first_winners
            
            # Save winners if requested
            if self.save_winners:
                area.saved_winners.append(area._new_winners.copy())
            
            # Save size if requested  
            if self.save_size:
                area.saved_w.append(area._new_w)
            
            # Update area state
            area.winners = area._new_winners
            area.w = area._new_w

    def _project_into_legacy(self, target_area, from_stimuli, from_areas, verbose=0, area_rng=None):
        """
        Direct implementation of the original brain.py project_into method.
        
        This ensures 100% behavioral parity with the original system.
        """
        rng = area_rng if area_rng is not None else self.rng
        target_area_name = target_area.name

        if verbose >= 1:
            print(f"Projecting {', '.join(from_stimuli)} and {', '.join(from_areas)} into {target_area.name}")

        # Validate source areas have assemblies
        for from_area_name in from_areas:
            from_area = self.areas[from_area_name]
            if from_area.winners.size == 0 or from_area.w == 0:
                raise ValueError(f"Projecting from area with no assembly: {from_area}")

        # Handle fixed assembly case
        if target_area.fixed_assembly:
            target_area._new_winners = target_area.winners.copy()
            target_area._new_w = target_area.w
            return 0, False

        # If sparse target receives no inputs this round, keep assembly unchanged
        if (not target_area.explicit) and (len(from_stimuli) == 0) and (len(from_areas) == 0):
            target_area._new_winners = target_area.winners.copy()
            target_area._new_w = target_area.w
            return 0, False

        # Initialize previous winner inputs
        if target_area.explicit:
            # For explicit areas, use the extracted explicit projection engine
            prev_winner_inputs = self.explicit_projection_engine.accumulate_prev_inputs_explicit(
                target_area.n, 
                from_stimuli, 
                from_areas, 
                self.connectomes_by_stimulus, 
                self.connectomes, 
                self.areas, 
                target_area_name
            )
        else:
            # For sparse areas, initialize with zeros of length w (matches original brain.py exactly)
            # This represents input strength to each of the current winners (indices 0 to w-1)
            prev_winner_inputs = np.zeros(target_area.w, dtype=np.float32)

        # Add stimulus inputs for sparse areas (matches original brain.py exactly)
        if not target_area.explicit:
            # Calculate input strength specific to each current winner
            for i in range(target_area.w):
                # Add stimulus inputs for this specific winner
                for stim in from_stimuli:
                    stim_inputs = self.connectomes_by_stimulus[stim][target_area_name].weights
                    if i < len(stim_inputs):
                        prev_winner_inputs[i] += stim_inputs[i]
                
                # Add area inputs for this specific winner
                for from_area_name in from_areas:
                    connectome = self.connectomes[from_area_name][target_area_name]
                    # Skip if connectome has no columns (no connections yet)
                    if connectome.weights.shape[1] == 0:
                        continue
                    
                    # Calculate total input from this source area to winner i
                    total_contrib = 0.0
                    for w in self.areas[from_area_name].winners:
                        # Map absolute winner index to area's internal index
                        if hasattr(self.areas[from_area_name], 'compact_to_neuron_id'):
                            # Find the internal index for this absolute winner
                            internal_idx = None
                            for j, neuron_id in enumerate(self.areas[from_area_name].compact_to_neuron_id):
                                if neuron_id == w:
                                    internal_idx = j
                                    break
                            if internal_idx is None or internal_idx >= connectome.weights.shape[0]:
                                continue
                        else:
                            # For explicit areas, winner index should directly correspond to connectome row
                            internal_idx = w
                            if internal_idx >= connectome.weights.shape[0]:
                                continue
                        
                        # Get contribution from this source winner to target winner i
                        if i < connectome.weights.shape[1]:
                            contrib = connectome.weights[internal_idx, i]
                            total_contrib += contrib
                    
                    prev_winner_inputs[i] += total_contrib

            # If there is absolutely no input signal, keep previous winners (avoid unintended resets)
            if len(prev_winner_inputs) > 0 and float(np.sum(prev_winner_inputs)) == 0.0:
                target_area._new_winners = target_area.winners.copy()
                target_area._new_w = target_area.w
                return (0, False)

        # Simulate new winners for sparse areas using proper statistical sampling
        if not target_area.explicit:
            # Calculate input statistics for new winner simulation
            input_size_by_from_area_index = []
            num_inputs = 0
            
            for stim in from_stimuli:
                local_k = self.stimuli[stim].size
                input_size_by_from_area_index.append(local_k)
                num_inputs += 1
            
            for from_area_name in from_areas:
                local_k = self.areas[from_area_name].k
                input_size_by_from_area_index.append(local_k)
                num_inputs += 1

            # Use proper statistical sampling like original brain.py
            total_k = sum(input_size_by_from_area_index)
            effective_n = target_area.n - target_area.w
            
            if effective_n <= target_area.k:
                raise RuntimeError(
                    f'Remaining size of area "{target_area_name}" too small to sample k new winners.')
            
            # Threshold for inputs that are above (n-k)/n quantile
            quantile = (effective_n - target_area.k) / effective_n
            alpha = binom.ppf(quantile, total_k, self.p)
            
            # Use normal approximation for sampling
            mu = total_k * self.p
            std = math.sqrt(total_k * self.p * (1.0 - self.p))
            a = (alpha - mu) / std
            
            # Generate potential new winner inputs using truncated normal distribution
            # Use scipy's truncnorm.rvs like the original brain.py
            from scipy.stats import truncnorm
            if area_rng is not None:
                potential_new_winner_inputs = (mu + truncnorm.rvs(a, np.inf, scale=std, size=target_area.k, random_state=area_rng)).round(0)
            else:
                potential_new_winner_inputs = (mu + truncnorm.rvs(a, np.inf, scale=std, size=target_area.k, random_state=self.rng)).round(0)
            for i in range(len(potential_new_winner_inputs)):
                if potential_new_winner_inputs[i] > total_k:
                    potential_new_winner_inputs[i] = total_k
            
            # CRITICAL FIX: Concatenate previous and new winner inputs like original brain.py
            # prev_winner_inputs has length target_area.w (current winners)
            # potential_new_winner_inputs has length target_area.k (new candidates)
            if len(prev_winner_inputs) > 0:
                all_potential_winner_inputs = np.concatenate([prev_winner_inputs, potential_new_winner_inputs])
            else:
                # No previous winners, only consider new candidates
                all_potential_winner_inputs = potential_new_winner_inputs
            
            # Select top k winners from combined list
            new_winner_indices = heapq.nlargest(target_area.k, range(len(all_potential_winner_inputs)), all_potential_winner_inputs.__getitem__)
            
            # Prepare per-input split containers (filled after selecting first winners)
            inputs_by_first_winner_index = []
        else:
            # For explicit areas, new winners are selected from all neurons
            new_winners = np.arange(target_area.n, dtype=np.uint32)

        # Process winner indices for sparse areas
        if not target_area.explicit:
            # Convert indices to actual neuron indices like the original brain.py
            # Modify new_winner_indices in place (like original brain.py)
            num_first_winners_processed = 0
            first_winner_inputs = []
            for i in range(target_area.k):
                if new_winner_indices[i] >= target_area.w:
                    # Winner-index larger than `w` means this winner was first-activated here
                    # Record its total input before remapping index
                    first_winner_inputs.append(int(all_potential_winner_inputs[new_winner_indices[i]]))
                    # Assign a stable actual neuron id from pre-shuffled pool, but keep compact index for simulation
                    if hasattr(target_area, 'neuron_id_pool') and target_area.neuron_id_pool is not None:
                        pid = target_area.neuron_id_pool_ptr
                        if pid >= len(target_area.neuron_id_pool):
                            raise RuntimeError(f"Neuron id pool exhausted for area {target_area.name}")
                        actual_id = int(target_area.neuron_id_pool[pid])
                        target_area.neuron_id_pool_ptr += 1
                    else:
                        actual_id = target_area.w + num_first_winners_processed
                    # Append mapping for this new compact column
                    if hasattr(target_area, 'compact_to_neuron_id'):
                        target_area.compact_to_neuron_id.append(actual_id)
                    # Use compact index for winner id in this simulation step
                    new_winner_indices[i] = target_area.w + num_first_winners_processed
                    num_first_winners_processed += 1
                # else: new_winner_indices[i] is already the correct previous winner index
            
            winners = new_winner_indices
            if verbose >= 2 or (os.environ.get("ASSEMBLIES_VERBOSE") == "1" and target_area_name == "C"):
                # Count previous vs new winners correctly
                # Indices 0 to target_area.w-1 are previous winners
                # Indices target_area.w to target_area.w+target_area.k-1 are new candidates
                prev_cnt = sum(1 for idx in winners if idx < target_area.w)
                new_cnt = len(winners) - prev_cnt
        else:
            # For explicit areas, use the extracted winner selector
            all_inputs = prev_winner_inputs
            winners, _, _, _ = self.winner_selector.select_combined_winners(
                all_inputs, target_area.w, target_area.k
            )
            num_first_winners_processed = 0

        # Update area state
        target_area._new_winners = winners
        # For sparse areas, accumulate ever-fired count like original brain.py
        if target_area.explicit:
            target_area._new_w = len(winners)
        else:
            target_area._new_w = target_area.w + num_first_winners_processed
        
        # Count first-time winners
        if target_area.explicit:
            num_first_winners = 0
        else:
            # We already tracked how many first-time winners we processed
            num_first_winners = num_first_winners_processed
        
        # Apply learning to all winners (like original brain.py)
        # This is the key difference - original applies learning every time, not just for new winners
        self._apply_plasticity_to_winners(target_area, from_stimuli, from_areas, winners)
        
        # Update connectomes if there are new winners
        if num_first_winners > 0:
            # Build per-input split for each new winner to avoid 2D matrices
            inputs_names = list(from_stimuli) + list(from_areas)
            # Compute splits proportional to input sizes
            splits_per_new = []
            for total_in in first_winner_inputs:
                remaining = int(total_in)
                proportions = np.array(input_size_by_from_area_index, dtype=np.float64) / float(total_k)
                base = np.floor(proportions * remaining).astype(int)
                base_sum = int(base.sum())
                remainder = remaining - base_sum
                if remainder > 0:
                    # Distribute leftover to inputs with largest fractional part
                    frac = (proportions * remaining) - base
                    order = np.argsort(-frac)
                    for j in range(remainder):
                        base[order[j % len(base)]] += 1
                splits_per_new.append(base.tolist())
            # New winners are those with indices >= prior w
            prior_w = target_area.w
            new_indices_for_update = [int(w) for w in winners if int(w) >= prior_w]
            self._update_connectomes_for_new_winners(target_area, inputs_names, new_indices_for_update, splits_per_new)
        
        # Return both num_first_winners and whether inputs were actually processed
        had_inputs = bool(from_stimuli or from_areas)
        return num_first_winners, had_inputs

    def _apply_plasticity_to_winners(self, target_area, from_stimuli, from_areas, winners):
        """Apply Hebbian learning to all winners like the original brain.py."""
        # Apply stimulus-to-area plasticity
        for stim_name in from_stimuli:
            connectome = self.connectomes_by_stimulus[stim_name][target_area.name]
            beta = target_area.beta_by_stimulus.get(stim_name, target_area.beta)
            if not self.disable_plasticity and beta != 0:
                # Apply learning: multiply weights by (1 + beta) for all winners
                for winner in winners:
                    if winner < len(connectome.weights):
                        connectome.weights[winner] *= (1 + beta)
        
        # Apply area-to-area plasticity
        for from_area_name in from_areas:
            connectome = self.connectomes[from_area_name][target_area.name]
            beta = target_area.beta_by_area.get(from_area_name, target_area.beta)
            if not self.disable_plasticity and beta != 0:
                from_area_winners = self.areas[from_area_name].winners
                # Apply learning: multiply weights by (1 + beta) for active connections
                if len(connectome.weights.shape) == 2:
                    # 2D connectome (area-to-area)
                    for winner in winners:
                        for from_winner in from_area_winners:
                            if from_winner < connectome.weights.shape[0] and winner < connectome.weights.shape[1]:
                                connectome.weights[from_winner, winner] *= (1 + beta)
                else:
                    # 1D connectome (stimulus-to-area)
                    for winner in winners:
                        if winner < len(connectome.weights):
                            connectome.weights[winner] *= (1 + beta)

    def _update_connectomes_for_new_winners(self, target_area, inputs_names, new_winners, splits_per_new):
        """Update connectomes for new winners using 1D sparse representations.

        - Stimulus→area: 1D vector per area of length w; for each new winner, set weight to its allocated stim inputs.
        - Area→area: maintain 1D per from-area of length w (columns), storing number of synapses from that from-area into each winner column.
        """
        # Update stimulus vectors
        stim_names = [name for name in inputs_names if name in self.stimuli]
        area_names = [name for name in inputs_names if name in self.areas]
        # Ensure all stimulus vectors have length _new_w
        for stim_name in self.stimuli.keys():
            conn = self.connectomes_by_stimulus[stim_name][target_area.name]
            if conn.sparse:
                old = len(conn.weights)
                if target_area._new_w > old:
                    # If this stim did NOT fire this round, populate new entries with binomial(stim.size, p)
                    add_len = target_area._new_w - old
                    if stim_name not in stim_names:
                        stim_size = self.stimuli[stim_name].size
                        add = self.rng.binomial(stim_size, self.p, size=add_len).astype(np.float32)
                    else:
                        # Temporary zeros; will be assigned per-winner below
                        add = np.zeros(add_len, dtype=np.float32)
                    conn.weights = np.concatenate([conn.weights, add])
        # Write allocations for firing stimuli
        for idx, win in enumerate(new_winners):
            if win >= target_area._new_w:
                continue
            # find split vector for this new winner
            split = splits_per_new[idx] if idx < len(splits_per_new) else None
            if split is None:
                continue
            # Map split entries to inputs_names order
            for j, name in enumerate(inputs_names):
                alloc = int(split[j])
                if name in self.stimuli:
                    conn = self.connectomes_by_stimulus[name][target_area.name]
                    if conn.sparse and win < len(conn.weights):
                        conn.weights[win] = alloc
        # Update area→area 2D sparse matrices per from-area using winner row sampling
        for from_area_name in area_names:
            conn = self.connectomes[from_area_name][target_area.name]
            if not conn.sparse:
                continue
            from_area = self.areas[from_area_name]
            # Ensure 2D matrix shape (from_area.w, target_area._new_w)
            if conn.weights.ndim != 2:
                # Initialize to 0x0
                conn.weights = np.empty((0, 0), dtype=np.float32)
            rows, cols = conn.weights.shape
            if rows < from_area.w:
                # Pad rows
                pad_rows = from_area.w - rows
                if cols == 0:
                    conn.weights = np.empty((from_area.w, 0), dtype=np.float32)
                else:
                    conn.weights = np.vstack([conn.weights, np.zeros((pad_rows, cols), dtype=np.float32)])
                rows = from_area.w
            if cols < target_area._new_w:
                # Pad columns
                pad_cols = target_area._new_w - cols
                if rows == 0:
                    conn.weights = np.empty((0, target_area._new_w), dtype=np.float32)
                else:
                    conn.weights = np.hstack([conn.weights, np.zeros((rows, pad_cols), dtype=np.float32)])
                cols = target_area._new_w
            # Allocate synapses for new columns for this from_area based on splits
            from_index = inputs_names.index(from_area_name)
            rng = np.random.default_rng(self.rng.integers(0, 2**32))
            for idx, win in enumerate(new_winners):
                alloc = int(splits_per_new[idx][from_index]) if idx < len(splits_per_new) else 0
                if alloc <= 0 or from_area.w == 0:
                    continue
                # Sample alloc rows from current from_area winners without replacement
                sample_size = min(alloc, len(from_area.winners))
                if sample_size <= 0:
                    continue
                chosen = rng.choice(from_area.winners, size=sample_size, replace=False)
                # Map winner index to column index (winners are consecutive from target_area.w)
                col_idx = win - target_area.w
                if 0 <= col_idx < conn.weights.shape[1]:
                    conn.weights[chosen, col_idx] = 1.0

    def _project_into(
        self,
        target_area: Area,
        from_stimuli: List[str],
        from_areas: List[str],
        verbose: int = 0,
    ):
        """
        Projects activations into a target area from stimuli and other areas.

        Args:
            target_area (Area): The target area.
            from_stimuli (List[str]): List of stimuli projecting into the target area.
            from_areas (List[str]): List of areas projecting into the target area.
            verbose (int): Verbosity level.
        """
        if verbose >= 1:
            print(f"Projecting {', '.join(from_stimuli)} and {', '.join(from_areas)} into {target_area.name}")

        # If projecting from area with no assembly, raise an error
        for from_area_name in from_areas:
            from_area = self.areas[from_area_name]
            if len(from_area.winners) == 0:
                raise ValueError(f"Projecting from area with no assembly: {from_area.name}")

        if target_area.fixed_assembly:
            # If the target area has a fixed assembly, use it
            new_winners = target_area.winners
        else:
            # Use sophisticated projection algorithms based on area type
            if target_area.explicit:
                # Use explicit projection engine for full simulation
                new_winners = self._project_into_explicit(target_area, from_stimuli, from_areas, verbose)
            else:
                # Use sparse simulation engine for statistical approximation
                new_winners = self._project_into_sparse(target_area, from_stimuli, from_areas, verbose)

        # Update the target area's winners
        target_area._update_winners(new_winners)

    def _project_into_explicit(self, target_area: Area, from_stimuli: List[str], from_areas: List[str], verbose: int = 0):
        """
        Project into explicit area using full simulation.
        
        Uses the extracted explicit projection engine for sophisticated
        explicit area handling including validation, input accumulation,
        and specialized plasticity.
        """
        # Validate source winners using class method
        for from_area_name in from_areas:
            connectome = self.connectomes[from_area_name][target_area.name]
            from_area = self.areas[from_area_name]
            self.explicit_projection_engine.validate_source_winners(connectome.weights, from_area.winners)
        
        # Accumulate previous inputs for explicit areas using class method
        stimuli_vectors = []
        for stim_name in from_stimuli:
            # Compute inputs from stimulus to target area
            connectome = self.connectomes_by_stimulus[stim_name][target_area.name]
            stim_inputs = connectome.compute_inputs(self.stimuli[stim_name].winners)
            stimuli_vectors.append(stim_inputs)
        
        area_connectomes_and_winners = []
        for from_area_name in from_areas:
            connectome = self.connectomes[from_area_name][target_area.name]
            winners = self.areas[from_area_name].winners
            area_connectomes_and_winners.append((connectome.weights, winners))
        
        # Accumulate previous inputs for explicit areas
        prev_winner_inputs = self.explicit_projection_engine.accumulate_prev_inputs_explicit(
            target_area.n, stimuli_vectors, area_connectomes_and_winners
        )
        
        # Select winners using sophisticated winner selection
        new_winners, _, _, _ = self.winner_selector.select_combined_winners(
            prev_winner_inputs, target_area.w, target_area.k
        )
        
        # Apply explicit area plasticity using class method
        for from_area_name in from_areas:
            connectome = self.connectomes[from_area_name][target_area.name]
            from_area_winners = self.areas[from_area_name].winners
            beta = 0 if self.disable_plasticity else target_area.beta_by_area.get(from_area_name, target_area.beta)
            self.explicit_projection_engine.apply_area_to_area_plasticity(
                connectome.weights, from_area_winners, new_winners, beta, self.disable_plasticity
            )
        
        return new_winners

    def _project_into_sparse(self, target_area: Area, from_stimuli: List[str], from_areas: List[str], verbose: int = 0):
        """
        Project into sparse area using statistical simulation.
        
        Uses the extracted sparse simulation engine for efficient
        large-scale neural simulation with statistical approximations.
        """
        # Compute inputs from stimuli and areas
        inputs = np.zeros(target_area.n, dtype=np.float32)
        for stim_name in from_stimuli:
            connectome = self.connectomes_by_stimulus[stim_name][target_area.name]
            inputs += connectome.compute_inputs(self.stimuli[stim_name].winners)

        for from_area_name in from_areas:
            connectome = self.connectomes[from_area_name][target_area.name]
            inputs += connectome.compute_inputs(self.areas[from_area_name].winners)

        # Use sophisticated winner selection with potential masking
        new_winners, _, _, _ = self.winner_selector.select_combined_winners(inputs, target_area.w, target_area.k)
        
        # Update connectomes using plasticity engine
        self._update_connectomes_advanced(target_area, from_stimuli, from_areas, new_winners)
        
        return new_winners

    def _update_connectomes_advanced(self, target_area: Area, from_stimuli: List[str], from_areas: List[str], new_winners: np.ndarray):
        """
        Advanced connectome updates using the plasticity engine.
        
        Uses the extracted plasticity engine for sophisticated
        Hebbian learning and weight updates.
        """
        # Update connectomes from stimuli
        for stim_name in from_stimuli:
            connectome = self.connectomes_by_stimulus[stim_name][target_area.name]
            beta = target_area.beta_by_stimulus.get(stim_name, target_area.beta)
            if not self.disable_plasticity and beta != 0:
                self.plasticity_engine.scale_stimulus_to_area(
                    connectome.weights, self.stimuli[stim_name].winners, new_winners, beta
                )

        # Update connectomes from areas
        for from_area_name in from_areas:
            connectome = self.connectomes[from_area_name][target_area.name]
            beta = target_area.beta_by_area.get(from_area_name, target_area.beta)
            if not self.disable_plasticity and beta != 0:
                self.plasticity_engine.scale_area_to_area(
                    connectome.weights, self.areas[from_area_name].winners, new_winners, beta
                )

    def _update_connectomes(
        self,
        target_area: Area,
        from_stimuli: List[str],
        from_areas: List[str],
        new_winners: np.ndarray,
    ):
        """
        Updates the synaptic weights in the connectomes due to plasticity.

        Args:
            target_area (Area): The target area.
            from_stimuli (List[str]): List of stimuli projecting into the target area.
            from_areas (List[str]): List of areas projecting into the target area.
            new_winners (np.ndarray): Indices of the new winners in the target area.
        """
        # Update connectomes from stimuli
        for stim_name in from_stimuli:
            connectome = self.connectomes_by_stimulus[stim_name][target_area.name]
            beta = target_area.beta_by_stimulus.get(stim_name, target_area.beta)
            if beta != 0:
                connectome.update_weights(self.stimuli[stim_name].winners, new_winners, beta)

        # Update connectomes from areas
        for from_area_name in from_areas:
            connectome = self.connectomes[from_area_name][target_area.name]
            beta = target_area.beta_by_area.get(from_area_name, target_area.beta)
            if beta != 0:
                connectome.update_weights(self.areas[from_area_name].winners, new_winners, beta)

    def _initialize_connectomes_for_area(self, area: Area):
        """
        Initializes connectomes related to a new area.

        Args:
            area (Area): The new area.
        """
        # Initialize connectomes from stimuli to this area
        for stim_name, stim in self.stimuli.items():
            if area.explicit:
                # For explicit areas, create actual connectome matrices
                connectome = Connectome(stim.size, area.n, self.p, sparse=False)
            else:
                # For sparse areas, start with empty 1D vector of length area.w (0)
                connectome = Connectome(stim.size, area.n, self.p, sparse=True)
                connectome.weights = np.empty(0, dtype=np.float32)
            self.connectomes_by_stimulus[stim_name][area.name] = connectome
            area.beta_by_stimulus[stim_name] = area.beta

        # Initialize self-connection for the area
        if area.explicit:
            self_connectome = Connectome(area.n, area.n, self.p, sparse=False)
        else:
            self_connectome = Connectome(area.n, area.n, self.p, sparse=True)
            # For sparse, represent area-to-area as 2D with 0 columns
            self_connectome.weights = np.empty((area.n, 0), dtype=np.float32)
        self.connectomes[area.name][area.name] = self_connectome
        
        # Initialize connectomes from existing areas to this area
        for other_area_name, other_area in self.areas.items():
            if other_area_name != area.name:
                if area.explicit or other_area.explicit:
                    # Create actual connectome matrices if either area is explicit
                    connectome = Connectome(other_area.n, area.n, self.p, sparse=False)
                    connectome_rev = Connectome(area.n, other_area.n, self.p, sparse=False)
                else:
                    # Both areas are sparse, represent compactly with 0x0 matrices initially
                    connectome = Connectome(other_area.n, area.n, self.p, sparse=True)
                    connectome.weights = np.empty((0, 0), dtype=np.float32)
                    connectome_rev = Connectome(area.n, other_area.n, self.p, sparse=True)
                    connectome_rev.weights = np.empty((0, 0), dtype=np.float32)
                
                self.connectomes[other_area_name][area.name] = connectome
                self.connectomes[area.name][other_area_name] = connectome_rev
                self.connectomes[area.name][other_area_name] = connectome_rev
                # Set beta values
                area.beta_by_area[other_area_name] = area.beta
                other_area.beta_by_area[area.name] = area.beta

    def _initialize_connectomes_for_stimulus(self, stimulus: Stimulus):
        """
        Initializes connectomes related to a new stimulus.

        Args:
            stimulus (Stimulus): The new stimulus.
        """
        # Initialize connectomes from stimulus to all areas
        for area_name, area in self.areas.items():
            connectome = Connectome(stimulus.size, area.n, self.p)
            self.connectomes_by_stimulus[stimulus.name][area_name] = connectome
            area.beta_by_stimulus[stimulus.name] = area.beta
            
    def update_plasticity(self, from_area: str, to_area: str, new_beta: float):
        """
        Updates the synaptic plasticity parameter between two areas.

        Args:
            from_area (str): Name of the area that the synapses come from.
            to_area (str): Name of the area that the synapses project to.
            new_beta (float): The new synaptic plasticity parameter.
        """
        self.areas[to_area].beta_by_area[from_area] = new_beta 

    def update_plasticities(
        self,
        area_update_map: Dict[str, List[Tuple[str, float]]] = {},
        stim_update_map: Dict[str, List[Tuple[str, float]]] = {},
    ):
        """
        Updates the synaptic plasticity parameter between multiple areas and stimuli.

        Args:
            area_update_map (Dict[str, List[Tuple[str, float]]]):
                A dictionary where the keys are the names of areas that the synapses project to.
                The values are lists of tuples, where each tuple contains the name of an area that the synapses come from
                and the new synaptic plasticity parameter.
            stim_update_map (Dict[str, List[Tuple[str, float]]]):
                A dictionary where the keys are the names of areas.
                The values are lists of tuples, where each tuple contains the name of a stimulus and the new synaptic plasticity parameter.
        """
        for to_area, update_rules in area_update_map.items():
            for from_area, new_beta in update_rules:
                self.update_plasticity(from_area, to_area, new_beta)
        for area_name, update_rules in stim_update_map.items():
            area = self.areas[area_name]
            for stim_name, new_beta in update_rules:
                area.beta_by_stimulus[stim_name] = new_beta

    def activate(self, area_name: str, index: int):
        """
        Activates a specific assembly in an area.

        Args:
            area_name (str): Name of the area to activate.
            index (int): Index of the assembly to activate.

        Notes:
            This function is a shortcut for activating a specific assembly in an area.
            It is equivalent to calling `area.fix_assembly()` after setting the winners to the desired assembly.
        """
        area = self.areas[area_name]
        k = area.k
        assembly_start = k * index
        area.winners = np.arange(assembly_start, assembly_start + k, dtype=np.uint32)
        area.fix_assembly()
    
    def activate_with_image(self, area_name: str, image: np.ndarray):
        """
        Activates neurons in the given area using raw image data.
        
        Uses the extracted image activation engine for sophisticated image processing
        including normalization, cropping, padding, and top-k selection.
        
        Args:
            area_name (str): The name of the brain area to activate.
            image (np.ndarray): The raw image data (flattened or 2D).
        """
        area = self.areas[area_name]
        
        # Use extracted image activation engine
        image_flat = self.image_activation_engine.preprocess_image(image, area.n)
        winners, _ = self.image_activation_engine.normalize_and_select_topk(image_flat, area.k)
        
        # Set the winners in the area
        area.winners = winners
        area.w = len(area.winners)
    
    # Comprehensive usage example
    @staticmethod
    def example_assembly_calculus_demo():
        """
        Demonstrate Assembly Calculus operations with a complete example.
        
        This example shows how to use the Brain class to implement the
        fundamental operations of the Assembly Calculus framework.
        
        Assembly Calculus Operations Demonstrated:
        1. Projection: Visual → Semantic (A → B)
        2. Association: Semantic + Motor (A + B → A' + B')
        3. Merge: Semantic + Motor → Action (A + B → C)
        
        Biological Context:
        - Visual area processes sensory input
        - Semantic area represents concepts
        - Motor area controls actions
        - Integration areas combine information
        
        Returns:
            Brain: Configured brain instance ready for Assembly Calculus operations
        """
        # Initialize brain with sparse connectivity
        brain = Brain(p=0.05, seed=42)
        
        # Add brain areas representing different functional regions
        brain.add_area("visual", n=1000, k=100, beta=0.1, explicit=True)
        brain.add_area("semantic", n=800, k=80, beta=0.1, explicit=True)
        brain.add_area("motor", n=600, k=60, beta=0.1, explicit=True)
        brain.add_area("integration", n=500, k=50, beta=0.1, explicit=True)
        
        # Add external stimuli
        brain.add_stimulus("image", size=200)
        brain.add_stimulus("sound", size=150)
        
        # Example 1: Projection (Visual → Semantic)
        # External image activates visual area
        visual_assembly = np.random.choice(1000, 100, replace=False)
        external_inputs = {"visual": visual_assembly}
        projections = {"visual": ["semantic"]}
        brain.project(external_inputs, projections)
        
        # Example 2: Association (Semantic + Motor)
        # Both areas activate simultaneously to strengthen association
        semantic_assembly = brain.areas["semantic"].winners
        motor_assembly = np.random.choice(600, 60, replace=False)
        external_inputs = {"semantic": semantic_assembly, "motor": motor_assembly}
        projections = {"semantic": ["motor"], "motor": ["semantic"]}
        brain.project(external_inputs, projections)
        
        # Example 3: Merge (Semantic + Motor → Integration)
        # Both areas project to integration area to form combined representation
        external_inputs = {"semantic": semantic_assembly, "motor": motor_assembly}
        projections = {"semantic": ["integration"], "motor": ["integration"]}
        brain.project(external_inputs, projections)
        
        return brain
