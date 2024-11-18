# brain.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from area import Area
from stimulus import Stimulus
from connectome import Connectome
import utils
import constants

class Brain:
    """
    Manages the brain simulation, including areas, stimuli, and projections.
    """

    def __init__(self, p: float = constants.DEFAULT_P, seed: int = 0):
        """
        Initializes a Brain instance.

        Args:
            p (float): Connection probability between neurons.
            seed (int): Random seed for reproducibility.
        """
        self.p = p
        self.areas: Dict[str, Area] = {}
        self.stimuli: Dict[str, Stimulus] = {}
        self.connectomes_by_stimulus: Dict[str, Dict[str, Connectome]] = {}
        self.connectomes: Dict[str, Dict[str, Connectome]] = {}
        self.rng = np.random.default_rng(seed)
        self.save_size = True
        self.save_winners = False
        self.disable_plasticity = False

    def add_area(self, area_name: str, n: int, k: int, beta: float = constants.DEFAULT_BETA, explicit: bool = False):
        """
        Adds an area to the brain.

        Args:
            area_name (str): Name of the area.
            n (int): Number of neurons.
            k (int): Number of firing neurons.
            beta (float): Synaptic plasticity parameter.
            explicit (bool): Whether the area is explicit.
        """
        area = Area(area_name, n, k, beta, explicit)
        self.areas[area_name] = area
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
        external_inputs: Dict[str, np.ndarray],
        projections: Dict[str, List[str]],
        verbose: int = 0,
    ):
        """
        Projects activations from stimuli or areas to target areas.

        Args:
            external_inputs (Dict[str, np.ndarray]): External inputs to areas.
            projections (Dict[str, List[str]]): Mapping from source areas to target areas.
            verbose (int): Verbosity level.
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

        # Perform projections into each target area
        for area_name in set(stim_in.keys()) | set(area_in.keys()):
            target_area = self.areas[area_name]
            from_stimuli = stim_in[area_name]
            from_areas = area_in[area_name]
            self._project_into(target_area, from_stimuli, from_areas, verbose)

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
            # Compute inputs from stimuli
            inputs = np.zeros(target_area.n, dtype=np.float32)
            for stim_name in from_stimuli:
                connectome = self.connectomes_by_stimulus[stim_name][target_area.name]
                inputs += connectome.compute_inputs(self.stimuli[stim_name].winners)

            # Compute inputs from areas
            for from_area_name in from_areas:
                connectome = self.connectomes[from_area_name][target_area.name]
                inputs += connectome.compute_inputs(self.areas[from_area_name].winners)

            # Select new winners based on inputs
            new_winners = utils.select_top_k_indices(inputs, target_area.k)

            # Update the connectomes (plasticity)
            self._update_connectomes(target_area, from_stimuli, from_areas, new_winners)

        # Update the target area's winners
        target_area._update_winners(new_winners)

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
            connectome = Connectome(stim.size, area.n, self.p)
            self.connectomes_by_stimulus[stim_name][area.name] = connectome
            area.beta_by_stimulus[stim_name] = area.beta

        # Initialize connectomes from existing areas to this area
        for other_area_name, other_area in self.areas.items():
            if other_area_name != area.name:
                # From other area to this area
                connectome = Connectome(other_area.n, area.n, self.p)
                self.connectomes[other_area_name][area.name] = connectome
                # From this area to other area
                connectome_rev = Connectome(area.n, other_area.n, self.p)
                self.connectomes[area.name][other_area_name] = connectome_rev
                # Set beta values
                area.beta_by_area[other_area_name] = area.beta
                other_area.beta_by_area[area.name] = other_area.beta

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

