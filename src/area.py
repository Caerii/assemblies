# area.py

import numpy as np
from typing import Dict, List, Optional

class Area:
    """
    Represents a brain area with neurons and activation logic.
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

        self._winners = np.array([], dtype=np.uint32)
        self.w = 0  # Number of neurons that have ever fired
        self.fixed_assembly = False

        if explicit:
            self.ever_fired = np.zeros(self.n, dtype=bool)
            self.num_ever_fired = 0

        self.beta_by_stimulus: Dict[str, float] = {}
        self.beta_by_area: Dict[str, float] = {}
        self.saved_winners: List[np.ndarray] = []
        self.saved_w: List[int] = []

    @property
    def winners(self) -> np.ndarray:
        return self._winners

    @winners.setter
    def winners(self, value: np.ndarray):
        self._winners = np.array(value, dtype=np.uint32)
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

    def _update_winners(self, new_winners: np.ndarray):
        """
        Updates the winners and records the state.

        Args:
            new_winners (np.ndarray): The new winners to set.
        """
        self.winners = new_winners
        if self.explicit:
            self.ever_fired[new_winners] = True
            self.num_ever_fired = np.sum(self.ever_fired)
        if self.saved_winners is not None:
            self.saved_winners.append(new_winners)
        if self.saved_w is not None:
            self.saved_w.append(self.w)
