# stimulus.py

import numpy as np
from .registration import validate_stimulus_registration

class Stimulus:
    """
    Represents an external stimulus that can activate brain areas.
    """

    def __init__(self, name: str, size: int):
        """
        Initializes the Stimulus.

        Args:
            name (str): Name of the stimulus.
            size (int): Number of firing neurons in the stimulus.
        """
        size = validate_stimulus_registration(name, size)
        self.name = name
        self.size = size
        self.winners = np.arange(size, dtype=np.uint32)
