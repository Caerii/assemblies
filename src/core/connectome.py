# connectome.py

import numpy as np

class Connectome:
    """
    Represents synaptic connections between neurons in different areas or stimuli.
    """

    def __init__(self, source_size: int, target_size: int, p: float, sparse: bool = False):
        """
        Initializes the Connectome.

        Args:
            source_size (int): Number of neurons in the source area.
            target_size (int): Number of neurons in the target area.
            p (float): Connection probability.
            sparse (bool): If True, creates empty weights for sparse simulation.
        """
        self.source_size = source_size
        self.target_size = target_size
        self.p = p
        self.sparse = sparse
        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> np.ndarray:
        """
        Initializes the synaptic weights.

        Returns:
            np.ndarray: The weights matrix.
        """
        if self.sparse:
            # For sparse simulation, create 2D array with 0 columns (matches original brain.py)
            return np.empty((self.source_size, 0), dtype=np.float32)
        else:
            # For explicit simulation, create full weight matrix
            return np.random.binomial(1, self.p, size=(self.source_size, self.target_size)).astype(np.float32)

    def compute_inputs(self, pre_neurons: np.ndarray) -> np.ndarray:
        """
        Computes inputs to the target neurons based on active pre-synaptic neurons.

        Args:
            pre_neurons (np.ndarray): Indices of active pre-synaptic neurons.

        Returns:
            np.ndarray: Input strengths to each target neuron.
        """
        if self.sparse:
            # For sparse simulation, return zeros (no actual connections)
            return np.zeros(self.target_size, dtype=np.float32)
        else:
            return self.weights[pre_neurons].sum(axis=0)

    def update_weights(self, pre_neurons: np.ndarray, post_neurons: np.ndarray, beta: float):
        """
        Updates the synaptic weights based on activations.

        Args:
            pre_neurons (np.ndarray): Indices of pre-synaptic neurons.
            post_neurons (np.ndarray): Indices of post-synaptic neurons.
            beta (float): Synaptic plasticity parameter.
        """
        for pre in pre_neurons:
            self.weights[pre, post_neurons] *= (1 + beta)

    def expand(self, new_source_size: int = 0, new_target_size: int = 0):
        """
        Expands the connectome to accommodate new neurons.

        Args:
            new_source_size (int): Number of new source neurons.
            new_target_size (int): Number of new target neurons.
        """
        if new_source_size > 0:
            new_rows = np.random.binomial(1, self.p, size=(new_source_size, self.weights.shape[1])).astype(np.float32)
            self.weights = np.vstack((self.weights, new_rows))
            self.source_size += new_source_size
        if new_target_size > 0:
            new_cols = np.random.binomial(1, self.p, size=(self.weights.shape[0], new_target_size)).astype(np.float32)
            self.weights = np.hstack((self.weights, new_cols))
            self.target_size += new_target_size
