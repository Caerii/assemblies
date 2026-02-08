# connectome.py

import numpy as np

from .backend import get_xp, to_xp


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

    def _initialize_weights(self):
        """
        Initializes the synaptic weights.

        Returns:
            The weights matrix (numpy or cupy array).
        """
        xp = get_xp()
        if self.sparse:
            # For sparse simulation, create 2D array with 0 columns (matches original brain.py)
            return xp.empty((self.source_size, 0), dtype=xp.float32)
        else:
            # For explicit simulation, create full weight matrix
            # Binomial sampling on CPU, then transfer to backend
            w = np.random.binomial(1, self.p, size=(self.source_size, self.target_size)).astype(np.float32)
            return to_xp(w)

    def compute_inputs(self, pre_neurons):
        """
        Computes inputs to the target neurons based on active pre-synaptic neurons.

        Args:
            pre_neurons: Indices of active pre-synaptic neurons.

        Returns:
            Input strengths to each target neuron.
        """
        xp = get_xp()
        if self.sparse:
            # For sparse simulation, return zeros (no actual connections)
            return xp.zeros(self.target_size, dtype=xp.float32)
        else:
            return self.weights[pre_neurons].sum(axis=0)

    def update_weights(self, pre_neurons, post_neurons, beta: float):
        """
        Updates the synaptic weights based on activations.

        Args:
            pre_neurons: Indices of pre-synaptic neurons.
            post_neurons: Indices of post-synaptic neurons.
            beta (float): Synaptic plasticity parameter.
        """
        xp = get_xp()
        pre_neurons = xp.asarray(pre_neurons)
        post_neurons = xp.asarray(post_neurons)
        if len(pre_neurons) > 0 and len(post_neurons) > 0:
            ix = xp.ix_(pre_neurons, post_neurons)
            self.weights[ix] *= (1 + beta)

    def expand(self, new_source_size: int = 0, new_target_size: int = 0):
        """
        Expands the connectome to accommodate new neurons.

        Args:
            new_source_size (int): Number of new source neurons.
            new_target_size (int): Number of new target neurons.
        """
        xp = get_xp()
        if new_source_size > 0:
            new_rows = to_xp(np.random.binomial(1, self.p, size=(new_source_size, self.weights.shape[1])).astype(np.float32))
            self.weights = xp.vstack((self.weights, new_rows))
            self.source_size += new_source_size
        if new_target_size > 0:
            new_cols = to_xp(np.random.binomial(1, self.p, size=(self.weights.shape[0], new_target_size)).astype(np.float32))
            self.weights = xp.hstack((self.weights, new_cols))
            self.target_size += new_target_size
