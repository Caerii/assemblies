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

    # -- Array-like delegation for backward compatibility --------------------
    # Legacy code accesses connectomes as if they were raw numpy arrays
    # (e.g. conn[:3, :] = 1.0).  Delegate to self.weights.

    def __getitem__(self, key):
        return self.weights[key]

    def __setitem__(self, key, value):
        self.weights[key] = value

    def copy(self):
        """Return a copy of the underlying weight matrix."""
        return self.weights.copy()

    def __getstate__(self):
        """Drop the over-allocated growth buffer when pickling/deep-copying.

        ``NumpySparseEngine`` may keep ``_cap_buf`` -- spare capacity for a
        growing sparse vector -- with ``weights`` as a view into it. Pickling
        both would store the slack twice and silently detach the view anyway,
        so serialize the logical weights only; the engine reallocates capacity
        on the next growth.
        """
        state = dict(self.__dict__)
        if "_cap_buf" in state:
            state.pop("_cap_buf")
            w = state.get("weights")
            if w is not None and getattr(w, "base", None) is not None:
                state["weights"] = w.copy()
        return state

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


def is_dense_connectome(conn) -> bool:
    """True for a dense ``Connectome`` usable in explicit↔sparse mixed edges."""
    return isinstance(conn, Connectome) and not conn.sparse


def is_sparse_connectome(conn) -> bool:
    """True for sparse ``Connectome`` or engine-native CSR connectomes."""
    if conn is None:
        return True
    if isinstance(conn, Connectome):
        return conn.sparse
    return True


def dense_connectome_or_new(
    conn,
    *,
    source_size: int,
    target_size: int,
    p: float,
) -> Connectome:
    """Return *conn* if it is dense; otherwise allocate a fresh dense connectome."""
    if is_dense_connectome(conn):
        return conn
    return Connectome(source_size, target_size, p, sparse=False)
