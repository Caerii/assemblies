# connectome.py

from typing import Any, Optional, cast

import numpy as np

from .backend import get_xp, to_xp


class Connectome:
    """
    Represents synaptic connections between neurons in different areas or stimuli.
    """

    # Optional growth/normalization metadata shared by sparse, hashed and
    # dense backends.  They are populated lazily by the owning engine.
    _log_rows: int
    _log_cols: int
    _deg_rows: int
    _deg_counts_arr: Any
    _cap_buf: Any

    def __init__(self, source_size: int, target_size: int, p: float,
                 sparse: bool = False, rng=None, pair_seed=None):
        """
        Initializes the Connectome.

        Args:
            source_size (int): Number of neurons in the source area.
            target_size (int): Number of neurons in the target area.
            p (float): Connection probability.
            sparse (bool): If True, creates empty weights for sparse simulation.
            rng: Seeded ``np.random.Generator`` to draw the initial wiring from.
                Pass the owning Brain's generator. Without it this falls back to
                the GLOBAL ``np.random``, which makes results depend on
                execution history rather than on the declared seed: two Brains
                built with the same ``seed=`` in one process get different
                wiring, and borderline experiments flip. Measured on the
                word-order learner -- identical config and seed, four builds in
                one process, produced OVS, VSO, OVS, VSO (alternating, i.e.
                reading a shared global stream) while stable configurations
                reproduced. Only dense connectomes draw here; sparse ones start
                empty, which is why the default engine was unaffected.
        """
        self.source_size = source_size
        self.target_size = target_size
        self.p = p
        self.sparse = sparse
        # Store the generator or None -- never the numpy MODULE. A module is
        # not picklable, and Brain is deep-copied in several places (parity and
        # trace tests do `copy.deepcopy(brain)`), so stashing `np.random` here
        # raises "cannot pickle 'module' object". Resolve the fallback at call
        # time instead; see `_gen`.
        self._rng = rng
        # CONTENT-ADDRESSED WIRING when a pair seed is supplied -- door 5 of
        # [[content-addressed-synapse-init]]. Without it, the dense path draws
        # from a STREAM, so a fiber's wiring depends on how many draws happened
        # before it, i.e. on the ORDER areas were created. Measured: two Brains
        # with the SAME seed and the same two areas, created in opposite order,
        # agree on X->X wiring at 0.905 -- exactly the chance level for p=0.05
        # (0.05^2 + 0.95^2), so the two wirings are independent draws.
        #
        # `hash_area_weights` keys each synapse on (row, col, pair_seed), so
        # creation order cannot reach it. This is the same primitive
        # `numpy_exact` is built on; door 5 stayed open only because the dense
        # path predates it.
        self._pair_seed = pair_seed
        self.weights = self._initialize_weights()

    @property
    def _gen(self):
        """Generator for wiring draws; falls back to the global stream.

        The fallback exists only for direct construction (e.g. unit tests).
        Production callers pass a seeded generator -- see the `rng` argument.
        """
        # ``np.random`` is a legacy module fallback while production callers
        # provide a Generator; keep that compatibility explicit at this
        # dynamic boundary rather than pretending both objects share a static
        # protocol that NumPy's stubs do not expose.
        return cast(Any, self._rng if self._rng is not None else np.random)

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
            if self._pair_seed is not None:
                from .numpy_engine._seeding import hash_area_weights
                w = np.asarray(
                    hash_area_weights(0, self.source_size,
                                      0, self.target_size,
                                      self._pair_seed, self.p),
                    dtype=np.float32)
                return to_xp(w)
            # Stream fallback: order-dependent, kept only for direct
            # construction without a fiber identity. See __init__.
            w = self._gen.binomial(
                1, self.p,
                size=(self.source_size, self.target_size),
            ).astype(np.float32)
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

    def update_weights(self, pre_neurons, post_neurons, beta: float,
                       w_max: Optional[float] = None):
        """
        Updates the synaptic weights based on activations.

        Args:
            pre_neurons: Indices of pre-synaptic neurons.
            post_neurons: Indices of post-synaptic neurons.
            beta (float): Synaptic plasticity parameter.
            w_max (float): Saturation ceiling. Hebbian potentiation here is
                MULTIPLICATIVE, so without a ceiling a synapse that is
                reinforced on every presentation grows as ``(1 + beta)^t`` and
                overflows float32 -- measured at roughly 120 training sentences
                on an explicit->sparse bridge, after which the weights are
                garbage and every downstream comparison is meaningless. The
                engines carry ``w_max`` for exactly this reason; clamping here,
                at the multiplication, means a caller cannot forget it.
        """
        xp = get_xp()
        pre_neurons = xp.asarray(pre_neurons)
        post_neurons = xp.asarray(post_neurons)
        if len(pre_neurons) > 0 and len(post_neurons) > 0:
            ix = xp.ix_(pre_neurons, post_neurons)
            self.weights[ix] *= (1 + beta)
            if w_max is not None:
                sub = self.weights[ix]
                xp.clip(sub, 0, w_max, out=sub)
                self.weights[ix] = sub

    def expand(self, new_source_size: int = 0, new_target_size: int = 0):
        """
        Expands the connectome to accommodate new neurons.

        Args:
            new_source_size (int): Number of new source neurons.
            new_target_size (int): Number of new target neurons.
        """
        xp = get_xp()
        if new_source_size > 0:
            new_rows = to_xp(self._gen.binomial(1, self.p, size=(new_source_size, self.weights.shape[1])).astype(np.float32))
            self.weights = xp.vstack((self.weights, new_rows))
            self.source_size += new_source_size
        if new_target_size > 0:
            new_cols = to_xp(self._gen.binomial(1, self.p, size=(self.weights.shape[0], new_target_size)).astype(np.float32))
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
    rng=None,
) -> Connectome:
    """Return *conn* if it is dense; otherwise allocate a fresh dense connectome.

    Pass the owning Brain's/engine's seeded generator as ``rng``; without it the
    fresh connectome draws its wiring from the GLOBAL ``np.random`` and the
    result stops depending only on the declared seed. This factory was the last
    global-RNG consumer during Brain construction (measured: 18 calls per build).
    """
    if is_dense_connectome(conn):
        return conn
    return Connectome(source_size, target_size, p, sparse=False, rng=rng)
