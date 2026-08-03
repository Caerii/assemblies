"""Dense wiring is a function of WHICH fiber it is, not of construction order.

DOOR 5 of [[content-addressed-synapse-init]], the last one open.

THE DEFECT. Dense `Connectome` drew its initial wiring from an RNG STREAM. A
stream advances, so a fiber's wiring depended on how many draws had happened
before it -- which is to say, on the ORDER areas and stimuli were created.
Measured before the fix: two Brains with the SAME seed and the same two areas,
built in opposite order, agreed on X->X wiring at 0.905. Chance agreement for
p=0.05 is 0.05^2 + 0.95^2 = 0.905. The two wirings were independent draws.

WHY IT SURVIVED FIVE OTHER DOORS. The sparse engines start their connectomes
EMPTY and materialise on demand through the content-addressed path, so the
default engine was never affected and nothing in the ordinary test suite could
see it. Only explicit/dense areas draw at construction.

WHERE THE FIX HAD TO GO, and it was not where it looked. Patching `Brain`'s
Connectome sites was not enough: for explicit areas the objects are BORN in
`NumpyExplicitEngine.add_area` and `Brain` adopts the same instances -- verified
by `id()` matching and `_pair_seed` still reading None after the Brain-side
patch. The engine is the owner.

THE FIX. `Connectome(pair_seed=...)` keys each synapse on (row, col,
pair_seed) via `hash_area_weights`, the same primitive `numpy_exact` is built
on. Order cannot reach a hash of coordinates. The stream path remains for
direct construction without a fiber identity, and its docstring says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.connectome import Connectome
from neural_assemblies.core.numpy_engine._seeding import fnv1a_pair_seed

P = 0.05


def _build(order, seed=7):
    b = Brain(p=P, seed=seed, norm_init=False, engine="numpy_sparse")
    for name, n in order:
        b.add_explicit_area(name, n, 10, P)
    return b


class TestConstructionOrderIndependence:

    @pytest.mark.parametrize("fiber", [("X", "X"), ("Y", "Y"), ("X", "Y")])
    def test_same_seed_different_order_same_wiring(self, fiber):
        a = _build([("X", 200), ("Y", 200)])
        b = _build([("Y", 200), ("X", 200)])
        src, tgt = fiber
        assert np.array_equal(a.connectomes[src][tgt].weights,
                              b.connectomes[src][tgt].weights), (
            f"{src}->{tgt} wiring depends on the order areas were created -- "
            f"the dense path is drawing from a stream again")

    def test_the_check_can_fail(self):
        """A different SEED must still give different wiring.

        Without this the test above passes trivially if wiring became a
        constant -- the failure mode that makes a determinism test useless.
        """
        a = _build([("X", 200), ("Y", 200)], seed=7)
        b = _build([("X", 200), ("Y", 200)], seed=8)
        assert not np.array_equal(a.connectomes["X"]["X"].weights,
                                  b.connectomes["X"]["X"].weights)

    def test_density_still_matches_p(self):
        """Content-addressing must not change the distribution it replaces."""
        a = _build([("X", 400), ("Y", 100)])
        density = float(np.asarray(a.connectomes["X"]["X"].weights).mean())
        assert abs(density - P) < 0.01, f"density {density:.4f} vs p={P}"

    def test_intervening_stimulus_does_not_shift_wiring(self):
        """The stream bug's other face: an unrelated allocation moved wiring."""
        a = Brain(p=P, seed=7, norm_init=False, engine="numpy_sparse")
        a.add_explicit_area("X", 200, 10, P)
        b = Brain(p=P, seed=7, norm_init=False, engine="numpy_sparse")
        b.add_stimulus("noise", 50)          # consumes stream draws
        b.add_explicit_area("X", 200, 10, P)
        assert np.array_equal(a.connectomes["X"]["X"].weights,
                              b.connectomes["X"]["X"].weights), (
            "adding an unrelated stimulus first changed X->X wiring")


class TestConnectomeDirectly:

    def test_pair_seed_is_deterministic_and_fiber_specific(self):
        ps = fnv1a_pair_seed(7, "X", "X")
        a = Connectome(200, 200, P, sparse=False, pair_seed=ps)
        b = Connectome(200, 200, P, sparse=False, pair_seed=ps)
        c = Connectome(200, 200, P, sparse=False,
                       pair_seed=fnv1a_pair_seed(7, "X", "Y"))
        assert np.array_equal(a.weights, b.weights)
        assert not np.array_equal(a.weights, c.weights)

    def test_stream_fallback_still_works_without_a_pair_seed(self):
        """Direct construction (unit tests, ad-hoc scripts) must not break."""
        rng = np.random.default_rng(0)
        conn = Connectome(50, 50, P, sparse=False, rng=rng)
        assert conn.weights.shape == (50, 50)

    def test_sparse_connectomes_are_untouched(self):
        """They start empty and materialise through the content-addressed
        path already; the pair seed must not make them allocate."""
        conn = Connectome(200, 200, P, sparse=True,
                          pair_seed=fnv1a_pair_seed(7, "X", "X"))
        assert conn.weights.shape[1] == 0
