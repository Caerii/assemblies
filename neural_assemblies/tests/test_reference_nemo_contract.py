"""Construction-level contracts for the NumPy literature reference backend."""

import numpy as np

from neural_assemblies.reference.nemo_numpy.areas import FFArea


def test_empty_input_has_area_vector_shape() -> None:
    area = FFArea(
        [3],
        n_neurons=5,
        cap_size=2,
        density=0.5,
        plasticity=0.1,
        rng=np.random.default_rng(0),
    )

    total = area.get_total_input()

    assert total.shape == (5,)
    assert np.array_equal(total, np.zeros(5))
