"""Construction-level contracts for the NumPy literature reference backend."""

import numpy as np
import pytest

from neural_assemblies.reference.nemo_numpy.areas import FFArea, k_cap


def test_k_cap_zero_is_empty_and_negative_is_rejected() -> None:
    drive = np.array([1.0, 2.0, 3.0])

    assert k_cap(drive, 0).size == 0
    assert k_cap(drive, np.int64(1)).size == 1
    with pytest.raises(ValueError, match="cap_size"):
        k_cap(drive, -1)


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
