"""True-negative and composition tests for the pure assembly attention readout."""

import numpy as np
import pytest

from neural_assemblies.assembly_calculus import Assembly, attend


def _assembly(area, ids):
    return Assembly(area, np.asarray(ids, dtype=np.uint32))


def test_attention_selects_compatible_key_and_aggregates_its_value():
    query = _assembly("Q", [1, 2, 3])
    keys = {"a": _assembly("Q", [1, 2, 3]), "b": _assembly("Q", [8, 9, 10])}
    values = {"a": _assembly("V", [20, 21]), "b": _assembly("V", [30, 31])}

    result = attend(query, keys, values, output_size=2)

    assert result.selected_labels == ("a",)
    assert result.value == _assembly("V", [20, 21])
    assert result.candidates[0].weight > result.candidates[1].weight
    assert np.isclose(sum(candidate.weight for candidate in result.candidates), 1.0)


def test_attention_multi_key_output_is_deterministic_and_bounded():
    query = _assembly("Q", [1, 2])
    keys = {"a": _assembly("Q", [1, 2]), "b": _assembly("Q", [1, 2])}
    values = {"a": _assembly("V", [7, 8]), "b": _assembly("V", [8, 9])}

    result = attend(query, keys, values, top_k=2, output_size=2)

    assert result.selected_labels == ("a", "b")
    assert tuple(result.value.neuron_ids) == (8, 7)
    assert len(result.value) == 2


@pytest.mark.parametrize("kwargs", [
    {"top_k": 0}, {"top_k": 3}, {"output_size": 0}, {"temperature": 0},
])
def test_attention_rejects_invalid_schedule(kwargs):
    query = _assembly("Q", [1])
    keys = {"a": _assembly("Q", [1])}
    values = {"a": _assembly("V", [2])}
    with pytest.raises(ValueError):
        attend(query, keys, values, **kwargs)


def test_attention_rejects_key_value_mismatch_and_mixed_value_areas():
    query = _assembly("Q", [1])
    with pytest.raises(ValueError, match="same labels"):
        attend(query, {"a": _assembly("Q", [1])},
               {"b": _assembly("V", [2])})
    with pytest.raises(ValueError, match="one area"):
        attend(
            query,
            {"a": _assembly("Q", [1]), "b": _assembly("Q", [2])},
            {"a": _assembly("V1", [2]), "b": _assembly("V2", [3])},
        )
    with pytest.raises(ValueError, match="share one area"):
        attend(
            query,
            {"a": _assembly("K", [1])},
            {"a": _assembly("V", [2])},
        )


@pytest.mark.parametrize("query,keys,values", [
    (_assembly("Q", []), {"a": _assembly("K", [1])},
     {"a": _assembly("V", [2])}),
    (_assembly("Q", [1]), {"a": _assembly("Q", [])},
     {"a": _assembly("V", [2])}),
    (_assembly("Q", [1]), {"a": _assembly("Q", [1])},
     {"a": _assembly("V", [])}),
])
def test_attention_rejects_empty_support(query, keys, values):
    with pytest.raises(ValueError, match="(query|nonempty)"):
        attend(query, keys, values)
