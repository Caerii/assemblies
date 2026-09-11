"""ERP measurements must expose partial index-space mappings as undefined."""

from types import SimpleNamespace

import numpy as np

from neural_assemblies.assembly_calculus.emergent.evaluation.erp import adapters
from neural_assemblies.assembly_calculus.assembly import Assembly


def test_predicted_energy_rejects_partial_neuron_id_mapping(monkeypatch):
    engine = SimpleNamespace(
        project_into=lambda *args, **kwargs: SimpleNamespace(
            pre_kwta_inputs=np.asarray([1.0, 2.0], dtype=np.float32),
        ),
    )
    brain = SimpleNamespace(
        areas={adapters.PREDICTION: SimpleNamespace()},
        record_activation=False,
        _engine_for=lambda _area: engine,
    )
    entry = Assembly(adapters.PREDICTION, [10, 11, 12])
    monkeypatch.setattr(adapters, "_compact_index", lambda *_args: {10: 0})

    measured = adapters._predicted_energy(brain, entry)

    assert not measured.defined
    assert measured.detail == {
        "legacy": 0.0,
        "entry_size": 3,
        "mapped_size": 1,
        "vector_size": 2,
    }
    assert "only part" in measured.why
