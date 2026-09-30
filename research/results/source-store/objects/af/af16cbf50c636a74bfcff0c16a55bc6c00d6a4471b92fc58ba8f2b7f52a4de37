"""Retired ERP compatibility parameters cannot change a measurement."""

import pytest

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    anchored_p600_live,
    phrase_stability,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.protocol import (
    ErpProtocol,
)


def test_phrase_stability_rejects_retired_settling_controls():
    with pytest.raises(ValueError, match="single-projection"):
        phrase_stability(
            object(), "ROLE", rounds=4, protocol=ErpProtocol()
        )
    with pytest.raises(ValueError, match="single-projection"):
        phrase_stability(
            object(), "ROLE", k=10, protocol=ErpProtocol()
        )


def test_anchored_p600_rejects_retired_source_and_settling_controls():
    with pytest.raises(ValueError, match="subject_core=None"):
        anchored_p600_live(object(), "CORE", "ROLE", subject_core="SUBJ")
    with pytest.raises(ValueError, match="pre-k-WTA"):
        anchored_p600_live(object(), "CORE", "ROLE", n_settling=2)
