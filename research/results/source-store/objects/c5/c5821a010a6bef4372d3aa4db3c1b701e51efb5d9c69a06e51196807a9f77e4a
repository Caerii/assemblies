"""The retired DIRECT instrument must fail before inspecting a brain."""

import pytest

from neural_assemblies import RetractedProtocol
from neural_assemblies.programs.direct import (
    direct_bind,
    intervention_bind_overlap,
    measure_directional_asymmetry,
    validate_direct_do_calculus,
)


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (direct_bind, (object(), "cause", "effect", "bind")),
        (measure_directional_asymmetry, (object(), "cause", "effect", "bind")),
        (intervention_bind_overlap, (object(), "fixed", "cue", "bind")),
        (validate_direct_do_calculus, (object(), "cause", "effect", "bind")),
    ],
)
def test_every_retired_direct_entry_point_refuses_without_brain_access(operation, args):
    with pytest.raises(RetractedProtocol, match="wipe negative control"):
        operation(*args)


def test_historical_recorder_cannot_overwrite_the_retracted_golden():
    from research.literature.parity.record_direct_pearl import main

    with pytest.raises(RetractedProtocol, match="historical evidence only"):
        main()
