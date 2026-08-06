"""`measure_stability` must not report 0.0 for an area it could not measure.

WHY THIS FILE LIVES HERE AND NOT IN `neural_assemblies/tests/`. It would be
better off in the main suite -- `pyproject.toml` sets
`testpaths = ["neural_assemblies/tests"]`, so nothing in this directory runs in
CI and this guard only fires when someone runs it deliberately. But importing
`..brain` pulls in `cupy`, and importing cupy before torch in the same process
breaks torch's CUDA with a bare Windows access violation (see the
`cupy-torch-import-order` finding). Adding that import to the main suite would
hand every torch test in the same process a crash that looks like a hardware
fault. So it stays here, and the coverage gap is stated rather than hidden.

WHAT IT PINS. `measure_stability` returned a bare `0.0` when the area held no
assembly. Both of its consumers read low stability as WOBBLY --
`test_merge_stability` calls it incompatible, and the decoder's "does this
pattern exist" check thresholds it at 0.3 -- so "there was nothing to measure"
arrived as a maximally confident wrong answer. Same defect as the ERP
`phrase_stability` readout in the other half of the codebase.

The distinction that matters is visible in the two cases below: a populated area
can legitimately measure 0.0, and an empty one now cannot produce a number at
all. Before the change those were the same value.
"""
import pytest

from neural_assemblies.core.measurement import UndefinedMeasurement
from ..areas import Area
from ..brain import EmergentNemoBrain
from ..params import EmergentParams


@pytest.fixture(scope="module")
def brain():
    return EmergentNemoBrain(EmergentParams(n=2000, k=20), verbose=False)


def test_an_empty_area_yields_an_undefined_stability_not_zero(brain):
    stability = brain.measure_stability(Area.VP, rounds=2)

    assert not stability.defined
    assert "holds no assembly" in stability.why, (
        "the reason must name the missing precondition, not restate 'undefined'")


def test_an_undefined_stability_refuses_to_be_used_as_a_number(brain):
    """`stability > 0.3` is the decoder's threshold, verbatim."""
    stability = brain.measure_stability(Area.VP, rounds=2)

    with pytest.raises(UndefinedMeasurement):
        float(stability)
    with pytest.raises(UndefinedMeasurement):
        _ = stability > 0.3


def test_a_populated_area_still_yields_a_defined_stability(brain):
    """The other direction -- and a DEFINED 0.0 is a real reading, not a gap."""
    phon = brain._get_or_create(Area.PHON, "dog")
    brain._project(Area.NOUN_CORE, phon, learn=True)

    stability = brain.measure_stability(Area.NOUN_CORE, rounds=2)

    assert stability.defined
    assert 0.0 <= float(stability) <= 1.0


def test_there_is_one_name_for_this_measurement(brain):
    """`get_phrase_stability` was an exact zero-caller alias. Two names for one
    measurement is how the two get fixed separately."""
    assert not hasattr(brain, "get_phrase_stability")
