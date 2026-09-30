import numpy as np
import pytest

from neural_assemblies.compute.hyperdimensional import FinalFixedHyperdimensionalAssembly


def _calculator():
    return FinalFixedHyperdimensionalAssembly.__new__(FinalFixedHyperdimensionalAssembly)


def test_calculus_rejects_mismatched_domain_and_values():
    calc = _calculator()
    with pytest.raises(ValueError, match="equal length"):
        calc._compute_derivative([np.array([1])], [])
    with pytest.raises(ValueError, match="equal length"):
        calc._compute_integral([np.array([1])], [])


def test_demo_rejects_truncated_zip_inputs():
    calc = _calculator()
    calc.dimension = 1000
    with pytest.raises(ValueError, match="equal length"):
        calc.assembly_calculus_demo([0.0, 1.0], [1.0])
