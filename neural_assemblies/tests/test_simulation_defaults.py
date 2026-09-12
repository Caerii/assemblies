from inspect import signature

from neural_assemblies.simulation.density_simulator import density_sim
from neural_assemblies.simulation.pattern_completion import pattern_com_alphas


def test_simulation_sweep_defaults_are_immutable():
    assert signature(density_sim).parameters["beta_values"].default is None
    assert signature(pattern_com_alphas).parameters["alphas"].default is None
