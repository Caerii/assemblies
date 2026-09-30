"""Reconstruction distinguishes missing substrate from recovered role evidence."""
import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import ops
from neural_assemblies.assembly_calculus.emergent.parser_mixins.roles import RoleBindingMixin
from neural_assemblies.assembly_calculus.emergent.core.areas import ROLE_AGENT, ROLE_PATIENT


class RoleHarness(RoleBindingMixin):
    rounds = 2

    def __init__(self, populated=False):
        self.brain = Brain(engine="numpy_sparse", p=.1, seed=13, norm_init=False)
        for name in ("LEX", ROLE_AGENT, ROLE_PATIENT):
            self.brain.add_area(name, 1000, 20)
        self.stim_map = {word: word for word in ("dog", "cat")}
        self.core_lexicons = {"LEX": {}}
        for word in self.stim_map:
            self.brain.add_stimulus(word, 20)
            self.core_lexicons["LEX"][word] = ops.project(self.brain, word, "LEX", rounds=2)
        if populated:
            for role in (ROLE_AGENT, ROLE_PATIENT):
                self.brain.materialize_area(role)
                self.brain.project({}, {"LEX": [role]})

    def classify_word_cached(self, word):
        return "NOUN", 1.0

    def _determine_role_order(self, words, cats):
        return (), False

    def _word_core_area(self, word):
        return "LEX"

    def _func_subcat_of(self, word):
        return None


@pytest.mark.parametrize("populated", [False, True])
def test_role_readout_reports_unavailable_populations_without_creating_them(populated):
    parser = RoleHarness(populated)
    before = {name: (area.winners.copy(), parser.brain._engine.get_num_ever_fired(name))
              for name, area in parser.brain.areas.items()}
    roles, diag = parser.parse_roles_by_reconstruction(["dog", "cat"])
    if populated:
        assert roles == {"dog": "AGENT", "cat": "PATIENT"}
        assert diag["gaps"] and all(gap[-1] > 0 for gap in diag["gaps"])
        assert diag["unavailable_areas"] == {}
    else:
        assert roles == {"dog": None, "cat": None}
        assert diag["gaps"] == []
        assert diag["unavailable_areas"] == {
            ROLE_AGENT: "population_not_materialized",
            ROLE_PATIENT: "population_not_materialized"}
    for name, (winners, count) in before.items():
        np.testing.assert_array_equal(parser.brain.areas[name].winners, winners)
        assert parser.brain._engine.get_num_ever_fired(name) == count


@pytest.mark.parametrize("engine,ready", [("numpy_sparse", False), ("numpy_exact", True),
                                         ("numpy_explicit", True)])
def test_probe_readiness_matches_guard_for_empty_populations(engine, ready):
    brain = Brain(engine=engine, norm_init=False)
    brain.add_area("T", 100, 10)
    assert brain._engine.probe_target_ready("T") is ready
    with brain.read_only():
        if ready:
            brain._engine.validate_probe_target("T")
        else:
            with pytest.raises(ValueError, match="materializ"):
                brain._engine.validate_probe_target("T")



def test_recursive_parser_keeps_inner_readout_availability():
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    parser = EmergentParser(n=3000, k=30, seed=42, rounds=3, fast_training=True)
    parser.train_lexicon(skip_known=False)
    result = parser.parse_recursive("the dog that chases the cat runs".split())
    assert result["inner_roles"]
    assert result["inner_role_diagnostics"]["unavailable_areas"]
    assert not result["inner_role_diagnostics"]["gaps"]
