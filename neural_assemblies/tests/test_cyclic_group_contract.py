"""Finite cyclic benchmarks must represent their requested group, not a subgroup."""
import pytest

from neural_assemblies.programs.word_problems import cyclic_group, word_problem_fsm


@pytest.mark.parametrize("order", [0, -1, True, 2.5])
def test_invalid_orders_are_rejected(order):
    with pytest.raises(ValueError, match="order"):
        cyclic_group(order)


@pytest.mark.parametrize("generators", [(True,), (1.5,), ("1",)])
def test_generator_coercions_are_rejected(generators):
    with pytest.raises(ValueError, match="integer residues"):
        cyclic_group(6, generators)


def test_proper_subgroup_cannot_masquerade_as_larger_benchmark():
    with pytest.raises(ValueError, match="proper subgroup"):
        cyclic_group(60, (2, 4))


def test_small_generator_pairs_against_explicit_linear_combinations():
    # An independent finite oracle: all integer combinations of two residues.
    for order in range(1, 13):
        for g in range(order):
            for h in range(order):
                span = {(a*g+b*h) % order for a in range(order) for b in range(order)}
                if len(span) != order:
                    with pytest.raises(ValueError, match="proper subgroup"):
                        cyclic_group(order, (g, h))
                else:
                    group = cyclic_group(order, (g, h))
                    assert set(group.elements) == span
                    for element in group.elements:
                        assert group.compose(element, -element) == group.identity


def test_normalized_generators_retain_alphabet_order():
    group = cyclic_group(5, (6, -3))
    assert group.generators == (1, 2)
    states, symbols, transitions = word_problem_fsm(group)
    assert states == ["0", "1", "2", "3", "4"]
    assert symbols == ["g0", "g1"]
    assert transitions[:2] == [("0", "g0", "1"), ("0", "g1", "2")]


def test_empty_alphabet_only_generates_trivial_group():
    assert cyclic_group(1, ()).elements == (0,)
    with pytest.raises(ValueError, match="proper subgroup"):
        cyclic_group(2, ())
