from fractions import Fraction
import itertools
import numpy as np
import pytest
from neural_assemblies.ir.selection import compare_winner_selection


def test_small_error_can_change_winners():
    ref = np.array([1.0, np.nextafter(1.0, 0.0)])
    result = compare_winner_selection(ref, ref[::-1], 1)
    assert np.allclose(ref, ref[::-1])
    assert not result.winners_agree and not result.margin_certified


def test_agreement_is_not_a_margin_certificate():
    result = compare_winner_selection([1., 1.], [1., 1.], 1)
    assert result.winners_agree and not result.margin_certified


def test_strict_bound_and_constructed_equality_counterexample():
    assert compare_winner_selection([0, 3], [1, 2], 1).margin_certified
    result = compare_winner_selection([0, 2], [1, 1], 1)
    assert not result.margin_certified and not result.winners_agree


@pytest.mark.parametrize('ref,got', [
    ([np.finfo(float).max, -np.finfo(float).max], [0., 1.]),
    ([0., np.nextafter(0., 1.)], [0., np.nextafter(0., 1.)]),
    (np.array([2**63, 2**63+1], dtype=np.uint64), np.array([2**63, 2**63+1], dtype=np.uint64)),
])
def test_exact_units_survive_overflow_subnormal_and_large_integer_inputs(ref, got):
    result = compare_winner_selection(ref, got, 1)
    expected = max(abs(Fraction(a.item() if isinstance(a, np.generic) else a) -
                       Fraction(b.item() if isinstance(b, np.generic) else b)) for a,b in zip(ref,got))
    assert Fraction(result.max_error_units, 2**result.scale_exponent) == expected
    assert not result.margin_certified or result.winners_agree


def test_exhaustive_small_integer_scores_certificates_are_sound():
    rows = list(itertools.product(range(-1, 2), repeat=3))
    certified = 0
    for ref, got in itertools.product(rows, repeat=2):
        for k in (1, 2):
            result = compare_winner_selection(ref, got, k)
            if result.margin_certified:
                certified += 1
                assert result.winners_agree
    assert certified > 0


@pytest.mark.parametrize('ref,got,k', [([1],[1,2],1), ([np.nan],[0],1),
    ([np.inf],[0],1), ([[1]],[[1]],1), ([True],[False],1),
    ([1],[1],True), ([1],[1],1.5), ([1],[1],2), ([1],[1],-1)])
def test_invalid_comparisons_raise(ref,got,k):
    with pytest.raises(ValueError):
        compare_winner_selection(ref,got,k)


@pytest.mark.parametrize('k', [0,2])
def test_trivial_sets_certify_only_selection_not_assembly_quality(k):
    result = compare_winner_selection([0,1],[100,-100],k)
    assert result.margin_units is None and result.margin_certified and result.winners_agree


def test_mixed_python_values_do_not_round_large_integers_before_audit():
    result = compare_winner_selection([2**63 + 1, 0.0], [2**63, 0.0], 1)
    assert result.scale_exponent == 0 and result.max_error_units == 1
    assert result.margin_certified
