import pytest
from research.experiments.seq_a1_learning_null import BARS, score_pair


def rows(accuracy, exact):
    return [{"seed": i+1, "accuracy": a, "exact_fraction": e}
            for i,(a,e) in enumerate(zip(accuracy,exact))]


def test_perfect_null_fails_instrument_sensitivity():
    perfect=rows([1.]*3,[1.]*3)
    result=score_pair({"trained":perfect,"null":perfect},[1,2,3],BARS)
    assert not result["passed"]
    assert result["checks"]["trained_accuracy_low"]
    assert not result["checks"]["accuracy_delta_low"]


def test_separated_control_passes_with_paired_seed_evidence():
    result=score_pair({"trained":rows([1.]*3,[1.]*3),
                       "null":rows([.3,.32,.34],[0.,0.,0.])},[1,2,3],BARS)
    assert result["passed"]
    assert result["ensembles"]["accuracy"]["delta"]["keys"] == (1,2,3)


def test_reordered_arm_rejects_before_pairing():
    trained=rows([1.]*3,[1.]*3)
    with pytest.raises(ValueError,match="seed order"):
        score_pair({"trained":trained,"null":trained[::-1]},[1,2,3],BARS)


@pytest.mark.parametrize("value", [float("nan"),1.1,-.1,True])
def test_non_probability_observations_reject(value):
    with pytest.raises(ValueError,match="probabilities"):
        score_pair({"trained":rows([value]*3,[1.]*3),"null":rows([.3]*3,[0.]*3)},[1,2,3],BARS)
