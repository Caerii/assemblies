"""The registered temporal-position runner keeps brains as the statistical unit."""
from copy import deepcopy

import pytest

from research.experiments import seq_temporal_positions as study
from research.experiments.study4 import ntp_agree


def reports(seeds, arm):
    levels = {"g0": (.12, .08), "g1": (.22, .18), "blind_g1": (0., 0.)}[arm]
    result = []
    for seed in seeds:
        rows = []
        for position in range(9):
            contrast = .5 if position in (0, 3, 6) else levels[0 if position in (1, 4, 7) else 1]
            rows.append({"position": position, "contrast": contrast})
        result.append({"seed": seed, "analysis": {"positions": rows}})
    return result


def test_score_arms_uses_paired_brains_and_fixed_distractor_positions():
    seeds = [11, 17, 23]
    arms = {arm: reports(seeds, arm) for arm in study.ARMS}
    scored = study.score_arms(arms, seeds, study.REGISTERED)
    assert scored["checks"] == {
        "TP-1 predicted-win amplification": True,
        "TP-2 state dependence": True,
        "TP-3 plain conjunction carry": True,
        "TP-4 state-blind negative": True,
    }
    assert scored["summaries"]["g0"]["D"]["values"] == pytest.approx((.1, .1, .1))
    assert scored["summaries"]["g0"]["D1_minus_D2"]["mean"] == pytest.approx(.04)
    assert set(scored["summaries"]["g1"]["positions"]) == {str(i) for i in range(9)}


def test_score_arms_rejects_seed_or_position_drift():
    seeds = [11, 17, 23]
    arms = {arm: reports(seeds, arm) for arm in study.ARMS}
    arms["g1"][0]["seed"] = 99
    with pytest.raises(ValueError, match="seed identities"):
        study.score_arms(arms, seeds, study.REGISTERED)
    arms = {arm: reports(seeds, arm) for arm in study.ARMS}
    arms["g1"][0]["analysis"]["positions"].pop()
    with pytest.raises(ValueError, match="complete gap-2"):
        study.score_arms(arms, seeds, study.REGISTERED)


def test_parameter_drift_is_rejected_and_smoke_is_separate():
    study._validate_parameters(deepcopy(study.REGISTERED), "study")
    study._validate_parameters(deepcopy(study.SMOKE), "smoke")
    changed = deepcopy(study.REGISTERED)
    changed["test_sentences"] += 1
    with pytest.raises(ValueError, match="fixed temporal-position"):
        study._validate_parameters(changed, "study")


def test_explicit_chain_gap_does_not_mutate_legacy_generator_state():
    old = ntp_agree.GAP
    short = ntp_agree.generate_chain(3, 7, gap=1)
    long = ntp_agree.generate_chain(3, 7, gap=3)
    assert all(len(sentence) == 7 for sentence in short)
    assert all(len(sentence) == 13 for sentence in long)
    assert ntp_agree.GAP == old


def test_main_routes_fixed_smoke_protocol_through_shared_runner(monkeypatch):
    called = {}

    def writer(**kwargs):
        called.update(kwargs)
        return "result.json"

    monkeypatch.setattr(study, "run_experiment", writer)
    study.main(["--tag", "unit-smoke", "--smoke", "--seeds", "1", "2", "3"])
    assert called["protocol"] == "sequence.temporal-positions"
    assert called["protocol_version"] == "1"
    assert called["registration"].endswith("PREREG_temporal_positions.md")
    assert called["parameters"] == study.SMOKE and called["minimum_study_seeds"] == 20


def test_study_seed_identity_fails_before_runner(monkeypatch):
    monkeypatch.setattr(study, "run_experiment", lambda **_kwargs: pytest.fail("must not run"))
    with pytest.raises(SystemExit):
        study.main(["--tag", "wrong-seeds", "--seeds", *map(str, range(20))])


def test_experiment_indexes_raw_frames_in_one_attachment(monkeypatch):
    seeds = [11, 17, 23]
    corpus = [["does", "dog", "dog", "sees", "cat", "cat", "it", "bird", "bird", "doesnt"]]

    def run_arm(group, _parameters, arm, _organ_semantics):
        rows = reports(group, arm)
        for row in rows:
            row.update(test_corpus=corpus, frames=[{"identity": row["seed"]}],
                       learned_state_digest="a" * 64, arm=arm)
        return rows

    monkeypatch.setattr(study, "run_arm", run_arm)
    parameters = deepcopy(study.SMOKE)
    output = study.experiment({
        "mode": "smoke", "seeds": seeds, "parameters": parameters,
        "execution_semantics": {"profiles": {
            name: study.describe_hashed_transducer(
                w_max=parameters["w_max"], norm_init=parameters["norm_init"],
                refracted_strength=parameters["refracted_strength"],
                state_mode=parameters["state_mode"],
                predict_gain=arm["predict_gain"],
            ).to_dict()
            for name, arm in study.ARMS.items()
        }},
    })
    assert output.observations["raw_observations"] == {
        "attachment": "raw-frames.json.gz", "format": "temporal-position-frames-v1",
        "brain_count": 3, "arm_count": 3, "frame_count": 9,
    }
    raw = output.json_attachments["raw-frames.json.gz"]
    assert raw["corpora"] == {str(seed): corpus for seed in seeds}
    assert all("test_corpus" not in row for rows in raw["arms"].values() for row in rows)
