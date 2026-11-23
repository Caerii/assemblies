"""Contract tests for the backend-independent paired teacher control harness."""
import numpy as np
import pytest
from research.multimodal.teacher_control_matrix import _teacher_labels, CONDITIONS, run_conditions
from research.multimodal.zero_shot_gate import Config, Dataset


class Memory:
    def __init__(self, cfg, condition):
        self.saved = {}
        self.writes = 0
        self.train_pairs = set()
        self.training_calls = 0
        self.condition = condition

    def train(self, item, role, cue, target):
        self.train_pairs.add((item, role))
        self.training_calls += 1
        if self.condition != "no_plasticity":
            self.saved[item, role] = target.copy()

    def predict(self, item, role, cue):
        return self.saved.get((item, role))


def test_control_names():
    assert CONDITIONS == ("correct_teacher", "shuffled_teacher", "no_plasticity")


def test_permuted_labels_only_come_from_training_pairs():
    d = Dataset(Config(seed=17, contents=8))
    original = _teacher_labels(d, "correct_teacher")
    shuffled = _teacher_labels(d, "shuffled_teacher")
    assert set(original) == set(shuffled) == set(d.train)
    assert not set(shuffled) & set(d.test)
    for role in d.cfg.roles:
        x = sorted(tuple(v) for (i, r), v in original.items() if r == role)
        y = sorted(tuple(v) for (i, r), v in shuffled.items() if r == role)
        assert x == y
        for i, r in d.train:
            if r == role:
                assert not np.array_equal(original[i, role], shuffled[i, role])


def test_pair_memory_controls_work_but_never_zero_shot():
    output = run_conditions(Config(contents=8), Memory)
    lookup = {x["condition"]: x for x in output["conditions"]}
    assert lookup["correct_teacher"]["summary"]["train"]["exact"] == 1.0
    assert lookup["correct_teacher"]["summary"]["heldout"]["exact"] == 0.0
    assert lookup["shuffled_teacher"]["summary"]["train"]["exact"] == 0.0
    assert lookup["no_plasticity"]["summary"]["train"]["exact"] == 0.0
    assert all(x["summary"]["heldout"]["exact"] == 0.0 for x in lookup.values())
    assert all(x["summary"]["train"]["repeatable"] == 1.0 for x in lookup.values())
    assert all(x["training_calls"] == 8 for x in lookup.values())
    assert (lookup["correct_teacher"]["train_labels_sha256"]
            != lookup["shuffled_teacher"]["train_labels_sha256"])


def test_deterministic_report_without_wallclock():
    cfg = Config(contents=8, seed=2)
    assert run_conditions(cfg, Memory) == run_conditions(cfg, Memory)


def test_model_mutates_query_does_not_mutate_dataset():
    class Malicious(Memory):
        def predict(self, item, role, cue):
            cue[:] = 0
            return None
    result = run_conditions(Config(contents=4), Malicious)
    assert all(x["summary"]["heldout"]["exact"] == 0 for x in result["conditions"])


def test_fail_on_incorrect_training_accounting():
    class DropPairs(Memory):
        def train(self, item, role, cue, target):
            self.training_calls += 1
    with pytest.raises(AssertionError, match="accounting"):
        run_conditions(Config(contents=4), DropPairs)


def test_invalid_condition_rejected():
    with pytest.raises(ValueError):
        _teacher_labels(Dataset(Config()), "invalid")


def test_fingerprint_detects_mutating_prediction():
    class Mutating(Memory):
        def __init__(self, cfg, condition):
            super().__init__(cfg, condition)
            self.ledger = 0

        def state_fingerprint(self):
            return str(self.ledger)

        def predict(self, item, role, cue):
            self.ledger += 1
            return super().predict(item, role, cue)

    with pytest.raises(AssertionError, match="read mutated synapses"):
        run_conditions(Config(contents=4), Mutating)


def test_fingerprint_reports_unchanged_state():
    class Frozen(Memory):
        def state_fingerprint(self):
            return str(sorted(
                (key, tuple(value)) for key, value in self.saved.items()))

    result = run_conditions(Config(contents=4), Frozen)
    for condition in result["conditions"]:
        assert condition["weight_audit_available"] is True
        assert isinstance(condition["weight_digest_after_training"], str)


def test_paired_control_factory_receives_only_public_model_seed():
    from research.multimodal.zero_shot_gate import ModelConfig
    observed = []
    class Spy(Memory):
        def __init__(self, public, condition):
            assert isinstance(public, ModelConfig)
            assert public.seed == 901
            assert not hasattr(public, "model_seed")
            observed.append((condition, public))
            super().__init__(public, condition)

    run_conditions(Config(seed=77, model_seed=901, contents=4), Spy)
    assert [condition for condition, _ in observed] == list(CONDITIONS)


def test_model_seed_does_not_change_teacher_training_labels():
    a = run_conditions(Config(seed=37, model_seed=911, contents=4), Memory)
    b = run_conditions(Config(seed=37, model_seed=912, contents=4), Memory)
    assert [x["train_labels_sha256"] for x in a["conditions"]] == [
        x["train_labels_sha256"] for x in b["conditions"]]
