"""R1d protocol tests. Mock tests do not validate numerical Brain behavior."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from research.multimodal import role_recombination as R


@pytest.mark.parametrize("cfg", [
    R.Protocol(n=20, k=20), R.Protocol(p=0),
    R.Protocol(beta=1), R.Protocol(contents=3),
    R.Protocol(contents=5), R.Protocol(roles=("S", "S")),
    R.Protocol(seeds=(1, 1)), R.Protocol(lex_rounds=0),
])
def test_invalid(cfg):
    with pytest.raises(ValueError):
        cfg.validate()


def test_split_proves_held_out_cells_are_not_held_out_entities():
    train, test = R.split(8, ("SUBJ", "OBJ"))
    assert not set(train).intersection(test)
    assert len(train) == len(test) == 8
    for role in ("SUBJ", "OBJ"):
        assert sum(r == role for _, r in train) == 4
        assert sum(r == role for _, r in test) == 4
    for item in range(8):
        assert sum(i == item for i, _ in train) == 1
        assert sum(i == item for i, _ in test) == 1


def test_rank_is_role_local_and_chance_is_real_n():
    def a(ids):
        return SimpleNamespace(neuron_ids=np.array(ids, dtype=np.uint32))
    references = {0: a([1, 2, 3]), 1: a([4, 5, 6])}
    good = R.score(a([4, 5, 6]), references, 1, 3, 30)
    assert good["top1"] == 1
    assert good["overlap"] == 1.
    assert good["chance_overlap"] == .1
    bad = R.score(a([1, 2, 3]), references, 1, 3, 30)
    assert bad["top1"] == 0


def test_native_api_schedule_is_not_train_on_test_before_one_shot(monkeypatch):
    events = []
    class FakeBrain:
        def __init__(self, **kwargs):
            self.events = events
            self.weights = {}
            self.frozen_depth = 0
        def add_area(self, *args, **kwargs): pass
        def add_stimulus(self, *args, **kwargs): pass
        def clear_activity(self, *args, **kwargs): pass
        @contextmanager
        def frozen(self):
            self.frozen_depth += 1
            try:
                yield self
            finally:
                self.frozen_depth -= 1

    def ass(ids):
        return SimpleNamespace(neuron_ids=np.asarray(ids, dtype=np.uint32))

    def project(brain, stimulus, area, rounds, recurrent):
        i = int(stimulus.split("_")[1])
        events.append(("lexical", i))
        return ass([i * 3 + v for v in range(3)])

    def bind(brain, src, role, source_assembly, tail_rounds):
        assert src == "LEX"
        assert brain.frozen_depth == 0
        key = (int(source_assembly.neuron_ids[0] // 3), role)
        events.append(("bind", key))
        # Distinct deterministic role-local assembly
        code = [key[0] * 3 + v for v in range(3)]
        brain.weights[key] = ass(code)
        return brain.weights[key]

    def read_binding(brain, src, role, source_assembly, tail_rounds):
        assert brain.frozen_depth
        key = (int(source_assembly.neuron_ids[0] // 3), role)
        events.append(("read", key))
        return brain.weights.get(key, ass([100, 101, 102]))

    monkeypatch.setattr(R, "_native", lambda: (FakeBrain, project, bind, read_binding))
    cfg = R.Protocol(n=160, k=3, contents=4, seeds=(0,))
    result = R.run(cfg)
    run = result["runs"][0]
    train, heldout = R.split(4, cfg.roles)
    assert run["familiar_before_top1"] == 1.
    assert run["heldout_after_one_binding_top1"] == 1.
    assert run["familiar_after_top1"] == 1.
    for pair in heldout:
        index = events.index(("bind", pair))
        assert any(j < index and event == ("read", pair)
                   for j, event in enumerate(events))
    assert [event for event in events if event[0] == "bind"] == [
        ("bind", pair) for pair in train + heldout]
