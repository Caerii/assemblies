from dataclasses import asdict, FrozenInstanceError
from types import SimpleNamespace
import sys
import pytest
from research.experiments import seq_a1_horizon_hashed as horizon


def protocol(**changes):
    values = dict(length=8, p_values=[.3], n_arc=200, n_state=60, k=10,
                  beta=.2, presentations=4, strength=.03, norm_init=True,
                  w_max=7., checkpoints=[2,8], device="cuda:0")
    values.update(changes)
    return horizon.HorizonProtocol(**values)


def test_protocol_copies_sequences_and_is_immutable():
    values = [.3]
    config = protocol(p_values=values)
    values.append(.4)
    assert config.p_values == (.3,)
    with pytest.raises(FrozenInstanceError):
        config.beta = 0


@pytest.mark.parametrize("changes", [{"length":0}, {"presentations":True},
    {"beta":float("nan")}, {"strength":-1}, {"norm_init":1}, {"w_max":.5},
    {"n_state":20}, {"k":201}, {"p_values":[.3,.3]}, {"p_values":[True]},
    {"checkpoints":[8,2]}, {"device":""}])
def test_invalid_model_and_schedule_reject(changes):
    with pytest.raises(ValueError):protocol(**changes)


def test_recorded_values_reach_actual_fsm_constructor(monkeypatch):
    calls=[]
    class Stop(Exception): pass
    def construct(*args, **kwargs):
        calls.append(kwargs)
        raise Stop
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    monkeypatch.setitem(sys.modules,"neural_assemblies.core.torch_engine._hashed_fsm",
                        SimpleNamespace(HashedArcFSM=construct))
    config=protocol()
    with pytest.raises(Stop):horizon.run_width([7,13,19],.3,config)
    call=calls[0]
    for key in ("n_arc","n_state","k","beta","w_max","norm_init","device"):
        assert call[key] == getattr(config,key)
    assert call["refracted_strength"] == .03
    assert call["max_potentiations"] == 4*4+8


def test_recorded_probability_grid_controls_calls(monkeypatch,tmp_path):
    path=tmp_path/"baseline.json"
    path.write_text('[{"seed":1,"p":0.3,"first_error":null}]')
    monkeypatch.setattr(horizon,"results_path",lambda *args:str(path))
    calls=[]
    def run(seeds,p,config,organ_semantics):
        calls.append((p,config))
        return [{"seed":s,"p":p,"first_error":None,"prefix_correct":{2:True,8:True}} for s in seeds]
    monkeypatch.setattr(horizon,"run_width",run)
    config=protocol()
    out=horizon.experiment({
        "mode":"smoke", "seeds":[1,2,3], "parameters":asdict(config),
        "execution_semantics": {"profiles": {"default":
            horizon.describe_hashed_arc_fsm(
                w_max=config.w_max, norm_init=config.norm_init,
                refracted_strength=config.strength,
            ).to_dict()}},
    })
    assert calls == [(.3,config)] and out["verdict"] == "VOID"
    with pytest.raises(ValueError,match="no reference"):
        horizon.experiment({"mode":"study","seeds":[1,2,3],"parameters":asdict(protocol(p_values=[.4]))})
    assert len(calls)==1
