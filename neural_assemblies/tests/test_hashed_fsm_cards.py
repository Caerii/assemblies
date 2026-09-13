"""Card F regressions for the assigned-state machine (`HashedArcFSM`).

Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-transition-machine

F1: the argmax label read out of STATE and all-k exact recovery are two
different quantities; a state with one intruder still labels correctly. The
register's exactness claims (SEQ-EXACT-RECOVERY) must use the latter, which
is what the A1 runner records as `exact_fraction`.

F2: `run()` masks the OUTPUT at idle (negative) symbol positions after
calling `step()`; it does not skip the step. A padded position therefore
still conjoins and advances the machine.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neural_assemblies.core.torch_engine import _fused_cuda                # noqa: E402
from neural_assemblies.programs.mod3_fsm import (                          # noqa: E402
    ALL_STATES, ALL_SYMBOLS, mod3_transition_table)

pytestmark = pytest.mark.gpu

N_ARC, N_STATE, K, P, BETA, REFR = 512, 128, 20, 0.3, 0.1, 0.1


@pytest.fixture(scope="module")
def mod():
    m = _fused_cuda.load()
    if m is None:
        pytest.skip(f"fused kernels unavailable: {_fused_cuda.last_error()}")
    return m


def _machine(seeds):
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
    fsm = HashedArcFSM(seeds, ALL_STATES, ALL_SYMBOLS, mod3_transition_table(),
                       n_arc=N_ARC, n_state=N_STATE, k=K, p=P, beta=BETA,
                       refracted_strength=REFR, w_max=20.0, norm_init=False,
                       max_potentiations=256, prefix="_cardF")
    fsm.train(presentations=2)
    return fsm


def test_f1_label_readout_tolerates_the_intruder_that_exactness_rejects(mod):
    fsm = _machine([7])
    fsm.cue_state(ALL_STATES[0])
    block = fsm.blocks[0]
    winners = fsm.state.winners.clone()
    assert torch.equal(torch.sort(winners[0]).values, block)
    # one member replaced by the first neuron of another block
    winners[0, 0] = fsm.blocks[1][0]
    fsm.state.winners = winners
    label = int(fsm.read_state()[0])
    exact = bool(torch.equal(torch.sort(fsm.state.winners[0]).values, block))
    assert label == 0, "argmax label must still name the majority block"
    assert not exact, "one intruder is not exact recovery"


def test_f2_idle_symbols_are_masked_in_the_output_not_skipped(mod, monkeypatch):
    fsm = _machine([7])
    stepped = []
    original = fsm.step

    def observed(symbol, freeze=True):
        stepped.append(int(symbol[0]))
        return original(symbol, freeze=freeze)

    monkeypatch.setattr(fsm, "step", observed)
    idx = fsm.symbol_index
    symbols = torch.tensor([[idx[ALL_SYMBOLS[3]], -1, idx[ALL_SYMBOLS[4]]]],
                           dtype=torch.int64, device=fsm.device)
    out = fsm.run(symbols, ALL_STATES[0])
    assert stepped == [idx[ALL_SYMBOLS[3]], -1, idx[ALL_SYMBOLS[4]]]
    assert int(out[0, 1]) == -1, "the idle position is masked in the output"
    assert int(out[0, 0]) >= 0 and int(out[0, 2]) >= 0
