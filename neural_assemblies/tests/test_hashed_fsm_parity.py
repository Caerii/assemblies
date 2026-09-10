"""GATE-1 and GATE-4 of DESIGN_sequence_port.md for the ASSIGNED-STATE
organ (`HashedArcFSM` against `programs/nemo_fsm.NemoArcFSM`).

FSM-1, DRIVE REPLAY: the numpy machine trains the mod-3 table and runs a
digit string with `record_activation`; its winner trajectory is replayed
through the hashed fibers -- the engine's symbol connectomes injected, the
two area fibers generated from the engine's own pair seeds -- and the NET
drive on ARC (after the refraction bias) and on STATE is compared on every
projection that computes one, the accumulated ARC bias at the end. The
teacher-forced write into the FIXED state block computes no drive in
either implementation; its effect is read through the run-phase STATE
drives.

FSM-4, IDENTITY ACROSS WIDTH: a brain run alone equals the same brain in
a launch of three on different digit strings, state for state.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neural_assemblies.core.brain import Brain                             # noqa: E402
from neural_assemblies.core.torch_engine import _fused_cuda                # noqa: E402
from neural_assemblies.programs.mod3_fsm import (                          # noqa: E402
    ALL_STATES, ALL_SYMBOLS, build_mod3_fsm, mod3_transition_table)
from neural_assemblies.tests import _parity_dump


@pytest.fixture(scope="module")
def mod():
    m = _fused_cuda.load()
    if m is None:
        pytest.skip(f"fused kernels unavailable: {_fused_cuda.last_error()}")
    return m


N_ARC, N_STATE, K, P, BETA, REFR, SEED = 512, 128, 20, 0.3, 0.1, 0.1, 7
PRESENTATIONS = 3
DIGITS = [3, 0, 4, 7, 1, 8, 2, 2, 9, 5, 6, 1]


def _numpy_fsm():
    random.seed(SEED)
    np.random.seed(SEED)
    b = Brain(p=P, save_winners=True, seed=SEED, engine="numpy_sparse", norm_init=False)
    fsm = build_mod3_fsm(b, n=N_ARC, k=K, n_state=N_STATE, beta=BETA,
                         refracted_strength=REFR)
    b.materialize_area(fsm.arc_area)
    return b, fsm


def _hashed(seeds, w_max, tie_jitter=0.0):
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
    return HashedArcFSM(seeds, ALL_STATES, ALL_SYMBOLS, mod3_transition_table(),
                        n_arc=N_ARC, n_state=N_STATE, k=K, p=P, beta=BETA,
                        refracted_strength=REFR, w_max=w_max, norm_init=False,
                        max_potentiations=256, tie_jitter=tie_jitter,
                        prefix="_mod3")   # the engine's fiber seeds are by NAME


def test_hashed_fsm_reproduces_numpy_drive(mod):
    b, fsm = _numpy_fsm()
    ARC, ST = fsm.arc_area, fsm.state_area
    eng = b._engine_for(b.areas[ARC])

    def winners(area):
        return np.asarray(eng.get_winners(area), dtype=np.int64)

    def proj(target, stims, froms, plastic):
        prev = {a: winners(a) for a in froms}
        r = eng.project_into(target, stims, froms, plasticity_enabled=plastic,
                             record_activation=True)
        return (target, tuple(stims), prev,
                np.asarray(r.pre_kwta_inputs, dtype=np.float64), winners(target),
                plastic)

    stim0 = {s: np.asarray(eng._stim_conns[fsm._sym_stim[s]][ARC].weights,
                           dtype=np.float64).copy() for s in ALL_SYMBOLS}
    trace = []
    for _ in range(PRESENTATIONS):                              # train
        for fr, sym, to in mod3_transition_table():
            b.inhibit_areas([ARC, ST])
            fsm._cue_state(fr)
            trace.append(proj(ARC, [fsm._sym_stim[sym]], [ST], True))
            fsm._unfix_state()
            fsm._cue_state(to)
            eng.project_into(ST, [], [ARC], plasticity_enabled=True)   # fixed target
            trace.append(("write", None, winners(ARC), None, winners(ST), False))
            fsm._unfix_state()
    bias_engine = np.asarray(eng._areas[ARC]._cumulative_bias, dtype=np.float64)
    assert float(np.abs(bias_engine).max()) > 0, "the engine never charged ARC"
    labels_ref = []
    with b.probe():                                             # run
        b.inhibit_areas([ARC, ST])
        fsm._cue_state("0")
        fsm._unfix_state()
        for d in DIGITS:
            trace.append(proj(ARC, [fsm._sym_stim[str(d)]], [ST], False))
            trace.append(proj(ST, [], [ARC], False))
            labels_ref.append(fsm.read_state())

    h = _hashed([SEED], b.w_max)
    for i, s in enumerate(ALL_SYMBOLS):
        h.sym.base[i, 0] = torch.from_numpy(stim0[s].astype(np.float32)).cuda()
    areas = {ARC: h.arc, ST: h.state}
    fibers = {(ST, ARC): h.state_arc, (ARC, ST): h.arc_state}
    sym_of = {fsm._sym_stim[s]: i for i, s in enumerate(ALL_SYMBOLS)}
    empty = torch.zeros(1, 0, dtype=torch.int64, device="cuda")
    cuda = lambda a: torch.from_numpy(np.asarray(a, dtype=np.int64)).cuda().view(1, -1)  # noqa: E731

    worst, first, n_compared = 0.0, [], 0
    for target, snames, prev, d_ref, new, plastic in trace:
        if target == "write":
            rows, nt = cuda(prev), cuda(new)
            h.arc_state.begin_episode()
            h.arc_state.observe(rows, nt)
            h.arc_state.end_episode()
            h.state.winners = nt
            continue
        area = areas[target]
        raw = torch.zeros(1, area.n, dtype=torch.float32, device="cuda")
        active = []
        for src, pw in prev.items():
            f = fibers[(src, target)]
            rows = cuda(pw) if len(pw) else empty
            f.contribute(raw, rows)
            active.append((f, rows))
        for sname in snames:
            h.sym.set_words(torch.tensor([sym_of[sname]], device="cuda"))
            h.sym.contribute(raw)
        net = area.apply_bias(raw)
        got = net[0].cpu().numpy().astype(np.float64)
        m = min(len(d_ref), len(got))
        err = float(np.abs(d_ref[:m] - got[:m]).max()) / max(
            float(np.abs(d_ref[:m]).max()), 1e-12)
        n_compared += 1
        if err > 1e-5 and len(first) < 4:
            j = int(np.argmax(np.abs(d_ref[:m] - got[:m])))
            first.append(f"{target} <- {list(prev)}+{list(snames)}: rel {err:.3g} "
                         f"at {j} ref {d_ref[j]:.5g} got {got[j]:.5g}")
        worst = max(worst, err)
        nt = cuda(new)
        if plastic:
            for f, rows in active:
                f.observe(rows, nt)
            for sname in snames:
                h.sym.set_words(torch.tensor([sym_of[sname]], device="cuda"))
                h.sym.observe(empty, nt)
            area.charge(raw, nt)
        area.winners = nt
    got_bias = h.arc.bias[0].cpu().numpy().astype(np.float64)
    m = min(len(bias_engine), len(got_bias))
    bias_err = float(np.abs(bias_engine[:m] - got_bias[:m]).max()) / max(
        float(np.abs(bias_engine[:m]).max()), 1e-12)
    assert n_compared == PRESENTATIONS * 33 + 2 * len(DIGITS)
    assert h.arc_state.store.max_count > 0, "ARC -> STATE never learned -- vacuous"
    _parity_dump.record("transition machine, drive", worst)
    _parity_dump.record("transition machine, bias", bias_err)
    assert worst < 5e-6, ("hashed FSM diverges from numpy_sparse on the drive: "
                          f"relative error {worst:.3g}; first: " + " | ".join(first))
    assert bias_err < 5e-6, f"ARC refraction bias diverges: {bias_err:.3g}"


def test_a_brain_in_a_launch_equals_the_brain_alone(mod):
    """FSM-4: the trajectory of brain 7 on its digit string is the same
    alone and beside two brains on other strings (one shorter, padded -1)."""
    rng = random.Random(3)
    strings = [[rng.randrange(10) for _ in range(40)] for _ in range(3)]
    strings[2] = strings[2][:25]
    L = max(len(s) for s in strings)
    sym = torch.full((3, L), -1, dtype=torch.int64)
    for bi, s in enumerate(strings):
        sym[bi, :len(s)] = torch.tensor(s)
    sym = sym.cuda()

    alone = _hashed([SEED], 20.0, tie_jitter=1e-6)
    alone.train(PRESENTATIONS)
    got_alone = alone.run(sym[:1], "0")
    wide = _hashed([SEED, SEED + 1, SEED + 2], 20.0, tie_jitter=1e-6)
    wide.train(PRESENTATIONS)
    got_wide = wide.run(sym, "0")
    assert torch.equal(got_alone[0], got_wide[0])
    assert bool((got_wide[2, 25:] == -1).all())
    assert torch.equal(alone.arc.bias[0], wide.arc.bias[0])
