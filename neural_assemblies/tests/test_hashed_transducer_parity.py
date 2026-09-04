"""GATE-1 of DESIGN_sequence_port.md: the hashed transducer reproduces the
numpy organ ON THE DRIVE, refraction included.

The numpy `SequenceTransducer` runs `ground` and `train_sentence` on a tiny
vocabulary with `record_activation`; its winner trajectory is replayed
through the hashed fibers -- the engine's stimulus connectomes injected,
the four area fibers generated from the engine's own pair seeds at the
organ's density -- and the NET drive (after the refraction bias on ARC) is
compared on every projection into LEX, ARC, STATE and OUT. At the end the
accumulated ARC bias is compared too. A test both implementations pass on
winner sets would verify neither ([[capacity-depends-on-n-over-k]]).
"""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neural_assemblies.core.brain import Brain                             # noqa: E402
from neural_assemblies.core.torch_engine import _fused_cuda                # noqa: E402
from neural_assemblies.programs.sequence_transducer import SequenceTransducer  # noqa: E402


@pytest.fixture(scope="module")
def mod():
    m = _fused_cuda.load()
    if m is None:
        pytest.skip(f"fused kernels unavailable: {_fused_cuda.last_error()}")
    return m


N, K, P, ORGAN_P, BETA, W_MAX, REFR, SEED = 512, 20, 0.1, 0.3, 0.1, 20.0, 0.1, 7
VOCAB = ["a", "b", "c", "d"]
SENTENCES = [["a", "b", "c"], ["b", "c", "d"], ["a", "c", "d", "b"]]
G_ROUNDS, ROUNDS = 3, 2


def _numpy_organ():
    random.seed(SEED)
    np.random.seed(SEED)
    b = Brain(p=P, seed=SEED, engine="numpy_sparse", w_max=W_MAX, norm_init=True)
    t = SequenceTransducer(b, VOCAB, n=N, k=K, beta=BETA, organ_p=ORGAN_P,
                           refracted_strength=REFR)
    for a in (t.lex_area, t.arc_area, t.state_area, t.out_area):
        b.materialize_area(a)
    return b, t


def test_hashed_transducer_reproduces_numpy_drive(mod):
    from neural_assemblies.core.torch_engine._hashed_transducer import HashedTransducer

    b, t = _numpy_organ()
    eng = b._engine_for(b.areas[t.lex_area])
    LEX, ARC, ST, OUT = t.lex_area, t.arc_area, t.state_area, t.out_area

    def winners(area):
        return np.asarray(eng.get_winners(area), dtype=np.int64)

    def proj(target, stims, froms):
        """One engine projection with everything recorded: (target, stims,
        {source: its pre-round winners}, net drive, new winners)."""
        prev = {a: winners(a) for a in froms}
        r = eng.project_into(target, stims, froms, plasticity_enabled=True,
                             record_activation=True)
        return (target, tuple(stims), prev,
                np.asarray(r.pre_kwta_inputs, dtype=np.float64), winners(target))

    stim0 = {}
    for w in VOCAB:
        stim0[t._s_stim[w]] = np.asarray(eng._stim_conns[t._s_stim[w]][LEX].weights,
                                         dtype=np.float64).copy()
        stim0[t._g_stim[w]] = np.asarray(eng._stim_conns[t._g_stim[w]][OUT].weights,
                                         dtype=np.float64).copy()

    # --- the numpy organ's exact projection pattern, recorded ---------------
    trace = []
    for w in VOCAB:                                             # ground
        b.inhibit_areas([LEX, OUT, ARC, ST])
        for _ in range(G_ROUNDS):
            trace.append(proj(LEX, [t._s_stim[w]], []))
        b.inhibit_areas([OUT])
        for _ in range(G_ROUNDS):
            trace.append(proj(OUT, [t._g_stim[w]], []))
    for sent in SENTENCES:                                      # train
        b.inhibit_areas([LEX, ARC, ST, OUT])
        for a, nxt in zip(sent, sent[1:]):
            for _ in range(ROUNDS):
                trace.append(proj(LEX, [t._s_stim[a]], []))
            trace.append(proj(ARC, [], [LEX, ST]))
            for _ in range(ROUNDS):
                # Brain.project computes both targets from the pre-round arc;
                # the arc does not change here, so in sequence is the same
                trace.append(proj(ST, [], [ARC]))
                trace.append(proj(OUT, [t._g_stim[nxt]], [ARC]))
    bias_engine = np.asarray(eng._areas[ARC]._cumulative_bias, dtype=np.float64)
    assert float(np.abs(bias_engine).max()) > 0, "the engine never charged ARC"

    # --- the hashed organ, the engine's stimuli injected --------------------
    h = HashedTransducer([SEED], VOCAB, n=N, k=K, p=P, beta=BETA, organ_p=ORGAN_P,
                         refracted_strength=REFR, w_max=W_MAX, norm_init=True,
                         max_potentiations=len(trace) + 1)
    for w in VOCAB:
        for sf, key in ((h.s[w], t._s_stim[w]), (h.g[w], t._g_stim[w])):
            base = torch.from_numpy(stim0[key].astype(np.float32)).cuda().view(1, -1)
            sf.base = base
            sf.dj = (base + P * (N - K)).clamp_min(1.0)
    areas = {LEX: h.lex, ARC: h.arc, ST: h.state, OUT: h.out}
    stims = {**{t._s_stim[w]: h.s[w] for w in VOCAB},
             **{t._g_stim[w]: h.g[w] for w in VOCAB}}
    fibers = {(LEX, ARC): h.lex_arc, (ST, ARC): h.state_arc,
              (ARC, ST): h.arc_state, (ARC, OUT): h.arc_out}
    empty = torch.zeros(1, 0, dtype=torch.int64, device="cuda")

    worst, first = 0.0, []
    for target, snames, prev, d_ref, new in trace:
        area = areas[target]
        raw = torch.zeros(1, area.n, dtype=torch.float32, device="cuda")
        active = []
        for src, pw in prev.items():
            f = fibers[(src, target)]
            rows = torch.from_numpy(pw).cuda().view(1, -1) if len(pw) else empty
            f.begin_episode()
            f.contribute(raw, rows)
            active.append((f, rows))
        for sname in snames:
            stims[sname].contribute(raw)
        net = area.apply_bias(raw)
        got = net[0].cpu().numpy().astype(np.float64)
        m = min(len(d_ref), len(got))
        err = float(np.abs(d_ref[:m] - got[:m]).max()) / max(
            float(np.abs(d_ref[:m]).max()), 1e-12)
        if err > 1e-5 and len(first) < 4:
            j = int(np.argmax(np.abs(d_ref[:m] - got[:m])))
            first.append(f"{target} <- {list(prev)}+{list(snames)}: rel {err:.3g} "
                         f"at {j} ref {d_ref[j]:.5g} got {got[j]:.5g}")
        worst = max(worst, err)
        nt = torch.from_numpy(new).cuda().view(1, -1)
        for f, rows in active:
            f.observe(rows, nt)
            f.end_episode()
        for sname in snames:
            sf = stims[sname]
            sf.begin_episode(); sf.observe(empty, nt); sf.end_episode()
        area.charge(raw, nt)
        area.winners = nt
    got_bias = h.arc.bias[0].cpu().numpy().astype(np.float64)
    m = min(len(bias_engine), len(got_bias))
    bias_err = float(np.abs(bias_engine[:m] - got_bias[:m]).max()) / max(
        float(np.abs(bias_engine[:m]).max()), 1e-12)
    assert h.arc_state.store.max_count > 0, "ARC -> STATE never learned -- vacuous"
    assert worst < 5e-6, ("hashed transducer diverges from numpy_sparse on the "
                          f"drive: relative error {worst:.3g}; first: " + " | ".join(first))
    assert bias_err < 5e-6, f"ARC refraction bias diverges: {bias_err:.3g}"
