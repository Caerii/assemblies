"""GATES 1 and 4 of DESIGN_sequence_port.md.

GATE-1: the hashed transducer reproduces the numpy organ ON THE DRIVE,
refraction included. The numpy `SequenceTransducer` runs `ground` and
`train_sentence` on a tiny vocabulary with `record_activation`; its winner
trajectory is replayed through the hashed fibers -- the engine's stimulus
connectomes injected, the four area fibers generated from the engine's own
pair seeds at the organ's density -- and the NET drive (after the
refraction bias on ARC) is compared on every projection into LEX, ARC,
STATE and OUT. At the end the accumulated ARC bias is compared too. A test
both implementations pass on winner sets would verify neither
([[capacity-depends-on-n-over-k]]).

GATE-4: identity across width -- a brain trained alone equals the same
brain trained in a launch beside brains on different schedules, winner for
winner, and an idle step is a no-op.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neural_assemblies.core.brain import Brain                             # noqa: E402
from neural_assemblies.core.torch_engine import _fused_cuda                # noqa: E402
from neural_assemblies.programs.sequence_transducer import SequenceTransducer  # noqa: E402
from neural_assemblies.tests import _parity_dump


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


def _hashed(seeds, tie_jitter=0.0):
    from neural_assemblies.core.torch_engine._hashed_transducer import HashedTransducer
    return HashedTransducer(seeds, VOCAB, n=N, k=K, p=P, beta=BETA, organ_p=ORGAN_P,
                            refracted_strength=REFR, w_max=W_MAX, norm_init=True,
                            max_potentiations=512, tie_jitter=tie_jitter)


def test_hashed_transducer_reproduces_numpy_drive(mod):
    b, t = _numpy_organ()
    eng = b._engine_for(b.areas[t.lex_area])
    LEX, ARC, ST, OUT = t.lex_area, t.arc_area, t.state_area, t.out_area

    def winners(area):
        return np.asarray(eng.get_winners(area), dtype=np.int64)

    def proj(target, stims, froms):
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
                trace.append(proj(ST, [], [ARC]))
                trace.append(proj(OUT, [t._g_stim[nxt]], [ARC]))
    bias_engine = np.asarray(eng._areas[ARC]._cumulative_bias, dtype=np.float64)
    assert float(np.abs(bias_engine).max()) > 0, "the engine never charged ARC"

    h = _hashed([SEED])
    for i, w in enumerate(VOCAB):
        for stack, key in ((h.S, t._s_stim[w]), (h.G, t._g_stim[w])):
            base = torch.from_numpy(stim0[key].astype(np.float32)).cuda()
            stack.base[i, 0] = base
            stack.dj[i, 0] = (base + P * (N - K)).clamp_min(1.0)
    areas = {LEX: h.lex, ARC: h.arc, ST: h.state, OUT: h.out}
    stims = {**{t._s_stim[w]: (h.S, i) for i, w in enumerate(VOCAB)},
             **{t._g_stim[w]: (h.G, i) for i, w in enumerate(VOCAB)}}
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
            f.contribute(raw, rows)
            active.append((f, rows))
        for sname in snames:
            stack, i = stims[sname]
            stack.set_words(torch.tensor([i], device="cuda"))
            stack.contribute(raw)
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
        for sname in snames:
            stack, i = stims[sname]
            stack.set_words(torch.tensor([i], device="cuda"))
            stack.observe(empty, nt)
        area.charge(raw, nt)
        area.winners = nt
    got_bias = h.arc.bias[0].cpu().numpy().astype(np.float64)
    m = min(len(bias_engine), len(got_bias))
    bias_err = float(np.abs(bias_engine[:m] - got_bias[:m]).max()) / max(
        float(np.abs(bias_engine[:m]).max()), 1e-12)
    assert h.arc_state.store.max_count > 0, "ARC -> STATE never learned -- vacuous"
    _parity_dump.record("transducer, drive", worst)
    _parity_dump.record("transducer, bias", bias_err)
    assert worst < 5e-6, ("hashed transducer diverges from numpy_sparse on the "
                          f"drive: relative error {worst:.3g}; first: " + " | ".join(first))
    assert bias_err < 5e-6, f"ARC refraction bias diverges: {bias_err:.3g}"


def _schedule(sentences, wi, S):
    """One brain's (words, targets, starts) from its sentences, padded to S."""
    W, T, St = [], [], []
    for s in sentences:
        for j, (a, nxt) in enumerate(zip(s, s[1:])):
            W.append(wi[a]); T.append(wi[nxt]); St.append(j == 0)
    W += [-1] * (S - len(W)); T += [-1] * (S - len(T)); St += [False] * (S - len(St))
    return W, T, St


def test_a_brain_in_a_launch_equals_the_brain_alone(mod):
    """GATE-4: winners of every area, and the OUT signatures, are identical
    whether brain 7 trains alone on its sentences or in a launch of three
    brains on different schedules (with idle steps)."""
    wi = {w: i for i, w in enumerate(VOCAB)}
    other = [["d", "c", "b", "a", "d"], ["c", "a"]]
    per = [SENTENCES, other, [["b", "a", "d"]]]
    S = max(sum(len(s) - 1 for s in p_) for p_ in per)
    sched = [_schedule(p_, wi, S) for p_ in per]
    W = torch.tensor([s[0] for s in sched]); T = torch.tensor([s[1] for s in sched])
    St = torch.tensor([s[2] for s in sched])

    def final(h):
        return {nm: a.winners.clone() for nm, a in
                (("lex", h.lex), ("arc", h.arc), ("state", h.state), ("out", h.out))}

    alone = _hashed([SEED], tie_jitter=1e-6)
    alone.ground(rounds=G_ROUNDS)
    alone.train_schedules(W[:1], T[:1], St[:1], rounds=ROUNDS)
    fa = final(alone)

    batch = _hashed([SEED, 11, 12], tie_jitter=1e-6)
    batch.ground(rounds=G_ROUNDS)
    batch.train_schedules(W, T, St, rounds=ROUNDS)
    fb = final(batch)
    for nm in fa:
        assert torch.equal(fa[nm][0], fb[nm][0]), f"{nm} winners differ across width"
    for w in VOCAB:
        assert torch.equal(alone.out_signature[w][0], batch.out_signature[w][0])
    torch.testing.assert_close(batch.arc.bias[0], alone.arc.bias[0], rtol=0, atol=0)
    assert torch.equal(batch.lex_arc.C[0], alone.lex_arc.C[0])
