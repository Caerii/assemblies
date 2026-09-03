"""The hashed aligner reproduces the numpy learner -- on the DRIVE.

Two arbiters, in the order the repo trusts them:

1. DRIVE REPLAY. The numpy engine runs the learner's exact projection pattern
   with `record_activation`; its winner trajectory is replayed through the
   hashed fibers and the pre-k-WTA drive is compared every round on both
   areas. Stimulus bases are injected from the engine (its stimulus
   connectomes are drawn in rng order); the LEX -> FEAT fiber is generated
   from the engine's own pair seed. A test both implementations pass on
   winner sets verifies neither ([[capacity-depends-on-n-over-k]]).

2. DECISIONS. `HashedAligner.train` run end to end on the same tiny corpus
   must align every word the numpy learner aligns.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine import _seeding

torch = pytest.importorskip("torch")
from neural_assemblies.core.torch_engine import _fused_cuda  # noqa: E402


@pytest.fixture(scope="module")
def mod():
    m = _fused_cuda.load()
    if m is None:
        pytest.skip(f"fused kernels unavailable: {_fused_cuda.last_error()}")
    return m


# UNCLIPPED: with column scaling on FEAT the clip and the scale do not
# commute, so the exact regime -- and the aligner's -- is w_max=None.
N, K, P, BETA, W_MAX, ROUNDS, SEED = 512, 20, 0.1, 0.1, None, 3, 7
WORDS = ["dog", "ball", "cat", "book"]
BUNDLES = {"dog": ("ANIMAL", "DOG"), "cat": ("ANIMAL", "CAT"),
           "ball": ("BALL", "OBJECT"), "book": ("BOOK", "OBJECT")}
FEATURES = sorted({f for b in BUNDLES.values() for f in b})
# THE COMPLETE PAIRING, by construction: every referent appears with every
# other exactly once per pass, so the only features that ALWAYS accompany a
# word are its own bundle's. The first toy paired each animal with each
# object only, which made ANIMAL a 100% co-occurrent of "ball" -- a corpus
# error the learner was then blamed for. `assert_identifiable` guards it.
_PAIRS = [("dog", "cat"), ("ball", "book"), ("dog", "ball"),
          ("dog", "book"), ("cat", "ball"), ("cat", "book")]
SCENES = [([a, b], [BUNDLES[a], BUNDLES[b]]) for a, b in _PAIRS]


def _to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _numpy_learner():
    random.seed(SEED)
    np.random.seed(SEED)
    b = Brain(p=P, seed=SEED, engine="numpy_sparse", w_max=W_MAX,
              norm_init=True, synaptic_scaling=frozenset({"FEAT"}))
    b.add_area("LEX", N, K, BETA)
    b.add_area("FEAT", N, K, BETA)
    b.materialize_area("LEX")
    b.materialize_area("FEAT")
    for w in WORDS:
        b.add_stimulus(f"phon_{w}", K)
    for f in FEATURES:
        b.add_stimulus(f"feat_{f}", K)
    return b


def test_the_toy_is_identifiable():
    """The corpus documents its own control (see SCENES)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                    "research", "experiments"))
    from unaligned_scenes import assert_identifiable
    assert assert_identifiable(SCENES, BUNDLES)


def test_hashed_aligner_reproduces_numpy_drive(mod):
    from neural_assemblies.core.torch_engine._hashed_aligner import HashedAligner

    b = _numpy_learner()
    eng = b._engine_for(b.areas["LEX"])
    stim0 = {}
    for w in WORDS:
        stim0[f"phon_{w}"] = np.asarray(
            eng._stim_conns[f"phon_{w}"]["LEX"].weights, dtype=np.float64).copy()
    for f in FEATURES:
        stim0[f"feat_{f}"] = np.asarray(
            eng._stim_conns[f"feat_{f}"]["FEAT"].weights, dtype=np.float64).copy()

    # --- numpy trace: the learner's projection pattern, target by target ---
    trace = []      # (kind, word, bundle, lex_prev, d_lex, new_lex, d_feat, new_feat)
    def winners(area):
        return np.asarray(eng.get_winners(area), dtype=np.int64)
    for words, bundles in SCENES:
        for w in words:
            for bun in bundles:
                b.inhibit_areas(["LEX", "FEAT"])
                feats = [f"feat_{f}" for f in bun]
                r = eng.project_into("LEX", [f"phon_{w}"], [],
                                     plasticity_enabled=True,
                                     record_activation=True)
                d_lex = np.asarray(r.pre_kwta_inputs, dtype=np.float64)
                rf = eng.project_into("FEAT", feats, [],
                                      plasticity_enabled=True,
                                      record_activation=True)
                trace.append(("stim", w, bun, None, d_lex, winners("LEX"),
                              np.asarray(rf.pre_kwta_inputs, dtype=np.float64),
                              winners("FEAT")))
                for _ in range(ROUNDS):
                    lex_prev = winners("LEX")
                    # FEAT first (reads LEX's pre-round winners), then LEX:
                    # equals Brain.project's batched simultaneous update.
                    rf = eng.project_into("FEAT", feats, ["LEX"],
                                          plasticity_enabled=True,
                                          record_activation=True)
                    d_feat = np.asarray(rf.pre_kwta_inputs, dtype=np.float64)
                    new_feat = winners("FEAT")
                    r = eng.project_into("LEX", [f"phon_{w}"], [],
                                         plasticity_enabled=True,
                                         record_activation=True)
                    trace.append(("cross", w, bun, lex_prev,
                                  np.asarray(r.pre_kwta_inputs, dtype=np.float64),
                                  winners("LEX"), d_feat, new_feat))

    # --- hashed replay with the engine's stimulus bases injected -----------
    al = HashedAligner([SEED], WORDS, FEATURES, n=N, k=K, feat_n=N, feat_k=K,
                       p=P, beta=BETA, w_max=W_MAX, rounds_word=ROUNDS,
                       stim_beta=BETA,      # the engine potentiates stimuli
                       stim_gain=1.0)       # and has no anchor gain
    for name, sf in list(al.phon.items()) + list(al.featf.items()):
        key = (f"phon_{name}" if name in al.phon and sf is al.phon[name]
               else f"feat_{name}")
        base = torch.from_numpy(stim0[key].astype(np.float32)).cuda().view(1, -1)
        sf.base = base
        sf.dj = (base + P * (N - K)).clamp_min(1.0)

    def drive(area, fibers, rows):
        d = torch.zeros(1, N, dtype=torch.float32, device="cuda")
        for f in fibers:
            f.contribute(d, rows)
        return d

    worst = 0.0
    first = []
    def cmp(ref, got, tag=""):
        nonlocal worst
        got = got[0].cpu().numpy().astype(np.float64)
        m = min(len(ref), len(got))
        err = float(np.abs(ref[:m] - got[:m]).max()) / max(
            float(np.abs(ref[:m]).max()), 1e-12)
        if err > 1e-5 and len(first) < 4:
            j = int(np.argmax(np.abs(ref[:m] - got[:m])))
            first.append(f"{tag}: rel {err:.3g} at col {j} ref {ref[j]:.5g} "
                         f"got {got[j]:.5g}; nonzero-diff cols "
                         f"{int((np.abs(ref[:m]-got[:m]) > 1e-6).sum())}/{m}")
        worst = max(worst, err)

    empty = torch.zeros(1, 0, dtype=torch.int64, device="cuda")
    for kind, w, bun, lex_prev, d_lex, new_lex, d_feat, new_feat in trace:
        phon, stims = al.phon[w], [al.featf[f] for f in bun]
        nl = torch.from_numpy(new_lex).cuda().view(1, -1)
        nf = torch.from_numpy(new_feat).cuda().view(1, -1)
        if kind == "stim":
            cmp(d_lex, drive("LEX", [phon], empty), f"stim LEX {w}")
            cmp(d_feat, drive("FEAT", stims, empty), f"stim FEAT {bun}")
            phon.begin_episode(); phon.observe(empty, nl); phon.end_episode()
            for s in stims:
                s.begin_episode(); s.observe(empty, nf); s.end_episode()
        else:
            lp = torch.from_numpy(lex_prev).cuda().view(1, -1)
            cmp(d_lex, drive("LEX", [phon], empty), f"cross LEX {w}")
            al.cross.begin_episode()
            d = torch.zeros(1, N, dtype=torch.float32, device="cuda")
            for s in stims:
                s.contribute(d, lp)
            al.cross.contribute(d, lp)
            cmp(d_feat, d, f"cross FEAT {w}/{bun}")
            phon.begin_episode(); phon.observe(empty, nl); phon.end_episode()
            for s in stims:
                s.begin_episode(); s.observe(lp, nf); s.end_episode()
            al.cross.observe(lp, nf)
            al.cross.end_episode()
    assert al.cross.nnz > 0, "the cross fiber never learned -- vacuous"
    assert worst < 5e-6, (
        f"hashed aligner diverges from numpy_sparse on the drive: relative "
        f"error {worst:.3g}; first divergences: " + " | ".join(first))


def test_hashed_aligner_aligns_the_identifiable_toy(mod):
    """End to end against GROUND TRUTH: on the complete-pairing toy at the
    U1 operating point, every brain aligns every word to its own bundle.

    Not a comparison with the numpy learner: the two substrates differ in
    stimulus model (binomial counts vs 0-or-size) and so in tie structure, and
    a winner-level agreement between them is a tie comparison. The drive
    replay above is the parity arbiter; this pins the LOOP -- one cross
    episode per step, anchored stimuli, unclipped relative pricing -- on what
    the registration actually asks of it.
    """
    from neural_assemblies.core.torch_engine._hashed_aligner import HashedAligner

    inventory = sorted(set(BUNDLES.values()))
    al = HashedAligner([SEED, SEED + 1, SEED + 2, SEED + 3], WORDS, FEATURES,
                       n=1000, k=50, feat_n=1000, feat_k=50, p=0.05,
                       beta=BETA, rounds_word=5)
    al.train(SCENES * 6, random.Random(SEED + 11))
    table = al.overlap_table(WORDS, inventory)            # [V, I, B]
    decisions = table.argmax(dim=1).cpu().numpy()         # [V, B]
    wrong = [(w, b, inventory[int(decisions[vi, b])])
             for vi, w in enumerate(WORDS) for b in range(4)
             if inventory[int(decisions[vi, b])] != BUNDLES[w]]
    assert not wrong, f"misaligned (word, brain, got): {wrong}"
