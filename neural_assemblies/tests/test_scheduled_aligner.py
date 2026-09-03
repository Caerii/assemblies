"""Layer 1 gate: the scheduled aligner equals the batched one.

Every brain on the SAME corpus, the same step order, the same fiber seeds:
the same kernels run in the same order, so the overlap tables must be
identical -- and then a brain on a DIFFERENT schedule in the same launch
must not disturb them.
"""
from __future__ import annotations

import os
import random
import sys

import pytest
import torch

from neural_assemblies.core.torch_engine import _fused_cuda

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "research", "experiments"))


@pytest.fixture(scope="module")
def mod():
    m = _fused_cuda.load()
    if m is None:
        pytest.skip(f"fused kernels unavailable: {_fused_cuda.last_error()}")
    return m


def _corpus():
    from unaligned_scenes import experience_of
    from grounded_corpus import build
    exp = experience_of(build())
    words = sorted({w for ws, _ in exp for w in ws})
    features = sorted({f for _, bs in exp for b in bs for f in b})
    inventory = sorted({b for _, bs in exp for b in bs})
    return exp, words, features, inventory


def _features_tensor(inventory, features, B):
    fi = {f: i for i, f in enumerate(features)}
    Fper = max(len(b) for b in inventory)
    t = torch.full((len(inventory), Fper), -1, dtype=torch.int64)
    for j, b in enumerate(inventory):
        for s, f in enumerate(b):
            t[j, s] = fi[f]
    return t.unsqueeze(0).expand(B, -1, -1).contiguous()


def test_scheduled_equals_batched_on_one_corpus(mod):
    from neural_assemblies.core.torch_engine._hashed_aligner import HashedAligner
    from neural_assemblies.core.torch_engine._scheduled_aligner import (
        ScheduledAligner, pad_schedules, schedule_of)
    from unaligned_scenes import P, BETA

    exp, words, features, inventory = _corpus()
    seeds = [42, 1, 2]
    order = list(range(len(exp)))
    random.Random(53).shuffle(order)

    # batched reference, trained in exactly `order`
    ref = HashedAligner(seeds, words, features, n=1000, k=50, feat_n=1000,
                        feat_k=50, p=P, beta=BETA, rounds_word=2)
    class _Rng:                                   # replay the same order
        def shuffle(self, lst):
            lst[:] = order
    ref.train(exp, _Rng())
    ref_tab = ref.overlap_table(words, inventory)              # [V, I, B]

    # scheduled, every brain on the same corpus and order
    wi = {w: i for i, w in enumerate(words)}
    bi = {b: j for j, b in enumerate(inventory)}
    sched = [schedule_of(exp, wi, bi, order)] * len(seeds)
    W, Bd = pad_schedules(sched)
    al = ScheduledAligner(seeds, n=1000, k=50, feat_n=1000, feat_k=50,
                          n_words=len(words), n_features=len(features),
                          word_names=[f"phon_{w}" for w in words],
                          feature_names=[f"feat_{f}" for f in features],
                          p=P, beta=BETA, rounds_word=2)
    al.prepare(_features_tensor(inventory, features, len(seeds)))
    al.train(W, Bd)
    tab = al.overlap_table()                                   # [B, V, I]
    torch.testing.assert_close(tab.permute(1, 2, 0), ref_tab, rtol=0, atol=0)


def test_a_different_schedule_in_the_same_launch_does_not_disturb(mod):
    from neural_assemblies.core.torch_engine._scheduled_aligner import (
        ScheduledAligner, pad_schedules, schedule_of)
    from unaligned_scenes import P, BETA

    exp, words, features, inventory = _corpus()
    wi = {w: i for i, w in enumerate(words)}
    bi = {b: j for j, b in enumerate(inventory)}
    order = list(range(len(exp)))
    random.Random(53).shuffle(order)
    half = order[: len(order) // 2]

    def run(scheds, seeds):
        W, Bd = pad_schedules(scheds)
        al = ScheduledAligner(seeds, n=1000, k=50, feat_n=1000, feat_k=50,
                              n_words=len(words), n_features=len(features),
                              word_names=[f"phon_{w}" for w in words],
                              feature_names=[f"feat_{f}" for f in features],
                              p=P, beta=BETA, rounds_word=2)
        al.prepare(_features_tensor(inventory, features, len(seeds)))
        al.train(W, Bd)
        return al.overlap_table()

    alone = run([schedule_of(exp, wi, bi, order)], [42])
    together = run([schedule_of(exp, wi, bi, order),
                    schedule_of(exp, wi, bi, half)], [42, 7])
    torch.testing.assert_close(together[0], alone[0], rtol=0, atol=0)
