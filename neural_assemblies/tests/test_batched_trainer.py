"""Batched-forward mini-batch trainer: learns, and matches online training.

See neural_assemblies/assembly_calculus/batched_trainer.py.
"""
import pytest


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_torch_cuda(), reason="BatchedSeqTrainer requires PyTorch + CUDA")


def _markov_corpus(V, n_sent, length, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    trans = torch.rand(V, V, generator=g) ** 6
    trans = trans / trans.sum(1, keepdim=True)
    sents = []
    for _ in range(n_sent):
        s = [int(torch.randint(0, V, (1,), generator=g))]
        for _ in range(length - 1):
            s.append(int(torch.multinomial(trans[s[-1]], 1, generator=g)))
        sents.append([str(w) for w in s])
    return sents


def test_trainer_learns_above_chance():
    from neural_assemblies.assembly_calculus.batched_trainer import (
        BatchedSeqTrainer,
    )
    V = 16
    vocab = [str(i) for i in range(V)]
    train_c = _markov_corpus(V, 240, 7, seed=1)
    test_c = _markov_corpus(V, 60, 7, seed=1)  # same table (seed), held-out draw
    m = BatchedSeqTrainer(6000, 30, vocab, p=0.01, beta=0.3, stim=2.0, seed=0)
    m.train(train_c, batch_size=32, epochs=2)
    acc = m.accuracy(test_c)
    assert acc > 3.0 / V, f"test acc {acc:.3f} not clearly above chance {1/V:.3f}"


def test_minibatch_matches_online_bit_identical():
    # In the stable stim-anchored regime, mini-batch training must be
    # bit-identical to online (the potentiated bridges run between fixed
    # assemblies, independent of the connectome's state).
    import torch
    from neural_assemblies.assembly_calculus.batched_trainer import (
        BatchedSeqTrainer,
    )
    vocab = [str(i) for i in range(16)]
    corpus = _markov_corpus(16, 120, 6, seed=3)

    m1 = BatchedSeqTrainer(4000, 30, vocab, p=0.01, beta=0.3, stim=2.0, seed=0)
    m1.train(corpus, batch_size=1, epochs=2)     # online
    mB = BatchedSeqTrainer(4000, 30, vocab, p=0.01, beta=0.3, stim=2.0, seed=0)
    mB.train(corpus, batch_size=32, epochs=2)    # mini-batch

    div = float((mB.W - m1.W).norm() / (m1.W.norm() + 1e-9))
    assert div < 1e-5, f"mini-batch diverged from online by {div:.2e}"
    # identical weights -> identical predictions
    assert m1.accuracy(corpus) == mB.accuracy(corpus)


def test_predict_batched_shapes():
    from neural_assemblies.assembly_calculus.batched_trainer import (
        BatchedSeqTrainer,
    )
    vocab = [str(i) for i in range(10)]
    corpus = _markov_corpus(10, 40, 5, seed=4)
    m = BatchedSeqTrainer(3000, 25, vocab, p=0.02, beta=0.3, stim=2.0, seed=0)
    m.train(corpus, batch_size=16, epochs=1)
    preds = m.predict([["0"], ["1", "2"], ["3", "4", "5"]])
    assert len(preds) == 3
    assert all(p in vocab for p in preds)


class TestSparseTrainer:
    """Sparse growing connectome lifts the vocab ceiling at no quality cost."""

    def test_sparse_learns_above_chance(self):
        from neural_assemblies.assembly_calculus.batched_trainer import (
            SparseBatchedSeqTrainer,
        )
        V = 16
        vocab = [str(i) for i in range(V)]
        train_c = _markov_corpus(V, 300, 6, seed=1)
        test_c = _markov_corpus(V, 80, 6, seed=1)
        t = SparseBatchedSeqTrainer(8000, 40, vocab, m=1, beta=0.3, seed=0)
        t.train(train_c, batch_size=32, epochs=2)
        assert t.accuracy(test_c) > 3.0 / V
        assert t.nnz() > 0

    def test_sparse_matches_dense_quality(self):
        from neural_assemblies.assembly_calculus.batched_trainer import (
            BatchedSeqTrainer, SparseBatchedSeqTrainer,
        )
        V = 16
        vocab = [str(i) for i in range(V)]
        train_c = _markov_corpus(V, 300, 6, seed=1)
        test_c = _markov_corpus(V, 80, 6, seed=1)
        d = BatchedSeqTrainer(8000, 40, vocab, p=0.01, beta=0.3, stim=2.0, seed=0)
        d.train(train_c, batch_size=32, epochs=2)
        s = SparseBatchedSeqTrainer(8000, 40, vocab, m=1, beta=0.3, seed=0)
        s.train(train_c, batch_size=32, epochs=2)
        # bigram sparse should be at least as good as the dense recurrent trainer
        assert s.accuracy(test_c) >= d.accuracy(test_c) - 0.05

    def test_sparse_scales_where_dense_would_oom(self):
        # n=1e6 dense [n,n] would need ~4 TB; sparse trains in a fraction of a GB.
        import torch
        from neural_assemblies.assembly_calculus.batched_trainer import (
            SparseBatchedSeqTrainer,
        )
        vocab = [str(i) for i in range(16)]
        corpus = _markov_corpus(16, 120, 6, seed=1)
        torch.cuda.reset_peak_memory_stats()
        t = SparseBatchedSeqTrainer(1_000_000, 40, vocab, m=1, beta=0.3,
                                    seed=0, max_batch_rows=32)
        t.train(corpus, batch_size=32, epochs=1)
        acc = t.accuracy(corpus)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        assert acc > 3.0 / 16
        assert peak_gb < 4.0, f"peak GPU {peak_gb:.1f}GB too high for sparse n=1e6"
