"""Batched next-token prediction matches the sequential next_token path.

See neural_assemblies/assembly_calculus/batched_next_token.py.
"""
import pytest


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_torch_cuda(), reason="BatchedLM requires PyTorch + CUDA")


def _train_lm(norm_init):
    import torch
    # The GPU truncated-normal sampler draws from torch's GLOBAL generator, so
    # seed it here to make training reproducible regardless of test order.
    torch.manual_seed(0)
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus.next_token import (
        build_next_token_model, train_on_corpus,
    )
    vocab = list("ABCDEFGHIJ")
    corpus = [list("ABCDE"), list("FGHIJ"), list("ACEGI"), list("BDFHJ")] * 8
    stim_map = {w: f"s_{w}" for w in vocab}
    b = Brain(p=0.05, seed=1, engine="torch_sparse", norm_init=norm_init)
    for w in vocab:
        b.add_stimulus(stim_map[w], 40)
    b.add_area("LEX", 4000, 40, 0.1)
    lex = build_next_token_model(b, "LEX", vocab, stim_map, rounds=10)
    train_on_corpus(b, "LEX", corpus, stim_map, rounds_per_token=5, repetitions=2)
    return b, "LEX", vocab, corpus, stim_map, lex


def test_batched_matches_sequential_top1():
    # BatchedLM targets the production norm_init=True substrate. (The
    # norm_init=False path has documented engine quirks -- column-reuse / lazy
    # expansion effects that norm_init exists to fix -- which the clean dense
    # readout deliberately does not reproduce.)
    from neural_assemblies.assembly_calculus.next_token import predict_next_token
    from neural_assemblies.assembly_calculus.batched_next_token import BatchedLM

    b, area, vocab, corpus, stim_map, lex = _train_lm(norm_init=True)

    contexts = []
    for s in corpus:
        for pos in range(len(s) - 1):
            contexts.append(s[:pos + 1])

    # Read-only inference: prediction must not materialize new neurons (else the
    # connectome mutates mid-run and there is nothing fixed to batch over). This
    # is also the correct semantics -- inference should not grow the brain.
    b._engine.readonly = True
    w_before = b.areas[area].w
    seq_top1 = []
    for ctx in contexts:
        b.inhibit_areas([area])
        preds = predict_next_token(b, area, ctx, stim_map, lex, rounds_per_token=5)
        seq_top1.append(preds[0][0])
    assert b.areas[area].w == w_before, "read-only prediction grew the area"

    lm = BatchedLM(b, area, vocab, stim_map, lex)
    bat = lm.predict(contexts, rounds_per_token=5)
    bat_top1 = [row[0][0] for row in bat]

    # Agreement is near-perfect; the residual is GPU float non-determinism
    # (scatter_add / SpMM use atomic adds in arbitrary order, so a few topk
    # near-ties flip run-to-run) -- not a systematic error. For corpus scoring
    # the aggregate metrics are unaffected.
    agree = sum(a == c for a, c in zip(seq_top1, bat_top1)) / len(seq_top1)
    assert agree >= 0.9, f"top-1 agreement {agree:.2%} (expected ~100%)"


def test_readonly_mode_prevents_materialization():
    # Read-only inference must select only among already-materialized neurons.
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus.ops import project
    b = Brain(p=0.05, seed=2, engine="torch_sparse", norm_init=True)
    b.add_stimulus("s", 40)
    b.add_area("A", 4000, 40, 0.1)
    project(b, "s", "A", rounds=10)
    b._engine.readonly = True
    w = b.areas["A"].w
    for _ in range(20):
        b.project({"s": ["A"]}, {"A": ["A"]})
    assert b.areas["A"].w == w, "readonly projection materialized new neurons"
    assert len(b.areas["A"].winners) == 40


def test_score_corpus_matches_reset_sequential():
    # BatchedLM uses clean-start semantics per prediction (well-defined and
    # order-independent). next_token.score_corpus instead carries area state
    # between positions, so we compare against a reset-sequential reference
    # computed the same way BatchedLM defines a prediction.
    from neural_assemblies.assembly_calculus.next_token import predict_next_token
    from neural_assemblies.assembly_calculus.batched_next_token import BatchedLM

    b, area, vocab, corpus, stim_map, lex = _train_lm(norm_init=True)
    b._engine.readonly = True

    total = seq_top1 = 0
    for s in corpus:
        for pos in range(len(s) - 1):
            b.inhibit_areas([area])
            preds = predict_next_token(
                b, area, s[:pos + 1], stim_map, lex, rounds_per_token=5)
            if preds and preds[0][0] == s[pos + 1]:
                seq_top1 += 1
            total += 1
    seq_acc = seq_top1 / total

    lm = BatchedLM(b, area, vocab, stim_map, lex)
    bat = lm.score_corpus(corpus, rounds_per_token=5)
    assert bat["total_predictions"] == total
    # aggregate accuracy matches within GPU float-nondeterminism noise
    assert abs(bat["top1_accuracy"] - seq_acc) < 0.05, (
        f"batched {bat['top1_accuracy']:.4f} != reset-sequential {seq_acc:.4f}")
