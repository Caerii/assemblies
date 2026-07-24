"""Batched-forward mini-batch training for assembly-calculus sequence models.

Turns the batched forward pass (Lever B, docs/gpu_scale_design.md) into a
TRAINING speedup. Assembly-calculus sequence training is normally online -- one
sentence at a time, each updating the connectome the next trains on. This trainer
processes B sentences' forward passes in parallel as ``[B, n]`` tensors, records
the Hebbian bridge each transition would form, accumulates them across the batch
via one ``SRC.T @ TGT`` edge-count matmul, and applies the update once.

This is exact where it matters: ``research/experiments/minibatch_training.py``
shows that in the stable, stimulus-anchored regime (the best-performing one)
mini-batch training is BIT-IDENTICAL to online, because bridges potentiate between
fixed stimulus-anchored assemblies and so do not depend on the connectome's
current state. So batching gives the GPU-parallel speedup with no quality loss in
the regime that matters.

Model: fixed per-word assemblies (a ``k``-subset of ``n`` neurons each) plus a
learned LEX->LEX bridge connectome. Context is built by feeding a prefix's words
with recurrence; the successor is recalled by projecting the context through the
learned bridges. Dense connectome (clear + fast to n~1e4 on a 10GB card); the same
accumulation works for a sparse connectome.

Requires PyTorch + CUDA.
"""

from typing import Dict, List, Sequence


class BatchedSeqTrainer:
    """GPU batched-forward mini-batch trainer for next-token sequence learning."""

    def __init__(self, n: int, k: int, vocab: Sequence[str], *, p: float = 0.01,
                 beta: float = 0.3, stim: float = 2.0, seed: int = 0,
                 device: str = "cuda"):
        import torch
        self._torch = torch
        self.device = device
        self.n, self.k = int(n), int(k)
        self.vocab = list(vocab)
        self.V = len(self.vocab)
        self.word_id: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}
        self.beta, self.stim = float(beta), float(stim)
        g = torch.Generator(device=device).manual_seed(seed)
        # fixed per-word assembly, [V, n] indicator (a random k-subset each)
        self.A = torch.zeros(self.V, self.n, device=device)
        for w in range(self.V):
            idx = torch.randperm(self.n, generator=g, device=device)[:self.k]
            self.A[w, idx] = 1.0
        # bridge connectome, random G(n,p) init
        self.W = (torch.rand(self.n, self.n, generator=g,
                             device=device) < p).float()

    # -- core batched dynamics ---------------------------------------------

    def _rec(self, act):
        """Scale-normalized recurrent drive (keeps a growing W from swamping the
        stimulus -- the role norm_init plays in the real engine). Batched over
        the leading dim."""
        r = act @ self.W
        mx = r.amax(dim=-1, keepdim=True)
        return self._torch.where(mx > 0, r / mx, r)

    def _topk_onehot(self, drive):
        torch = self._torch
        idx = torch.topk(drive, self.k, dim=-1).indices
        out = torch.zeros_like(drive)
        out.scatter_(-1, idx, 1.0)
        return out

    def _ids(self, sentences: List[List[str]]):
        return [[self.word_id[w] for w in s] for s in sentences]

    # -- batched forward: collect transitions to potentiate ----------------

    def _batch_transitions(self, sentences: List[List[str]]):
        """Forward all B sentences in parallel; return stacked ``(SRC, TGT)``
        potentiation pairs -- ``SRC[t]`` is the context before a token, ``TGT[t]``
        the next word's assembly. Ragged lengths handled by masking finished
        sentences."""
        torch = self._torch
        B = len(sentences)
        ids = self._ids(sentences)
        lens = [len(s) for s in sentences]
        maxlen = max(lens)
        first = torch.tensor([row[0] for row in ids], device=self.device)
        ctx = self.A[first].clone()                      # [B, n]
        SRC, TGT = [], []
        for i in range(1, maxlen):
            active = torch.tensor([i < lens[b] for b in range(B)],
                                  device=self.device)
            wi = torch.tensor([ids[b][i] if i < lens[b] else 0
                               for b in range(B)], device=self.device)
            tgt = self.A[wi]                             # [B, n]
            if active.any():
                SRC.append(ctx[active])                 # context before word i
                TGT.append(tgt[active])
            drive = self.stim * tgt + self._rec(ctx)
            new_ctx = self._topk_onehot(drive)
            ctx = torch.where(active.view(B, 1), new_ctx, ctx)
        if not SRC:
            empty = torch.empty(0, self.n, device=self.device)
            return empty, empty
        return torch.cat(SRC), torch.cat(TGT)

    # -- training ----------------------------------------------------------

    def train(self, corpus: List[List[str]], *, batch_size: int = 16,
              epochs: int = 1) -> None:
        """Mini-batch train: freeze W across each batch of ``batch_size``
        sentences, accumulate their bridge updates, apply once. ``batch_size=1``
        is online. Returns nothing; the connectome ``self.W`` is updated."""
        for _ in range(epochs):
            for start in range(0, len(corpus), batch_size):
                batch = corpus[start:start + batch_size]
                SRC, TGT = self._batch_transitions(batch)
                if SRC.shape[0] > 0:
                    # C[i,j] = # of transitions potentiating edge i->j
                    self.W = self.W + self.beta * (SRC.t() @ TGT)

    # -- batched inference -------------------------------------------------

    def _build_contexts(self, prefixes: List[List[str]]):
        """Batched context assemblies [P, n] for P prefixes (ragged, masked)."""
        torch = self._torch
        P = len(prefixes)
        ids = self._ids(prefixes)
        lens = [len(s) for s in prefixes]
        maxlen = max(lens)
        first = torch.tensor([row[0] for row in ids], device=self.device)
        ctx = self.A[first].clone()
        for i in range(1, maxlen):
            active = torch.tensor([i < lens[b] for b in range(P)],
                                  device=self.device)
            wi = torch.tensor([ids[b][i] if i < lens[b] else 0
                               for b in range(P)], device=self.device)
            drive = self.stim * self.A[wi] + self._rec(ctx)
            new_ctx = self._topk_onehot(drive)
            ctx = torch.where(active.view(P, 1), new_ctx, ctx)
        return ctx

    def predict(self, prefixes: List[List[str]]) -> List[str]:
        """Top-1 next word for each of P prefixes (batched recall + readout)."""
        ctx = self._build_contexts(prefixes)
        pred = self._topk_onehot(self._rec(ctx))         # recall successors
        top = (pred @ self.A.t()).argmax(dim=-1)         # [P]
        return [self.vocab[int(i)] for i in top]

    def accuracy(self, corpus: List[List[str]], batch_size: int = 512) -> float:
        """Next-token top-1 accuracy over every (sentence, position) point."""
        prefixes, actuals = [], []
        for s in corpus:
            for i in range(1, len(s)):
                prefixes.append(s[:i]); actuals.append(s[i])
        correct = 0
        for j in range(0, len(prefixes), batch_size):
            preds = self.predict(prefixes[j:j + batch_size])
            for p, a in zip(preds, actuals[j:j + batch_size]):
                correct += (p == a)
        return correct / max(len(prefixes), 1)
