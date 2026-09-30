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

from ..core._torch_ops import torch_ops


class BatchedSeqTrainer:
    """GPU batched-forward mini-batch trainer for next-token sequence learning."""

    def __init__(self, n: int, k: int, vocab: Sequence[str], *, p: float = 0.01,
                 beta: float = 0.3, stim: float = 2.0, seed: int = 0,
                 device: str = "cuda"):
        if n < 1 or k < 1 or k > n:
            raise ValueError("require 1 <= k <= n for batched training")
        if not vocab:
            raise ValueError("vocab must contain at least one token")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must lie in [0, 1]")
        self._torch = torch_ops
        self.device = device
        self.n, self.k = int(n), int(k)
        self.vocab = list(vocab)
        self.V = len(self.vocab)
        self.word_id: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}
        self.beta, self.stim = float(beta), float(stim)
        g = torch_ops.Generator(device=device).manual_seed(seed)
        # fixed per-word assembly, [V, n] indicator (a random k-subset each)
        self.A = torch_ops.zeros(self.V, self.n, device=device)
        for w in range(self.V):
            idx = torch_ops.randperm(self.n, generator=g, device=device)[:self.k]
            self.A[w, idx] = 1.0
        # bridge connectome, random G(n,p) init
        self.W = (torch_ops.rand(self.n, self.n, generator=g,
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
        idx = torch_ops.topk(drive, self.k, dim=-1).indices
        out = torch_ops.zeros_like(drive)
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
        B = len(sentences)
        ids = self._ids(sentences)
        lens = [len(s) for s in sentences]
        maxlen = max(lens)
        first = torch_ops.tensor([row[0] for row in ids], device=self.device)
        ctx = self.A[first].clone()                      # [B, n]
        SRC, TGT = [], []
        for i in range(1, maxlen):
            active = torch_ops.tensor([i < lens[b] for b in range(B)],
                                  device=self.device)
            wi = torch_ops.tensor([ids[b][i] if i < lens[b] else 0
                               for b in range(B)], device=self.device)
            tgt = self.A[wi]                             # [B, n]
            if active.any():
                SRC.append(ctx[active])                 # context before word i
                TGT.append(tgt[active])
            drive = self.stim * tgt + self._rec(ctx)
            new_ctx = self._topk_onehot(drive)
            ctx = torch_ops.where(active.view(B, 1), new_ctx, ctx)
        if not SRC:
            empty = torch_ops.empty(0, self.n, device=self.device)
            return empty, empty
        return torch_ops.cat(SRC), torch_ops.cat(TGT)

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
        P = len(prefixes)
        ids = self._ids(prefixes)
        lens = [len(s) for s in prefixes]
        maxlen = max(lens)
        first = torch_ops.tensor([row[0] for row in ids], device=self.device)
        ctx = self.A[first].clone()
        for i in range(1, maxlen):
            active = torch_ops.tensor([i < lens[b] for b in range(P)],
                                  device=self.device)
            wi = torch_ops.tensor([ids[b][i] if i < lens[b] else 0
                               for b in range(P)], device=self.device)
            drive = self.stim * self.A[wi] + self._rec(ctx)
            new_ctx = self._topk_onehot(drive)
            ctx = torch_ops.where(active.view(P, 1), new_ctx, ctx)
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
            for p, a in zip(preds, actuals[j:j + batch_size], strict=True):
                correct += (p == a)
        return correct / max(len(prefixes), 1)


class SparseBatchedSeqTrainer:
    """Large-vocab batched trainer: a GROWING SPARSE bridge connectome plus a
    BOUNDED m-gram state, so both vocab and context scale.

    Two lessons from the existing codebase are baked in (see
    ``emergent/parser_mixins/state_prediction.py`` and ``core/torch_engine/
    _csr.py``):

    1. **Sparse, growing connectome.** The dense ``[n, n]`` W in
       :class:`BatchedSeqTrainer` caps at n~1e4 (400 MB). Here W is a sparse
       edge set that starts EMPTY and grows only the bridge edges actually
       formed -- the same "materialize edges lazily" model the engine's
       ``CSRConn`` uses -- so n can reach ~1e6 (large vocabulary).

    2. **Bounded state, not an accumulating buffer.** ``StatePredictionMixin``
       documents that folding a whole prefix into one area saturates it (one
       ``k``-area cannot encode an unbounded prefix -> near-bigram). So the
       predictor is keyed on a BOUNDED state, never a growing buffer.

    MEASURED RESULT. The sparse connectome lifts the vocab ceiling with ZERO
    quality loss: at equal V, sparse bigram matches the dense trainer's accuracy
    (0.340 vs 0.340), and holds that accuracy at n=1e6 using ~0.4 GB where a
    dense ``[n,n]`` would need ~4000 GB. But a bounded m-gram *union* state
    (m>1: bag of the last m words' assemblies) does NOT beat bigram -- it dilutes
    the signal, exactly the codebase's point that naive aggregation is not the
    answer. So ``m=1`` (bigram) is the effective default; genuinely richer
    context needs a STRUCTURED bounded state (core x syntactic-slot x mood, as in
    ``StatePredictionMixin``), which is the next architectural step, not a bigger
    bag. ``m`` is kept as a knob to reproduce that finding.

    Assemblies are stored as indices ``[V, k]`` (a dense ``[V, n]`` would blow up
    at large n). Activity/drive are dense ``[B, n]`` with B capped, so peak
    memory stays ~B*n floats regardless of vocabulary size.
    """

    def __init__(self, n, k, vocab, *, m=1, beta=0.3, seed=0, device="cuda",
                 max_batch_rows=64):
        self._torch = torch_ops
        self.device = device
        self.n, self.k, self.m = int(n), int(k), int(m)
        self.beta = float(beta)
        self.max_batch_rows = int(max_batch_rows)
        self.vocab = list(vocab)
        self.V = len(self.vocab)
        self.word_id = {w: i for i, w in enumerate(self.vocab)}
        g = torch_ops.Generator(device=device).manual_seed(seed)
        # per-word assembly as INDICES [V, k] (no dense [V, n])
        self.A_idx = torch_ops.stack([
            torch_ops.randperm(self.n, generator=g, device=device)[:self.k]
            for _ in range(self.V)])
        # flat word-assembly index table for batched readout
        self._A_flat = self.A_idx.reshape(-1)              # [V*k]
        # sparse bridge connectome, starts EMPTY (grows only real bridges)
        self.W = torch_ops.sparse_coo_tensor(
            torch_ops.empty(2, 0, dtype=torch_ops.long, device=device),
            torch_ops.empty(0, device=device), (self.n, self.n)).coalesce()

    # -- bounded m-gram state ----------------------------------------------

    def _state_idx(self, word_ids):
        """Indices of the bounded state = union of the last m words' assemblies."""
        recent = word_ids[-self.m:]
        return torch_ops.unique(self.A_idx[torch_ops.tensor(recent, device=self.device)])

    # -- sparse forward / recall -------------------------------------------

    def _recall_scores(self, states):
        """[P, V] readout scores for P bounded states (as index tensors)."""
        P = len(states)
        act = torch_ops.zeros(P, self.n, device=self.device)
        for i, s in enumerate(states):
            act[i, s] = 1.0
        Wt = self.W.t().to_sparse_csr()
        drive = torch_ops.sparse.mm(Wt, act.t()).t()          # [P, n] successor drive
        # readout: gather drive at each word's assembly, sum over its k neurons
        cols = drive[:, self._A_flat].view(P, self.V, self.k)
        return cols.sum(dim=2)                             # [P, V]

    # -- training ----------------------------------------------------------

    def _batch_edges(self, sentences):
        """(row, col) bridge edges every transition in the batch would grow:
        state(<=m*k) x next-word-assembly(k), for each position."""
        rows, cols = [], []
        for sent in sentences:
            ids = [self.word_id[w] for w in sent]
            for i in range(1, len(ids)):
                s = self._state_idx(ids[:i])              # bounded state indices
                t = self.A_idx[ids[i]]                    # next word assembly
                rows.append(s.repeat_interleave(t.numel()))
                cols.append(t.repeat(s.numel()))
        if not rows:
            return None
        return torch_ops.cat(rows), torch_ops.cat(cols)

    def train(self, corpus, *, batch_size=16, epochs=1):
        """Mini-batch: freeze W across each batch, accumulate the bridge edges,
        add them once (grows/strengthens the sparse connectome)."""
        for _ in range(epochs):
            for start in range(0, len(corpus), batch_size):
                edges = self._batch_edges(corpus[start:start + batch_size])
                if edges is None:
                    continue
                r, c = edges
                add = torch_ops.sparse_coo_tensor(
                    torch_ops.stack([r, c]),
                    torch_ops.full((r.numel(),), self.beta, device=self.device),
                    (self.n, self.n))
                # coalesce sums duplicate edges -> W[i,j] += beta * count
                self.W = (self.W + add).coalesce()

    # -- inference ---------------------------------------------------------

    def predict(self, prefixes):
        """Top-1 next word for each prefix (batched, memory-capped)."""
        out = []
        for j in range(0, len(prefixes), self.max_batch_rows):
            chunk = prefixes[j:j + self.max_batch_rows]
            states = [self._state_idx([self.word_id[w] for w in p]) for p in chunk]
            scores = self._recall_scores(states)
            for row in scores:
                out.append(self.vocab[int(row.argmax())])
        return out

    def accuracy(self, corpus):
        prefixes, actuals = [], []
        for s in corpus:
            for i in range(1, len(s)):
                prefixes.append(s[:i]); actuals.append(s[i])
        preds = self.predict(prefixes)
        return sum(p == a for p, a in zip(preds, actuals, strict=True)) / max(len(prefixes), 1)

    def nnz(self):
        return int(self.W._nnz())
