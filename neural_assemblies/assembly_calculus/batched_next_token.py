"""Batched next-token prediction on the GPU (scale up language inference).

``next_token.score_corpus`` runs one FROZEN prediction per corpus position, each
independent of the others -- so they are embarrassingly parallel through the one
shared (frozen) connectome. This module runs all of them at once: B context
prefixes are driven through the trained area as a single ``[B, n]`` activity
tensor with one batched sparse-dense matmul + one ``topk`` per step, then read
out against the vocabulary in one matmul.

It replicates ``_predict_next_token_inner`` step-for-step over the FROZEN
connectome, so predictions match the sequential read-only path (validated:
~100% top-1 agreement; the residual is GPU float non-determinism -- scatter_add /
SpMM atomic adds reorder, flipping a few topk near-ties -- not a systematic
error), while running 50x+ faster on a trained model (RTX 3080). See
docs/gpu_scale_design.md (Lever B) and core/torch_engine/_batched.py.

BatchedLM is read-only inference by construction (it selects only among
materialized neurons). The matching sequential semantics is the engine's
``readonly`` mode (``brain._engine.readonly = True``), which suppresses candidate
sampling so a projection never materializes new neurons -- inference should not
grow the brain, and it gives a fixed connectome to batch over.

Targets the production ``norm_init=True`` substrate. Requires the ``torch_sparse``
engine (CUDA). Build a ``BatchedLM`` once from a trained brain, then call
``score_corpus`` / ``predict``.
"""

from typing import Dict, List, Tuple


class BatchedLM:
    """Frozen-connectome batched next-token predictor extracted from a trained
    ``next_token`` brain on the torch_sparse engine.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-batched-next-token

    Args:
        brain: trained Brain on the ``torch_sparse`` engine.
        area: the LEX area name.
        vocab: vocabulary words (defines the output index order).
        stimuli_map: word -> stimulus name.
        lexicon: word -> Assembly (from ``build_next_token_model``).
    """

    def __init__(self, brain, area, vocab, stimuli_map, lexicon):
        eng = brain._engine
        if not getattr(eng, "supports_batched_next_token", False):
            raise TypeError(
                f"BatchedLM requires an engine with batched next-token support; "
                f"{type(eng).__name__} does not provide that capability.")
        import torch

        self._torch = torch
        self.area = area
        self.vocab = list(vocab)
        self.k = int(brain.areas[area].k)
        n = int(brain.areas[area].n)
        self.n = n
        dev = eng._device
        self.device = dev
        p = float(brain.p)
        norm_init = bool(getattr(brain, "norm_init", False))

        from neural_assemblies.core.torch_engine._batched import (
            csrconn_to_torch_csr,
        )

        # Recurrent LEX->LEX connectome, with the frozen norm_init column scale
        # (1/d_j) baked in as a per-column vector applied to the drive.
        csr = eng._area_conns[area][area]
        self.W = csrconn_to_torch_csr(csr, n)
        self.Wt = self.W.t().to_sparse_csr()
        if norm_init:
            deg = csr.column_indegree(n)
            unknown = max(n - int(csr._nrows), 0)
            self.nscale = 1.0 / torch.clamp(deg + unknown * p, min=1.0)
        else:
            self.nscale = torch.ones(n, device=dev)

        # Per-word stimulus drive D[V, n], with the frozen stim norm scale.
        V = len(self.vocab)
        self.D = torch.zeros(V, n, device=dev)
        for wi, w in enumerate(self.vocab):
            sc = eng._stim_conns[stimuli_map[w]][area]
            sw = sc.weights.float()
            m = int(sw.numel())
            if norm_init:
                ndb = getattr(sc, "_norm_deg_base", None)
                base = ndb.float() if ndb is not None else sw
                bm = int(base.numel())
                ssc = 1.0 / torch.clamp(base + (n - self.k) * p, min=1.0)
                self.D[wi, :bm] = sw[:bm] * ssc
                if m > bm:
                    self.D[wi, bm:m] = sw[bm:m] / max((n - self.k) * p, 1.0)
            else:
                self.D[wi, :m] = sw

        # Lexicon indicator L[V, n] in COMPACT index space (Assembly.winners are
        # STABLE neuron ids; map them back through compact_to_neuron_id).
        c2n = eng._areas[area].compact_to_neuron_id
        stable_to_compact = {int(s): c for c, s in enumerate(c2n)}
        self.L = torch.zeros(V, n, device=dev)
        for wi, w in enumerate(self.vocab):
            comp = [stable_to_compact[int(s)] for s in lexicon[w].winners
                    if int(s) in stable_to_compact]
            if comp:
                self.L[wi, torch.tensor(comp, device=dev, dtype=torch.long)] = 1.0
        self._word_of = {w: i for i, w in enumerate(self.vocab)}

    def _scores(self, contexts: List[List[str]], rounds_per_token: int):
        """Return [B, V] overlap scores, replicating _predict_next_token_inner."""
        if not contexts:
            raise ValueError("contexts must contain at least one prefix")
        if rounds_per_token < 1:
            raise ValueError("rounds_per_token must be positive")
        if any(not context for context in contexts):
            raise ValueError("contexts cannot contain empty prefixes")
        torch = self._torch
        B = len(contexts)
        n, K, dev = self.n, self.k, self.device
        maxlen = max(len(c) for c in contexts)
        act = torch.zeros(B, n, device=dev)

        def step(stim, active):
            nonlocal act
            rec = torch.sparse.mm(self.Wt, act.t()).t() * self.nscale
            drive = stim + rec if stim is not None else rec
            idx = torch.topk(drive, K, dim=1).indices
            newact = torch.zeros_like(act).scatter_(1, idx, 1.0)
            act = torch.where(active.view(B, 1), newact, act)

        for t in range(maxlen):
            wids = torch.tensor(
                [self._word_of[c[t]] if t < len(c) else -1 for c in contexts],
                device=dev)
            active = wids >= 0
            stim = torch.zeros(B, n, device=dev)
            stim[active] = self.D[wids[active].clamp(min=0)]
            # t==0: one stim-only start step (act is 0 so recurrence is a no-op);
            # then rounds_per_token-1 stim+recurrence steps, same as t>0.
            if t == 0:
                step(stim, active)
            for _ in range(rounds_per_token - 1):
                step(stim, active)
        # autonomous prediction step (stimulus removed) for all items
        step(None, torch.ones(B, dtype=torch.bool, device=dev))
        return act @ self.L.t()   # [B, V] intersection counts (== overlap rank)

    def predict(self, contexts: List[List[str]], rounds_per_token: int = 5,
                ) -> List[List[Tuple[str, float]]]:
        """Ranked ``(word, score)`` predictions for each of B contexts."""
        scores = self._scores(contexts, rounds_per_token)
        out = []
        for b in range(len(contexts)):
            row = scores[b]
            order = self._torch.argsort(row, descending=True)
            out.append([(self.vocab[int(i)], float(row[int(i)])) for i in order])
        return out

    def score_corpus(self, corpus: List[List[str]], rounds_per_token: int = 5,
                     batch_size: int = 512) -> Dict[str, float]:
        """Batched equivalent of ``next_token.score_corpus`` (top1/top3/mrr).

        Enumerates every (sentence, position) prediction point and runs them in
        batches through the GPU; identical predictions to the sequential path.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if rounds_per_token < 1:
            raise ValueError("rounds_per_token must be positive")
        torch = self._torch
        contexts, actuals = [], []
        for s in corpus:
            for pos in range(len(s) - 1):
                contexts.append(s[:pos + 1])
                actuals.append(s[pos + 1])
        top1 = top3 = 0
        rr = 0.0
        for i in range(0, len(contexts), batch_size):
            chunk = contexts[i:i + batch_size]
            acts = actuals[i:i + batch_size]
            scores = self._scores(chunk, rounds_per_token)
            order = torch.argsort(scores, dim=1, descending=True)  # [b, V]
            for b, actual in enumerate(acts):
                ranked = [self.vocab[int(j)] for j in order[b]]
                if ranked and ranked[0] == actual:
                    top1 += 1
                if actual in ranked[:3]:
                    top3 += 1
                if actual in ranked:
                    rr += 1.0 / (ranked.index(actual) + 1)
        nn = max(len(contexts), 1)
        return {
            "top1_accuracy": top1 / nn,
            "top3_accuracy": top3 / nn,
            "mrr": rr / nn,
            "total_predictions": len(contexts),
        }
