"""Does mini-batch (deferred-update) Hebbian training preserve what online
training learns?  -- the one path to training ONE assembly-calculus language
model at GPU scale (see docs/gpu_scale_design.md).

Assembly-calculus sequence training is normally ONLINE: each sentence's Hebbian
updates immediately change the connectome the next sentence trains on. To batch
it we must FREEZE the connectome across a batch of B sentences, accumulate the
updates each would make, and apply them once -- exactly mini-batch vs online SGD.
Whether that preserves learning is not obvious: winner selection is a nonlinear
topk and the potentiated assemblies depend on the current weights (recurrence),
so batching changes which edges get strengthened.

This is a controlled model (fixed word assemblies, recurrent multi-word context,
Hebbian x_{i-1}->x_i bridges) -- the essential dynamics of next_token training,
with the SAME update rule for online and mini-batch so the only variable is WHEN
updates are applied. We sweep batch size and the stimulus/recurrence balance, and
report next-token top-1 accuracy plus the relative divergence of the learned
weights (mini-batch vs online). Runs on GPU (dense connectome for clarity).

FINDING (RTX 3080). The stimulus-anchored regime -- high STIM, where each word's
assembly is pinned by its stimulus and only modulated by recurrence -- is BOTH
the best-performing regime AND perfectly batchable: mini-batch training is
BIT-IDENTICAL to online (0.00% weight divergence at every batch size), because the
potentiated bridges run between fixed, stimulus-anchored assemblies and so do not
depend on the connectome's current state. As STIM drops and recurrence (the
learned W) starts to move the winners, training becomes genuinely order-dependent
and the learned weights diverge (20-58%), yet final task accuracy stays comparable
(mini-batch neither systematically helps nor hurts -- like mini-batch vs online
SGD). The operative lesson: the STABILITY assembly calculus relies on (and that
norm_init enforces) is exactly what makes training order-independent, so one model
CAN be trained mini-batch at GPU scale with no quality loss in the regime that
matters. Speedup comes from batching the forward passes (validated at ~91x in
BatchedLM); this script isolates the *correctness* question.
"""
import time
import torch

DEV = "cuda"


def make_trans(V, seed):
    """A peaked first-order Markov transition table (learnable dominant successor)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    trans = torch.rand(V, V, generator=g) ** 6
    return trans / trans.sum(1, keepdim=True)


def sample_sentences(trans, n_sent, length, seed):
    """Sentences from a fixed transition table (train/test share the table)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    V = trans.shape[0]
    sents = []
    for _ in range(n_sent):
        s = [int(torch.randint(0, V, (1,), generator=g))]
        for _ in range(length - 1):
            s.append(int(torch.multinomial(trans[s[-1]], 1, generator=g)))
        sents.append(s)
    return sents


class SeqModel:
    """Fixed word assemblies + a learned LEX->LEX bridge connectome (dense)."""

    def __init__(self, n, k, V, p, beta, stim_strength, ctx_rounds, seed):
        self.n, self.k, self.V = n, k, V
        self.beta, self.stim = beta, stim_strength
        self.ctx_rounds = ctx_rounds
        g = torch.Generator(device=DEV).manual_seed(seed)
        # fixed random k-assembly per word, as [V, n] indicator
        self.A = torch.zeros(V, n, device=DEV)
        for w in range(V):
            idx = torch.randperm(n, generator=g, device=DEV)[:k]
            self.A[w, idx] = 1.0
        # recurrent connectome, random G(n,p) init (float weights)
        self.W = (torch.rand(n, n, generator=g, device=DEV) < p).float()

    def _rec(self, act):
        """Recurrent drive, scale-normalized so a growing W cannot swamp the
        stimulus (the role norm_init plays in the real engine)."""
        r = act @ self.W
        mx = r.max()
        return r / mx if mx > 0 else r

    def _context(self, words):
        """Recurrent multi-word context assembly [n] for a prefix (W-dependent)."""
        act = self.A[words[0]].clone()
        for w in words[1:]:
            drive = self.stim * self.A[w] + self._rec(act)
            act = self._topk(drive)
        return act

    def _topk(self, drive):
        idx = torch.topk(drive, self.k).indices
        out = torch.zeros(self.n, device=DEV)
        out[idx] = 1.0
        return out

    def transitions(self, sentence):
        """(src_ctx, tgt_assembly) pairs a sentence would potentiate."""
        pairs = []
        for i in range(1, len(sentence)):
            src = self._context(sentence[:i])         # context up to i-1
            tgt = self.A[sentence[i]]                  # next word assembly
            pairs.append((src, tgt))
        return pairs

    def apply_updates(self, SRC, TGT):
        """Grow bridges: W[i,j] += beta * count(i->j).  C = SRC^T @ TGT gives the
        per-edge potentiation counts. Additive edge creation mirrors how assembly
        calculus materializes new x_{i-1}->x_i synapses (a multiplicative rule
        only strengthens the few pre-existing random edges and barely learns)."""
        if SRC.numel() == 0:
            return
        C = SRC.t() @ TGT                              # [n, n] counts
        self.W = self.W + self.beta * C

    def predict(self, context_words):
        """Top-1 next word: recall through W, read out vs word assemblies."""
        src = self._context(context_words)
        pred = self._topk(self._rec(src))              # recall successor
        return int((pred @ self.A.t()).argmax())


def train(model, corpus, batch_size):
    """batch_size=1 -> online; >1 -> freeze W across `batch_size` sentences,
    accumulate updates, apply once."""
    for start in range(0, len(corpus), batch_size):
        batch = corpus[start:start + batch_size]
        SRC, TGT = [], []
        for sent in batch:
            for src, tgt in model.transitions(sent):   # all on the frozen W
                SRC.append(src); TGT.append(tgt)
        if SRC:
            model.apply_updates(torch.stack(SRC), torch.stack(TGT))


def accuracy(model, corpus):
    correct = total = 0
    for sent in corpus:
        for i in range(1, len(sent)):
            if model.predict(sent[:i]) == sent[i]:
                correct += 1
            total += 1
    return correct / max(total, 1)


def run():
    N, K, V, P = 6000, 30, 16, 0.01
    BETA, CTX = 0.3, 3
    trans = make_trans(V, seed=7)
    train_corpus = sample_sentences(trans, 120, 6, seed=1)
    test_corpus = sample_sentences(trans, 40, 6, seed=2)   # held-out, same dist

    # Sweep stimulus strength: high STIM -> stim-anchored, W-independent (mini-
    # batch trivially exact); lower STIM -> the learned bridges (W) genuinely
    # shift the context winners, so training becomes order-dependent and mini-
    # batch COULD diverge from online. We test whether it does.
    for STIM in [2.0, 0.5, 0.1, 0.03]:
        print(f"\n=== STIM={STIM} (n={N} k={K} V={V} beta={BETA}, "
              f"chance={1/V:.3f}) ===")
        print(f"{'batch':>6} {'train_acc':>10} {'test_acc':>9} {'time':>8} "
              f"{'d_test':>8} {'W-div%':>8}")
        online_test = None
        W_online = None
        for B in [1, 4, 16, 64]:
            m = SeqModel(N, K, V, P, BETA, STIM, CTX, seed=0)
            t0 = time.perf_counter()
            train(m, train_corpus, batch_size=B)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tr = accuracy(m, train_corpus)
            te = accuracy(m, test_corpus)
            if B == 1:
                online_test = te
                W_online = m.W.clone()
            # relative divergence of the learned weights vs online
            wdiv = float((m.W - W_online).norm() / (W_online.norm() + 1e-9)) * 100
            print(f"{B:>6} {tr:>10.3f} {te:>9.3f} {dt:>6.2f}s "
                  f"{te - online_test:>+8.3f} {wdiv:>7.2f}%", flush=True)


if __name__ == "__main__":
    run()
