"""Reduced replay model with MEASURED interference statistics (sufficiency check).
Outsider synapse counts: Poisson(mu * G_j), G_j ~ Gamma(a, 1/a) per neuron; a fitted so the
outsider excess variance matches the engine's. Targets' own synapses: 1 + Poisson(own - 1)."""
import math
import numpy as np

def excess_stats(mu, a, k, p, beta, rng, N=100000):
    m = rng.binomial(k, p, N)
    G = rng.gamma(a, 1.0 / a, N) if a < 1e6 else np.ones(N)
    cnt = rng.poisson((mu * G)[:, None], (N, k))
    mask = np.arange(k)[None, :] < m[:, None]
    e = ((np.minimum((1 + beta) ** cnt, 20.0) - 1) * mask).sum(1)
    return e.mean(), e.var()

def F(o, n, k, p, beta, mu, a, own, trials=60, rng=None):
    rng = rng or np.random.default_rng(0)
    c = int(round(o * k)); out = []
    for _ in range(trials):
        d = rng.binomial(n, p, size=n).astype(float)
        S = rng.binomial(k, p, size=n); R = rng.binomial(k, p, size=n)
        score = (S + R) / d
        target = np.zeros(n, bool); target[np.argpartition(-score, k)[:k]] = True
        Rc = rng.hypergeometric(R, k - R, c) if c > 0 else np.zeros(n, int)
        W = rng.binomial(k - c, p, size=n); m = Rc + W
        G = rng.gamma(a, 1.0 / a, n) if a < 1e6 else np.ones(n)
        cols = np.arange(k)[None, :]
        cnt = rng.poisson((mu * G)[:, None], (n, k))
        ownmask = (cols < Rc[:, None]) & target[:, None]
        cnt = np.where(ownmask, 1 + rng.poisson(own - 1, (n, k)), cnt)
        price = np.minimum((1.0 + beta) ** cnt, 20.0)
        drive = (price * (cols < m[:, None])).sum(1) / d
        win = np.argpartition(-drive, k)[:k]
        out.append(target[win].mean())
    return float(np.mean(out))

n, k, p = 4000, 60, 0.5
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
rng = np.random.default_rng(1)
cases = {1600: (0.414, 3.220, 1.493, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.241, 0.375, 0.518, 0.652, 0.77, 0.856, 0.923, 0.963]),
         3200: (0.844, 13.650, 2.034, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.212, 0.327, 0.447, 0.569, 0.678, 0.774, 0.851, 0.904])}
for L, (mu, var_target, own, os_, fs) in cases.items():
    best = None
    for a in (1e9, 20, 10, 6, 4, 3, 2, 1.5, 1.0, 0.7, 0.5):
        mean, var = excess_stats(mu, a, k, p, beta, rng)
        if best is None or abs(var - var_target) < abs(best[2] - var_target):
            best = (a, mean, var)
    a = best[0]
    model = [F(o, n, k, p, beta, mu, a, own) for o in os_]
    print(f"L={L}: fitted gamma shape a={a} (excess mean {best[1]:.2f}, var {best[2]:.2f} vs {var_target})")
    print(f"   measured {fs}")
    print(f"   model    {[round(v, 3) for v in model]}")
    print(f"   diff     {[round(x - y, 3) for x, y in zip(model, fs)]}", flush=True)
