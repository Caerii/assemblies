"""A reduced model of one replay step (no n x n matrix): the overlap map F(o).

Per neuron j of n:
  write   the next element A' is the top k of  S_j + R_j  among neurons not
          excluded by refraction, with S_j ~ Bin(k, p) the element's stimulus and
          R_j ~ Bin(k, p) the base drive from the current element A (present
          synapses), both divided by the in-degree-like divisor (np): the
          SELECTION makes A' members over-connected from A.
  replay  input = c = o k members of A (a random subset) + (k - c) random others.
          drive_j = [ (1 + beta) Rc_j  +  W_j  +  beta X_j ] / d_j   for j in A'
          drive_j = [            Rc_j  +  W_j  +  beta X_j ] / d_j   otherwise
          Rc_j ~ Hypergeom(k, R_j, c): present synapses from the c true inputs;
          W_j ~ Bin(k - c, p): present synapses from the random inputs;
          X_j: excess from the other transitions -- each present input synapse
          carries a count ~ Poisson(lam), lam = L k^2 / n^2, priced
          ((1 + beta)^c - 1) / beta, clipped at w_max (the target's own
          count-1 synapse is priced in the (1 + beta) term);
          d_j ~ Bin(n, p) the in-degree (norm_init).
  F(o) = |top k of drive  intersect  A'| / k.
"""
import math
import numpy as np


def F(o, n, k, p, beta, L, *, excluded=0, w_max=20.0, trials=200, rng=None):
    rng = rng or np.random.default_rng(0)
    lam = L * k * k / (n * n)
    c = int(round(o * k))
    out = []
    for _ in range(trials):
        d = rng.binomial(n, p, size=n).astype(float)
        S = rng.binomial(k, p, size=n)
        R = rng.binomial(k, p, size=n)
        score = (S + R) / d
        if excluded:
            score[rng.choice(n, excluded, replace=False)] = -np.inf
        target = np.zeros(n, bool)
        target[np.argpartition(-score, k)[:k]] = True
        Rc = rng.hypergeometric(R, k - R, c) if c > 0 else np.zeros(n, int)
        W = rng.binomial(k - c, p, size=n)
        m = Rc + W                                           # present synapses from the input
        # interference: each present input synapse has count ~ Poisson(lam)
        counts = rng.poisson(lam * m)                       # total extra counts (approx. one per synapse)
        X = np.minimum((1 + beta) ** 1 - 1, w_max) * counts / beta if beta > 0 else counts * 0.0
        drive = (m + beta * X) + np.where(target, beta * Rc, 0.0)
        drive = drive / d
        win = np.argpartition(-drive, k)[:k]
        out.append(target[win].mean())
    return float(np.mean(out)), float(np.std(out))


if __name__ == "__main__":
    n, k, p = 4000, 60, 0.5
    beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
    meas = {
        ("tau33", 400): ([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.541, 0.676, 0.793, 0.879, 0.938, 0.974]),
        ("tau64", 1600): ([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.241, 0.375, 0.518, 0.652, 0.77, 0.856, 0.923, 0.963]),
        ("tau64", 2300): ([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.231, 0.353, 0.501, 0.627, 0.742, 0.833, 0.902, 0.945]),
        ("tau64", 3200): ([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.212, 0.327, 0.447, 0.569, 0.678, 0.774, 0.851, 0.904]),
    }
    for (tag, L), (os_, fs) in meas.items():
        model = [F(o, n, k, p, beta, L, trials=60)[0] for o in os_]
        print(f"{tag} L={L}: measured {fs}")
        print(f"            model    {[round(v, 3) for v in model]}")
        print(f"            diff     {[round(a - b, 3) for a, b in zip(model, fs)]}", flush=True)
