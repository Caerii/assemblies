"""beta=0 idempotence: the sparse engine invents new candidates every call.

THE INVARIANT.  A neuron's afferent drive from stimulus x is
|{j in x : synapse j->i}| -- a fixed property of the random graph. With
beta=0 and no recurrence, nothing in a projection can change it, so
re-projecting the same stimulus MUST elect exactly the same k winners.

WHAT ACTUALLY HAPPENS (n=20000, k=50, p=0.1, overlap vs the first round):

    explicit  beta=0.0    1.000 1.000 1.000 1.000 1.000
    sparse    beta=0.0    1.000 0.620 0.340 0.320 0.260
    explicit  beta=0.1    1.000 1.000 1.000 1.000 1.000
    sparse    beta=0.1    1.000 0.620 0.620 0.620 0.620

Identical under projection_fidelity 'exact' and 'compiled'.

THE MECHANISM.  `SparseSimulationEngine.sample_new_winner_inputs` draws k
candidate drive values from Binomial(total_k, p) truncated to the top-(k/n)
quantile, and it does so on EVERY projection. Those candidates are near the top
of the distribution by construction, so they displace materialised incumbents
whose only advantage is accumulated potentiation. At beta=0 the incumbents have
no advantage at all and decay toward chance; at beta=0.1 they plateau part-way
instead of converging to 1.0.

This is inherited from the reference brain.py, so it is not a porting error --
but it means the sparse path does not implement the model the PNAS paper
describes, and the sparse path is the production engine (`add_area`).

WHAT IT PLAUSIBLY EXPLAINS.  Offered as hypotheses to test, not conclusions:
PNAS Fig. 2 B1-B3 predicts overlap(y1,y2) ~ 50% and we measure 0.0000; #39
associate() at exactly chance; #47 materialised blocks changing 56.6%
retroactively; the very high w/n churn in trained parser areas (NOUN_CORE
2501/3000 neurons ever fired); and the standing reading that "beta buys
maintenance" -- beta may simply be paying to outrun phantom candidates.

Run:  python research/experiments/projection_idempotence.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")

from neural_assemblies.core.brain import Brain


def run(n: int, k: int, p: float, *, explicit: bool, beta: float,
        rounds: int, fidelity: str = "exact", seed: int = 0):
    np.random.seed(seed)
    b = Brain(p=p, seed=seed, save_winners=True, norm_init=False,
              recurrent_projection=False, engine="numpy_sparse",
              projection_fidelity=fidelity)
    b.add_stimulus("x", k)
    (b.add_explicit_area if explicit else b.add_area)("B", n, k, beta)
    ws = []
    for _ in range(rounds):
        b.project({"x": ["B"]}, {})
        ws.append(np.sort(np.asarray(b.areas["B"].winners).copy()))
    return [len(np.intersect1d(ws[0], w)) / k for w in ws]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[2000, 20000])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.1])
    a = ap.parse_args()

    failures = 0
    for n in a.n:
        print(f"\n===== n={n} k={a.k} p={a.p}  (chance {a.k/n:.4f}) =====")
        for beta in a.betas:
            for explicit in (True, False):
                tag = "explicit" if explicit else "sparse  "
                ov = run(n, a.k, a.p, explicit=explicit, beta=beta,
                         rounds=a.rounds)
                ok = abs(ov[-1] - 1.0) < 1e-9
                failures += (not ok) and (not explicit)
                print(f"  {tag} beta={beta}  " + " ".join(f"{o:.3f}" for o in ov)
                      + ("" if ok else "   <-- NOT IDEMPOTENT"))
    print(f"\nnon-idempotent sparse configurations: {failures}")


if __name__ == "__main__":
    main()
