"""Exploratory (not registered): the window of replay-time habituation (probe_control.py's 'habit'),
and whether it also helps the REPETITION edge. Same cell and seeds as probe_control.py. Arms:
recurrence (U = 40 random successors) and repetition (U = 10, b = 3, R ~ 3.7); habituation
c in {0.03, 0.05, 0.1, 0.15, 0.2} x T in {10, 25, 50, 100} steps, on the odd half."""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_control.py"), encoding="utf-8").read()
head = src.split("even, odd = ")[0]
for U, b in ((40, None), (10, 3)):
    code = head.replace("n, k, p, tau, U = 10000, 75, 0.48, 67, 40", f"n, k, p, tau, U = 10000, 75, 0.48, 67, {U}")
    code = code.replace("words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]",
                        f"words = [rg.walks(sd, M, V, {b}, L * 1000 + U) for sd in seeds]")
    g = {"__file__": __file__, "__name__": "probe"}
    exec(code, g)
    odd = list(range(1, g["M"], 2))
    torch, B, n = g["torch"], g["B"], g["n"]
    zero = torch.zeros(B, n, device="cuda")
    base = sum(g["replay"](q, zero.clone())[0].float() for q in odd) / len(odd)
    print(f"U={U} b={b or 'V'}: base {float(base.mean()):.3f} (brains < 0.2: {int((base < 0.2).sum())})", flush=True)
    for c in (0.03, 0.05, 0.1, 0.15, 0.2):
        row = []
        for T in (10, 25, 50, 100):
            bb = zero.clone()
            acc = sum(g["replay"](q, bb, charge=c, decay=math.exp(-1 / T))[0].float() for q in odd) / len(odd)
            row.append(f"T{T} {float(acc.mean()):.3f}")
        print(f"   c={c}: " + "  ".join(row), flush=True)
    g["area"].bias = g["saved_bias"]
