"""Exploratory: the shape of the tau peak (seeds 980-999), and tau = n/k at n/k = 300 (seeds 980-989)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_taunk.py")).read().split("for n, k, p in ((4000")[0])
for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
    n, k, p = 4000, 200, 0.15
    tau = round(f * n / k); r5, r9 = rho50(n, k, p, tau)
    print(f"({n}, {k}, {p}) n/k=20 tau={tau} tau/(n/k)={f}: rho_50 {r5 and round(r5,4)} rho_90 {r9 and round(r9,4)}", flush=True)
seeds[:] = list(range(980, 990))
for tau in (300, 200, 400):
    r5, r9 = rho50(15000, 50, 0.7, tau)
    print(f"(15000, 50, 0.7) n/k=300 tau={tau} tau/(n/k)={tau/300:.2f}: rho_50 {r5 and round(r5,4)} rho_90 {r9 and round(r9,4)}", flush=True)
