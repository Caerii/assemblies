import os, sys
sys.argv = [sys.argv[0], "none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_vocab.py")).read().split("mode = sys.argv[1]")[0])
for rho in (0.03, 0.05, 0.08):
    L = max(1, round(rho * ml.unit(n, k, p) / l)) * l
    for uses in (5, 10, 20, 40):
        V = max(2, round(L / uses))
        L_, M, w, same, ff = run(rho, V=V)
        print(f"rho={rho} uses/word={uses} V={V} L={L_}: whole {w:.3f} same-word overlap {same and round(same,3)}", flush=True)
