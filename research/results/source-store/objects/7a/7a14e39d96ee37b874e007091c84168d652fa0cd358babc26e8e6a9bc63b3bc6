"""Does a STORED assembly receive privileged recurrent drive from an UNRELATED one?

Engine-neutral formulation of the `_expansion_col` question. Density is not
comparable across engines (explicit has no recruitment at all, so its nonzero
pattern is Bernoulli(p) by construction). What IS comparable is the dynamical
quantity: drive an assembly the area has never seen, and ask how much recurrent
drive the previously-stored assembly collects relative to the area average.

Ground truth is 1.0 -- an unrelated input has no reason to prefer a stored
assembly. Anything above that is a hair-trigger attractor.
"""
import sys, numpy as np
sys.path.insert(0, r"F:\Github\assemblies")
from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import ensemble

N, K, P, ROUNDS = 1000, 50, 0.05, 12


def build(engine, norm_init, seed, beta, explicit):
    b = Brain(p=P, seed=seed, engine=engine, norm_init=norm_init)
    b.add_stimulus("S1", K)
    b.add_stimulus("S2", K)
    if explicit:
        b.add_explicit_area("A", N, K, beta)
    else:
        b.add_area("A", N, K, beta)
    return b


def self_fiber(b):
    eng = b._engine_for(b.areas["A"])
    conn = eng._area_conns["A"]["A"]
    return eng, conn, np.asarray(conn.weights)


def run(engine, norm_init, seed, beta, explicit, probe='frozen'):
    b = build(engine, norm_init, seed, beta, explicit)
    # --- train a recurrent assembly on S1 ---
    b.project({"S1": ["A"]}, {})
    for _ in range(ROUNDS):
        b.project({"S1": ["A"]}, {"A": ["A"]})
    stored = np.asarray(b.areas["A"].winners).astype(int)

    # --- drive an INDEPENDENT stimulus as a PROBE ---
    # frozen() stops plasticity but NOT recruitment, so the probe itself grows
    # the area -- and the self fiber, not being a source here, is not expanded
    # with it. read_only() is the one that also suppresses growth.
    w_before = int(b._engine_for(b.areas["A"]).materialized_count("A") or 0)
    ctx = b.read_only() if probe == "read_only" else b.frozen()
    with ctx:
        b.project({"S2": ["A"]}, {})
        other = np.asarray(b.areas["A"].winners).astype(int)
    w_after = int(b._engine_for(b.areas["A"]).materialized_count("A") or 0)

    eng, conn, W = self_fiber(b)
    if W.ndim != 2 or W.size == 0:
        return None
    # THE EXTENT IS LOAD-BEARING and the two candidates are not the same
    # quantity, so report BOTH rather than picking one.
    #   w         -- neurons the engine has actually materialized. On the
    #                EXPLICIT engine this is len(winners)==k, which is not an
    #                extent at all.
    #   _log_cols -- the fiber's logical column watermark. Can run ahead of w:
    #                allocated columns that are not yet neurons.
    cand = {
        "w": int(eng.materialized_count("A") or W.shape[1]),
        "log": int(eng.fiber_extent("A", "A") or W.shape[1]),
        "shape": int(W.shape[1]),
    }
    out = {}
    for tag, live in cand.items():
        live = min(int(live), W.shape[0], W.shape[1])
        st = stored[stored < live]
        ot = other[other < live]
        if st.size == 0 or ot.size == 0:
            out[tag] = np.nan
            continue
        drive = W[ot, :live].sum(axis=0)
        out[tag] = float(drive[st].mean() / max(drive.mean(), 1e-12))
    overlap = len(set(stored.tolist()) & set(other.tolist())) / max(len(stored), 1)
    return dict(ratio=out["w"], r_log=out["log"], r_shape=out["shape"],
                grew=w_after - w_before,
                overlap=float(overlap),
                w=cand["w"], log=cand["log"], shape=cand["shape"])


def summarize(label, eng, ni, ex, probe, beta, seeds):
    """One line per arm, via `diagnostics.ensemble`.

    NOT a hand-rolled mean+SE. `ensemble` refuses fewer than 3 seeds and
    reports a t-based 95% interval rather than a standard error, so the
    printed number cannot be read as tighter than it is. The methodology
    ratchet catches the hand-rolled version -- it caught this file.
    """
    def metric(key):
        def _run(s):
            r = run(eng, ni, s, beta, ex, probe)
            return float("nan") if r is None else r[key]
        return _run

    parts = []
    for key, tag in (("ratio", "w"), ("r_log", "log"), ("r_shape", "shape")):
        e = ensemble(metric(key), list(seeds), f"{label}/{tag}")
        parts.append(f"{tag} {e.mean:5.3f}+/-{e.ci:5.3f}")
    grew = ensemble(metric("grew"), list(seeds), f"{label}/grew")
    ovl = ensemble(metric("overlap"), list(seeds), f"{label}/ovl")
    print(f"{label:32s}  " + "  ".join(parts)
          + f"   ovl {ovl.mean:5.3f}  GREW {grew.mean:+5.1f}")


ARMS = [
    ("sparse area    norm_init=False", "numpy_sparse", False, False),
    ("sparse area    norm_init=True",  "numpy_sparse", True,  False),
    ("EXPLICIT area  norm_init=False", "numpy_sparse", False, True),
    ("EXPLICIT area  norm_init=True",  "numpy_sparse", True,  True),
]

if __name__ == "__main__":
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    print(f"n={N} k={K} p={P} rounds={ROUNDS}   (+/- is a 95% CI, not an SE)")
    print("ratio = recurrent drive an UNRELATED assembly delivers to the STORED")
    print("        assembly, over the area mean.  Ground truth 1.0.")
    print("beta=0 is the NULL: no potentiation, so any excess is pure WIRING.\n")
    for probe in ("frozen", "read_only"):
        for beta in (0.0,):
            print(f"--- probe = {probe}()   beta = {beta} ---")
            for label, eng, ni, ex in ARMS:
                summarize(label, eng, ni, ex, probe, beta, seeds)
            print()
