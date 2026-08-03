"""#41 as literally titled: does read_only() roll back connectome MATERIALIZATION?

Three quantities, snapshotted before and after a probe:
  w        -- neurons the engine has materialized
  log_cols -- the self fiber's logical column watermark
  bytes    -- a hash of every connectome's weights, so ANY numeric change shows

frozen() is expected to change all three (it stops plasticity, not growth).
read_only() claims to change none of them. That is the claim under test.
"""
import sys, hashlib, numpy as np
sys.path.insert(0, r"F:\Github\assemblies")
from neural_assemblies.core.brain import Brain

N, K, P, BETA, ROUNDS = 1000, 50, 0.05, 0.05, 12


def snapshot(b):
    eng = b._engine_for(b.areas["A"])
    h = hashlib.blake2b(digest_size=8)
    shapes = {}
    for attr in ("_stim_conns", "_area_conns"):
        for src, per in sorted(getattr(eng, attr, {}).items()):
            for dst, conn in sorted(per.items()):
                w = np.ascontiguousarray(np.asarray(conn.weights, dtype=np.float64))
                h.update(f"{src}>{dst}".encode()); h.update(w.tobytes())
                shapes[f"{src}>{dst}"] = (w.shape, float(w.sum()),
                                          int((w != 0).sum()))
    return dict(w=int(eng.materialized_count("A")),
                log=int(eng.fiber_extent("A", "A")),
                digest=h.hexdigest(), shapes=shapes)


def trial(probe, seed):
    b = Brain(p=P, seed=seed, engine="numpy_sparse", norm_init=True)
    b.add_stimulus("S1", K); b.add_stimulus("S2", K)
    b.add_area("A", N, K, BETA)
    b.project({"S1": ["A"]}, {})
    for _ in range(ROUNDS):
        b.project({"S1": ["A"]}, {"A": ["A"]})

    before = snapshot(b)
    ctx = b.read_only() if probe == "read_only" else b.frozen()
    with ctx:
        b.project({"S2": ["A"]}, {"A": ["A"]})
    after = snapshot(b)

    changed = [k for k in before["shapes"]
               if before["shapes"][k] != after["shapes"].get(k)]
    return dict(dw=after["w"] - before["w"],
                dlog=after["log"] - before["log"],
                same=before["digest"] == after["digest"],
                changed=changed,
                detail={k: (before["shapes"][k], after["shapes"][k])
                        for k in changed})


if __name__ == "__main__":
    for probe in ("frozen", "read_only"):
        rows = [trial(probe, s) for s in range(6)]
        dw = np.mean([r["dw"] for r in rows])
        dlog = np.mean([r["dlog"] for r in rows])
        nsame = sum(r["same"] for r in rows)
        print(f"{probe+'()':13s}  d(w) {dw:+6.1f}   d(log_cols) {dlog:+6.1f}   "
              f"connectome BYTE-IDENTICAL in {nsame}/{len(rows)} trials")
        if rows[0]["changed"]:
            print(f"   fibers that changed (seed 0): {rows[0]['changed']}")
            for k, (b0, a0) in rows[0]["detail"].items():
                print(f"     {k:12s} shape {b0[0]}->{a0[0]}  "
                      f"sum {b0[1]:.1f}->{a0[1]:.1f}  nnz {b0[2]}->{a0[2]}")
        print()
