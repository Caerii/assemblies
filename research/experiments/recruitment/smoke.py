"""Smoke test: can `refracted` and `synaptic_scaling` actually be engaged?

Run:  python -m research.experiments.recruitment.smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from neural_assemblies.core.brain import Brain  # noqa: E402


def build(n=1000, k=30, beta=0.05, rs=0.0, ss=False, lri=(0, 0.0)):
    b = Brain(p=0.05, seed=0, engine="numpy_sparse", synaptic_scaling=ss)
    b.add_explicit_area("PHON", 600, 30)
    b.add_area("LEX", n, k, beta=beta,
               refracted=(rs > 0), refracted_strength=rs,
               refractory_period=lri[0], inhibition_strength=lri[1])
    return b


def drive(b, pat, rounds=6):
    b.inhibit_areas(["LEX"])
    for _ in range(rounds):
        b.project(external_inputs={"PHON": pat},
                  projections={"PHON": ["LEX"]})
    return np.array(b.areas["LEX"].winners, dtype=np.int64)


def main():
    rng = np.random.default_rng(0)
    pats = [np.sort(rng.choice(600, 30, replace=False)).astype(np.uint32)
            for _ in range(20)]

    for label, kw in [("baseline", {}),
                      ("refracted rs=0.05", {"rs": 0.05}),
                      ("refracted rs=1.0", {"rs": 1.0}),
                      ("lri p=5 s=0.5", {"lri": (5, 0.5)}),
                      ("synaptic_scaling", {"ss": True})]:
        try:
            b = build(**kw)
            recs = []
            for i, pat in enumerate(pats):
                wb = int(b.areas["LEX"].w)
                a = drive(b, pat)
                recs.append(int(np.sum(a >= wb)) / max(1, len(a)))
            st = b._engine._areas["LEX"]
            bias = getattr(st, "_cumulative_bias", None)
            conn = b._engine._area_conns.get("PHON", {}).get("LEX")
            wshape = None
            wsum = None
            if conn is not None and getattr(conn, "weights", None) is not None:
                W = conn.weights
                wshape = getattr(W, "shape", None)
                try:
                    wsum = float(np.asarray(W).sum())
                except Exception:
                    wsum = "n/a"
            print(f"{label:22s} w={st.w:5d} recruit[0:5]="
                  f"{np.round(recs[:5],2)} recruit[-5:]={np.round(recs[-5:],2)} "
                  f"bias_len={0 if bias is None else len(bias)} "
                  f"bias_max={0.0 if bias is None or len(bias)==0 else float(np.max(bias)):.3f} "
                  f"PHON->LEX shape={wshape} sum={wsum}")
        except Exception as exc:  # noqa: BLE001
            print(f"{label:22s} FAILED: {type(exc).__name__}: {exc}")

    # Does synaptic scaling change PHON->LEX weights relative to baseline?
    outs = {}
    for label, ss in [("ss=False", False), ("ss=True", True)]:
        b = build(ss=ss)
        for pat in pats:
            drive(b, pat)
        conn = b._engine._area_conns.get("PHON", {}).get("LEX")
        W = np.asarray(conn.weights) if conn is not None else None
        outs[label] = (None if W is None else (float(W.sum()), float(W.max())))
    print("PHON->LEX (sum,max):", outs)
    print("identical:", outs["ss=False"] == outs["ss=True"])

    # what does the sparse engine think PHON's w is?
    b = build()
    drive(b, pats[0])
    print("sparse-engine PHON state w =", b._engine._areas["PHON"].w,
          " explicit_source =", getattr(b._engine._areas["PHON"],
                                        "explicit_source", None))
    conn = b._engine._area_conns.get("PHON", {}).get("LEX")
    print("PHON->LEX conn type:", type(conn).__name__,
          "sparse:", getattr(conn, "sparse", None),
          "weights type:", type(getattr(conn, "weights", None)).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
