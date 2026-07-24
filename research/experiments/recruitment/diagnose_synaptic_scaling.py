"""Is synaptic_scaling a no-op on an explicit->sparse (PHON->LEX) fiber?

Instruments `_normalize_area_columns` (monkeypatched in the EXPERIMENT
process only; `neural_assemblies/` is untouched on disk) to record which
fibers it actually rescales.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from neural_assemblies.core.brain import Brain  # noqa: E402
from neural_assemblies.core.numpy_engine._sparse import (  # noqa: E402
    NumpySparseEngine,
)

CALLS = []


def main():
    orig = NumpySparseEngine._normalize_area_columns

    def traced(self, target, from_areas, winners):
        for src in from_areas:
            conn = self._area_conns[src][target]
            w = getattr(conn, "weights", None)
            rows = int(self._areas[src].w)
            before = None if w is None or getattr(w, "ndim", 0) != 2 else float(
                np.asarray(w).sum())
            CALLS.append((src, target, rows,
                          None if w is None else getattr(w, "shape", None),
                          self.synaptic_scaling))
        out = orig(self, target, from_areas, winners)
        return out

    NumpySparseEngine._normalize_area_columns = traced

    rng = np.random.default_rng(0)
    pats = [np.sort(rng.choice(600, 30, replace=False)).astype(np.uint32)
            for _ in range(10)]

    for mode in ("ff", "rec"):
        for ss in (False, True):
            CALLS.clear()
            b = Brain(p=0.05, seed=0, engine="numpy_sparse",
                      synaptic_scaling=ss)
            b.add_explicit_area("PHON", 600, 30)
            b.add_area("LEX", 1000, 30, beta=0.05)
            for pat in pats:
                b.inhibit_areas(["LEX"])
                for r in range(6):
                    proj = ({"PHON": ["LEX"]} if mode == "ff" or r == 0
                            else {"PHON": ["LEX"], "LEX": ["LEX"]})
                    b.project(external_inputs={"PHON": pat}, projections=proj)
            seen = {}
            for src, tgt, rows, shape, flag in CALLS:
                seen.setdefault((src, tgt), []).append(rows)
            print(f"mode={mode} ss={ss}")
            for kk, rr in seen.items():
                acts = sum(1 for r in rr if r > 0)
                print(f"   fiber {kk[0]}->{kk[1]}: {len(rr)} calls, "
                      f"src.w>0 on {acts} of them  (rows sample={rr[:3]})")
            conn = b._engine._area_conns["PHON"]["LEX"]
            print("   PHON->LEX weight sum:",
                  float(np.asarray(conn.weights).sum()))


if __name__ == "__main__":
    raise SystemExit(main())
