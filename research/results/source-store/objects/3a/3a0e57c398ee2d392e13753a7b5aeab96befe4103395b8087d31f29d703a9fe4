import os
import math, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import torch
from research.experiments import memory_load_drift as md, memory_load_law as ml, memory_threshold_law as tl, memory_sequences as sq
n, k, p = 4000, 400, 0.5
spec = {"n": n, "k": k, "p": p, "beta": round(tl.theta(n, k, p), 5), "batch": 187, "tau": 5}
print(md.reliability(spec, 40, [980, 981, 982], "cuda"))
spec["tau"] = 33; print(md.reliability(spec, 40, [980, 981, 982], "cuda"))
spec2 = dict(spec, k=60, beta=round(tl.theta(n, 60, p), 5)); print(md.reliability(spec2, 40, [980, 981, 982], "cuda"))
st = torch.arange(400, device="cuda"); c = md.cue(st, 980, 40, 400, "cuda"); print(c.shape, c[:10])
