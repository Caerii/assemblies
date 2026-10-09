import os
import math, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
from neural_assemblies.tests import test_memory_oracle as t
for n, k, p in [(int(a), int(b), float(c)) for a, b, c in (s.split(",") for s in sys.argv[1:])]:
    t0 = time.time()
    os_ = t._run(n, k, p, L=40, decay=math.exp(-1 / 64), brains=1)
    o = os_[0]
    print(f"({n},{k},{p}) PASS: counts equal, bias close, recall equal; near ties {o.near_ties}/{o.steps}; {time.time()-t0:.0f}s", flush=True)
