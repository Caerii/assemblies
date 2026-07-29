"""What is DIFFERENT inside a spawned worker? (task #49)

The ladder's runtime self-check fired at n=4000 M=512 D=3: serial produced 512
distinct assemblies at margin 6.34, the worker produced 1 at margin 1.00 and
spread 0.9998. Same code, same seed, same arguments -- so the divergence is in
PROCESS STATE, not in the experiment.

STAGE 1 (single stimulus, n=4000) found ZERO differing fields: engine, module,
norm_init, fidelity, numpy version, and the resulting winners all matched
exactly. So the divergence needs MANY stimuli to appear, which is consistent
with where it was seen (M=512 and n=10000, never M=64).

STAGE 2 therefore builds the real thing -- M stimuli feed-forward into one
shared area -- and reports the distinctness of the STIMULUS CONNECTIVITY
itself, not just of the assemblies. If two stimuli have identical incoming
weight vectors they cannot produce different winners, and every downstream
collapse follows from that one fact.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def probe(n: int, m_items: int, seed: int) -> dict:
    """Build M stimuli into one shared area and report where distinctness dies."""
    import numpy as np

    from neural_assemblies.core.brain import Brain

    brain = Brain(p=0.05, seed=seed, n_hint=n)
    brain.add_area("L", n, 50, beta=0.20)
    for i in range(m_items):
        brain.add_stimulus(f"w{i}", 50)

    assemblies = []
    for i in range(m_items):
        for _ in range(2):
            brain.project({f"w{i}": ["L"]}, {})
        assemblies.append(tuple(sorted(int(x) for x in brain.areas["L"].winners)))

    # Fingerprint the stimulus->area connectivity. Identical vectors across
    # stimuli would make identical winners arithmetically unavoidable.
    eng = brain._engine
    sigs = []
    for i in range(min(m_items, 64)):
        try:
            conn = eng._stim_connectomes[f"w{i}"]["L"]  # type: ignore[attr-defined]
            wv = np.asarray(getattr(conn, "weights", conn))
            sigs.append((float(wv[:200].sum()), int(wv.size)))
        except Exception as exc:  # noqa: BLE001
            sigs.append(("ERR", str(exc)[:40]))

    return {
        "pid": os.getpid(),
        "distinct_assemblies": len(set(assemblies)),
        "distinct_stim_sigs": len(set(sigs)),
        "n_stim_sigs_checked": len(sigs),
        "first_sig": sigs[0] if sigs else None,
        "second_sig": sigs[1] if len(sigs) > 1 else None,
        "area_w": int(brain.areas["L"].w),
        "first_assembly_head": list(assemblies[0][:6]),
        "last_assembly_head": list(assemblies[-1][:6]),
    }


def _child(args):
    return probe(*args)


if __name__ == "__main__":
    import multiprocessing as mp

    N = int(os.environ.get("PROBE_N", "4000"))
    M = int(os.environ.get("PROBE_M", "512"))
    SEED = 42

    parent = probe(N, M, SEED)
    # STAGE 3: reproduce `parallel_seeds`'s own asymmetry. It assigns
    # PYTHONHASHSEED in the PARENT after startup, which cannot change the
    # parent's own hashing but IS inherited by every spawned child. So parent
    # and children randomize hashes differently -- the exact failure class this
    # project has already been bitten by once.
    if os.environ.get("PROBE_HASHSEED_ASYMMETRY", "1") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
    with mp.get_context("spawn").Pool(1) as pool:
        child = pool.map(_child, [(N, M, SEED)])[0]

    print(f"\n  worker divergence probe  (n={N} M={M} seed={SEED})\n")
    keys = [k for k in parent if k != "pid"]
    width = max(len(k) for k in keys)
    ndiff = 0
    for k in keys:
        same = parent[k] == child[k]
        ndiff += not same
        flag = "   " if same else "<<<"
        print(f"  {flag} {k:<{width}}  parent={parent[k]!r:<26} child={child[k]!r}")
    print(f"\n  {ndiff} field(s) differ between parent and spawned child")
