"""E6: name the step's undeclared memory channel. [[SEQ-EXACT-RECOVERY]]

Implements the Addendum in `research/notes/PREREG_s5_cliff_anatomy.md`. The
census proved exact-in -> exact-out for every (state, symbol) pair, yet
trajectories deviate -- so the step is not memoryless, and the only structural
difference between the two contexts is RESIDUAL CONTENT: the census inhibits
the arc before every step, the sequence does not.

On each derailed seed: replay (deterministic, verified), find the FIRST step d
where the live state leaves the block of its own label, then re-run that
single step in three ways:

  P-census    inhibit arc+state, cue block, step.          Must be exact.
  P-residual  same, but the arc is first restored to its step-(d-1) winners.
              If this reproduces the in-sequence deviation, the residual is
              the channel.
  P-arc       is the in-sequence ARC at step d already different from the
              census arc for the same pair? Locates where deviation enters.

TWO INDEX SPACES, deliberately handled: the arc residual is captured as
COMPACT winners (what `set_winners` takes; stable here because probe() forbids
recruitment and the arc is saturated), while every comparison across contexts
uses NEURON IDS via `_snap`. See [[two-index-spaces-compact-vs-neuron-id]].
"""
from __future__ import annotations

import json
import os
import sys
import random

import numpy as np

from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.diagnostics import assembly_overlap
from neural_assemblies.programs.word_problems import true_trajectory

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from seq_s5_word_problem import build, run_tiered  # noqa: E402

RESULTS = os.path.join(_HERE, "seq_s5_cliff_anatomy_results.json")
LONGEST = 500


def _ov(a, b):
    return assembly_overlap(np.asarray(a.winners), np.asarray(b.winners))


def worker(group_name, seed, _arm="trained"):
    group, fsm, symbols = build(group_name, seed, "trained")
    b = fsm.brain
    rng = random.Random(seed + 4242)
    word = [rng.choice(symbols) for _ in range(LONGEST)]
    start = group.label(group.identity)

    # Replay, recording label, state snap (neuron IDs), arc snap (neuron IDs)
    # and arc COMPACT winners per step.
    labels, state_snaps, arc_snaps, arc_compact = [], [], [], []
    with b.probe():
        b.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state(start)
        fsm._unfix_state()
        for sym in word:
            labels.append(fsm.step(sym))
            state_snaps.append(_snap(b, fsm.state_area))
            arc_snaps.append(_snap(b, fsm.arc_area))
            arc_compact.append(b.areas[fsm.arc_area].winners.copy())

    truth = true_trajectory(group, word)
    first_bad = next((i for i, (a, t) in enumerate(zip(labels, truth))
                      if a != t), LONGEST)
    d = next((i for i, (l, s) in enumerate(zip(labels, state_snaps))
              if _ov(s, fsm.state_assembly(l)) < 1.0), None)
    if d is None:
        return {"first_bad": first_bad, "first_dev": None, "note": "clean"}

    prev_label = labels[d - 1] if d > 0 else start
    sym_d = word[d]
    inseq_state, inseq_arc = state_snaps[d], arc_snaps[d]

    # P-census: the pair in isolation. SNAPPED INSIDE THE PROBE -- the first
    # version snapped after fsm.run's internal probe returned, and probe()
    # RESTORES winners on exit, so it read the pre-run residue and produced
    # the contradiction that exposed the dead census (see prereg Addendum 2).
    with b.probe():
        b.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state(prev_label)
        fsm._unfix_state()
        fsm.step(sym_d)
        cen_state, cen_arc = _snap(b, fsm.state_area), _snap(b, fsm.arc_area)

    # P-residual: identical, except the arc holds its step-(d-1) winners when
    # the step's first projection fires.
    with b.probe():
        b.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state(prev_label)
        fsm._unfix_state()
        if d > 0:
            w = np.asarray(arc_compact[d - 1], dtype=np.uint32)
            b.areas[fsm.arc_area].unfix_assembly()
            b.areas[fsm.arc_area]._winners = w
            b._engine.set_winners(fsm.arc_area, w)
        fsm.step(sym_d)
        res_state, res_arc = _snap(b, fsm.state_area), _snap(b, fsm.arc_area)

    exp_block = fsm.state_assembly(labels[d])
    return {
        "first_bad": int(first_bad),
        "first_dev": int(d),
        "dev_onblock": float(_ov(inseq_state, exp_block)),
        "census_exact": bool(_ov(cen_state, exp_block) == 1.0),
        "residual_reproduces_state": bool(
            _ov(res_state, inseq_state) == 1.0),
        "residual_reproduces_arc": bool(_ov(res_arc, inseq_arc) == 1.0),
        "arc_differs_from_census": bool(_ov(inseq_arc, cen_arc) < 1.0),
        "arc_inseq_vs_census": float(_ov(inseq_arc, cen_arc)),
        "state_res_vs_census": float(_ov(res_state, cen_state)),
    }


def main():
    with open(RESULTS, encoding="utf-8") as fh:
        prior = json.load(fh)
    cells = [(g, s, "trained")
             for g, rows in prior["groups"].items()
             for s, v in zip(prior["seeds"], rows) if v["first_bad"] < 500]
    print("=== E6: the memory channel ===")
    print(f"    {len(cells)} derailed seeds from the registered study\n")
    r = run_tiered(cells, worker_fn=worker)

    print(f"\n    {'group':7s} {'seed':>4s} {'dev@':>5s} {'onblock':>8s} "
          f"{'census':>7s} {'res==seq':>9s} {'arc==seq':>9s} {'arcdiff':>8s}")
    out = []
    for (g, s, _a), v in sorted(r.items()):
        out.append({"group": g, "seed": s, **v})
        if v.get("note") == "clean":
            print(f"    {g:7s} {s:4d}  REPLAY CLEAN -- determinism violated?")
            continue
        print(f"    {g:7s} {s:4d} {v['first_dev']:5d} "
              f"{v['dev_onblock']:8.4f} {str(v['census_exact'])[0]:>7s} "
              f"{str(v['residual_reproduces_state'])[0]:>9s} "
              f"{str(v['residual_reproduces_arc'])[0]:>9s} "
              f"{v['arc_inseq_vs_census']:8.4f}", flush=True)

    live = [v for v in out if v.get("note") != "clean"]
    print("\n=== VERDICT ===")
    print(f"  census exact everywhere:      "
          f"{all(v['census_exact'] for v in live)}")
    print(f"  residual reproduces sequence: "
          f"{sum(v['residual_reproduces_state'] for v in live)}/{len(live)}")
    print(f"  deviation enters at the ARC:  "
          f"{sum(v['arc_differs_from_census'] for v in live)}/{len(live)}")

    path = os.path.join(_HERE, "seq_s5_memory_channel_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
