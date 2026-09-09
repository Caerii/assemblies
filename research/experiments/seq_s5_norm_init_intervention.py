"""Does norm_init delete the soft transitions? [[SEQ-EXACT-RECOVERY]]

Implements `research/notes/sequence/PREREG_s5_norm_init_intervention.md`. Two arms on
the registered S5 protocol -- norm_init=False (control, re-establishing the
baseline in-study) and norm_init=True (intervention) -- with the soft census
now recording IDENTITIES: intruder and displaced neuron per soft pair, plus
the intruder's afferent-mass percentile among all state-area columns.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.diagnostics import assembly_overlap
from neural_assemblies.programs.word_problems import (
    true_trajectory, word_problem_fsm,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from seq_s5_word_problem import (  # noqa: E402
    GROUP_NAMES, SEEDS, build, run_tiered,
)

LONGEST = 500


def _ov(a, b):
    return assembly_overlap(np.asarray(a.winners), np.asarray(b.winners))


def _column_mass(brain, src, tgt):
    """Afferent weight mass per TARGET column of the src -> tgt fiber.

    Handles both storages without densifying CSR: `CSRWeights` wraps a scipy
    matrix at `._m`, whose axis-0 sum is sparse-native; the dense buffer sums
    directly. Trailing padding columns beyond the logical width are dropped.
    """
    conn = brain._engine._area_conns[src][tgt]
    w = conn.weights
    if hasattr(w, "_m"):
        sums = np.asarray(w._m.sum(axis=0)).ravel()
    else:
        sums = np.asarray(w, dtype=np.float64).sum(axis=0)
    cols = getattr(conn, "_log_cols", None) or len(sums)
    return sums[:cols]


def worker(group_name, seed, norm_flag):
    norm_init = norm_flag == "norm"
    group, fsm, symbols = build(group_name, seed, "trained",
                                norm_init=norm_init)
    b = fsm.brain
    rng = random.Random(seed + 4242)
    word = [rng.choice(symbols) for _ in range(LONGEST)]
    start = group.label(group.identity)
    truth = true_trajectory(group, word)

    labels, onblock = [], []
    with b.probe():
        b.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state(start)
        fsm._unfix_state()
        for sym in word:
            labels.append(fsm.step(sym))
            onblock.append(_ov(_snap(b, fsm.state_area),
                               fsm.state_assembly(labels[-1])))
    first_bad = next((i for i, (a, t) in enumerate(zip(labels, truth))
                      if a != t), LONGEST)
    first_dev = next((i for i, o in enumerate(onblock) if o < 1.0), None)

    _states, _syms, transitions = word_problem_fsm(group)
    table = {(fr, sym): to for fr, sym, to in transitions}
    soft, hard = [], []
    exact = {}
    for st in fsm.states:
        for sym in symbols:
            with b.probe():
                b.inhibit_areas([fsm.arc_area, fsm.state_area])
                fsm._cue_state(st)
                fsm._unfix_state()
                label = fsm.step(sym)
                live = _snap(b, fsm.state_area)
            intended = fsm.state_assembly(table[(st, sym)])
            if label != table[(st, sym)]:
                hard.append((st, sym))
                continue
            got = set(np.asarray(live.winners).tolist())
            want = set(np.asarray(intended.winners).tolist())
            if got != want:
                soft.append({
                    "pair": [st, sym],
                    "overlap": len(got & want) / len(want),
                    "intruders": sorted(got - want),
                    "displaced": sorted(want - got),
                })
    for L in (10, 50, 100, 500):
        exact[str(L)] = bool(labels[:L] == truth[:L])

    # Hub diagnostics: intruder afferent-mass percentile over state columns.
    intruder_pct, shared = [], None
    if soft:
        mass = _column_mass(b, fsm.arc_area, fsm.state_area)
        eng_area = b._engine._areas[fsm.state_area]
        inv = {int(n): i for i, n in
               enumerate(eng_area.compact_to_neuron_id)} \
            if eng_area.compact_to_neuron_id else None
        for rec in soft:
            for nid in rec["intruders"]:
                col = inv[int(nid)] if inv else int(nid)
                if col < len(mass):
                    pct = float((mass < mass[col]).mean())
                    intruder_pct.append(pct)
        all_intruders = [tuple(rec["intruders"]) for rec in soft]
        shared = len(set(all_intruders)) < len(all_intruders)

    bad_pairs = {tuple(rec["pair"]) for rec in soft} | set(hard)
    prev, predicted = start, None
    for i, sym in enumerate(word):
        if (prev, sym) in bad_pairs:
            predicted = i
            break
        prev = table[(prev, sym)]

    return {
        "arm": norm_flag, "first_bad": int(first_bad),
        "first_dev": first_dev, "predicted_first_dev": predicted,
        "v2_exact": bool(first_dev == predicted),
        "n_soft": len(soft), "n_hard": len(hard), "soft": soft,
        "exact": exact, "intruder_pct": intruder_pct,
        "intruder_shared_within_organ": shared,
    }


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print("=== norm_init intervention on the S5 soft transitions ===\n")

    from _results import results_path
    with open(results_path("sequence", "seq_s5_soft_census_results.json"),
              encoding="utf-8") as fh:
        registered = {(v["group"], v["seed"]): v["n_soft"]
                      for v in json.load(fh)}

    # Control FIRST, as committed.
    print("  [control] norm_init=False -- must reproduce the registered census")
    rc = run_tiered([(g, s, "base") for g in GROUP_NAMES for s in seeds],
                    worker_fn=worker)
    n1 = True
    for (g, s, _f), v in sorted(rc.items()):
        match = v["n_soft"] == registered.get((g, s))
        n1 &= match
        if not match:
            print(f"    {g}/{s}: n_soft {v['n_soft']} vs registered "
                  f"{registered.get((g, s))}  MISMATCH")
    print(f"    N1 {'PASS' if n1 else 'FAIL'} (registered soft counts "
          f"reproduced on all {len(rc)} organs)")
    if not n1:
        print("    N1 failed -- instrument moved; intervention NOT run.")
        _write({"n1": False})
        return

    print("\n  [intervention] norm_init=True")
    ri = run_tiered([(g, s, "norm") for g in GROUP_NAMES for s in seeds],
                    worker_fn=worker)

    out = {"control": {}, "intervention": {}}
    total_soft_i = total_hard_i = 0
    exact500 = {}
    print(f"\n    {'group':7s} {'seed':>4s} {'ctrl soft':>9s} "
          f"{'norm soft':>9s} {'norm hard':>9s} {'norm exact@500':>14s}")
    for g in GROUP_NAMES:
        e500 = 0
        for s in seeds:
            c, i = rc[(g, s, "base")], ri[(g, s, "norm")]
            out["control"][f"{g}/{s}"] = c
            out["intervention"][f"{g}/{s}"] = i
            total_soft_i += i["n_soft"]
            total_hard_i += i["n_hard"]
            e500 += i["exact"]["500"]
            print(f"    {g:7s} {s:4d} {c['n_soft']:9d} {i['n_soft']:9d} "
                  f"{i['n_hard']:9d} {str(i['exact']['500']):>14s}",
                  flush=True)
        exact500[g] = e500

    pcts = [p for v in rc.values() for p in v["intruder_pct"]]
    print("\n=== BARS ===")
    print(f"  PASS  N1 control reproduces registered census")
    n2 = total_soft_i == 0
    print(f"  {'PASS' if n2 else 'FAIL'}  N2 intervention soft rate is ZERO "
          f"(total soft {total_soft_i})")
    n3 = all(v == len(seeds) for v in exact500.values())
    print(f"  {'PASS' if n3 else 'FAIL'}  N3 exact@500 10/10 everywhere "
          f"under norm_init  {exact500}")
    n4 = bool(pcts) and min(pcts) >= 0.99
    print(f"  {'PASS' if n4 else 'FAIL'}  N4 control intruders above 99th "
          f"pct of afferent mass (min {min(pcts) if pcts else float('nan'):.4f}, "
          f"n={len(pcts)})")
    n5 = total_hard_i == 0
    print(f"  {'PASS' if n5 else 'FAIL'}  N5 no hard defects introduced "
          f"(total hard {total_hard_i})")
    shared = [v["intruder_shared_within_organ"] for v in rc.values()
              if v["intruder_shared_within_organ"] is not None]
    print(f"  info: organs whose soft pairs share an intruder: "
          f"{sum(shared)}/{len(shared)}")

    out["verdicts"] = {"N1": n1, "N2": n2, "N3": n3, "N4": n4, "N5": n5}
    _write(out)


def _write(out):
    path = os.path.join(_HERE, "seq_s5_norm_init_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
