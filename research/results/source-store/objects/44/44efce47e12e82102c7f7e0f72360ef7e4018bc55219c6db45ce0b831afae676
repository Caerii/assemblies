"""The substitution fires. Is it a LEVEL SHIFT or does it decide the number?

`erp_or_else_census.py` established that `phrase_stability` is undefined on
84/135 readings, always for the same reason (`VP` has no self-fiber), and that
the rate is NOT equal across arms: ~50% on grammatical and category_violation,
~75% on novel_noun.

A rate is not yet an impact. The aggregation at `erp/adapters.py:637` is

    stabilities   = [r.or_else(0.0) for r in readings]
    mean          = sum(stabilities)/len(stabilities) if stabilities else 1.0
    instability   = 1.0 - mean

so the two aggregations differ in a way that is NOT a shift:

  * one undefined among several  -> substituting 0.0 drags the mean DOWN, i.e.
    raises instability, i.e. raises p600;
  * ALL readings undefined       -> substituting gives mean 0.0, instability
    1.0, the MAXIMUM. Dropping them gives an empty list, which takes the
    `else 1.0` branch: mean 1.0, instability 0.0, the MINIMUM.

Same probe, opposite ends of the range. So the honest question is not "how big
is the correction" but "on how many probes does the substituted term carry the
whole answer", and whether those probes are concentrated in one arm.

This records, per probe: the areas asked, which were undefined, and the p600 the
other aggregation would have produced -- recomputed from the SAME captured
`anchored` term, so the delta isolates the stability term alone.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation import (   # noqa: E402
    calibrate_erp_thresholds,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp import (  # noqa: E402
    adapters, frames as frames_mod, runner as runner_mod,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)

SEEDS = [11, 12, 42]

_LABEL_OF = {
    tuple(words): label
    for label, _d, words in frames_mod.DEFAULT_CALIBRATION_FRAMES
}


def _label_for(known):
    kt = tuple(known)
    if kt in _LABEL_OF:
        return _LABEL_OF[kt]
    for words, label in _LABEL_OF.items():
        if kt == tuple(w for w in words if w in set(known)):
            return label
    return "?unattributed"


class Probes:
    def __init__(self):
        self.rows = []
        self.cur = None

    def open(self, label, word):
        self.cur = {"label": label, "word": word, "readings": [],
                    "anchored": None, "p600": None, "role_area": None}

    def close(self):
        if self.cur is not None and self.cur["p600"] is not None:
            self.rows.append(self.cur)
        self.cur = None


def _install(P):
    undo = []

    orig_stab = adapters.phrase_stability

    def stab(*a, **kw):
        m = orig_stab(*a, **kw)
        if P.cur is not None:
            area = a[1] if len(a) > 1 else kw.get("area")
            P.cur["readings"].append((area, m.defined,
                                      float(m) if m.defined else None))
        return m

    adapters.phrase_stability = stab
    undo.append(lambda: setattr(adapters, "phrase_stability", orig_stab))

    orig_anch = adapters.anchored_p600_live

    def anch(*a, **kw):
        m = orig_anch(*a, **kw)
        if P.cur is not None:
            P.cur["anchored"] = m.or_else(
                float((m.detail or {}).get("legacy", 0.0)))
        return m

    adapters.anchored_p600_live = anch
    undo.append(lambda: setattr(adapters, "anchored_p600_live", orig_anch))

    orig_mli = adapters.measure_live_integration

    def mli(parser, word, category, **kw):
        out = orig_mli(parser, word, category, **kw)
        if P.cur is not None:
            P.cur["p600"], P.cur["role_area"] = out[0], out[1]
        return out

    # EVERY namespace that imported it by name. `runner.py:145` is the live one
    # on the non-warm path (warm_start is only on under sweep mode), so patching
    # `adapters` and `frames` alone captured ZERO probes on the first run --
    # the one-sibling bug, in the diagnostic.
    for mod in (adapters, frames_mod, runner_mod):
        setattr(mod, "measure_live_integration", mli)
        undo.append(
            lambda m=mod: setattr(m, "measure_live_integration", orig_mli))

    for pname in ("_probe_at_critical_position",
                  "_probe_at_critical_position_warm"):
        op = getattr(frames_mod, pname)

        def make(o):
            def wrapped(parser, known, pos, **kw):
                P.open(_label_for(known), known[pos] if pos < len(known) else "")
                try:
                    return o(parser, known, pos, **kw)
                finally:
                    P.close()
            return wrapped

        setattr(frames_mod, pname, make(op))
        undo.append(lambda n=pname, o=op: setattr(frames_mod, n, o))

    return lambda: [f() for f in reversed(undo)]


def main():
    rows = []
    for seed in SEEDS:
        P = Probes()
        restore = _install(P)
        try:
            parser = get_parser_cache().fork("SENTENCES", seed=seed)
            calibrate_erp_thresholds(parser)
        finally:
            restore()
        for r in P.rows:
            r["seed"] = seed
        rows.extend(P.rows)

    W_STAB = adapters.N400_WEIGHT
    W_ANCH = adapters.P600_WEIGHT
    print(f"p600 = {W_STAB} * phrase_instability + {W_ANCH} * anchored")
    print(f"{len(rows)} probes over seeds {SEEDS}")
    print()

    by_label = defaultdict(list)
    for r in rows:
        n = len(r["readings"])
        u = sum(1 for _a, d, _v in r["readings"] if not d)
        defined = [v for _a, d, v in r["readings"] if d]
        mean_sub = (sum(0.0 if not d else v for _a, d, v in r["readings"]) / n
                    if n else 1.0)
        mean_drop = sum(defined) / len(defined) if defined else 1.0
        anchored = r["anchored"] or 0.0
        p_sub = W_STAB * (1.0 - mean_sub) + W_ANCH * anchored
        p_drop = W_STAB * (1.0 - mean_drop) + W_ANCH * anchored
        r.update(n=n, u=u, all_undef=(n > 0 and u == n),
                 p_sub=p_sub, p_drop=p_drop, delta=p_drop - p_sub)
        by_label[r["label"]].append(r)

    hdr = (f"{'arm':<20} {'probes':>7} {'areas/probe':>12} {'undef':>8} "
           f"{'ALL undef':>10} {'p600 shipped':>13} {'p600 honest':>12} "
           f"{'delta':>9}")
    print(hdr)
    print("-" * len(hdr))
    for label in sorted(by_label):
        rs = by_label[label]
        npr = len(rs)
        ar = sum(r["n"] for r in rs) / npr
        ud = sum(r["u"] for r in rs)
        tot = sum(r["n"] for r in rs)
        allu = sum(1 for r in rs if r["all_undef"])
        ps = sum(r["p_sub"] for r in rs) / npr
        pd = sum(r["p_drop"] for r in rs) / npr
        print(f"{label:<20} {npr:>7} {ar:>12.2f} {ud:>4}/{tot:<3} "
              f"{allu:>6}/{npr:<3} {ps:>13.4f} {pd:>12.4f} {pd - ps:>+9.4f}")

    print()
    print("PROBES WHERE EVERY READING WAS UNDEFINED -- the substituted term is")
    print("the WHOLE stability answer, and the two aggregations sit at opposite")
    print("ends of its range:")
    allu = [r for r in rows if r["all_undef"]]
    if not allu:
        print("  none.")
    else:
        seen = set()
        for r in allu:
            key = (r["label"], r["word"], r["role_area"],
                   tuple(a for a, _d, _v in r["readings"]))
            if key in seen:
                continue
            seen.add(key)
            areas = ", ".join(a for a, _d, _v in r["readings"])
            print(f"  {r['label']:<20} word={r['word']:<8} "
                  f"role={r['role_area']:<14} areas=[{areas}]  "
                  f"shipped p600={r['p_sub']:.4f} -> honest {r['p_drop']:.4f}")
        print(f"  ({len(allu)} such probes of {len(rows)})")


if __name__ == "__main__":
    main()
