"""How often does an ERP number contain a SUBSTITUTED value, and does it matter?

WHY. `e8455cc` found that `area_health`'s margin went undefined exactly when
separation was PERFECT, so the NaN-filter that looked like hygiene averaged the
worse half of the ensemble. **Undefinedness was correlated with the outcome.**

Every `.or_else(fallback)` on the ERP path is the same bet, and none of them has
been measured:

  * `erp/adapters.py:637`  stability readings -> `.or_else(0.0)`, then MEANED.
    A 0.0 is not neutral here: the caller reads `1 - mean_stability` as phrase
    INSTABILITY, so every substituted reading pushes p600 UP.
  * `erp/adapters.py:650`  `anchored_p600_live` -> `.or_else(legacy)`.
  * `erp/frames.py:277`, `erp/runner.py:114`  `measure_lexical_surprise` ->
    `.or_else(legacy)`, and an absent prefix -> a hard 0.0 N400.
  * `parser_mixins/roles.py:305`  role margins -> `.or_else(0.0)` into a
    NORMALIZED distribution, where a zero is not an abstention, it is a vote.

WHAT THIS SCRIPT DOES NOT DO. It does not restate the calibration driver. It
wraps the two probe entry points in `erp/frames.py` to learn WHICH FRAME is in
flight, then calls the real `calibrate_erp_thresholds`. The frame->label map is
READ from the frame table, not re-derived (rule 2c: a diagnostic that
reimplements its subject will eventually disagree with it).

THE QUESTION IS TWO QUESTIONS, and the second is the one that matters:

  1. Does the fallback ever fire? If never, the honest aggregation is free --
     exactly the #110 outcome, and worth having in writing.
  2. If it fires, is it INDEPENDENT OF CONDITION? A substitution that fires
     equally on all three arms shifts a level. One that fires on the violation
     arm and not the grammatical arm IS the effect being reported.
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
from neural_assemblies.assembly_calculus.emergent.parser_mixins.roles import (  # noqa: E402
    RoleBindingMixin,
)

SEEDS = [11, 12, 42]

#: label -> the frames that carry it, read from the production table.
_LABEL_OF = {
    tuple(words): label
    for label, _desc, words in frames_mod.DEFAULT_CALIBRATION_FRAMES
}


def _label_for(known):
    """Which arm is this? `known` is the IN-VOCABULARY prefix of a frame."""
    kt = tuple(known)
    if kt in _LABEL_OF:
        return _LABEL_OF[kt]
    for words, label in _LABEL_OF.items():
        if list(known) == [w for w in words if w in kt]:
            return label
        if kt == tuple(w for w in words if w in set(known)):
            return label
    return "?unattributed"


class Census:
    """Tally of (site, label) -> calls / undefined, plus the reasons given."""

    def __init__(self):
        self.calls = defaultdict(int)
        self.undef = defaultdict(int)
        self.why = defaultdict(set)
        self.current = "?outside-frame"

    def record(self, site, measured):
        key = (site, self.current)
        self.calls[key] += 1
        if not measured.defined:
            self.undef[key] += 1
            self.why[site].add(measured.why)
        return measured

    def sites(self):
        return sorted({s for s, _ in self.calls})

    def labels(self):
        return sorted({lb for _, lb in self.calls})


def _install(census):
    """Wrap producers in every namespace that IMPORTED them by name.

    `frames.py` and `runner.py` do `from .adapters import measure_lexical_surprise`,
    so patching `adapters.measure_lexical_surprise` alone would miss both -- the
    one-sibling bug, in the diagnostic this time.
    """
    undo = []

    def wrap(mod, name, site):
        orig = getattr(mod, name)

        def wrapped(*a, **kw):
            return census.record(site, orig(*a, **kw))

        setattr(mod, name, wrapped)
        undo.append(lambda: setattr(mod, name, orig))

    wrap(adapters, "phrase_stability", "phrase_stability")
    wrap(adapters, "anchored_p600_live", "anchored_p600_live")
    for mod in (adapters, frames_mod, runner_mod):
        if hasattr(mod, "measure_lexical_surprise"):
            wrap(mod, "measure_lexical_surprise", "measure_lexical_surprise")
    wrap(RoleBindingMixin, "_role_binding_margin", "_role_binding_margin")

    # Attribution: learn which frame is in flight from the probe entry points.
    for probe_name in ("_probe_at_critical_position",
                       "_probe_at_critical_position_warm"):
        orig_probe = getattr(frames_mod, probe_name)

        def make(op):
            def wrapped_probe(parser, known, pos, **kw):
                prev = census.current
                census.current = _label_for(known)
                try:
                    return op(parser, known, pos, **kw)
                finally:
                    census.current = prev
            return wrapped_probe

        setattr(frames_mod, probe_name, make(orig_probe))
        undo.append(
            lambda n=probe_name, o=orig_probe: setattr(frames_mod, n, o))

    return lambda: [f() for f in reversed(undo)]


def main():
    per_seed = []
    for seed in SEEDS:
        census = Census()
        restore = _install(census)
        try:
            parser = get_parser_cache().fork("SENTENCES", seed=seed)
            report = calibrate_erp_thresholds(parser)
        finally:
            restore()
        per_seed.append((seed, census, report))

    print(f"or_else census over seeds {SEEDS}")
    print()

    merged = Census()
    for _s, c, _r in per_seed:
        for k, v in c.calls.items():
            merged.calls[k] += v
        for k, v in c.undef.items():
            merged.undef[k] += v
        for k, v in c.why.items():
            merged.why[k] |= v

    labels = merged.labels()
    width = max(len(s) for s in merged.sites()) if merged.calls else 20
    header = f"{'site':<{width}}  " + "  ".join(f"{lb:>22}" for lb in labels)
    print(header)
    print("-" * len(header))
    for site in merged.sites():
        cells = []
        for lb in labels:
            n = merged.calls[(site, lb)]
            u = merged.undef[(site, lb)]
            cells.append(f"{u:>4}/{n:<4} undef" if n else f"{'-':>15}")
        print(f"{site:<{width}}  " + "  ".join(f"{c:>22}" for c in cells))

    print()
    for site in merged.sites():
        total = sum(v for (s, _), v in merged.calls.items() if s == site)
        und = sum(v for (s, _), v in merged.undef.items() if s == site)
        verdict = ("NEVER FIRES -- substitution is free"
                   if und == 0 else f"FIRES {und}/{total}")
        print(f"{site}: {verdict}")
        for w in sorted(merged.why.get(site, ())):
            print(f"    reason: {w}")

    print()
    print("PER-ARM ASYMMETRY is the question that matters. A substitution that")
    print("fires equally on all arms shifts a level; one that fires on ONE arm")
    print("is the reported effect.")


if __name__ == "__main__":
    main()
