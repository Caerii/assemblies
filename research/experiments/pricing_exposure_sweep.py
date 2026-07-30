"""Which recorded goldens could the k-WTA pricing fixes have moved?

Two engine fixes (`c7ce506`, `54e6c00`, unified in `5fa91ed`) changed how top-k
prices materialized incumbents against sampled candidates under `norm_init`.
Every golden recorded before them is potentially affected -- but only if its
brain actually reached an affected path. This partitions the corpus so that
re-recording effort goes only where a number could have moved.

HOW.  `neural_assemblies.diagnostics.pricing_exposure` reads topology off a live
brain, but the goldens are produced inside pytest cases, not by importable
entry points. So instead of reverse-engineering 23 constructions, this installs
a tracer on `Brain.add_area` / `Brain.project` and lets the golden tests
themselves drive it. The tracer records, per Brain instance:

  * every area's (n, explicit) and the brain's norm_init, and
  * every (src -> dst) area fiber a `project()` call actually DROVE.

Exposure is then evaluated over DRIVEN fibers only, which is stricter than the
topology reading in `diagnostics.pricing_exposure`: a connectome the engine
materialized but nothing ever drove cannot have distorted a result.

Usage::

    uv run python research/experiments/pricing_exposure_sweep.py
    uv run python research/experiments/pricing_exposure_sweep.py -k Colt2022
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

REGISTRY = "research/literature/parity/registry.json"
GOLDEN_TESTS = "neural_assemblies/tests/test_literature_golden.py"


class Tracer:
    """Per-Brain topology + driven-fiber log, keyed by id(brain)."""

    def __init__(self):
        self.areas = defaultdict(dict)       # bid -> {name: (n, explicit)}
        self.driven = defaultdict(set)       # bid -> {(src, dst)}
        self.norm_init = {}                  # bid -> bool
        self.engine = {}                     # bid -> str
        self.current_test = None
        self.by_test = defaultdict(set)      # test id -> {bid}

    def note_brain(self, b):
        bid = id(b)
        if bid not in self.norm_init:
            self.norm_init[bid] = bool(getattr(b, "norm_init", False))
            self.engine[bid] = type(getattr(b, "_engine", None)).__name__
        if self.current_test:
            self.by_test[self.current_test].add(bid)

    def note_area(self, b, name, n, explicit):
        self.note_brain(b)
        self.areas[id(b)][name] = (int(n), bool(explicit))

    def note_project(self, b, dst_by_src):
        self.note_brain(b)
        for src, dsts in (dst_by_src or {}).items():
            for dst in dsts:
                self.driven[id(b)].add((src, dst))

    # -- verdicts ----------------------------------------------------------

    def findings(self, bid):
        """(heterogeneous_fibers, explicit_source_fibers) among DRIVEN fibers."""
        areas = self.areas[bid]
        het, exp = [], []
        for src, dst in sorted(self.driven[bid]):
            if src not in areas or dst not in areas:
                continue
            n_src, src_explicit = areas[src]
            n_dst, _ = areas[dst]
            if n_src != n_dst:
                het.append((src, dst, n_src, n_dst))
            if src_explicit:
                exp.append((src, dst))
        return het, exp

    def exposed(self, bid):
        if not self.norm_init.get(bid, False):
            return False
        het, exp = self.findings(bid)
        return bool(het or exp)


TRACER = Tracer()


def install():
    from neural_assemblies.core.brain import Brain

    orig_add_area = Brain.add_area
    orig_project = Brain.project

    def add_area(self, name, n, k, *a, **kw):
        out = orig_add_area(self, name, n, k, *a, **kw)
        TRACER.note_area(self, name, n, bool(kw.get("explicit", False)))
        return out

    def project(self, areas_by_stim=None, dst_areas_by_src_area=None,
                *a, **kw):
        TRACER.note_project(self, dst_areas_by_src_area)
        return orig_project(self, areas_by_stim, dst_areas_by_src_area,
                            *a, **kw)

    Brain.add_area = add_area
    Brain.project = project


class Plugin:
    def pytest_runtest_setup(self, item):
        TRACER.current_test = item.nodeid.split("::", 1)[-1]

    def pytest_runtest_teardown(self, item):
        TRACER.current_test = None


def main():
    install()
    import pytest

    args = [GOLDEN_TESTS, "-q", "-p", "no:randomly"] + sys.argv[1:]
    code = pytest.main(args, plugins=[Plugin()])

    reg = json.load(open(REGISTRY))
    backend = {p["protocol_id"]: p.get("backend", "?")
               for p in reg["protocols"]}

    print("\n" + "=" * 78)
    print("PRICING EXPOSURE -- which goldens could the k-WTA fixes have moved")
    print("=" * 78)
    print("  exposed = norm_init ON and a DRIVEN fiber is either")
    print("            heterogeneous (n_src != n_dst) or from an EXPLICIT area\n")

    exposed_tests, clean_tests, silent = [], [], []
    for test, bids in sorted(TRACER.by_test.items()):
        hits = [b for b in bids if TRACER.exposed(b)]
        if not bids or not any(TRACER.areas[b] for b in bids):
            silent.append(test)
        elif hits:
            exposed_tests.append((test, hits))
        else:
            clean_tests.append(test)

    for test, bids in exposed_tests:
        print(f"  [EXPOSED] {test}")
        for bid in bids:
            het, exp = TRACER.findings(bid)
            print(f"            engine={TRACER.engine.get(bid)} "
                  f"norm_init={TRACER.norm_init.get(bid)}")
            for src, dst, ns, nd in het[:6]:
                sign = "SEAL (candidates over-divided)" if ns < nd else \
                       "CHURN (candidates under-divided)"
                print(f"              het  {src}(n={ns}) -> {dst}(n={nd})"
                      f"   expect {sign}")
            for src, dst in exp[:6]:
                print(f"              expl {src} -> {dst}"
                      f"   (drive skipped _norm_scale entirely)")
    print()
    for test in clean_tests:
        print(f"  [clean  ] {test}")
    if silent:
        print("\n  built no Brain via the traced API (reference port or "
              "cached golden):")
        for test in silent:
            print(f"    - {test}")

    print(f"\n  {len(exposed_tests)} exposed, {len(clean_tests)} clean, "
          f"{len(silent)} untraced, of {len(TRACER.by_test)} tests")
    print(f"  registry backends: "
          f"{sorted(set(backend.values()))}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
