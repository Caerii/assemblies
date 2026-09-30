"""Which recorded golden values does the suite ever even LOOK at?

SUPERSEDES the static half of `golden_coverage_audit.py`, which read the golden
JSON plus `verify_against_golden` and concluded that 57% of recorded result
metrics are compared to nothing. That figure was wrong in a specific way: only 7
of the 20 golden tests go through `verify_protocol`. The other 13 have inline
bodies that assert against `g["metrics"][...]` directly, and a static read of
the runner cannot see those. It reported `nemo2025_curriculum` as asserting
nothing when its test body pins two metrics to +-0.05 and +-0.01.

So measure it at runtime instead, and measure something unarguable: wrap each
loaded golden in a dict that records every key ACCESS, then run the suite. A
recorded value that is never read during its own test cannot possibly be
asserted. That is a sound lower bound on decorativeness regardless of whether
the assertion lives in the runner or the test body.

Read-but-not-asserted is still possible (a key can be read for a print), so this
UNDER-reports rather than over-reports -- the opposite failure to the static
version, and the safer one.

RESULT: 5 of 26, or 19%. My estimate of this number moved 74 -> 66 -> 57 -> 59
-> 43 -> 19 as each confound was removed, and every step was downward:

    74%  static, modelling only the generic verifier
    66%  ... but 7 protocols use hand-written verifiers that DO pin values
    57%  ... and 26 entries are provenance (seed, n, k, p) nothing should assert
    59%  runtime measurement, different denominator
    43%  ... minus tests whose read-set is truncated by failing or xfailing,
         and minus keys duplicated into `expected` and asserted through it
    19%  ... minus decoration the code documents as deliberate

The alarming version of this finding did not survive contact with the details.
The corpus is in good shape; what remains is five values, listed below the
per-protocol lines when this runs.

    uv run python research/experiments/golden_readkey_audit.py
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PARITY = os.path.join(ROOT, "research", "literature", "parity")
REGISTRY = os.path.join(PARITY, "registry.json")
GOLDEN_TESTS = "neural_assemblies/tests/test_literature_golden.py"

PROVENANCE = {
    "n", "k", "p", "beta", "seed", "data_source", "expected_roles",
    "language", "engine", "recorded", "source", "protocol", "parameters",
    "config_ref", "notes", "thresholds", "expected", "tolerance",
}

# Decorative ON PURPOSE, with the rationale in the code that skips them.
# `_verify_coin`: "Coin protocols: assert qualitative `expected` fields, not
# flip counts." Pinning a stochastic flip count would be brittle, so these are
# a documented decision rather than a coverage gap, and flagging them as work
# would manufacture a finding.
DELIBERATE = {
    "coin2024_demo", "coin2024_compete", "coin2024_softmax",
    "coin2024_markov_arc",
}

READS = defaultdict(set)          # golden name -> {"metrics.role_probes.accuracy"}
CURRENT = {"name": "<unknown>"}
OUTCOME = {}                      # golden name -> "passed" / "failed" / "skipped"
TEST_GOLDENS = defaultdict(set)   # test nodeid -> {golden name}


class Tracked(dict):
    """A dict that records the dotted path of every key it is asked for."""

    def __init__(self, data, name, prefix=""):
        super().__init__(data)
        self._name = name
        self._prefix = prefix

    def _note(self, key):
        READS[self._name].add(f"{self._prefix}{key}")
        test = CURRENT.get("test")
        if test:
            TEST_GOLDENS[test].add(self._name)

    def _wrap(self, key, value):
        if isinstance(value, dict):
            return Tracked(value, self._name, f"{self._prefix}{key}.")
        return value

    def __getitem__(self, key):
        self._note(key)
        return self._wrap(key, super().__getitem__(key))

    def get(self, key, default=None):
        self._note(key)
        if key not in self:
            return default
        return self._wrap(key, super().__getitem__(key))


def install():
    """Wrap both golden loaders so every access is recorded."""
    import neural_assemblies.tests.test_literature_golden as gt
    from neural_assemblies.parity import runner as pr

    orig_load = gt._load

    def _load(name, *a, **kw):
        CURRENT["name"] = name
        return Tracked(orig_load(name, *a, **kw), name)

    gt._load = _load

    orig_load_golden = pr.load_golden

    def load_golden(proto, *a, **kw):
        name = os.path.basename(getattr(proto, "golden", "") or "")
        CURRENT["name"] = name or getattr(proto, "protocol_id", "?")
        return Tracked(orig_load_golden(proto, *a, **kw), CURRENT["name"])

    pr.load_golden = load_golden
    # runner.py imported the symbol directly, so rebind there too.
    if hasattr(pr, "load_golden"):
        pr.load_golden = load_golden


class OutcomePlugin:
    """A FAILING test stops reading its golden, which would read as decorative.

    Without this, `nemo2025_curriculum` reports 29/29 unread purely because its
    role-probe threshold assertion fails three lines ABOVE the two assertions
    that do pin recorded metrics. Truncated read-sets are reported separately,
    not counted.
    """

    def __init__(self):
        self._current = None

    def pytest_runtest_setup(self, item):
        self._current = item.nodeid
        CURRENT["test"] = item.nodeid

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        for name in TEST_GOLDENS.get(report.nodeid, ()):
            OUTCOME[name] = report.outcome


def leaves(d, prefix=""):
    """Every dotted path to a scalar, skipping provenance keys at any depth."""
    out = []
    for key, value in d.items():
        if key in PROVENANCE:
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out += leaves(value, f"{path}.")
        else:
            out.append(path)
    return out


def covered(path, read, duplicated=frozenset()):
    """A leaf counts as reached if it or any ancestor block was accessed.

    Also reached if the SAME KEY is duplicated into `expected` and that copy was
    read: `nemo2025_fsm_mod3_numpy` records positive_accepted under both
    `metrics` and `expected`, and its test asserts the `expected` copy. Counting
    the `metrics` copy as unchecked would be true but misleading -- the value is
    pinned, just through the other spelling.
    """
    parts = path.split(".")
    if parts[-1] in duplicated:
        return True
    return any(".".join(parts[:i]) in read for i in range(1, len(parts) + 1))


def main():
    install()
    import pytest
    code = pytest.main([GOLDEN_TESTS, "-q", "-p", "no:randomly"]
                       + sys.argv[1:], plugins=[OutcomePlugin()])

    reg = json.load(open(REGISTRY))
    print("\n" + "=" * 78)
    print("GOLDEN READ-KEY AUDIT -- recorded values the suite never looks at")
    print("=" * 78)
    print("  a value never READ during its own test cannot be asserted;")
    print("  read-but-not-asserted is possible, so this UNDER-reports\n")

    tot_read = tot_unread = 0
    tot_deliberate = 0
    truncated = []
    other = []
    for proto in sorted(reg["protocols"], key=lambda x: x["protocol_id"]):
        path = os.path.join(PARITY, proto["golden"])
        if not os.path.exists(path):
            continue
        g = json.load(open(path))
        name = os.path.basename(proto["golden"])
        read = READS.get(name, set())
        if not read:
            continue                       # test skipped or never ran
        # "skipped" covers unavailable and retracted records. They have no
        # active metric block to audit and cannot be treated as parity passes.
        if OUTCOME.get(name) in ("failed", "skipped"):
            truncated.append(f"{proto['protocol_id']} ({OUTCOME[name]})")
            continue                       # read-set truncated
        metrics = {"metrics": g.get("metrics", g.get("regimes", {}))}
        all_leaves = leaves(metrics)
        if not all_leaves:
            # Records its results under a paper-specific key (e.g. hoff2026's
            # table1_*), so there is nothing here to attribute. Not "all read".
            other.append(proto["protocol_id"])
            continue
        # Keys duplicated into a block the test DID read are pinned already.
        dup = {k for k in g.get("expected", {}) if f"expected.{k}" in read
               or "expected" in read}
        unread = [p for p in all_leaves if not covered(p, read, dup)]
        tot_read += len(all_leaves) - len(unread)
        deliberate = proto["protocol_id"] in DELIBERATE
        if deliberate:
            tot_deliberate += len(unread)
        else:
            tot_unread += len(unread)
        if unread:
            tag = " (BY DESIGN -- see _verify_coin)" if deliberate else ""
            print(f"  {proto['protocol_id']:<32} "
                  f"{len(unread)}/{len(all_leaves)} never read{tag}")
            for p in unread[:8]:
                print(f"      {p}")
            if len(unread) > 8:
                print(f"      ... and {len(unread) - 8} more")
        else:
            print(f"  {proto['protocol_id']:<32} all "
                  f"{len(all_leaves)} read")

    n = tot_read + tot_unread
    print("\n" + "-" * 78)
    if n:
        print(f"  {tot_unread} of {n} recorded values are never read "
              f"({100 * tot_unread / n:.0f}%)")
    if truncated:
        print("\n  EXCLUDED -- the test FAILED, so its read-set is truncated"
              " and\n  cannot distinguish decorative from simply unreached:")
        for pid in truncated:
            print(f"    - {pid}")
    if other:
        print("\n  EXCLUDED -- results recorded under a paper-specific key,"
              " nothing\n  under `metrics` to attribute:")
        for pid in other:
            print(f"    - {pid}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
