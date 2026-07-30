"""How much of each recorded golden is actually CHECKED?

`verify_against_golden` compares two blocks only -- `golden["expected"]` (exact
or within tolerance) and `golden["thresholds"]` (one-sided `_min` / `_max`
bounds). The `golden["metrics"]` block, which is what the record scripts
carefully measure and write down, is loaded into `result.golden_metrics` for
display and **never compared to anything**.

So a metric recorded in `metrics` but absent from `expected`/`thresholds` can
drift arbitrarily and every parity test stays green. That is not hypothetical:
`tacl2021_parser_f1` recorded `roles_found = [ADVERB, OBJ, SUBJ]` and now
produces `[OBJ, SUBJ]` -- the spurious ADVERB is gone, which is an improvement,
but `diffs` is empty and the suite could not tell you either way.

This prints, per golden, which recorded metrics are load-bearing and which are
decorative. A one-sided `_min` on a metric whose recorded value sits far above
it is counted as WEAK: it pins a floor, not the value, so it cannot detect
drift downward until the floor is crossed.

    uv run python research/experiments/golden_coverage_audit.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PARITY = os.path.join(ROOT, "research", "literature", "parity")
REGISTRY = os.path.join(PARITY, "registry.json")

# Seven protocols bypass `verify_against_golden` for a hand-written verifier in
# runner.py. Modelling only the generic path OVER-REPORTS them as decorative --
# `_verify_pnas_scaling` pins three metrics per regime to +-0.05, which a naive
# read of the golden file cannot see. Keys these special cases actually compare
# to the RECORDED value (not merely to a threshold):
SPECIAL_PINS = {
    "pnas2020_scaling": {"project_persistence", "separate_overlap",
                         "chance_overlap"},          # +-0.05, per regime
    "direct2026_pearl": {"forward_overlap", "reverse_overlap"},       # +-0.05
    "colt2022_mnist_brain": {"mean_accuracy"},       # metrics_match_tolerance
    "hoff2026_size_dist": set(),                     # band from paper, no pin
    "colt2022_mnist": set(),                         # thresholds only
    "coin2024_demo": set(), "coin2024_compete": set(),
    "coin2024_softmax": set(),                       # `expected` block only
}


# Recorded for PROVENANCE, not as results. Counting these as "unchecked" would
# inflate the headline: a golden should record the seed it ran at, and nothing
# is meant to assert against it.
PROVENANCE = {
    "n", "k", "p", "beta", "seed", "data_source", "expected_roles",
    "language", "engine", "recorded", "source",
}


def classify(key, value, expected, thresholds, pins=frozenset()):
    """PINNED / weak / DECORATIVE / provenance for one recorded metric."""
    if key in PROVENANCE:
        return "provenance", ""
    if key in pins or key in expected:
        return "PINNED", ""
    for suffix, cmp in (("_min", "floor"), ("_max", "ceiling")):
        tkey = f"{key}{suffix}"
        if tkey in thresholds:
            bound = thresholds[tkey]
            try:
                slack = (float(value) - float(bound)) if suffix == "_min" \
                    else (float(bound) - float(value))
                margin = f"{cmp} {bound}, recorded {value} (slack {slack:+.4g})"
            except (TypeError, ValueError):
                margin = f"{cmp} {bound}"
            return "weak", margin
    return "DECORATIVE", ""


def walk(metrics, expected, thresholds, pins=frozenset(), prefix=""):
    """Flatten one level of nesting; goldens use both flat and per-regime."""
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            rows += walk(value, expected, thresholds, pins,
                         f"{prefix}{key}.")
            continue
        verdict, note = classify(key, value, expected, thresholds, pins)
        rows.append((f"{prefix}{key}", verdict, note))
    return rows


def main():
    reg = json.load(open(REGISTRY))
    protocols = sorted(reg["protocols"], key=lambda p: p["protocol_id"])

    tot = {"PINNED": 0, "weak": 0, "DECORATIVE": 0, "provenance": 0}
    fully_decorative = []

    for proto in protocols:
        path = os.path.join(PARITY, proto["golden"])
        if not os.path.exists(path):
            print(f"\n  {proto['protocol_id']:<34} MISSING {proto['golden']}")
            continue
        g = json.load(open(path))
        metrics = g.get("metrics", g.get("regimes", {}))
        expected = g.get("expected", {})
        thresholds = g.get("thresholds", {})
        pins = SPECIAL_PINS.get(proto["protocol_id"], frozenset())
        rows = walk(metrics, expected, thresholds, pins)
        if not rows:
            continue

        counts = {"PINNED": 0, "weak": 0, "DECORATIVE": 0,
                  "provenance": 0}
        for _, v, _ in rows:
            counts[v] += 1
            tot[v] += 1
        if (counts["PINNED"] == 0 and counts["weak"] == 0
                and counts["DECORATIVE"] > 0):
            fully_decorative.append(proto["protocol_id"])

        print(f"\n  {proto['protocol_id']:<34} "
              f"pinned {counts['PINNED']}  weak {counts['weak']}  "
              f"decorative {counts['DECORATIVE']}")
        for key, verdict, note in rows:
            if verdict in ("PINNED", "provenance"):
                continue
            tag = "weak      " if verdict == "weak" else "DECORATIVE"
            print(f"      [{tag}] {key}" + (f"   {note}" if note else ""))

    n = tot["PINNED"] + tot["weak"] + tot["DECORATIVE"]
    print("\n" + "=" * 74)
    print(f"  {tot['PINNED']} pinned / {tot['weak']} weak / "
          f"{tot['DECORATIVE']} decorative, of {n} recorded RESULT metrics")
    print(f"  (+{tot['provenance']} provenance entries -- seed, n, k, p, beta --"
          f" which are not meant to be asserted)")
    if n:
        print(f"  {100 * tot['DECORATIVE'] / n:.0f}% of the measured results are"
              f" compared to NOTHING; a further {100 * tot['weak'] / n:.0f}%"
              f" only to a one-sided bound")
    if fully_decorative:
        print("\n  goldens with NO checked metric at all "
              "(the file records, the test asserts nothing):")
        for pid in fully_decorative:
            print(f"    - {pid}")


if __name__ == "__main__":
    main()
