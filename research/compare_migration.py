"""Compare migrated observations with a historical artifact, without running a study.

Numerical agreement is necessary, not proof of matching model/protocol semantics.
Capacity's historical files lack seed metadata: supply independently verified
reference seed order explicitly. Never infer it from the filename.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from research.evidence import validate_artifact


def _unique_pairs(pairs):
    """Reject ambiguous evidence before constructing an index or JSON object."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate evidence key: {key!r}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)


def _equal(actual, expected):
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=5e-6, abs_tol=1e-7))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_equal(a, b) for a, b in zip(actual, expected)))
    return actual == expected


def compare(candidate, baseline, kind, reference_seeds=None):
    # Specification: neural_assemblies/ir/VERIFICATION.md#contract-migration-identity
    errors, checked = [], 0
    observations, record = candidate["observations"], candidate["run"]
    if kind == "a1":
        for row in baseline["rows"]:
            for field in ("seed", "length", "first_error"):
                value = row.get(field)
                if value is not None and type(value) is not int:
                    raise ValueError(f"historical A1 {field} must be an integer")
        old = _unique_pairs(((r["seed"], r["p"]), r) for r in baseline["rows"])
        rows = observations["rows"]
        expected_keys = {(s, p) for s in record["seeds"] for p in record["parameters"]["p_values"]}
        actual_keys = [(r["seed"], r["p"]) for r in rows]
        if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected_keys:
            errors.append("missing, extra, or duplicate A1 seed/width cells")
        for r in rows:
            key = (r["seed"], r["p"])
            if key not in old or not _equal(r, old[key]):
                errors.append(f"A1 cell {key} differs (including length and exactness)")
            checked += 1
    elif kind == "capacity":
        if (not reference_seeds or any(type(seed) is not int for seed in reference_seeds)
                or len(set(reference_seeds)) != len(reference_seeds)):
            raise ValueError("capacity requires independently verified, unique reference seed order")
        indices = {seed: i for i, seed in enumerate(reference_seeds)}
        if any(seed not in indices for seed in record["seeds"]):
            raise ValueError("candidate seed is absent from the reference seed order")
        expected_keys = {(a, n, k) for a in record["parameters"]["arms"]
                         for n, k in record["parameters"]["nk"]}
        cells = observations["cells"]
        actual_keys = [(c["arm"], c["n"], c["k"]) for c in cells]
        if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected_keys:
            errors.append("missing, extra, or duplicate capacity cells")
        for c in cells:
            key = f'{c["arm"]}/{c["n"]}'
            if key not in baseline or not _equal(c["k"], baseline.get(key + "/ceiling", {}).get("k")):
                errors.append(f"{key}: missing reference or ambiguous k")
                continue
            expected_ms = {str(m) for m in record["parameters"]["configuration"]["checkpoints"]}
            if set(c["checkpoints"]) != expected_ms:
                errors.append(f"{key}: missing or extra checkpoints")
            for m, metrics in c["checkpoints"].items():
                old = baseline[key].get(str(m))
                if old is None or set(metrics) != set(old):
                    errors.append(f"{key}/{m}: missing checkpoint or metric")
                    continue
                for metric, values in metrics.items():
                    prior = old[metric]
                    if len(prior) != len(reference_seeds):
                        raise ValueError("reference seed order does not match observation count")
                    expected = [prior[indices[s]] for s in record["seeds"]]
                    if len(values) != len(expected) or any(
                            not _equal(a, b) for a, b in zip(values, expected)):
                        errors.append(f"{key}/{m}/{metric}: per-seed values differ")
                    checked += len(expected)
            # A subset of brains cannot reproduce a full-ensemble ceiling.
            # Compare aggregate values only when both seeds and grid match.
            if (set(record["seeds"]) == set(reference_seeds)
                    and set(c["checkpoints"]) == set(baseline[key])):
                aliases = {"m_star": "m_star", "supported": "supported", "alpha": "alpha",
                           "fill": "fill_at_ceiling", "censored": "fill_censored"}
                for old_name, new_name in aliases.items():
                    prior = baseline[key + "/ceiling"]
                    if old_name in prior:
                        actual = c.get("ceiling", {}).get(new_name)
                        if not _equal(actual, prior[old_name]):
                            errors.append(f"{key}/ceiling/{old_name}: aggregate differs")
                        checked += 1
    else:
        raise ValueError(f"unknown comparison: {kind}")
    if not checked:
        errors.append("no observations compared")
    return {"numerical_match": not errors, "comparisons": checked, "errors": errors,
            "scope": "observation comparison only; no scientific adoption or protocol-equivalence verdict"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("a1", "capacity"))
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--reference-seeds", nargs="+", type=int)
    args = parser.parse_args()
    errors = validate_artifact(args.candidate)
    if errors:
        print(json.dumps({"numerical_match": False, "errors": errors}, indent=2))
        return 1
    try:
        candidate = _load_json(args.candidate)
        baseline = _load_json(args.reference)
        result = compare(candidate, baseline, args.kind, args.reference_seeds)
    except ValueError as exc:
        print(json.dumps({"numerical_match": False, "errors": [str(exc)]}, indent=2))
        return 1
    result["comparison_version"] = 2
    result["comparator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["candidate_sha256"] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    result["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))
    return 0 if result["numerical_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
