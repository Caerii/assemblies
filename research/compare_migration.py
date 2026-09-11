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
import subprocess

from research.evidence import validate_artifact
from research.json_documents import (
    load_document as _load_json,
    unique_pairs as _unique_pairs,
    write_new_document,
)

ROOT = Path(__file__).resolve().parents[1]


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


def _compare_capacity(candidate, baseline, reference_seeds, *, condition=None,
                      allow_extra_checkpoints=False):
    observations, record = candidate["observations"], candidate["run"]
    indices = {seed: i for i, seed in enumerate(reference_seeds)}
    if any(seed not in indices for seed in record["seeds"]):
        raise ValueError("candidate seed is absent from the reference seed order")
    expected_keys = {(a, n, k) for a in record["parameters"]["arms"]
                     for n, k in record["parameters"]["nk"]}
    if condition is None:
        cells = observations["cells"]
        checkpoints = record["parameters"]["configuration"]["checkpoints"]
    else:
        mapped = observations["conditions"][condition]["cells"]
        cells = list(mapped.values())
        checkpoints = record["parameters"]["conditions"][condition]["checkpoints"]
        expected_names = {f"{arm}/{n}/{k}" for arm, n, k in expected_keys}
        if set(mapped) != expected_names:
            return [f"{condition}: missing, extra, or duplicate keyed capacity cells"], 0
    errors, checked = [], 0
    actual_keys = [(c["arm"], c["n"], c["k"]) for c in cells]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected_keys:
        errors.append("missing, extra, or duplicate capacity cells")
    for c in cells:
        key = f'{c["arm"]}/{c["n"]}'
        if key not in baseline or not _equal(c["k"], baseline.get(key + "/ceiling", {}).get("k")):
            errors.append(f"{key}: missing reference or ambiguous k")
            continue
        expected_ms = {str(m) for m in checkpoints}
        actual_ms = set(c["checkpoints"])
        reference_ms = set(baseline[key])
        if actual_ms != expected_ms:
            errors.append(f"{key}: output disagrees with recorded checkpoints")
        if ((allow_extra_checkpoints and not reference_ms <= actual_ms)
                or (not allow_extra_checkpoints and actual_ms != reference_ms)):
            errors.append(f"{key}: missing or extra checkpoints")
        for m in sorted(reference_ms & actual_ms, key=int):
            metrics = c["checkpoints"][m]
            old = baseline[key][m]
            if set(metrics) != set(old):
                errors.append(f"{key}/{m}: missing checkpoint metric")
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
        if set(record["seeds"]) == set(reference_seeds):
            aliases = {"m_star": "m_star", "supported": "supported", "alpha": "alpha",
                       "fill": "fill_at_ceiling", "censored": "fill_censored"}
            for old_name, new_name in aliases.items():
                prior = baseline[key + "/ceiling"]
                if old_name in prior:
                    actual = c.get("ceiling", {}).get(new_name)
                    if not _equal(actual, prior[old_name]):
                        errors.append(f"{key}/ceiling/{old_name}: aggregate differs")
                    checked += 1
    return errors, checked


def compare(candidate, baseline, kind, reference_seeds=None, treatment_baseline=None):
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
    elif kind in {"capacity", "capacity-paired"}:
        if (not reference_seeds or any(type(seed) is not int for seed in reference_seeds)
                or len(set(reference_seeds)) != len(reference_seeds)):
            raise ValueError("capacity requires independently verified, unique reference seed order")
        if kind == "capacity":
            errors, checked = _compare_capacity(candidate, baseline, reference_seeds)
        else:
            if treatment_baseline is None:
                raise ValueError("capacity-paired requires a treatment reference")
            if set(observations.get("conditions", {})) != {"control", "refracted"}:
                errors.append("paired candidate must contain control and refracted conditions")
            else:
                for condition, reference, allow_extra in (
                        ("control", baseline, True),
                        ("refracted", treatment_baseline, False)):
                    found, count = _compare_capacity(
                        candidate, reference, reference_seeds,
                        condition=condition, allow_extra_checkpoints=allow_extra)
                    errors.extend(f"{condition}: {error}" for error in found)
                    checked += count
    else:
        raise ValueError(f"unknown comparison: {kind}")
    if not checked:
        errors.append("no observations compared")
    return {"numerical_match": not errors, "comparisons": checked, "errors": errors,
            "scope": "observation comparison only; no scientific adoption or protocol-equivalence verdict"}


def _repo_relative(path: Path, root: Path = ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"comparison input must be inside the repository: {path}") from exc


def validate_receipt(path: Path, root: Path = ROOT) -> list[str]:
    """Recompute and authenticate a version-4 migration comparison receipt."""
    try:
        receipt = _load_json(path)
    except (OSError, ValueError) as exc:
        return [f"cannot read comparison receipt: {exc}"]
    required = {
        "comparison_version", "kind", "candidate", "reference",
        "treatment_reference", "reference_seeds", "numerical_match",
        "comparisons", "errors", "scope", "comparator_source",
        "comparator_sha256", "candidate_sha256", "reference_sha256",
        "treatment_reference_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        return ["comparison receipt has an incomplete or unknown schema"]
    errors = []
    if (receipt["comparison_version"] != 4
            or receipt["kind"] != "capacity-paired"
            or receipt["numerical_match"] is not True
            or receipt["errors"] != []
            or type(receipt["comparisons"]) is not int
            or receipt["comparisons"] < 1):
        errors.append("comparison receipt does not record a successful version-4 comparison")
    paths = {}
    for field in ("candidate", "reference", "treatment_reference"):
        name = receipt[field]
        target = (root / name).resolve() if isinstance(name, str) else root.parent
        if (not isinstance(name, str) or not target.is_relative_to(root.resolve())
                or not target.is_file()):
            errors.append(f"comparison receipt has unsafe or missing {field}")
        else:
            paths[field] = target
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != receipt[f"{field}_sha256"]:
                errors.append(f"comparison receipt {field} digest differs")
    source = receipt["comparator_source"]
    if (not isinstance(source, dict)
            or set(source) != {"inventory", "git_commit", "source_sha256"}):
        errors.append("comparison receipt has invalid comparator source identity")
    else:
        try:
            blob = subprocess.check_output(
                ["git", "show", f"{source['git_commit']}:research/compare_migration.py"],
                cwd=root,
            )
            if hashlib.sha256(blob).hexdigest() != receipt["comparator_sha256"]:
                errors.append("comparison receipt comparator digest differs from its commit")
        except (OSError, subprocess.CalledProcessError):
            errors.append("comparison receipt comparator commit is unavailable")
    if "candidate" in paths:
        errors.extend(f"candidate: {error}" for error in
                      validate_artifact(paths["candidate"], root=root))
    if not errors and len(paths) == 3:
        try:
            recomputed = compare(
                _load_json(paths["candidate"]), _load_json(paths["reference"]),
                receipt["kind"], receipt["reference_seeds"],
                treatment_baseline=_load_json(paths["treatment_reference"]),
            )
            for field in ("numerical_match", "comparisons", "errors", "scope"):
                if recomputed[field] != receipt[field]:
                    errors.append(f"comparison receipt recomputation differs at {field}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"comparison receipt cannot be recomputed: {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("a1", "capacity", "capacity-paired"))
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--reference-seeds", nargs="+", type=int)
    parser.add_argument("--treatment-reference", type=Path)
    parser.add_argument("--output", type=Path,
                        help="exclusive JSON receipt path; existing files are refused")
    args = parser.parse_args()
    errors = validate_artifact(args.candidate)
    if errors:
        print(json.dumps({"numerical_match": False, "errors": errors}, indent=2))
        return 1
    try:
        candidate = _load_json(args.candidate)
        baseline = _load_json(args.reference)
        treatment = (_load_json(args.treatment_reference)
                     if args.treatment_reference is not None else None)
        result = compare(candidate, baseline, args.kind, args.reference_seeds,
                         treatment_baseline=treatment)
    except ValueError as exc:
        print(json.dumps({"numerical_match": False, "errors": [str(exc)]}, indent=2))
        return 1
    from research.runner import SOURCE_INVENTORY, _source_identity
    result["comparison_version"] = 4
    result["kind"] = args.kind
    result["candidate"] = _repo_relative(args.candidate)
    result["reference"] = _repo_relative(args.reference)
    result["treatment_reference"] = (
        _repo_relative(args.treatment_reference)
        if args.treatment_reference is not None else None)
    result["reference_seeds"] = args.reference_seeds
    result["comparator_source"] = {"inventory": SOURCE_INVENTORY, **_source_identity()}
    result["comparator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["candidate_sha256"] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    result["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    if args.treatment_reference is not None:
        result["treatment_reference_sha256"] = hashlib.sha256(
            args.treatment_reference.read_bytes()).hexdigest()
    if args.treatment_reference is None:
        result["treatment_reference_sha256"] = None
    if args.output is None:
        print(json.dumps(result, indent=2))
    else:
        write_new_document(args.output, result)
        print(f"wrote {args.output}")
    return 0 if result["numerical_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
