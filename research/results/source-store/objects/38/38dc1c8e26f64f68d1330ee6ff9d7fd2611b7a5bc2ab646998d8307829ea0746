"""Prevent new experiment result writes from bypassing provenance controls.

The baseline records historical direct JSON publication sites. Existing
studies must be migrated deliberately; a new site is an accidental bypass and
fails the maintained test gate.
"""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
BASELINE = ROOT / "research" / "experiments" / "result_writer_baseline.json"


def _direct_json_dump_sites():
    sites = {}
    for path in (ROOT / "research" / "experiments").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count = sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dump"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            for node in ast.walk(tree)
        )
        if count:
            sites[path.relative_to(ROOT).as_posix()] = count
    return sites


def _direct_json_write_text_sites():
    sites = {}
    for path in (ROOT / "research" / "experiments").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count = sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write_text"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and isinstance(node.args[0].func.value, ast.Name)
            and node.args[0].func.value.id == "json"
            and node.args[0].func.attr == "dumps"
            for node in ast.walk(tree)
        )
        if count:
            sites[path.relative_to(ROOT).as_posix()] = count
    return sites


def test_direct_result_writes_do_not_grow_without_disposition():
    inventory = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline = inventory["direct_json_dump_sites"]
    current = _direct_json_dump_sites()
    unexpected = sorted(set(current) - set(baseline))
    changed = sorted(path for path in set(current) & set(baseline) if current[path] > baseline[path])
    assert not unexpected and not changed, (
        "new direct experiment result writes bypass the shared runner/writer: "
        f"unexpected={unexpected}, increased={changed}. Migrate to the canonical "
        "writer or add an explicit legacy disposition."
    )


def test_direct_json_write_text_sites_do_not_grow_without_disposition():
    inventory = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline = inventory["direct_json_write_text_sites"]
    current = _direct_json_write_text_sites()
    unexpected = sorted(set(current) - set(baseline))
    changed = sorted(path for path in set(current) & set(baseline) if current[path] > baseline[path])
    assert not unexpected and not changed, (
        "new Path.write_text(json.dumps(...)) experiment writes bypass the "
        "shared runner/writer: "
        f"unexpected={unexpected}, increased={changed}. Migrate to the "
        "canonical writer or add an explicit legacy disposition."
    )
