"""research.refactor_proof proves a pure move a no-op and catches a move that changes meaning.

A small package is written three ways in separate roots and snapshotted in separate processes,
as the tool is used (the commit in a worktree, the edit in the checkout): as it was, after a pure
move (a constant and a function moved to a library module, re-exported, and a caller switched to
the library), and after the same move with the constant's value changed."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from research import refactor_proof

ROOT = pathlib.Path(__file__).resolve().parents[2]

BEFORE = {
    "a.py": "W = 20.0\n\n\ndef f(x):\n    return x * W\n",
    "b.py": "from proofpkg import a\n\n\ndef g(x):\n    return a.f(x) + a.W\n\n\n"
            "def h(x):\n    from proofpkg.a import f\n    return f(x)\n",
}
MOVED = {
    "lib.py": "W = 20.0\n\n\ndef f(x):\n    \"\"\"now documented\"\"\"\n    return x * W\n",
    "a.py": "from proofpkg.lib import W, f  # noqa: F401  re-exported\n",
    "b.py": "from proofpkg import lib\n\n\ndef g(x):\n    return lib.f(x) + lib.W\n\n\n"
            "def h(x):\n    from proofpkg.lib import f\n    return f(x)\n",
}


def _snap(tmp_path, name, files):
    root = tmp_path / name
    (root / "proofpkg").mkdir(parents=True)
    (root / "proofpkg" / "__init__.py").write_text("", encoding="utf-8")
    for f, text in files.items():
        (root / "proofpkg" / f).write_text(text, encoding="utf-8")
    out = tmp_path / f"{name}.json"
    subprocess.run([sys.executable, "-m", "research.refactor_proof", "snapshot", str(out), "proofpkg.*",
                    "--root", str(root), "--follow", "proofpkg"], cwd=ROOT, check=True, capture_output=True)
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_pure_move_changes_no_meaning(tmp_path):
    before, after = _snap(tmp_path, "before", BEFORE), _snap(tmp_path, "after", MOVED)
    result = refactor_proof.compare(before, after)
    assert result["changed"] == []
    assert ("proofpkg.b", "a", "module:proofpkg.a", None) in result["removed"]
    assert ("proofpkg.b", "lib", None, "module:proofpkg.lib") in result["added"]
    assert before["proofpkg.b"]["g"] == after["proofpkg.b"]["g"]       # alias switched, same objects
    assert before["proofpkg.b"]["h"] == after["proofpkg.b"]["h"]       # local import moved, same object


def test_a_method_moved_into_a_mixin_keeps_its_entry(tmp_path):
    cls = ("W = 2.0\n\n\nclass Engine:\n    def step(self, x):\n        return self.scale(x) * W\n\n"
           "    def scale(self, x):\n        return x + 1\n")
    before = _snap(tmp_path, "before", {"engine.py": cls})
    mixed = {"_steps.py": "W = 2.0\n\n\nclass Steps:\n    def step(self, x):\n        return self.scale(x) * W\n",
             "engine.py": "from proofpkg._steps import Steps\n\n\nclass Engine(Steps):\n"
                          "    def scale(self, x):\n        return x + 1\n"}
    after = _snap(tmp_path, "after", mixed)
    assert after["proofpkg.engine"]["Engine.step"] == before["proofpkg.engine"]["Engine.step"]
    assert after["proofpkg.engine"]["Engine.scale"] == before["proofpkg.engine"]["Engine.scale"]
    broken = dict(mixed, **{"_steps.py": mixed["_steps.py"].replace("* W", "* W * 2")})
    worse = _snap(tmp_path, "broken", broken)
    assert worse["proofpkg.engine"]["Engine.step"] != before["proofpkg.engine"]["Engine.step"]


def test_a_class_moved_to_another_module_keeps_its_callers(tmp_path):
    one = {"m.py": "from dataclasses import dataclass\n\n\n@dataclass\nclass Verdict:\n    ok: bool\n\n"
                   "    def text(self):\n        return 'ok' if self.ok else 'no'\n\n\n"
                   "def judge(x):\n    return Verdict(x > 0)\n"}
    two = {"verdict.py": "from dataclasses import dataclass\n\n\n@dataclass\nclass Verdict:\n    ok: bool\n\n"
                         "    def text(self):\n        return 'ok' if self.ok else 'no'\n",
           "m.py": "from proofpkg.verdict import Verdict\n\n\ndef judge(x):\n    return Verdict(x > 0)\n"}
    before, after = _snap(tmp_path, "before", one), _snap(tmp_path, "after", two)
    assert after["proofpkg.m"]["judge"] == before["proofpkg.m"]["judge"]
    assert after["proofpkg.m"]["Verdict"] == before["proofpkg.m"]["Verdict"]
    changed = dict(two, **{"verdict.py": two["verdict.py"].replace("'no'", "'not ok'")})
    assert _snap(tmp_path, "changed", changed)["proofpkg.m"]["judge"] != before["proofpkg.m"]["judge"]


def test_a_move_that_changes_a_value_is_caught(tmp_path):
    before = _snap(tmp_path, "before", BEFORE)
    broken = dict(MOVED, **{"lib.py": MOVED["lib.py"].replace("W = 20.0", "W = 20.5")})
    after = _snap(tmp_path, "broken", broken)
    changed = {(m, k) for m, k, _, _ in refactor_proof.compare(before, after)["changed"]}
    assert {("proofpkg.a", "W"), ("proofpkg.a", "f"), ("proofpkg.b", "g"), ("proofpkg.b", "h")} <= changed
