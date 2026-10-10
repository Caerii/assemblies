"""research/amend.py applies an amendment to the documents mechanically, in each file's own line
endings; research/experiments/memory_lib/spec.py renders what it applies. Checked on copies (never
the real documents), and the claim generator the strongest way: A55's register entry, regenerated
from its fields, equals the hand-written one."""
from __future__ import annotations

import pytest

from research import amend
from research.experiments import memory_lib as lib


def _copy(tmp_path, name, text, crlf):
    p = tmp_path / name
    p.write_bytes(text.replace("\n", "\r\n" if crlf else "\n").encode("utf-8"))
    return p


@pytest.mark.parametrize("crlf", [True, False])
def test_append_and_scorecard_keep_the_files_line_endings(tmp_path, crlf):
    doc = ("# PREREG\n\n| Bar | Registered | Verdict | Deciding number |\n|-----|---|---|---|\n"
           "| A1 x (A1) | >= 1 | PASS | 2 |\n\nprose after the table\n\n## Amendment 1\n\ntext\n")
    p = _copy(tmp_path, "p.md", doc, crlf)
    amend.add_scorecard_rows(p, "| B1 y (A2) | <= 3 | FAIL | 4 |\n| B2 z (A2) | >= 0 | PASS | 1 |\n")
    amend.append_section(p, "## Amendment 2\n\nnew text\n")
    raw = p.read_bytes().decode("utf-8")
    assert ("\r\n" in raw) == crlf and ("\n" in raw.replace("\r\n", "")) == (not crlf)
    lines = raw.replace("\r\n", "\n").split("\n")
    i = lines.index("| A1 x (A1) | >= 1 | PASS | 2 |")
    assert lines[i + 1:i + 3] == ["| B1 y (A2) | <= 3 | FAIL | 4 |", "| B2 z (A2) | >= 0 | PASS | 1 |"]
    assert lines[i + 3] == "" and raw.replace("\r\n", "\n").endswith("text\n\n## Amendment 2\n\nnew text\n")
    with pytest.raises(ValueError, match="table rows"):
        amend.add_scorecard_rows(p, "not a row")


def test_anchor_is_githubs():
    assert amend.anchor("### Amendment 55 result (2026-10-10)") == "amendment-55-result-2026-10-10"


def test_the_claim_generator_regenerates_a55s_entry(tmp_path):
    from neural_assemblies import theory
    old = theory.RESULTS["BIRTH-SETPOINT-GATES-SLEEP"]
    checks = [{"sample_path": c.sample_path, "treatment_path": c.treatment_path, "control_path": c.control_path,
               "relation": c.relation, "minimum_effect": c.minimum_effect, "mechanism": c.mechanism}
              for c in old.sensitivity_checks]
    block = amend.theory_result(id=old.id, engine=old.engine, claim=old.claim, source=old.source,
                                preconditions=old.preconditions, evidence=old.evidence[0],
                                artifact=old.evidence_refs[0].path, artifact_role=old.evidence_refs[0].limitation,
                                checks=checks, caveat=old.caveat)
    ns = {k: getattr(theory, k) for k in ("Result", "Status", "EvidenceRef", "SensitivityCheck")}
    new = eval(block.strip().rstrip(","), ns)                            # noqa: S307 -- our own code
    assert new == old


@pytest.mark.parametrize("crlf", [True, False])
def test_claims_and_citations_go_where_they_belong(tmp_path, crlf):
    th = _copy(tmp_path, "theory.py", 'X = [\n    Result(\n        id="OLD",\n    ),\n]\n', crlf)
    amend.add_claim(th, '    Result(\n        id="NEW",\n    ),\n', before_id="OLD")
    text = th.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert text.index('id="NEW"') < text.index('id="OLD"')
    tc = _copy(tmp_path, "t.py", 'ACTIVE = (\n            "OLD",\n)\n', crlf)
    amend.allow_citation(tc, "NEW", after_id="OLD", comment="why")
    assert tc.read_bytes().decode("utf-8").replace("\r\n", "\n") == \
        'ACTIVE = (\n            "OLD",\n            # why\n            "NEW",\n)\n'
    with pytest.raises(ValueError, match="expected one"):
        amend.add_claim(th, "x", before_id="MISSING")


def _registration():
    bars = (lib.Bar("Q1", "REACH", "At both cells, x >= 0.5.", (lib.Check(lib.brains("x"), ">=", 0.5),)),
            lib.Bar("Q2", "FEW COLLAPSED", "At both cells, at most 1 brain below 0.2.",
                    (lib.Check(lib.below("x", 0.2), "<=", 1),)))
    return lib.Registration(
        amendment=99, title="a synthetic check", preamble="Why we ask.", seen="A probe saw 0.6.",
        protocol="Twenty brains write and replay.", module="memory_synthetic", runner_key="synthetic",
        tag="synthetic-20261010", cells=(lib.Cell.of(9900, 77, 0.5), lib.Cell.of(13300, 95, 0.4)),
        seeds=tuple(range(5000, 5020)), reference_seeds=tuple(range(5020, 5040)), bars=bars,
        reported="the spread.", interpretation=("Q1 and Q2 pass: it works.",))


def test_a_registration_renders_its_text_judges_and_records():
    reg = _registration()
    assert reg.problems() == []
    text = reg.registration_text("2026-10-10")
    for piece in ("## Amendment 99 (2026-10-10, before running): a synthetic check",
                  "**Seen before registering.** A probe saw 0.6.", "### Protocol", "### Bars",
                  "    Q1  REACH. At both cells, x >= 0.5.", "subject seeds 5000 to 5019",
                  "reference brains 5020 to 5039", "(9900, 77, 0.5) (n/k = 129, tau = 64)",
                  "* A failed bar is recorded as failed and not moved.",
                  "The run is UNJUDGED until Q1 to Q2 are evaluated and recorded below.",
                  "--tag synthetic-20261010 --seeds 5000 ... 5019"):
        assert piece in text, piece
    assert all(len(line) <= 90 for line in text.split("\n"))
    good = [0.7, 0.72, 0.69, 0.71, 0.7] * 4
    obs = {"cells": {"9900/77/0.5": {"x": good}, "13300/95/0.4": {"x": good[:-1] + [0.1]}}}
    judged = reg.judge(obs)
    assert judged["Q1"]["pass"] and judged["Q2"]["pass"]
    result = reg.result_text(judged, "2026-10-11", "abc123", "r.json", "r.log", "Summary.", "It holds.")
    assert "### Amendment 99 result (2026-10-11)" in result and "Q1  REACH" in result and "PASS" in result
    rows = reg.scorecard_rows(judged, {"Q1": ("reach", "x >= 0.5"), "Q2": ("few collapsed", "<= 1 below 0.2")})
    assert rows.startswith("| Q1 reach (A99) | x >= 0.5 | PASS | 0.") and rows.count("\n") == 2


def test_a_registration_on_used_cells_or_brains_has_problems():
    import dataclasses
    reg = dataclasses.replace(_registration(), cells=(lib.Cell.of(10000, 75, 0.48),), seeds=tuple(range(1000, 1020)))
    assert len(reg.problems()) == 2


# ----------------------------------------------------------------- the pre-registration's sources
def test_the_preregistration_is_its_sources_concatenated():
    """PREREG_refraction_memory.md is built from research/notes/memory/prereg/ (amend.Prereg);
    edit the sources and rebuild, never the built file"""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    pre = amend.Prereg(root)
    built = pre.file.read_bytes().decode("utf-8")
    assert pre.text().replace("\r\n", "\n") == built.replace("\r\n", "\n")
    names = pre.order()
    assert names[0] == "A00.md" and "scorecard.md" in names and len(set(names)) == len(names)
    assert sorted(p.name for p in pre.dir.glob("*.md") if p.name != "README.md") == sorted(names)


def test_register_record_and_scorecard_on_a_copy(tmp_path):
    import pathlib
    import shutil
    root = pathlib.Path(__file__).resolve().parents[2]
    src = root / "research" / "notes" / "memory"
    dst = tmp_path / "research" / "notes" / "memory"
    shutil.copytree(src / "prereg", dst / "prereg")
    shutil.copy(src / "PREREG_refraction_memory.md", dst / "PREREG_refraction_memory.md")
    pre = amend.Prereg(tmp_path)
    before = pre.text()
    pre.register(99, "## Amendment 99 (2026-10-11, before running): a test\n\nRegistered.\n")
    pre.record(99, "### Amendment 99 result (2026-10-12)\n\nPassed.\n")
    pre.scorecard("| Q1 test (A99) | >= 1 | PASS | 2 |\n")
    after = pre.file.read_bytes().decode("utf-8")
    assert "\r\n" in after and "\n" not in after.replace("\r\n", "")         # still CRLF throughout
    flat = after.replace("\r\n", "\n")
    assert flat.endswith("\n\n## Amendment 99 (2026-10-11, before running): a test\n\nRegistered.\n\n"
                         "### Amendment 99 result (2026-10-12)\n\nPassed.\n")
    assert flat.index("| Q1 test (A99) |") < flat.index("## Amendment 7 ")
    assert before.replace("\r\n", "\n").rstrip("\n") in flat.replace("| Q1 test (A99) | >= 1 | PASS | 2 |\n", "")
    with pytest.raises(ValueError, match="already registered"):
        pre.register(99, "## Amendment 99 again\n")


def test_the_preregistration_index_is_fresh():
    import pathlib
    pre = amend.Prereg(pathlib.Path(__file__).resolve().parents[2])
    assert (pre.dir / "README.md").read_text(encoding="utf-8") == pre.index()
