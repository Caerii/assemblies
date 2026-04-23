from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scientific_status_doc_exists_and_marks_boundaries():
    doc = REPO_ROOT / "docs" / "scientific_status.md"
    assert doc.exists(), "docs/scientific_status.md should exist"

    text = doc.read_text(encoding="utf-8")
    assert "Package-Backed Capabilities" in text
    assert "Heuristic Or Benchmark-Dependent Claims" in text
    assert "Research-Only Or Aspirational Areas" in text
