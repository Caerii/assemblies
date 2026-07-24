from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scientific_status_doc_exists_and_marks_boundaries():
    doc = REPO_ROOT / "docs" / "scientific_status.md"
    assert doc.exists(), "docs/scientific_status.md should exist"

    text = doc.read_text(encoding="utf-8")
    # The doc marks claim-strength boundaries under these section headings
    # (renamed from the older verbose forms in "docs: tighten public writing
    # style"): Package Claims (package-backed), Qualified Claims (heuristic /
    # benchmark-dependent), Research Claims (research-only / aspirational).
    assert "Package Claims" in text
    assert "Qualified Claims" in text
    assert "Research Claims" in text
