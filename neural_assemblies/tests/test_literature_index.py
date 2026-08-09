"""The bibliography and the code must agree about what a citation means.

Docstrings across this repo cite claims as ``[PNAS20]``, ``[HOFF26]``,
``[NEMOREF]``. Until those tags were added to ``research/literature/index.json``
they resolved to nothing -- a citation was a bare string in a comment, so a typo,
a renamed paper, or an invented tag read exactly like a real reference. Nothing
could tell the difference, and nothing ran the index validator either.

These tests are cheap and pin three things: the index is structurally valid, the
citation scanner actually detects a bad tag (rather than approving everything),
and the tags used in code all resolve.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "research" / "literature" / "index.json"
VALIDATOR_PATH = ROOT / "research" / "literature" / "validate_index.py"


def _load_validator():
    """Import validate_index.py by path (research/ is not an importable package)."""
    spec = importlib.util.spec_from_file_location(
        "_validate_index", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    if not VALIDATOR_PATH.is_file():
        pytest.skip("literature validator not present")
    return _load_validator()


@pytest.fixture(scope="module")
def index():
    if not INDEX_PATH.is_file():
        pytest.skip("literature index not present")
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def test_index_passes_validation(validator, index):
    """The whole check, as CI would run it: structure, tags, paths, citations.

    Skipped in a fresh worktree (untracked PDFs absent -- see
    test_declared_local_pdfs_exist); the live tree runs the full check.
    """
    declared = [pdf for e in index["entries"]
                if (pdf := e["urls"].get("local_pdf"))]
    if declared and not any((ROOT / pdf).is_file() for pdf in declared):
        import pytest

        pytest.skip("fresh worktree: untracked PDFs absent; full "
                    "validation runs in the main tree")
    assert validator.main() == 0, (
        "research/literature/index.json failed validation -- run "
        "`python research/literature/validate_index.py` for the reasons"
    )


def test_every_entry_has_a_unique_cite_tag(index):
    records = index["entries"] + index["reference_implementations"]
    tags = [r["cite_tag"] for r in records]
    assert len(tags) == len(set(tags)), f"duplicate cite_tag among {tags}"
    for r in records:
        assert r["cite_tag"], f"{r['id']}: empty cite_tag"


def test_declared_local_pdfs_exist(index):
    """A local_pdf that is not there makes the citation unfollowable.

    PDFs are gitignored, so a FRESH WORKTREE (the suite now runs from
    snapshots) legitimately has none -- that is skipped, not failed. The
    failure this guards is PARTIAL drift: some PDFs present, some
    declared-but-missing, which means the index and the disk disagree.
    """
    declared = [
        (e["id"], pdf)
        for e in index["entries"]
        if (pdf := e["urls"].get("local_pdf"))
    ]
    missing = [(i, pdf) for i, pdf in declared
               if not (ROOT / pdf).is_file()]
    if declared and len(missing) == len(declared):
        import pytest

        pytest.skip("no declared PDFs on disk at all -- fresh worktree "
                    "(PDFs are untracked); drift check needs the main tree")
    assert not missing, f"declared PDFs absent from disk: {missing}"


def test_citations_in_code_resolve(validator, index):
    declared = {r["cite_tag"] for r in
                index["entries"] + index["reference_implementations"]}
    _used, unknown = validator.scan_citations(ROOT, declared)
    assert not unknown, (
        f"code cites tags with no index entry: "
        f"{ {t: v[:2] for t, v in unknown.items()} }"
    )


def test_scanner_detects_an_unknown_citation(validator, tmp_path):
    """NEGATIVE CONTROL -- without this, a scanner that finds nothing at all
    passes every other test in this file.

    Builds a throwaway tree containing one real tag and one fabricated one, and
    requires the scanner to separate them.
    """
    # ASSEMBLED AT RUNTIME so the bracketed form never appears literally in this
    # file. Written out, the scan of the real tree would find it here and
    # test_citations_in_code_resolve would fail on this test's own fixture.
    fake = "BOGUS99"
    real = "PNAS20"
    pkg = tmp_path / "neural_assemblies"
    pkg.mkdir()
    (pkg / "mod.py").write_text(
        f'"""Cites [{real}] which is real and [{fake}] which is not."""\n',
        encoding="utf-8",
    )
    used, unknown = validator.scan_citations(tmp_path, {real})
    assert used == {real}, f"real citation not found: {used}"
    assert fake in unknown, f"fabricated citation not flagged: {unknown}"
    assert "neural_assemblies/mod.py" in unknown[fake]


def test_scanner_ignores_non_citation_bracket_tags(validator, tmp_path):
    """The codebase is full of bracketed tags that are NOT citations -- [AREA],
    [MOOD], [SEMANTIC], [PREDICTION], [P600], [SUBJ]. Flagging those would make
    the check unusable, so the shape rule (>=3 letters + exactly 2 digits) is
    pinned here rather than left to the regex being read correctly.
    """
    pkg = tmp_path / "neural_assemblies"
    pkg.mkdir()
    (pkg / "mod.py").write_text(
        "# [AREA] [HIGH] [MOOD] [SEMANTIC] [CONTEXT] [PREDICTION] [CLASS]\n"
        "# [P600] [N400] [SUBJ] [VERB] [TENSE] [POLARITY] [ROUNDS] [ERROR]\n",
        encoding="utf-8",
    )
    used, unknown = validator.scan_citations(tmp_path, set())
    assert not used and not unknown, (
        f"non-citation bracket tags were treated as citations: "
        f"used={used} unknown={unknown}"
    )
