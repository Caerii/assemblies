"""Source-to-spec navigation is checked independently of semantic correctness."""
from pathlib import Path

import pytest

from research.evidence import specification_links

ROOT = Path(__file__).resolve().parents[2]


def test_reviewed_operations_keep_their_specification_links():
    edges, errors = specification_links(ROOT)
    assert not errors, errors
    linked = {edge["from"].rsplit(":", 1)[-1] for edge in edges}
    assert {"project", "reciprocal_project", "associate", "merge", "pattern_complete",
            "read_only", "AssemblyMemory", "HashedArcFSM", "HashedTransducer"} <= linked


@pytest.mark.parametrize("reference", ["missing.md#contract-test", "spec.md#missing", "../outside.md#x"])
def test_dangling_specifications_are_rejected(tmp_path, reference):
    package = tmp_path / "neural_assemblies"
    package.mkdir()
    (package / "operation.py").write_text(
        f'def operation():\n    """Specification: {reference}"""\n', encoding="utf-8")
    (tmp_path / "spec.md").write_text('<a id="contract-test"></a>', encoding="utf-8")
    edges, errors = specification_links(tmp_path)
    assert len(edges) == len(errors) == 1
