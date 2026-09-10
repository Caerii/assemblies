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
            "read_only", "project_rounds", "parse_roles_by_reconstruction", "classify_word",
            "classify_word_evidence", "ClassificationEvidence",
            "clone", "ExplicitRound",
            "fork_parser_instance", "_pristine_copy", "_calibrate",
            "resolve_holdout_set",
            "load_backbone_cache", "save_backbone_cache",
            "_sparse_sources_drive_to_explicit",
            "_reset_context_for_bridge", "AssemblyMemory", "HashedArcFSM", "HashedTransducer"} <= linked


@pytest.mark.parametrize("reference", ["missing.md#contract-test", "spec.md#missing", "../outside.md#x"])
def test_dangling_specifications_are_rejected(tmp_path, reference):
    package = tmp_path / "neural_assemblies"
    package.mkdir()
    (package / "operation.py").write_text(
        f'def operation():\n    """Specification: {reference}"""\n', encoding="utf-8")
    (tmp_path / "spec.md").write_text('<a id="contract-test"></a>', encoding="utf-8")
    edges, errors = specification_links(tmp_path)
    assert len(edges) == len(errors) == 1


def test_rust_ir_source_links_its_wire_contract():
    edges, errors = specification_links(ROOT)
    assert not errors, errors
    assert any(edge["from"] == "neural_assemblies/ir/rust/lib.rs:<module>"
               and edge["to"].endswith("#contract-protocol-wire") for edge in edges)


def test_lean_domain_links_its_checked_execution_contract():
    edges, errors = specification_links(ROOT)
    assert not errors, errors
    assert {"from": "formal/AssemblyIR/Domain.lean:<module>",
            "to": "neural_assemblies/ir/VERIFICATION.md#contract-checked-domain"} in edges


def test_dangling_lean_contract_is_rejected(tmp_path):
    formal = tmp_path / "formal" / "AssemblyIR"
    formal.mkdir(parents=True)
    (formal / "Domain.lean").write_text(
        "/-!\nSpecification: missing.md#contract-domain\n-/\n", encoding="utf-8")
    edges, errors = specification_links(tmp_path)
    assert len(edges) == len(errors) == 1
    assert "dangling specification file" in errors[0]
