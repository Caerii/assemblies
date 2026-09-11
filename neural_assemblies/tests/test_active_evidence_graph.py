"""New runner evidence must be valid and navigable from its own registration."""

from research.evidence import validate_active_evidence_graph


def test_tracked_runner_evidence_has_no_dangling_edges():
    errors = validate_active_evidence_graph()
    assert not errors, "\n".join(errors)
